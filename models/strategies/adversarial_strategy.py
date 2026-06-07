"""Adversarial strategy for RotatE-style training."""

from __future__ import annotations

import os
import time
from typing import Iterable

import torch

from base.evaluator import Evaluator
from data.dataset import Example, load_data, reverse_triplet
from data.dict_hub import get_entity_dict
from models.losses.adversarial_bce_loss import compute_adversarial_bce_loss
from models.samplers.filtered_1_to_n_sampler import FilteredSubsampler
from utils.checkpoint import best_model_path, delete_old_ckt, last_model_path, save_checkpoint
from utils.device import get_model_obj
from utils.logger import logger


class AdversarialStrategy:
    """Standalone adversarial RotatE training loop with evaluation and checkpointing."""

    def __init__(self, encoder, args, all_train_triples):
        self.encoder = encoder
        self.args = args
        self.all_train_triples = all_train_triples.long()
        self.entity_dict = get_entity_dict()
        self.entity_ids = [ex.entity_id for ex in self.entity_dict.entity_exs]
        self.evaluator = Evaluator(args)
        self.best_metric = None
        self.best_checkpoint_path = None
        self.train_time = 0.0
        self.valid_time = 0.0
        self.total_time = 0.0

        self.base_lr = float(getattr(args, "lr", getattr(args, "learning_rate", 5e-5)))
        weight_decay = getattr(args, "weight_decay", 0.0)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.base_lr, weight_decay=weight_decay)
        self.global_step = 0
        self.next_lr_decay_step = getattr(args, "warm_up_steps", None)
        if self.next_lr_decay_step is not None:
            self.next_lr_decay_step = int(self.next_lr_decay_step)
        self.lr_decay_factor = float(getattr(args, "lr_decay_factor", 0.1))
        self.max_steps = getattr(args, "max_steps", None)
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
        self.shuffle_train = bool(getattr(args, "shuffle_train", False))

        nentity = getattr(args, "nentity", getattr(args, "ent_total", None))
        if nentity is None and hasattr(encoder, "entity_embedding"):
            nentity = encoder.entity_embedding.size(0)
        if nentity is None:
            raise ValueError("`nentity` or `ent_total` is required for FilteredSubsampler")

        self.sampler = FilteredSubsampler(self.all_train_triples, int(nentity), int(getattr(args, "n_sample", 1)))

        if torch.cuda.is_available():
            self.encoder.cuda()
        self.device = next(self.encoder.parameters()).device

    def _maybe_decay_learning_rate(self) -> None:
        """Decay the optimizer learning rate at configured step boundaries."""

        if self.next_lr_decay_step is None or self.global_step < self.next_lr_decay_step:
            return

        decay_factor = max(self.lr_decay_factor, 0.0)
        new_lr = self.optimizer.param_groups[0]["lr"] * decay_factor
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        logger.info(
            "Change learning rate to %.8f at step %d",
            new_lr,
            self.global_step,
        )
        self.next_lr_decay_step = int(self.next_lr_decay_step * 3)

    def _iter_train_batches(self, epoch: int):
        """Iterate over tensorized train triples using configured batch size."""

        batch_size = max(getattr(self.args, "batch_size", 1024), 1)
        triples = self.all_train_triples
        if self.shuffle_train:
            generator = torch.Generator()
            generator.manual_seed(int(getattr(self.args, "seed", 0) or 0) + int(epoch))
            permutation = torch.randperm(triples.size(0), generator=generator)
            triples = triples[permutation]
        for start in range(0, len(triples), batch_size):
            end = start + batch_size
            yield triples[start:end]

    def _extract_monitor_value(self, metric_dict, train_loss):
        """Prefer validation MRR for checkpointing, fallback to negative train loss."""

        if metric_dict and "mrr" in metric_dict:
            return metric_dict["mrr"]
        return -float(train_loss)

    def train_epoch(self, dataloader: Iterable, epoch: int) -> float:
        """Train the model for one epoch and return the average training loss."""

        self.encoder.train()
        total_loss = 0.0
        step = 0
        modes = ["head-batch", "tail-batch"]

        for batch in dataloader:
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break

            self.optimizer.zero_grad()
            current_mode = modes[step % 2]
            step += 1

            pos_sample, neg_sample, weights, current_mode = self.sampler.sample(batch, current_mode)

            pos_sample = pos_sample.to(self.device)
            neg_sample = neg_sample.to(self.device)
            weights = weights.to(self.device)

            outputs = self.encoder(pos_sample, neg_sample, current_mode)
            pos_scores = outputs["positive_scores"]
            neg_scores = outputs["negative_scores"]

            adv_temp = getattr(self.args, "adversarial_temp", getattr(self.args, "adversarial_temperature", 1.0))
            loss = compute_adversarial_bce_loss(pos_scores, neg_scores, adv_temp, weights)

            loss.backward()
            self.optimizer.step()
            self.global_step += 1
            self._maybe_decay_learning_rate()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(step, 1)
        logger.info("[EPOCH %s] Train | Loss: %.4f", epoch + 1, avg_loss)
        return avg_loss

    @torch.no_grad()
    def eval_epoch(self, epoch):
        """Evaluate the model using the common evaluator path."""

        self.encoder.eval()
        metric_dict = {}
        valid_path = getattr(self.args, "valid_path", "")

        if valid_path and os.path.exists(valid_path):
            valid_exs = load_data(valid_path, add_forward_triplet=True, add_backward_triplet=False)
            if getattr(get_model_obj(self.encoder), "bidirectional_score_batch", False):
                valid_backward_exs = valid_exs
            else:
                valid_backward_exs = [
                    Example(**reverse_triplet({
                        "head_id": ex.head_id,
                        "head": ex.head,
                        "relation": ex.relation,
                        "tail_id": ex.tail_id,
                        "tail": ex.tail,
                    }))
                    for ex in valid_exs
                ]

            valid_output_path = os.path.join(self.args.output_dir, "valid_link_prediction.log")
            forward_metrics = self.evaluator.evaluate_link_prediction_inplace(
                self.encoder,
                valid_path,
                self.entity_dict,
                valid_output_path,
                eval_forward=True,
                examples=valid_exs,
            )
            backward_metrics = self.evaluator.evaluate_link_prediction_inplace(
                self.encoder,
                valid_path,
                self.entity_dict,
                valid_output_path,
                eval_forward=False,
                examples=valid_backward_exs,
            )

            if forward_metrics and backward_metrics:
                mrr = (forward_metrics.get("mrr", 0.0) + backward_metrics.get("mrr", 0.0)) / 2
                metric_dict["mrr"] = mrr
                logger.info("[EPOCH %s] Valid | MRR: %.4f", epoch + 1, mrr)

        return metric_dict

    def train_loop(self):
        """Run full adversarial training with periodic evaluation and checkpointing."""

        total_start = time.time()
        max_epochs = max(getattr(self.args, "epochs", 1), 1)
        for epoch in range(max_epochs):
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break

            train_start = time.time()
            train_loss = self.train_epoch(self._iter_train_batches(epoch), epoch)
            self.train_time += time.time() - train_start

            eval_start = time.time()
            metric_dict = self.eval_epoch(epoch)
            self.valid_time += time.time() - eval_start

            monitor_value = self._extract_monitor_value(metric_dict, train_loss)
            is_best = self.best_metric is None or monitor_value > self.best_metric.get("score", float("-inf"))
            if is_best:
                self.best_metric = {"score": monitor_value, "metrics": metric_dict, "epoch": epoch}

            saved_checkpoint_path = save_checkpoint(
                {
                    "epoch": epoch,
                    "best_epoch": epoch if is_best else None,
                    "best_metric": self.best_metric,
                    "args": self.args.__dict__,
                    "state_dict": get_model_obj(self.encoder).state_dict(),
                },
                is_best=is_best,
                filename=last_model_path(self.args.output_dir),
            )
            if is_best:
                self.best_checkpoint_path = best_model_path(self.args.output_dir)
            elif self.best_checkpoint_path is None:
                self.best_checkpoint_path = saved_checkpoint_path
            delete_old_ckt(path_pattern="{}/checkpoint_*.mdl".format(self.args.output_dir), keep=getattr(self.args, "max_to_keep", 5))

            if self.max_steps is not None and self.global_step >= self.max_steps:
                break

        self.total_time = time.time() - total_start
        return {
            "best_epoch": None if self.best_metric is None else self.best_metric.get("epoch", 0) + 1,
            "best_mrr": None if self.best_metric is None else self.best_metric.get("score"),
            "train_time": self.train_time,
            "valid_time": self.valid_time,
            "total_time": self.total_time,
        }


Strategy = AdversarialStrategy
