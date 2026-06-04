"""Pointwise training strategy for DaBR training."""

import os
import torch
from torch import optim
from base.evaluator import Evaluator
from data.dict_hub import get_entity_dict
from data.dataset import load_data, Example, reverse_triplet
from utils.checkpoint import best_model_path, last_model_path, save_checkpoint
from utils.device import get_model_obj
from utils.logger import logger

from models.samplers.uniform_pointwise_sampler import get_pointwise_negatives
from models.losses.pointwise_logistic_loss import compute_softplus_loss


class PointwiseStrategy:
    """Pointwise training loop for DaBR KG encoders."""

    def __init__(self, encoder: torch.nn.Module, args):
        self.encoder = encoder
        self.args = args
        self.evaluator = Evaluator(args)
        self.entity_dict = get_entity_dict()
        self.entity_ids = [ex.entity_id for ex in self.entity_dict.entity_exs]
        self.best_metric = None
        self.best_checkpoint_path = None
        self.train_time = 0.0
        self.valid_time = 0.0
        self.total_time = 0.0

        lr = getattr(args, 'lr', getattr(args, 'learning_rate', 0.1))
        optim_name = getattr(args, 'optim', 'sgd').lower()

        if optim_name == 'adam':
            self.optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        elif optim_name == 'adagrad':
            self.optimizer = optim.Adagrad(self.encoder.parameters(), lr=lr)
        else:
            self.optimizer = optim.SGD(self.encoder.parameters(), lr=lr)

        if torch.cuda.is_available():
            self.encoder.cuda()
        self.device = next(self.encoder.parameters()).device
        self._valid_forward_exs = None
        self._valid_backward_exs = None

    def _get_valid_examples(self):
        """Load and cache validation examples (forward + optional backward)."""

        valid_path = getattr(self.args, 'valid_path', '')
        if not valid_path or not os.path.exists(valid_path):
            return None, None
        if self._valid_forward_exs is None:
            self._valid_forward_exs = load_data(valid_path, add_forward_triplet=True, add_backward_triplet=False)
            self._valid_backward_exs = [
                Example(**reverse_triplet({
                    'head_id': ex.head_id,
                    'head': ex.head,
                    'relation': ex.relation,
                    'tail_id': ex.tail_id,
                    'tail': ex.tail,
                }))
                for ex in self._valid_forward_exs
            ]
        return self._valid_forward_exs, self._valid_backward_exs

    def train_epoch(self, dataloader, epoch: int) -> float:
        self.encoder.train()
        total_loss = 0.0
        step = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            ent_total = self.encoder.ent_embeddings.num_embeddings

            sampled = get_pointwise_negatives(batch, getattr(self.args, 'n_sample', 1), int(ent_total))
            for key in sampled:
                sampled[key] = sampled[key].to(self.device)

            outputs = self.encoder(sampled)
            scores = outputs['scores']
            labels = sampled['labels']

            base_loss = compute_softplus_loss(scores, labels)

            h, t = outputs['ent_emb']
            r, dr = outputs['rel_emb']
            reg_ent = self.encoder.regularization(h) + self.encoder.regularization(t)
            reg_rel = self.encoder.regularization(r) + self.encoder.regularization(dr)

            entity_reg_weight = getattr(self.args, 'entity_reg_weight', 0.0)
            relation_reg_weight = getattr(self.args, 'relation_reg_weight', 0.0)
            total = base_loss + (entity_reg_weight * reg_ent) + (relation_reg_weight * reg_rel)

            total.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
            self.optimizer.step()
            total_loss += total.item()
            step += 1

        avg_loss = total_loss / max(step, 1)
        logger.info(f"[EPOCH {epoch + 1}] Train | Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def eval_epoch(self, epoch):
        import time

        self.encoder.eval()
        metric_dict = {}
        valid_exs, valid_backward_exs = self._get_valid_examples()
        if not valid_exs:
            return metric_dict

        valid_path = getattr(self.args, 'valid_path', '')
        valid_output_path = os.path.join(self.args.output_dir, 'valid_link_prediction.log')

        eval_start = time.time()
        forward_metrics = self.evaluator.evaluate_link_prediction_inplace(
            self.encoder, valid_path, self.entity_dict, valid_output_path,
            eval_forward=True, examples=valid_exs,
        )
        backward_metrics = self.evaluator.evaluate_link_prediction_inplace(
            self.encoder, valid_path, self.entity_dict, valid_output_path,
            eval_forward=False, examples=valid_backward_exs,
        )
        eval_seconds = time.time() - eval_start

        if forward_metrics and backward_metrics:
            mrr = (forward_metrics.get('mrr', 0) + backward_metrics.get('mrr', 0)) / 2
            metric_dict['mrr'] = mrr
            logger.info(
                '[EPOCH %s] Valid | MRR: %.4f (fwd=%.4f, bwd=%.4f, eval_s=%.1f)',
                epoch + 1, mrr, forward_metrics.get('mrr', 0.0), backward_metrics.get('mrr', 0.0), eval_seconds,
            )

        return metric_dict

    def _validation_interval(self) -> int:
        """Epochs between validation runs (``epoch_per_eval``; 0 or unset → every epoch)."""

        raw = getattr(self.args, 'epoch_per_eval', None)
        interval = int(raw) if raw is not None else 1
        if interval <= 0:
            return 1
        return interval

    def _should_evaluate(self, epoch: int, total_epochs: int) -> bool:
        """Return True when validation should run after this epoch."""

        interval = self._validation_interval()
        epoch_number = epoch + 1
        return epoch_number % interval == 0 or epoch_number >= total_epochs

    def train_loop(self, train_dataloader):
        import time

        total_epochs = max(getattr(self.args, 'epochs', 1), 1)
        total_start = time.time()
        for epoch in range(total_epochs):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_dataloader, epoch)
            self.train_time += time.time() - epoch_start

            metric_dict = {}
            if self._should_evaluate(epoch, total_epochs):
                eval_start = time.time()
                metric_dict = self.eval_epoch(epoch)
                self.valid_time += time.time() - eval_start

            monitor_value = metric_dict.get('mrr', -train_loss)
            is_best = self.best_metric is None or monitor_value > self.best_metric.get('score', float('-inf'))
            if is_best:
                self.best_metric = {'score': monitor_value, 'metrics': metric_dict, 'epoch': epoch}

            saved_checkpoint_path = save_checkpoint(
                {
                    'epoch': epoch,
                    'best_epoch': epoch if is_best else None,
                    'best_metric': self.best_metric,
                    'args': self.args.__dict__,
                    'state_dict': get_model_obj(self.encoder).state_dict(),
                },
                is_best=is_best,
                filename=last_model_path(self.args.output_dir),
            )
            if is_best:
                self.best_checkpoint_path = best_model_path(self.args.output_dir)
            elif self.best_checkpoint_path is None:
                self.best_checkpoint_path = saved_checkpoint_path

        self.total_time = time.time() - total_start
        if self.best_checkpoint_path is None or not os.path.exists(self.best_checkpoint_path):
            self.best_checkpoint_path = last_model_path(self.args.output_dir)
        return {
            'best_epoch': None if self.best_metric is None else self.best_metric.get('epoch', 0) + 1,
            'best_mrr': None if self.best_metric is None else self.best_metric.get('score'),
            'train_time': self.train_time,
            'valid_time': self.valid_time,
            'total_time': self.total_time,
            'best_checkpoint_path': self.best_checkpoint_path,
        }

Strategy = PointwiseStrategy
