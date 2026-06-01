"""Pointwise training strategy for DaBR training."""

import os
import torch
from torch import optim
from base.evaluator import Evaluator
from data.dict_hub import get_entity_dict
from data.dataset import load_data, Example, reverse_triplet
from utils.checkpoint import save_checkpoint, best_model_path
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

    def train_epoch(self, dataloader) -> float:
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

            total = base_loss + (getattr(self.args, 'lmbda', getattr(self.args, 'lam', 0.0)) * reg_ent) + (getattr(self.args, 'lmbda_two', getattr(self.args, 'lmbda2', 0.0)) * reg_rel)

            total.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
            self.optimizer.step()
            total_loss += total.item()
            step += 1

        avg_loss = total_loss / max(step, 1)
        logger.info(f"Train | Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def eval_epoch(self, epoch):
        self.encoder.eval()
        metric_dict = {}
        valid_path = getattr(self.args, 'valid_path', '')

        if valid_path and os.path.exists(valid_path):
            valid_exs = load_data(valid_path, add_forward_triplet=True, add_backward_triplet=False)
            valid_backward_exs = [
                Example(**reverse_triplet({'head_id': ex.head_id, 'head': ex.head, 'relation': ex.relation, 'tail_id': ex.tail_id, 'tail': ex.tail}))
                for ex in valid_exs
            ]

            valid_output_path = os.path.join(self.args.output_dir, 'valid_link_prediction.log')
            forward_metrics = self.evaluator.evaluate_link_prediction_inplace(self.encoder, valid_path, self.entity_dict, valid_output_path, eval_forward=True, examples=valid_exs)
            backward_metrics = self.evaluator.evaluate_link_prediction_inplace(self.encoder, valid_path, self.entity_dict, valid_output_path, eval_forward=False, examples=valid_backward_exs)

            if forward_metrics and backward_metrics:
                mrr = (forward_metrics.get('mrr', 0) + backward_metrics.get('mrr', 0)) / 2
                metric_dict['mrr'] = mrr
                logger.info(f"[EPOCH {epoch}] Valid | MRR: {mrr:.4f}")

        return metric_dict

    def train_loop(self, train_dataloader):
        import time

        total_start = time.time()
        for epoch in range(getattr(self.args, 'epochs', 1)):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_dataloader)
            self.train_time += time.time() - epoch_start

            if (epoch + 1) % getattr(self.args, 'eval_every_n_step', 50) == 0:
                eval_start = time.time()
                metric_dict = self.eval_epoch(epoch)
                self.valid_time += time.time() - eval_start

                monitor_value = metric_dict.get('mrr', -train_loss)
                is_best = self.best_metric is None or monitor_value > self.best_metric.get('score', float('-inf'))
                if is_best:
                    self.best_metric = {'score': monitor_value, 'epoch': epoch}
                    saved_checkpoint_path = save_checkpoint({'state_dict': self.encoder.state_dict(), 'args': self.args.__dict__}, is_best=True, filename=best_model_path(self.args.output_dir))
                    self.best_checkpoint_path = best_model_path(self.args.output_dir)
                elif self.best_checkpoint_path is None:
                    self.best_checkpoint_path = best_model_path(self.args.output_dir)

        self.total_time = time.time() - total_start
        return {
            'best_epoch': None if self.best_metric is None else self.best_metric.get('epoch', 0) + 1,
            'best_mrr': None if self.best_metric is None else self.best_metric.get('score'),
            'train_time': self.train_time,
            'valid_time': self.valid_time,
            'total_time': self.total_time,
            'best_checkpoint_path': self.best_checkpoint_path,
        }

Strategy = PointwiseStrategy
