"""Pointwise training strategy for DaBR training."""

import torch
from torch import optim
from models.samplers.pointwise_sampler import get_pointwise_negatives
from models.losses.softplus_loss import compute_softplus_loss


class PointwiseStrategy:
    """Pointwise training loop for DaBR KG encoders."""

    def __init__(self, encoder: torch.nn.Module, args):
        self.encoder = encoder
        self.args = args
        lr = getattr(args, 'lr', getattr(args, 'learning_rate', 0.1))
        self.optimizer = optim.SGD(self.encoder.parameters(), lr=lr)

    def train_epoch(self, dataloader) -> float:
        """Train the model for one epoch and return the average training loss."""

        self.encoder.train()
        total_loss = 0.0
        for batch in dataloader:
            self.optimizer.zero_grad()
            ent_total = getattr(self.args, 'ent_total', getattr(self.args, 'entTotal', None))
            if ent_total is None:
                # try derive from encoder embeddings
                try:
                    ent_total = self.encoder.ent_embeddings.num_embeddings
                except Exception:
                    raise ValueError('Number of entities (ent_total) is required in args or encoder')

            sampled = get_pointwise_negatives(batch, getattr(self.args, 'n_sample', 1), int(ent_total))
            outputs = self.encoder(sampled)
            scores = outputs['scores']
            labels = sampled['labels'].to(scores.device)

            base_loss = compute_softplus_loss(scores, labels)

            h, t = outputs['ent_emb']
            r, dr = outputs['rel_emb']
            reg_ent = self.encoder.regularization(h) + self.encoder.regularization(t)
            reg_rel = self.encoder.regularization(r) + self.encoder.regularization(dr)

            total = base_loss + (getattr(self.args, 'lmbda', getattr(self.args, 'lam', 0.0)) * reg_ent) + (getattr(self.args, 'lmbda_two', getattr(self.args, 'lmbda2', 0.0)) * reg_rel)

            total.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
            self.optimizer.step()

            total_loss += total.item() * (batch['head_id'].size(0) if isinstance(batch['head_id'], torch.Tensor) else 1)

        return total_loss
