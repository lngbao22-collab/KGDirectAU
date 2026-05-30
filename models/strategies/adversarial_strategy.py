"""Adversarial strategy for RotatE-style training."""

from __future__ import annotations

from typing import Iterable

import torch

from models.losses.adversarial_bce_loss import compute_adversarial_bce_loss
from models.samplers.filtered_1_to_n_sampler import FilteredSubsampler


class AdversarialStrategy:
    def __init__(self, encoder, args, all_train_triples):
        self.encoder = encoder
        self.args = args
        lr = getattr(args, "lr", getattr(args, "learning_rate", 5e-5))
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

        nentity = getattr(args, "nentity", getattr(args, "ent_total", None))
        if nentity is None and hasattr(encoder, "entity_embedding"):
            nentity = encoder.entity_embedding.size(0)
        if nentity is None:
            raise ValueError("`nentity` or `ent_total` is required for FilteredSubsampler")

        self.sampler = FilteredSubsampler(all_train_triples, int(nentity), int(getattr(args, "n_sample", 1)))

    def train_epoch(self, dataloader: Iterable) -> float:
        """Train the model for one epoch and return the average training loss."""

        self.encoder.train()
        total_loss = 0.0
        step = 0
        modes = ["head-batch", "tail-batch"]

        for batch in dataloader:
            self.optimizer.zero_grad()

            current_mode = modes[step % 2]
            step += 1

            pos_sample, neg_sample, weights, current_mode = self.sampler.sample(batch, current_mode)

            device = next(self.encoder.parameters()).device
            pos_sample = pos_sample.to(device)
            neg_sample = neg_sample.to(device)
            weights = weights.to(device)

            outputs = self.encoder(pos_sample, neg_sample, current_mode)
            pos_scores = outputs["positive_scores"]
            neg_scores = outputs["negative_scores"]

            loss = compute_adversarial_bce_loss(
                pos_scores,
                neg_scores,
                getattr(self.args, "adversarial_temp", getattr(self.args, "adversarial_temperature", 1.0)),
                weights,
            )

            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
        return total_loss / max(step, 1)

Strategy = AdversarialStrategy
