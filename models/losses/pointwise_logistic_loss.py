"""Pointwise logistic (softplus) loss for DaBR."""

import torch.nn.functional as F
import torch


def compute_softplus_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """DaBR uses Softplus on (scores * labels) where labels are +1 / -1."""

    scores = scores.view(-1)
    labels = labels.view(-1).to(scores.device)
    return F.softplus(scores * labels).mean()
