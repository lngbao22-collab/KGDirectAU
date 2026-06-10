"""Adversarial negative-sampling BCE loss for RotatE training."""

import torch
import torch.nn.functional as F


def compute_adversarial_bce_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    adversarial_temp: float,
    subsampling_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted RotatE-style adversarial BCE loss."""

    pos_scores = pos_scores.squeeze(-1)
    if subsampling_weight is None:
        subsampling_weight = torch.ones_like(pos_scores)
    subsampling_weight = subsampling_weight.to(pos_scores.device)

    pos_loss = -F.logsigmoid(pos_scores)

    neg_weights = F.softmax(neg_scores * adversarial_temp, dim=-1).detach()
    neg_loss = -(neg_weights * F.logsigmoid(-neg_scores)).sum(dim=-1)

    total = (subsampling_weight * (pos_loss + neg_loss) / 2.0).sum() / subsampling_weight.sum().clamp_min(1e-12)
    return total


def build_negsamp_loss_fn(args):
    """Factory for adversarial negative-sampling training."""

    temp = float(
        getattr(args, 'adversarial_temp', None)
        or getattr(args, 'adversarial_temperature', 1.0)
        or 1.0
    )

    def loss_fn(pos_scores, neg_scores, weights=None, **_kwargs):
        return compute_adversarial_bce_loss(pos_scores, neg_scores, temp, weights)

    return loss_fn


build_loss_fn = build_negsamp_loss_fn


def compute_loss(args):
	return build_negsamp_loss_fn(args)
