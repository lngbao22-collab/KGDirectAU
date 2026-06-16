"""Shared loss utilities for softmax-family and BCE-family objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _ensure_index_targets(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(targets):
        return torch.as_tensor(targets, device=scores.device, dtype=torch.long)
    return targets.to(device=scores.device, dtype=torch.long)


def _labels_as_matrix(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Convert index labels to a one-hot/multi-hot matrix when needed."""

    if labels.dim() == 2:
        return labels.to(device=scores.device, dtype=scores.dtype)
    index_labels = labels.long().to(scores.device)
    matrix = torch.zeros(scores.shape, device=scores.device, dtype=scores.dtype)
    matrix[torch.arange(scores.size(0), device=scores.device), index_labels] = 1.0
    return matrix


def compute_softmax_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute CE/KL-style softmax loss from hard or soft targets."""

    if targets.dim() == 1:
        index_targets = _ensure_index_targets(scores, targets)
        return F.cross_entropy(scores, index_targets, reduction=reduction)

    target_matrix = targets.to(device=scores.device, dtype=scores.dtype)
    log_probs = F.log_softmax(scores, dim=1)
    target_probs = F.normalize(target_matrix, p=1, dim=1)
    return F.kl_div(log_probs, target_probs, reduction=reduction)


def compute_bce_loss(
    scores: torch.Tensor,
    targets: torch.Tensor | None = None,
    *,
    pos_scores: torch.Tensor | None = None,
    neg_scores: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    offset: float = 0.0,
    reduction: str = "mean",
    adversarial_temp: float | None = None,
) -> torch.Tensor:
    """Compute BCE loss for broadcast or negative-sampling training.

    Broadcast mode:
        - Provide ``scores`` and ``targets``.
    Negative-sampling mode:
        - Provide ``pos_scores`` and ``neg_scores`` (or ``scores`` as ``pos_scores``).
        - Optional ``weights`` for weighted batch averaging.
        - Optional ``adversarial_temp`` for adversarial negative weighting.
    """

    # Negative-sampling mode
    if neg_scores is not None or pos_scores is not None:
        if pos_scores is None:
            pos_scores = scores
        if pos_scores is None or neg_scores is None:
            raise ValueError("Negative-sampling BCE requires both pos_scores and neg_scores")

        pos_scores = pos_scores.reshape(-1)
        if neg_scores.dim() == 3:
            neg_scores = neg_scores.squeeze(-1)
        neg_scores = neg_scores.to(pos_scores.device)
        batch_size = max(pos_scores.size(0), 1)

        if offset != 0.0:
            pos_scores = pos_scores + offset
            neg_scores = neg_scores + offset

        if adversarial_temp is not None:
            if weights is None:
                weights = torch.ones_like(pos_scores)
            weights = weights.to(pos_scores.device).reshape(-1)
            pos_loss = -F.logsigmoid(pos_scores)
            neg_weights = F.softmax(neg_scores * adversarial_temp, dim=-1).detach()
            neg_loss = -(neg_weights * F.logsigmoid(-neg_scores)).sum(dim=-1)
            per_row = (pos_loss + neg_loss) / 2.0
            return (per_row * weights).sum() / weights.sum().clamp_min(1e-12)

        scores_mat = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        labels = torch.zeros_like(scores_mat)
        labels[:, 0] = 1.0
        per_row = F.binary_cross_entropy_with_logits(scores_mat, labels, reduction="none").sum(dim=1)
        if weights is not None:
            weights = weights.to(scores_mat.device).reshape(-1)
            return (per_row * weights).sum() / weights.sum().clamp_min(1e-12)
        return per_row.sum() / batch_size

    # Broadcast mode
    if scores is None or targets is None:
        raise ValueError("Broadcast BCE requires scores and targets")
    target_matrix = _labels_as_matrix(scores, targets)
    if offset != 0.0:
        scores = scores + offset
    return F.binary_cross_entropy_with_logits(scores, target_matrix, reduction=reduction)
