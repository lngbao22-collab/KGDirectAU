"""KL divergence loss for KvsAll multi-hot targets (LibKGE ``kl`` loss)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_kl_loss(
	scores: torch.Tensor,
	targets: torch.Tensor,
	*,
	reduction: str = 'sum',
) -> torch.Tensor:
	"""Compute KL divergence between softmax scores and a label distribution.

	For index targets (1vsAll), this reduces to cross-entropy with ``reduction``.

	For multi-hot / smoothed KvsAll labels, matches LibKGE ``KLDivWithSoftmaxKgeLoss``.
	"""

	if targets.dim() == 1:
		index_targets = targets.long().to(device=scores.device)
		return F.cross_entropy(scores, index_targets, reduction=reduction)

	target_matrix = targets.to(device=scores.device, dtype=scores.dtype)
	log_probs = F.log_softmax(scores, dim=1)
	target_probs = F.normalize(target_matrix, p=1, dim=1)
	return F.kl_div(log_probs, target_probs, reduction=reduction)


def build_kl_loss_fn(args):
	"""Factory for LibKGE-style KvsAll KL training (sum reduction; divide by batch_size in strategy)."""

	reduction = str(getattr(args, 'kl_reduction', 'sum'))

	def loss_fn(scores: torch.Tensor, targets: torch.Tensor, **_kwargs) -> torch.Tensor:
		return compute_kl_loss(scores, targets, reduction=reduction)

	return loss_fn


build_kvsall_loss_fn = build_kl_loss_fn
build_loss_fn = build_kl_loss_fn


def compute_loss(args):
	return build_kl_loss_fn(args)
