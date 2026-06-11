"""Standard binary cross entropy loss for KvsAll and multi-hot targets (LibKGE ``bce`` loss)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _labels_as_matrix(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
	"""Convert index labels to a multi-hot matrix when needed."""

	if labels.dim() == 2:
		return labels.to(device=scores.device, dtype=scores.dtype)
	index_labels = labels.long().to(scores.device)
	matrix = torch.zeros(scores.shape, device=scores.device, dtype=scores.dtype)
	matrix[torch.arange(scores.size(0), device=scores.device), index_labels] = 1.0
	return matrix


def compute_bce_loss(
	scores: torch.Tensor,
	targets: torch.Tensor,
	*,
	offset: float = 0.0,
	reduction: str = 'mean',
) -> torch.Tensor:
	"""Compute BCE-with-logits for multi-hot or index targets.

	:param scores: ``[batch_size, num_entities]`` or ``[batch_size, 1 + num_neg]`` logits
	:param targets: Same shape as ``scores`` (multi-hot) or ``[batch_size]`` entity indices
	:param offset: Optional score offset (LibKGE ``train.loss_arg`` for BCE)
	:param reduction: ``mean``, ``sum``, or ``none``
	"""

	target_matrix = _labels_as_matrix(scores, targets)
	if offset != 0.0:
		scores = scores + offset
	return F.binary_cross_entropy_with_logits(scores, target_matrix, reduction=reduction)


def build_bce_loss_fn(args):
	"""Factory for standard BCE-with-logits training."""

	offset = getattr(args, 'bce_offset', None)
	if offset is None:
		raw = getattr(args, 'loss_arg', None)
		offset = 0.0 if raw is None or (isinstance(raw, float) and math.isnan(raw)) else float(raw)
	reduction = str(getattr(args, 'bce_reduction', 'mean'))

	def loss_fn(scores: torch.Tensor, targets: torch.Tensor, **_kwargs) -> torch.Tensor:
		return compute_bce_loss(scores, targets, offset=offset, reduction=reduction)

	return loss_fn


build_kvsall_loss_fn = build_bce_loss_fn
build_loss_fn = build_bce_loss_fn


def compute_loss(args):
	return build_bce_loss_fn(args)
