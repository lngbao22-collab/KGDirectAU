"""Standard binary cross entropy loss for KvsAll and multi-hot targets (LibKGE ``bce`` loss)."""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F


def uses_bce_logit_offset(args) -> bool:
	"""Return True when inference should add the LibKGE BCE logit offset to raw scores."""

	loss_path = str(getattr(args, 'model_loss_path', '') or '').lower()
	basename = os.path.basename(loss_path)
	return basename in {'bce_loss.py', 'bce_loss'}


def bce_logit_offset(args) -> float:
	"""Return the BCE logit offset (LibKGE ``train.loss_arg``) for inference."""

	if not uses_bce_logit_offset(args):
		return 0.0
	offset = getattr(args, 'bce_offset', None)
	if offset is not None:
		return float(offset)
	raw = getattr(args, 'loss_arg', None)
	if raw is None or (isinstance(raw, float) and math.isnan(raw)):
		return 0.0
	return float(raw)


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

	offset = bce_logit_offset(args)
	reduction = str(getattr(args, 'bce_reduction', 'mean'))

	def loss_fn(scores: torch.Tensor, targets: torch.Tensor, **_kwargs) -> torch.Tensor:
		return compute_bce_loss(scores, targets, offset=offset, reduction=reduction)

	return loss_fn


build_kvsall_loss_fn = build_bce_loss_fn
build_loss_fn = build_bce_loss_fn


def build_negsamp_loss_fn(args):
	"""Factory for LibKGE triple negative-sampling BCE (``train.loss: bce``)."""

	offset = bce_logit_offset(args)

	def loss_fn(pos_scores: torch.Tensor, neg_scores: torch.Tensor, weights=None, **_kwargs) -> torch.Tensor:
		pos_scores = pos_scores.reshape(-1)
		if neg_scores.dim() == 3:
			neg_scores = neg_scores.squeeze(-1)
		batch_size = max(pos_scores.size(0), 1)
		scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
		labels = torch.zeros_like(scores)
		labels[:, 0] = 1.0
		if offset != 0.0:
			scores = scores + offset
		per_row = F.binary_cross_entropy_with_logits(scores, labels, reduction='none').sum(dim=1)
		if weights is not None:
			weights = weights.to(scores.device).reshape(-1)
			return (per_row * weights).sum() / weights.sum().clamp_min(1e-12)
		return per_row.sum() / batch_size

	return loss_fn


def compute_loss(args):
	return build_bce_loss_fn(args)
