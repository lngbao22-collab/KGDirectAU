"""Standard binary cross entropy loss for KvsAll and multi-hot targets (``bce`` loss)."""

import math
import os

import torch
from models.losses.loss_utilities import compute_bce_loss as _compute_bce_loss


def uses_bce_logit_offset(args) -> bool:
	"""Return True when inference should add the BCE logit offset to raw scores."""

	loss_path = str(getattr(args, 'model_loss_path', '') or '').lower()
	basename = os.path.basename(loss_path)
	return basename in {'bce_loss.py', 'bce_loss'}


def bce_logit_offset(args) -> float:
	"""Return the BCE logit offset (``train.loss_arg``) for inference."""

	if not uses_bce_logit_offset(args):
		return 0.0
	offset = getattr(args, 'bce_offset', None)
	if offset is not None:
		return float(offset)
	raw = getattr(args, 'loss_arg', None)
	if raw is None or (isinstance(raw, float) and math.isnan(raw)):
		return 0.0
	return float(raw)

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
	:param offset: Optional score offset (``train.loss_arg`` for BCE)
	:param reduction: ``mean``, ``sum``, or ``none``
	"""

	return _compute_bce_loss(scores, targets, offset=offset, reduction=reduction)


def build_bce_loss_fn(args):
	"""Factory for standard BCE-with-logits training."""

	offset = bce_logit_offset(args)
	reduction = str(getattr(args, 'bce_reduction', 'mean'))

	def loss_fn(scores: torch.Tensor, targets: torch.Tensor, **_kwargs) -> torch.Tensor:
		return _compute_bce_loss(scores, targets, offset=offset, reduction=reduction)

	return loss_fn


build_kvsall_loss_fn = build_bce_loss_fn
build_loss_fn = build_bce_loss_fn


def build_negsamp_loss_fn(args):
	"""Factory for triple negative-sampling BCE (``train.loss: bce``)."""

	offset = bce_logit_offset(args)

	def loss_fn(pos_scores: torch.Tensor, neg_scores: torch.Tensor, weights=None, **_kwargs) -> torch.Tensor:
		return _compute_bce_loss(
			scores=pos_scores,
			pos_scores=pos_scores,
			neg_scores=neg_scores,
			weights=weights,
			offset=offset,
		)

	return loss_fn


def compute_loss(args):
	return build_bce_loss_fn(args)
