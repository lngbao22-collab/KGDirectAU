"""Pairwise margin ranking loss for translational models (LibKGE ``margin_ranking`` loss)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def compute_margin_loss(
	pos_scores: torch.Tensor,
	neg_scores: torch.Tensor,
	margin: float,
) -> torch.Tensor:
	"""Compute ``max(0, margin - pos_score + neg_score)`` averaged over pairs.

	Assumes higher scores are better (similarity). For distance-based scores,
	flip the sign convention before calling this function.
	"""

	if pos_scores.dim() == 1 and neg_scores.dim() == 2:
		pos_scores = pos_scores.unsqueeze(1).expand_as(neg_scores)
	loss = F.relu(float(margin) - pos_scores + neg_scores)
	return loss.mean()


def build_margin_ranking_loss_fn(args):
	"""Factory for pairwise margin ranking on negative-sampling batches."""

	margin = getattr(args, 'margin', None)
	if margin is None:
		raw = getattr(args, 'loss_arg', None)
		margin = 1.0 if raw is None or (isinstance(raw, float) and math.isnan(raw)) else float(raw)

	def loss_fn(pos_scores: torch.Tensor, neg_scores: torch.Tensor, **_kwargs) -> torch.Tensor:
		return compute_margin_loss(pos_scores, neg_scores, margin)

	return loss_fn


build_negsamp_loss_fn = build_margin_ranking_loss_fn
build_loss_fn = build_margin_ranking_loss_fn


def compute_loss(args):
	return build_margin_ranking_loss_fn(args)
