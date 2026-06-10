"""Pure DistMult scorer operating on raw tensors only."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_model(args) -> nn.Module:
	"""Factory helper kept for compatibility with the model loader."""

	return DistMultScorer(args)


class DistMultScorer(nn.Module):
	"""DistMult score function with explicit 1-to-1 and 1-vs-All tensor paths."""

	def __init__(self, args=None):
		super().__init__()
		self.args = args

	def score_spo(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return standard DistMult scores for matching batches of triples."""

		return torch.sum(h_emb * r_emb * t_emb, dim=-1)

	def score_sp_(self, h_emb: torch.Tensor, r_emb: torch.Tensor, all_t_embs: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all DistMult scores using LibKGE-style sp_ broadcasting."""

		return torch.mm(h_emb * r_emb, all_t_embs.t())

	def forward(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Alias for score_spo to keep the module callable."""

		return self.score_spo(h_emb, r_emb, t_emb)
