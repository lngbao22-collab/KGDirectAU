"""Pure DistMult scorer operating on raw tensors only."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_scorer(args) -> DistMultScorer:
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

	def score_po_(self, all_h_embs: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all DistMult head scores for each (relation, tail) query."""

		return torch.mm(t_emb * r_emb, all_h_embs.t())

	def build_query(self, h_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
		return h_emb * r_emb
