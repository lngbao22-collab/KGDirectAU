"""Pure pRotatE scorer operating on raw tensors only."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_model(args) -> nn.Module:
	"""Factory helper kept for compatibility with the model loader."""

	return pRotatEScorer(args)


class pRotatEScorer(nn.Module):
	"""pRotatE score function with explicit 1-to-1 and 1-vs-All tensor paths."""

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		self.dim = int(getattr(args, "dim", 0) or 0)
		margin_value = getattr(args, "margin", None)
		self.margin = float(6.0 if margin_value is None else margin_value)
		epsilon = float(getattr(args, "epsilon", 2.0))
		self.embedding_range = float((self.margin + epsilon) / max(self.dim, 1))
		self.modulus = float(getattr(args, "modulus", 0.5 * self.embedding_range))

	def _phase(self, embeddings: torch.Tensor) -> torch.Tensor:
		"""Map raw tensors into the phase space used by pRotatE."""

		return embeddings / (self.embedding_range / math.pi)

	def score_spo(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return standard pRotatE scores for matching batches of triples."""

		phase = self._phase(h_emb) + self._phase(r_emb) - self._phase(t_emb)
		return self.margin - torch.abs(torch.sin(phase)).sum(dim=-1) * self.modulus

	def score_sp_(self, h_emb: torch.Tensor, r_emb: torch.Tensor, all_t_embs: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all pRotatE scores using raw tensor broadcasting."""

		phase = self._phase(h_emb).unsqueeze(1) + self._phase(r_emb).unsqueeze(1) - self._phase(all_t_embs).unsqueeze(0)
		return self.margin - torch.abs(torch.sin(phase)).sum(dim=-1) * self.modulus

	def forward(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Alias for score_spo to keep the module callable."""

		return self.score_spo(h_emb, r_emb, t_emb)
