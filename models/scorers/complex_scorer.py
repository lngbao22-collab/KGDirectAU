"""Pure ComplEx scorer operating on raw tensors only."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_scorer(args) -> ComplExScorer:
	return ComplExScorer(args)


class ComplExScorer(nn.Module):
	"""ComplEx score function with explicit 1-to-1 and 1-vs-All tensor paths."""

	def __init__(self, args=None):
		super().__init__()
		self.args = args

	@staticmethod
	def _split_complex(embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Split concatenated real and imaginary representations."""

		return torch.chunk(embeddings, 2, dim=-1)

	@staticmethod
	def query_encoder(
		h_re: torch.Tensor,
		h_im: torch.Tensor,
		r_re: torch.Tensor,
		r_im: torch.Tensor,
	) -> torch.Tensor:
		"""Fuse head and relation into a query vector (h ⊗ r).

		Inputs shape: ``[batch, dim]``. Output shape: ``[batch, dim * 2]``.
		"""

		q_re = (h_re * r_re) - (h_im * r_im)
		q_im = (h_re * r_im) + (h_im * r_re)
		return torch.cat([q_re, q_im], dim=-1)

	@staticmethod
	def target_encoder(t_re: torch.Tensor, t_im: torch.Tensor) -> torch.Tensor:
		"""Encode an entity into a target vector: ``cat([re, im])``.

		Inputs shape: ``[batch, dim]``. Output shape: ``[batch, dim * 2]``.
		"""

		return torch.cat([t_re, t_im], dim=-1)

	@staticmethod
	def inv_query_encoder(
		r_re: torch.Tensor,
		r_im: torch.Tensor,
		t_re: torch.Tensor,
		t_im: torch.Tensor,
	) -> torch.Tensor:
		"""Fuse relation and tail into a head-prediction query (conj(r) ⊗ t).

		Inputs shape: ``[batch, dim]``. Output shape: ``[batch, dim * 2]``.
		"""

		q_re = (r_re * t_re) + (r_im * t_im)
		q_im = (r_re * t_im) - (r_im * t_re)
		return torch.cat([q_re, q_im], dim=-1)

	def score_spo(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return standard ComplEx scores for matching batches of triples."""

		h_re, h_im = self._split_complex(h_emb)
		r_re, r_im = self._split_complex(r_emb)
		t_re, t_im = self._split_complex(t_emb)
		query = self.query_encoder(h_re, h_im, r_re, r_im)
		target = self.target_encoder(t_re, t_im)
		return torch.sum(query * target, dim=-1)

	def score_po(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return ComplEx scores for head candidates with fixed (relation, tail)."""

		return self.score_spo(h_emb, r_emb, t_emb)

	def score_sp_(self, h_emb: torch.Tensor, r_emb: torch.Tensor, all_t_embs: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all ComplEx scores using LibKGE-style sp_ broadcasting."""

		h_re, h_im = self._split_complex(h_emb)
		r_re, r_im = self._split_complex(r_emb)
		t_re, t_im = self._split_complex(all_t_embs)
		query = self.query_encoder(h_re, h_im, r_re, r_im)
		target = self.target_encoder(t_re, t_im)
		return torch.mm(query, target.t())

	def score_po_(self, all_h_embs: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all ComplEx head scores for each (relation, tail) query."""

		h_re, h_im = self._split_complex(all_h_embs)
		r_re, r_im = self._split_complex(r_emb)
		t_re, t_im = self._split_complex(t_emb)
		query = self.inv_query_encoder(r_re, r_im, t_re, t_im)
		target = self.target_encoder(h_re, h_im)
		return torch.mm(query, target.t())

	def build_query(self, h_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
		h_re, h_im = self._split_complex(h_emb)
		r_re, r_im = self._split_complex(r_emb)
		return self.query_encoder(h_re, h_im, r_re, r_im)
