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

	def score_spo(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return standard ComplEx scores for matching batches of triples."""

		h_re, h_im = self._split_complex(h_emb)
		r_re, r_im = self._split_complex(r_emb)
		t_re, t_im = self._split_complex(t_emb)
		query_re = h_re * r_re - h_im * r_im
		query_im = h_re * r_im + h_im * r_re
		return torch.sum(query_re * t_re + query_im * t_im, dim=-1)

	def score_po(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return ComplEx scores for head candidates with fixed (relation, tail)."""

		return self.score_spo(h_emb, r_emb, t_emb)

	def score_sp_(self, h_emb: torch.Tensor, r_emb: torch.Tensor, all_t_embs: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all ComplEx scores using LibKGE-style sp_ broadcasting."""

		h_re, h_im = self._split_complex(h_emb)
		r_re, r_im = self._split_complex(r_emb)
		t_re, t_im = self._split_complex(all_t_embs)
		query_re = h_re * r_re - h_im * r_im
		query_im = h_re * r_im + h_im * r_re
		return torch.mm(query_re, t_re.t()) + torch.mm(query_im, t_im.t())

	def score_po_(self, all_h_embs: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all ComplEx head scores for each (relation, tail) query."""

		h_re, h_im = self._split_complex(all_h_embs)
		r_re, r_im = self._split_complex(r_emb)
		t_re, t_im = self._split_complex(t_emb)
		query_re = r_re * t_re + r_im * t_im
		query_im = r_re * t_im - r_im * t_re
		return torch.mm(query_re, h_re.t()) + torch.mm(query_im, h_im.t())

	def build_query(self, h_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
		h_re, h_im = self._split_complex(h_emb)
		r_re, r_im = self._split_complex(r_emb)
		return torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)
