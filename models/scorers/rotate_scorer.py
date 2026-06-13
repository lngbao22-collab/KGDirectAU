"""Pure RotatE scorer operating on raw tensors only."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_scorer(args) -> RotatEScorer:
	return RotatEScorer(args)


class RotatEScorer(nn.Module):
	"""RotatE score function with explicit 1-to-1 and 1-vs-All tensor paths."""

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		self.dim = int(getattr(args, "dim", 0) or 0)
		margin_value = getattr(args, "margin", None)
		self.margin = float(6.0 if margin_value is None else margin_value)
		epsilon = float(getattr(args, "epsilon", 2.0))
		self.embedding_range = float((self.margin + epsilon) / max(self.dim, 1))

	def _phase(self, relation_emb: torch.Tensor) -> torch.Tensor:
		"""Map raw relation tensors to the RotatE phase space."""

		return relation_emb / (self.embedding_range / math.pi)

	@staticmethod
	def _split_complex(embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Split concatenated real and imaginary parts."""

		return torch.chunk(embeddings, 2, dim=-1)

	def score_spo(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return standard RotatE tail scores for matching batches of triples."""

		return self._distance_score(h_emb, r_emb, t_emb, predict_head=False)

	def score_po(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return standard RotatE head scores for matching batches of triples."""

		return self._distance_score(h_emb, r_emb, t_emb, predict_head=True)

	def _margin_distance_1vsall(
		self,
		q_re: torch.Tensor,
		q_im: torch.Tensor,
		cand_re: torch.Tensor,
		cand_im: torch.Tensor,
	) -> torch.Tensor:
		"""1-vs-all RotatE distance without materializing ``[batch, num_candidates, dim]`` tensors."""

		query_sq = (q_re ** 2 + q_im ** 2).sum(dim=-1, keepdim=True)
		candidate_sq = (cand_re ** 2 + cand_im ** 2).sum(dim=-1)
		cross = torch.mm(q_re, cand_re.t()) + torch.mm(q_im, cand_im.t())
		dist_sq = query_sq + candidate_sq.unsqueeze(0) - 2.0 * cross
		return self.margin - torch.sqrt(dist_sq.clamp_min(0.0))

	def score_sp_(self, h_emb: torch.Tensor, r_emb: torch.Tensor, all_t_embs: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all RotatE tail scores using LibKGE-style sp_ broadcasting."""

		h_re, h_im = self._split_complex(h_emb)
		t_re, t_im = self._split_complex(all_t_embs)
		phase = self._phase(r_emb)
		r_re = torch.cos(phase)
		r_im = torch.sin(phase)
		q_re = h_re * r_re - h_im * r_im
		q_im = h_re * r_im + h_im * r_re
		return self._margin_distance_1vsall(q_re, q_im, t_re, t_im)

	def score_po_(self, all_h_embs: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all RotatE head scores (LibKGE ``_po`` combine)."""

		h_re, h_im = self._split_complex(all_h_embs)
		t_re, t_im = self._split_complex(t_emb)
		phase = self._phase(r_emb)
		r_re = torch.cos(phase)
		r_im = torch.sin(phase)
		q_re = r_re * t_re + r_im * t_im
		q_im = r_re * t_im - r_im * t_re
		return self._margin_distance_1vsall(q_re, q_im, h_re, h_im)

	def _distance_score(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		*,
		predict_head: bool,
	) -> torch.Tensor:
		"""Shared RotatE distance for tail- or head-prediction."""

		h_re, h_im = self._split_complex(h_emb)
		t_re, t_im = self._split_complex(t_emb)
		phase = self._phase(r_emb)
		r_re = torch.cos(phase)
		r_im = torch.sin(phase)
		if predict_head:
			re_score = r_re * t_re + r_im * t_im - h_re
			im_score = r_re * t_im - r_im * t_re - h_im
		else:
			re_score = h_re * r_re - h_im * r_im - t_re
			im_score = h_re * r_im + h_im * r_re - t_im
		return self.margin - torch.sqrt(re_score ** 2 + im_score ** 2).sum(dim=-1)
