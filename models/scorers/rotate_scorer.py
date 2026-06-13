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

	def _entity_chunk_size(self) -> int:
		"""Candidate chunk size for 1-vs-all scoring (controls peak GPU memory)."""

		return int(getattr(self.args, 'eval_entity_chunk_size', 1024) or 1024)

	def _margin_distance_1vsall(
		self,
		q_re: torch.Tensor,
		q_im: torch.Tensor,
		cand_re: torch.Tensor,
		cand_im: torch.Tensor,
	) -> torch.Tensor:
		"""1-vs-all RotatE distance with chunked broadcasting.

		RotatE uses ``sum_i sqrt(re_i^2 + im_i^2)``, which cannot be reduced to a
		single entity matrix multiply, so candidates are processed in chunks to avoid
		materializing ``[batch, num_candidates, dim]`` for the full entity vocabulary.
		"""

		num_candidates = cand_re.size(0)
		chunk_size = self._entity_chunk_size()
		if num_candidates <= chunk_size:
			re_score = q_re.unsqueeze(1) - cand_re.unsqueeze(0)
			im_score = q_im.unsqueeze(1) - cand_im.unsqueeze(0)
			return self.margin - torch.sqrt(re_score ** 2 + im_score ** 2).sum(dim=-1)

		scores = q_re.new_empty(q_re.size(0), num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			re_score = q_re.unsqueeze(1) - cand_re[start:end].unsqueeze(0)
			im_score = q_im.unsqueeze(1) - cand_im[start:end].unsqueeze(0)
			scores[:, start:end] = self.margin - torch.sqrt(re_score ** 2 + im_score ** 2).sum(dim=-1)
		return scores

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
