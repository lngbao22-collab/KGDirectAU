"""Pure RotatE scorer operating on raw tensors only."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_model(args) -> nn.Module:
	"""Factory helper kept for compatibility with the model loader."""

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
		"""Return standard RotatE scores for matching batches of triples."""

		h_re, h_im = self._split_complex(h_emb)
		t_re, t_im = self._split_complex(t_emb)
		phase = self._phase(r_emb)
		r_re = torch.cos(phase)
		r_im = torch.sin(phase)
		re_score = h_re * r_re - h_im * r_im - t_re
		im_score = h_re * r_im + h_im * r_re - t_im
		return self.margin - torch.sqrt(re_score ** 2 + im_score ** 2).sum(dim=-1)

	def score_sp_(self, h_emb: torch.Tensor, r_emb: torch.Tensor, all_t_embs: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all RotatE scores using raw tensor broadcasting."""

		h_re, h_im = self._split_complex(h_emb)
		t_re, t_im = self._split_complex(all_t_embs)
		phase = self._phase(r_emb)
		r_re = torch.cos(phase)
		r_im = torch.sin(phase)
		q_re = h_re * r_re - h_im * r_im
		q_im = h_re * r_im + h_im * r_re
		re_score = q_re.unsqueeze(1) - t_re.unsqueeze(0)
		im_score = q_im.unsqueeze(1) - t_im.unsqueeze(0)
		return self.margin - torch.sqrt(re_score ** 2 + im_score ** 2).sum(dim=-1)

	def forward(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Alias for score_spo to keep the module callable."""

		return self.score_spo(h_emb, r_emb, t_emb)
