"""Pure DaBR scorer operating on raw tensors only."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_model(args) -> nn.Module:
	"""Factory helper kept for compatibility with the model loader."""

	return DaBRScorer(args)


class DaBRScorer(nn.Module):
	"""DaBR score function with explicit 1-to-1 and 1-vs-All tensor paths."""

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		self.para = float(getattr(args, "para", 0.1))

	@staticmethod
	def _split_quaternion(embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Split concatenated quaternion components."""

		return torch.chunk(embeddings, 4, dim=-1)

	@staticmethod
	def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
		"""Quaternion multiply two tensors with matching final dimensions."""

		l_r, l_i, l_j, l_k = DaBRScorer._split_quaternion(left)
		r_r, r_i, r_j, r_k = DaBRScorer._split_quaternion(right)
		q_r = l_r * r_r - l_i * r_i - l_j * r_j - l_k * r_k
		q_i = l_r * r_i + l_i * r_r + l_j * r_k - l_k * r_j
		q_j = l_r * r_j - l_i * r_k + l_j * r_r + l_k * r_i
		q_k = l_r * r_k + l_i * r_j - l_j * r_i + l_k * r_r
		return torch.cat([q_r, q_i, q_j, q_k], dim=-1)

	@staticmethod
	def _quat_inv(embeddings: torch.Tensor) -> torch.Tensor:
		"""Return the multiplicative inverse of a quaternion tensor."""

		r, i, j, k = DaBRScorer._split_quaternion(embeddings)
		norm = (r ** 2 + i ** 2 + j ** 2 + k ** 2).clamp_min(1e-12)
		return torch.cat([r / norm, -i / norm, -j / norm, -k / norm], dim=-1)

	@staticmethod
	def _quat_sum(embeddings: torch.Tensor) -> torch.Tensor:
		"""Sum the quaternion components to match the current DaBR AU math."""

		return torch.stack(DaBRScorer._split_quaternion(embeddings), dim=1).sum(dim=1)

	def score_spo(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Return standard DaBR scores for matching batches of triples."""

		if dr_emb is None:
			dr_emb = torch.zeros_like(h_emb)
		hr = self._quat_mul(h_emb, r_emb)
		t_inv = self._quat_inv(r_emb)
		tr = self._quat_mul(t_emb, t_inv)
		score_s = -(hr * tr).sum(dim=-1)
		score_d = self._quat_sum(h_emb + dr_emb - t_emb)
		return score_s - self.para * score_d.abs().sum(dim=-1)

	def score_sp_(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		all_t_embs: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Return 1-vs-all DaBR scores using raw tensor broadcasting."""

		if dr_emb is None:
			dr_emb = torch.zeros_like(h_emb)
		q_mult = self._quat_mul(h_emb, r_emb).unsqueeze(1)
		r_inv = self._quat_inv(r_emb).unsqueeze(1)
		t_mult = self._quat_mul(all_t_embs.unsqueeze(0), r_inv)
		mult_score = -(q_mult * t_mult).sum(dim=-1)
		q_add = self._quat_sum(h_emb + dr_emb).unsqueeze(1)
		t_add = self._quat_sum(all_t_embs).unsqueeze(0)
		add_score = torch.abs(q_add - t_add).sum(dim=-1)
		return mult_score - self.para * add_score

	def forward(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Alias for score_spo to keep the module callable."""

		return self.score_spo(h_emb, r_emb, t_emb, dr_emb=dr_emb)
