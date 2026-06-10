"""Pure DaBR scorer operating on raw tensors only."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_scorer(args) -> DaBRScorer:
	return DaBRScorer(args)


class DaBRScorer(nn.Module):
	"""DaBR score function with explicit 1-to-1 and 1-vs-All tensor paths."""

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		self.para = float(getattr(args, "para", 0.1))

	@staticmethod
	def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
		"""Normalize each quaternion block (matches classic DaBR)."""

		size = quaternion.size(-1) // 4
		reshaped = quaternion.reshape(-1, 4, size)
		norm = torch.sqrt(torch.sum(reshaped ** 2, dim=1, keepdim=True).clamp_min(1e-12))
		return (reshaped / norm).reshape(-1, 4 * size)

	@staticmethod
	def _split_quaternion(embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Split concatenated quaternion components."""

		return torch.chunk(embeddings, 4, dim=-1)

	@staticmethod
	def _make_wise_quaternion(quaternion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Build the four quaternion-wise views used by DaBR multiplication."""

		if quaternion.dim() == 1:
			quaternion = quaternion.unsqueeze(0)
		size = quaternion.size(-1) // 4
		r, i, j, k = torch.split(quaternion, size, dim=-1)
		r2 = torch.cat([r, -i, -j, -k], dim=-1)
		i2 = torch.cat([i, r, -k, j], dim=-1)
		j2 = torch.cat([j, k, r, -i], dim=-1)
		k2 = torch.cat([k, -j, i, r], dim=-1)
		return r2, i2, j2, k2

	@staticmethod
	def _quaternion_wise_sum(quaternion: torch.Tensor) -> torch.Tensor:
		"""Sum the four quaternion components (DaBR ``get_quaternion_wise_mul``)."""

		size = quaternion.size(-1) // 4
		reshaped = quaternion.view(*quaternion.shape[:-1], 4, size)
		return torch.sum(reshaped, dim=-2)

	@classmethod
	def _quat_mul_q(cls, left: torch.Tensor, right_normalized: torch.Tensor) -> torch.Tensor:
		"""Quaternion multiply when the right operand is already normalized."""

		l_r, l_i, l_j, l_k = cls._make_wise_quaternion(left)
		qp_r = cls._quaternion_wise_sum(l_r * right_normalized)
		qp_i = cls._quaternion_wise_sum(l_i * right_normalized)
		qp_j = cls._quaternion_wise_sum(l_j * right_normalized)
		qp_k = cls._quaternion_wise_sum(l_k * right_normalized)
		return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=-1)

	@classmethod
	def _quat_mul(cls, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
		"""Quaternion multiply ``left * right`` with normalization of ``right``."""

		return cls._quat_mul_q(left, cls._normalize_quaternion(right))

	@staticmethod
	def _quat_inv(embeddings: torch.Tensor) -> torch.Tensor:
		"""Return the multiplicative inverse of a quaternion tensor."""

		r, i, j, k = DaBRScorer._split_quaternion(embeddings)
		norm = (r ** 2 + i ** 2 + j ** 2 + k ** 2).clamp_min(1e-12)
		return torch.cat([r / norm, -i / norm, -j / norm, -k / norm], dim=-1)

	@classmethod
	def _additive_penalty(cls, h_emb: torch.Tensor, dr_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""L1 penalty on the additive DaBR branch."""

		hrt = h_emb + dr_emb - t_emb
		s_d, x_d, y_d, z_d = cls._split_quaternion(hrt)
		score_d = s_d + x_d + y_d + z_d
		return torch.norm(score_d, p=1, dim=-1)

	@classmethod
	def _score_from_hr(
		cls,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor,
		para: float,
	) -> torch.Tensor:
		"""Score rows when ``hr = h⊗r`` is already computed or derived inline."""

		hr = cls._quat_mul(h_emb, r_emb)
		r_inv_norm = cls._normalize_quaternion(cls._quat_inv(r_emb))
		tr = cls._quat_mul_q(t_emb, r_inv_norm)
		score_s = -torch.sum(hr * tr, dim=-1)
		return score_s - para * cls._additive_penalty(h_emb, dr_emb, t_emb)

	@staticmethod
	def _coalesce_para(para: float | torch.Tensor | None, default: float) -> float:
		if para is None:
			return default
		if torch.is_tensor(para):
			return float(para.item())
		return float(para)

	def score_spo(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		para: float | torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Return standard DaBR scores for matching batches of triples."""

		if dr_emb is None:
			dr_emb = torch.zeros_like(h_emb)
		return self._score_from_hr(h_emb, r_emb, t_emb, dr_emb, self._coalesce_para(para, self.para))

	def score_sp_(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		all_t_embs: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		para: float | torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Return 1-vs-all DaBR scores using raw tensor broadcasting."""

		if dr_emb is None:
			dr_emb = torch.zeros_like(h_emb)
		para_value = self._coalesce_para(para, self.para)
		hr = self._quat_mul(h_emb, r_emb).unsqueeze(1)
		r_inv_norm = self._normalize_quaternion(self._quat_inv(r_emb)).unsqueeze(1)
		tr = self._quat_mul_q(all_t_embs.unsqueeze(0), r_inv_norm)
		score_s = -torch.sum(hr * tr, dim=-1)
		add_penalty = self._additive_penalty(
			h_emb.unsqueeze(1),
			dr_emb.unsqueeze(1),
			all_t_embs.unsqueeze(0),
		)
		return score_s - para_value * add_penalty
