"""Pure DaBR scorer operating on raw tensors only."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.kge_scorer import KGEScorer


def build_scorer(args) -> DaBRScorer:
	return DaBRScorer(args)


class DaBRScorer(KGEScorer):
	"""DaBR score function with explicit 1-to-1 and 1-vs-All tensor paths."""

	bidirectional_score_batch = True

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		self.para = nn.Parameter(torch.tensor([float(getattr(args, 'para', 0.1))]))
		norm_p = int(getattr(args, 'dabr_distance_norm', 1) or 1)
		if norm_p not in (1, 2):
			raise ValueError(f'dabr_distance_norm must be 1 or 2, got {norm_p}')
		self.distance_norm = norm_p

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
	def _additive_penalty(
		cls,
		h_emb: torch.Tensor,
		dr_emb: torch.Tensor,
		t_emb: torch.Tensor,
		norm_p: int = 1,
	) -> torch.Tensor:
		"""Lp penalty on the quaternion-wise sum of the additive DaBR branch.

		``norm_p=1`` reproduces the paper's L1 term; ``norm_p=2`` measures the same
		``score_d`` vector (after the quaternion-wise sum) with L2 (Option A).
		"""

		hrt = h_emb + dr_emb - t_emb
		s_d, x_d, y_d, z_d = cls._split_quaternion(hrt)
		score_d = s_d + x_d + y_d + z_d
		return torch.norm(score_d, p=norm_p, dim=-1)

	@classmethod
	def _score_from_hr(
		cls,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor,
		para: float,
		norm_p: int = 1,
	) -> torch.Tensor:
		"""Score rows when ``hr = h⊗r`` is already computed or derived inline.

		Returns the paper's plausibility ``phi = <h⊗r, t⊗r^-1> + lambda * ||h+dr-t||_1``
		(Eq. 12-13), where **higher is better**. This matches the codebase loss/eval
		convention. The reference ``_calc`` returns ``-phi`` (OpenKE lower-is-better
		energy); using ``-phi`` here would optimize the non-negative L1 distance term in
		the wrong direction, so we keep the higher-is-better paper form.
		"""

		hr = cls._quat_mul(h_emb, r_emb)
		r_inv_norm = cls._normalize_quaternion(cls._quat_inv(r_emb))
		tr = cls._quat_mul_q(t_emb, r_inv_norm)
		score_s = torch.sum(hr * tr, dim=-1)
		return score_s + para * cls._additive_penalty(h_emb, dr_emb, t_emb, norm_p)

	@staticmethod
	def regularization(quaternion: torch.Tensor) -> torch.Tensor:
		"""Mean squared norm per quaternion component (official DaBR ``regularization``)."""

		size = quaternion.size(-1) // 4
		r, i, j, k = torch.split(quaternion, size, dim=-1)
		return torch.mean(r ** 2) + torch.mean(i ** 2) + torch.mean(j ** 2) + torch.mean(k ** 2)

	@staticmethod
	def _coalesce_para(para: float | torch.Tensor | None, default: float) -> float:
		if para is None:
			return default
		if torch.is_tensor(para):
			return float(para.item())
		return float(para)

	def _entity_chunk_size(self, batch_size: int, embed_dim: int) -> int:
		"""Candidate chunk size for 1-vs-all scoring (controls peak GPU memory)."""

		configured = int(getattr(self.args, 'eval_entity_chunk_size', 256) or 256)
		bytes_budget = int(
			getattr(self.args, 'eval_entity_chunk_bytes', 128 * 1024 * 1024) or 128 * 1024 * 1024
		)
		# DaBR quaternion 1-vs-all scoring materializes several [B, C, D] tensors.
		per_candidate = max(1, batch_size * embed_dim * 4 * 12)
		memory_limit = max(1, bytes_budget // per_candidate)
		return max(1, min(configured, memory_limit))

	def _score_sp_candidate_chunk(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb_chunk: torch.Tensor,
		dr_emb: torch.Tensor,
		para_value: float,
	) -> torch.Tensor:
		hr = self._quat_mul(h_emb, r_emb).unsqueeze(1)
		r_inv_norm = self._normalize_quaternion(self._quat_inv(r_emb)).unsqueeze(1)
		tr = self._quat_mul_q(t_emb_chunk.unsqueeze(0), r_inv_norm)
		score_s = torch.sum(hr * tr, dim=-1)
		add_penalty = self._additive_penalty(
			h_emb.unsqueeze(1),
			dr_emb.unsqueeze(1),
			t_emb_chunk.unsqueeze(0),
			self.distance_norm,
		)
		return score_s + para_value * add_penalty

	def _score_po_candidate_chunk(
		self,
		h_emb_chunk: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor,
		para_value: float,
	) -> torch.Tensor:
		num_heads = h_emb_chunk.size(0)
		batch_size = r_emb.size(0)
		all_h_exp = h_emb_chunk.unsqueeze(0).expand(batch_size, num_heads, -1)
		r_exp = self._normalize_quaternion(r_emb).unsqueeze(1).expand(-1, num_heads, -1)
		flat_h = all_h_exp.reshape(batch_size * num_heads, -1)
		flat_r = r_exp.reshape(batch_size * num_heads, -1)
		hr = self._quat_mul(flat_h, flat_r).view(batch_size, num_heads, -1)
		r_inv_norm = self._normalize_quaternion(self._quat_inv(r_emb)).unsqueeze(1)
		tr = self._quat_mul_q(t_emb.unsqueeze(1), r_inv_norm)
		score_s = torch.sum(hr * tr, dim=-1)
		add_penalty = self._additive_penalty(
			all_h_exp,
			dr_emb.unsqueeze(1).expand(-1, num_heads, -1),
			t_emb.unsqueeze(1).expand(-1, num_heads, -1),
			self.distance_norm,
		)
		return score_s + para_value * add_penalty

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
		return self._score_from_hr(
			h_emb, r_emb, t_emb, dr_emb, self._coalesce_para(para, self.para), self.distance_norm
		)

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
		num_candidates = all_t_embs.size(0)
		batch_size = h_emb.size(0)
		embed_dim = all_t_embs.size(-1)
		chunk_size = self._entity_chunk_size(batch_size, embed_dim)
		if num_candidates <= chunk_size:
			return self._score_sp_candidate_chunk(h_emb, r_emb, all_t_embs, dr_emb, para_value)

		scores = h_emb.new_empty(batch_size, num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			scores[:, start:end] = self._score_sp_candidate_chunk(
				h_emb,
				r_emb,
				all_t_embs[start:end],
				dr_emb,
				para_value,
			)
		return scores

	def score_po_(
		self,
		all_h_embs: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		para: float | torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Return 1-vs-all head scores with fixed relation and tail (``po_forward`` eval)."""

		if dr_emb is None:
			dr_emb = torch.zeros_like(t_emb)
		para_value = self._coalesce_para(para, self.para)
		num_heads = all_h_embs.size(0)
		batch_size = r_emb.size(0)
		embed_dim = all_h_embs.size(-1)
		chunk_size = self._entity_chunk_size(batch_size, embed_dim)
		if num_heads <= chunk_size:
			return self._score_po_candidate_chunk(all_h_embs, r_emb, t_emb, dr_emb, para_value)

		scores = r_emb.new_empty(batch_size, num_heads)
		for start in range(0, num_heads, chunk_size):
			end = min(start + chunk_size, num_heads)
			scores[:, start:end] = self._score_po_candidate_chunk(
				all_h_embs[start:end],
				r_emb,
				t_emb,
				dr_emb,
				para_value,
			)
		return scores

	@staticmethod
	def _normalized_pair_score(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
		left = F.normalize(left, p=2, dim=-1)
		right = F.normalize(right, p=2, dim=-1)
		return torch.sum(left * right, dim=-1)

	@staticmethod
	def _normalized_1vsall_score(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
		query = F.normalize(query, p=2, dim=-1)
		candidates = F.normalize(candidates, p=2, dim=-1)
		return torch.sum(query.unsqueeze(1) * candidates, dim=-1)

	def _coalesce_dr(self, h_emb: torch.Tensor, dr_emb: torch.Tensor | None) -> torch.Tensor:
		if dr_emb is None:
			return torch.zeros_like(h_emb)
		return dr_emb

	def _au_head_vector(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Tail/head query side: ``cat(h⊗r, h+dr)``."""

		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		q_mult = self._quat_mul(h_emb, r_emb)
		return torch.cat([q_mult, h_emb + dr_emb], dim=-1)

	def _au_tail_vector(self, t_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
		"""Tail alignment / head-query side: ``cat(t⊗r⁻¹, t)`` (matches native ``⟨h⊗r, t⊗r⁻¹⟩``)."""

		t_mult = self._quat_mul(t_emb, self._quat_inv(r_emb))
		return torch.cat([t_mult, t_emb], dim=-1)

	def _au_tail_vectors_batch(self, entity_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
		"""Relation-aware tail AU vectors ``[B, C, 2D]`` for 1-vs-all cosine LP."""

		num_ent = entity_emb.size(0)
		if r_emb.dim() == 1:
			r_emb = r_emb.unsqueeze(0)
		batch_size = r_emb.size(0)
		ent_exp = entity_emb.unsqueeze(0).expand(batch_size, num_ent, -1)
		r_exp = r_emb.unsqueeze(1).expand(batch_size, num_ent, -1)
		flat_ent = ent_exp.reshape(batch_size * num_ent, -1)
		flat_r = r_exp.reshape(batch_size * num_ent, -1)
		t_mult = self._quat_mul(flat_ent, flat_r).view(batch_size, num_ent, -1)
		return torch.cat([t_mult, ent_exp], dim=-1)

	def _au_head_vectors_batch(
		self,
		entity_emb: torch.Tensor,
		r_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Relation-aware head AU vectors ``[B, C, 2D]`` for 1-vs-all cosine LP."""

		num_ent = entity_emb.size(0)
		if r_emb.dim() == 1:
			r_emb = r_emb.unsqueeze(0)
		batch_size = r_emb.size(0)
		dr_emb = self._coalesce_dr(r_emb, dr_emb)
		ent_exp = entity_emb.unsqueeze(0).expand(batch_size, num_ent, -1)
		r_exp = r_emb.unsqueeze(1).expand(batch_size, num_ent, -1)
		dr_exp = dr_emb.unsqueeze(1).expand(batch_size, num_ent, -1)
		flat_ent = ent_exp.reshape(batch_size * num_ent, -1)
		flat_r = r_exp.reshape(batch_size * num_ent, -1)
		q_mult = self._quat_mul(flat_ent, flat_r).view(batch_size, num_ent, -1)
		return torch.cat([q_mult, ent_exp + dr_exp], dim=-1)

	def build_query(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Tail-prediction query vectors for cosine / Lp-distance link prediction."""

		return self._au_head_vector(h_emb, r_emb, dr_emb)

	def build_po_query(self, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Head-prediction query vectors for cosine / Lp-distance link prediction."""

		return self._au_tail_vector(t_emb, r_emb)

	def normalized_score_spo(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		return self._normalized_pair_score(
			self.build_query(h_emb, r_emb, dr_emb=dr_emb),
			self._au_tail_vector(t_emb, r_emb),
		)

	def normalized_score_po(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		return self._normalized_pair_score(
			self._au_head_vector(h_emb, r_emb, dr_emb),
			self.build_po_query(r_emb, t_emb),
		)

	def _group_batch_by_relation(
		self,
		r_emb: torch.Tensor,
		*tensors: torch.Tensor,
	) -> list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]:
		"""Group batch rows that share the same relation embedding."""

		batch_size = r_emb.size(0)
		if batch_size <= 1:
			return [(torch.arange(batch_size, device=r_emb.device), r_emb, list(tensors))]

		unique_r, inverse = torch.unique(r_emb, dim=0, return_inverse=True)
		groups: list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]] = []
		for rel_idx in range(unique_r.size(0)):
			row_mask = inverse == rel_idx
			row_indices = row_mask.nonzero(as_tuple=True)[0]
			grouped = [tensor[row_mask] for tensor in tensors]
			groups.append((row_indices, unique_r[rel_idx:rel_idx + 1], grouped))
		return groups

	def _normalized_sp_scores_chunked(
		self,
		query: torch.Tensor,
		all_entity_embs: torch.Tensor,
		r_emb: torch.Tensor,
		*,
		predict_head: bool,
		dr_emb: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Cosine 1-vs-all scores with relation-aware AU candidates (chunked over entities)."""

		num_candidates = all_entity_embs.size(0)
		batch_size = query.size(0)
		embed_dim = all_entity_embs.size(-1)
		chunk_size = self._entity_chunk_size(max(batch_size, 1), embed_dim * 2)
		scores = query.new_empty(batch_size, num_candidates)
		vector_fn = self._au_head_vectors_batch if predict_head else self._au_tail_vectors_batch
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			entity_chunk = all_entity_embs[start:end]
			if predict_head:
				targets = vector_fn(entity_chunk, r_emb, dr_emb)
			else:
				targets = vector_fn(entity_chunk, r_emb)
			if targets.size(0) == 1 and batch_size > 1:
				targets = targets.expand(batch_size, -1, -1)
			scores[:, start:end] = self._normalized_1vsall_score(query, targets)
		return scores

	def normalized_score_sp_(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		all_t_embs: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		batch_size = h_emb.size(0)
		num_candidates = all_t_embs.size(0)
		scores = h_emb.new_empty(batch_size, num_candidates)
		for row_indices, r_row, (h_sub, dr_sub) in self._group_batch_by_relation(r_emb, h_emb, dr_emb):
			r_sub = r_row.expand(h_sub.size(0), -1)
			query = self.build_query(h_sub, r_sub, dr_emb=dr_sub)
			group_scores = self._normalized_sp_scores_chunked(
				query,
				all_t_embs,
				r_row,
				predict_head=False,
			)
			scores.index_copy_(0, row_indices, group_scores)
		return scores

	def normalized_score_po_(
		self,
		all_h_embs: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		dr_emb = self._coalesce_dr(t_emb, dr_emb)
		batch_size = t_emb.size(0)
		num_candidates = all_h_embs.size(0)
		scores = t_emb.new_empty(batch_size, num_candidates)
		for row_indices, r_row, (t_sub, dr_sub) in self._group_batch_by_relation(r_emb, t_emb, dr_emb):
			r_sub = r_row.expand(t_sub.size(0), -1)
			query = self.build_po_query(r_sub, t_sub)
			group_scores = self._normalized_sp_scores_chunked(
				query,
				all_h_embs,
				r_row,
				predict_head=True,
				dr_emb=dr_sub[:1] if dr_sub is not None else None,
			)
			scores.index_copy_(0, row_indices, group_scores)
		return scores

	def au_representations(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		*,
		predict_head: bool = False,
		**kwargs,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		if predict_head:
			return (
				self.build_po_query(r_emb, t_emb),
				self._au_head_vector(h_emb, r_emb, dr_emb),
				h_emb,
			)
		return (
			self.build_query(h_emb, r_emb, dr_emb=dr_emb),
			self._au_tail_vector(t_emb, r_emb),
			h_emb,
		)
