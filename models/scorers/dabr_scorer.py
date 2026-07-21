"""Pure DaBR scorer operating on raw tensors only."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.model import KGEScorer


def build_scorer(args) -> 'DaBRScorer':
	return DaBRScorer(args)


class DaBRScorer(KGEScorer):
	"""DaBR score function with explicit 1-to-1 and 1-vs-All tensor paths.

	Quaternion feature ops use the last dimension (``dim=-1`` / ``size(-1)``) so the
	same kernels work for rank-2 ``[B, D]`` batches and rank-3 ``[B, C, D]`` broadcasts.
	That is equivalent to the original DaBR ``dim=1`` / ``size(1)`` convention on 2D tensors.
	"""

	bidirectional_score_batch = True
	kgau_alignment_mode = 'dabr_blocks'

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		para_init = getattr(args, 'para', None)
		if para_init is None:
			para_init = getattr(args, 'lmbda', 0.1)
		self.para = nn.Parameter(torch.tensor([float(para_init)]))
		norm_p = int(getattr(args, 'dabr_distance_norm', 1) or 1)
		if norm_p not in (1, 2):
			raise ValueError(f'dabr_distance_norm must be 1 or 2, got {norm_p}')
		self.distance_norm = norm_p

	# ------------------------------------------------------------------
	# Core DaBR quaternion ops (llqy123/DaBR ``models/DaBR.py``)
	# ------------------------------------------------------------------

	@classmethod
	def _distance_score(
		self,
		h_emb: torch.Tensor,
		dr_emb: torch.Tensor,
		t_emb: torch.Tensor,
		norm_p: int = 1,
	) -> torch.Tensor:
		"""Geometric distance term from DaBR ``_calc``.

		``norm_p=1`` matches the paper's L1; ``norm_p=2`` uses L2 on the same
		quaternion-summed ``score_d`` vector.
		"""

		hrt = h_emb + dr_emb - t_emb
		s_d, x_d, y_d, z_d = torch.chunk(hrt, 4, dim=-1)
		return torch.norm(s_d + x_d + y_d + z_d, p=norm_p, dim=-1)

	@staticmethod
	def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
		"""Unit-normalize each quaternion slot. DaBR: ``normalization``.

		Preserves leading batch dims (``[..., 4s] → [..., 4s]``). The original
		implementation flattened to 2D via ``reshape(-1, 4, size)``; that breaks
		``[B, C, D]`` broadcasting used by 1-vs-all scoring.
		"""

		size = quaternion.size(-1) // 4
		leading = quaternion.shape[:-1]

		reshaped = quaternion.reshape(*leading, 4, size)
		norm = torch.sqrt(torch.sum(reshaped ** 2, dim=-2, keepdim=True).clamp_min(1e-12))
		return (reshaped / norm).reshape(*leading, 4 * size)

	@staticmethod
	def _wise_quaternion(
		quaternion: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Build the four Hamilton-product views. DaBR: ``make_wise_quaternion``."""

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
	def _quaternion_wise_mul(quaternion: torch.Tensor) -> torch.Tensor:
		"""Sum the four component blocks after element-wise products.

		DaBR: ``get_quaternion_wise_mul``. Uses ``view(*shape[:-1], 4, size)`` and
		``sum(dim=-2)`` so leading dims are preserved (origin: ``view(-1, 4, size)``,
		``sum(..., 1)``).
		"""

		size = quaternion.size(-1) // 4
		reshaped = quaternion.view(*quaternion.shape[:-1], 4, size)
		return torch.sum(reshaped, dim=-2)

	@classmethod
	def _vec_vec_wise_multiplication(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
		"""Quaternion multiply ``left ⊗ right`` with unit-normalized ``right``.

		DaBR: ``vec_vec_wise_multiplication`` (always normalizes the right operand).
		"""

		normalized_right = self._normalize_quaternion(right)
		l_r, l_i, l_j, l_k = self._wise_quaternion(left)

		qp_r = self._quaternion_wise_mul(l_r * normalized_right)
		qp_i = self._quaternion_wise_mul(l_i * normalized_right)
		qp_j = self._quaternion_wise_mul(l_j * normalized_right)
		qp_k = self._quaternion_wise_mul(l_k * normalized_right)
		return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=-1)

	@staticmethod
	def _quat_inv(embeddings: torch.Tensor) -> torch.Tensor:
		"""Multiplicative inverse ``q⁻¹ = conjugate(q) / |q|²``. DaBR: ``get_inv``."""

		r, i, j, k = torch.chunk(embeddings, 4, dim=-1)
		norm = (r ** 2 + i ** 2 + j ** 2 + k ** 2).clamp_min(1e-12)
		return torch.cat([r / norm, -i / norm, -j / norm, -k / norm], dim=-1)

	@classmethod
	def _semantic_score(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		"""Semantic matching term ``⟨h⊗r, t⊗r⁻¹⟩``. DaBR: semantic part of ``_calc``.

		``r⁻¹`` is passed raw; unit normalization happens inside
		``_vec_vec_wise_multiplication`` (same as the original).
		"""

		hr = self._vec_vec_wise_multiplication(h_emb, r_emb)
		tr = self._vec_vec_wise_multiplication(t_emb, self._quat_inv(r_emb))
		return torch.sum(hr * tr, dim=-1)

	@staticmethod
	def regularization(quaternion: torch.Tensor) -> torch.Tensor:
		"""Mean squared magnitude of the four quaternion components. DaBR: ``regularization``."""

		size = quaternion.size(-1) // 4
		r, i, j, k = torch.split(quaternion, size, dim=-1)
		return torch.mean(r ** 2) + torch.mean(i ** 2) + torch.mean(j ** 2) + torch.mean(k ** 2)

	@staticmethod
	def _coalesce_para(
		para: float | torch.Tensor | None,
		default: float | torch.Tensor,
	) -> float | torch.Tensor:
		"""Resolve optional ``para`` override; keep tensor defaults for autograd."""

		if para is None:
			return default
		if torch.is_tensor(para):
			return para
		return float(para)

	def score_hrt(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		para: float | torch.Tensor | None = None,
	) -> torch.Tensor:
		"""1-to-1 DaBR score ``⟨h⊗r, t⊗r⁻¹⟩ + λ‖(h+dr−t)_Σ‖``.

		Sign is flipped relative to DaBR ``_calc`` (framework: higher is better).
		"""

		if dr_emb is None:
			dr_emb = torch.zeros_like(h_emb)
		score_s = self._semantic_score(h_emb, r_emb, t_emb)
		score_d = self._distance_score(h_emb, dr_emb, t_emb, self.distance_norm)
		para_value = self._coalesce_para(para, self.para)
		return score_s + para_value * score_d

	# ------------------------------------------------------------------
	# AU / DirectAU block vectors
	# ------------------------------------------------------------------

	def _coalesce_dr(
		self,
		h_emb: torch.Tensor,
		dr_emb: torch.Tensor | None,
	) -> torch.Tensor:
		"""Return ``dr_emb`` or zeros shaped like ``h_emb``."""

		if dr_emb is None:
			return torch.zeros_like(h_emb)
		return dr_emb

	def _au_head_vector(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Head-side AU vector ``cat(h⊗r, h+dr)``."""

		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		q_mult = self._vec_vec_wise_multiplication(h_emb, r_emb)
		return torch.cat([q_mult, h_emb + dr_emb], dim=-1)

	def _au_tail_vector(
		self,
		t_emb: torch.Tensor,
		r_emb: torch.Tensor,
	) -> torch.Tensor:
		"""Tail-side AU vector ``cat(t⊗r⁻¹, t)`` (matches ``⟨h⊗r, t⊗r⁻¹⟩``)."""

		t_mult = self._vec_vec_wise_multiplication(t_emb, self._quat_inv(r_emb))
		return torch.cat([t_mult, t_emb], dim=-1)

	def build_query(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Tail-prediction query: ``cat(h⊗r, h+dr)``."""

		return self._au_head_vector(h_emb, r_emb, dr_emb)

	def build_inv_query(
		self,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		"""Head-prediction query: ``cat(t⊗r⁻¹, t)``."""

		return self._au_tail_vector(t_emb, r_emb)

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
		"""Return ``(anchor, positive, entity_for_uniformity)`` AU triples."""

		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		if predict_head:
			return (
				self.build_inv_query(r_emb, t_emb),
				self._au_head_vector(h_emb, r_emb, dr_emb),
				h_emb
			)
		return (
			self.build_query(h_emb, r_emb, dr_emb=dr_emb),
			self._au_tail_vector(t_emb, r_emb),
			h_emb
		)

	def au_entity_embeddings(self, entity_emb: torch.Tensor) -> torch.Tensor:
		"""Widen raw entities to two-block AU width via ``cat(e, e)``.

		Used for entity-uniformity / LP table wrappers. Alignment and cosine LP
		still build relation-aware targets via ``_au_tail_vector`` / ``_au_head_vector``.
		"""

		return torch.cat([entity_emb, entity_emb], dim=-1)

	# ------------------------------------------------------------------
	# Native 1-vs-all DaBR scoring
	# ------------------------------------------------------------------

	def _entity_chunk_size(self, batch_size: int, embed_dim: int) -> int:
		"""Candidate chunk size for 1-vs-all scoring (peak GPU memory control)."""

		configured = int(getattr(self.args, 'eval_entity_chunk_size', 256) or 256)
		bytes_budget = int(
			getattr(self.args, 'eval_entity_chunk_bytes', 128 * 1024 * 1024) or 128 * 1024 * 1024
		)
		# Several [B, C, D] quaternion intermediates are materialized per chunk.
		per_candidate = max(1, batch_size * embed_dim * 4 * 12)
		memory_limit = max(1, bytes_budget // per_candidate)
		return max(1, min(configured, memory_limit))

	def _score_hr_candidate_chunk(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb_chunk: torch.Tensor,
		dr_emb: torch.Tensor,
		para_value: float | torch.Tensor,
	) -> torch.Tensor:
		"""1-vs-all tail scores for one entity chunk (broadcast over candidates)."""

		hr = self._vec_vec_wise_multiplication(h_emb, r_emb).unsqueeze(1)
		tr = self._vec_vec_wise_multiplication(
			t_emb_chunk.unsqueeze(0),
			self._quat_inv(r_emb).unsqueeze(1)
		)

		score_s = torch.sum(hr * tr, dim=-1)
		score_d = self._distance_score(
			h_emb.unsqueeze(1),
			dr_emb.unsqueeze(1),
			t_emb_chunk.unsqueeze(0),
			self.distance_norm
		)
		return score_s + para_value * score_d

	def _score_rt_candidate_chunk(
		self,
		h_emb_chunk: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor,
		para_value: float | torch.Tensor,
	) -> torch.Tensor:
		"""1-vs-all head scores for one entity chunk (broadcast over candidates)."""

		num_heads = h_emb_chunk.size(0)
		hr = self._vec_vec_wise_multiplication(
			h_emb_chunk.unsqueeze(0),
			r_emb.unsqueeze(1),
		)
		tr = self._vec_vec_wise_multiplication(
			t_emb.unsqueeze(1),
			self._quat_inv(r_emb).unsqueeze(1),
		)

		score_s = torch.sum(hr * tr, dim=-1)
		score_d = self._distance_score(
			h_emb_chunk.unsqueeze(0).expand(r_emb.size(0), num_heads, -1),
			dr_emb.unsqueeze(1).expand(-1, num_heads, -1),
			t_emb.unsqueeze(1).expand(-1, num_heads, -1),
			self.distance_norm,
		)
		return score_s + para_value * score_d

	def score_hr_(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		all_t_embs: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		para: float | torch.Tensor | None = None,
	) -> torch.Tensor:
		"""1-vs-all tail prediction scores ``[B, |E|]``."""

		if dr_emb is None:
			dr_emb = torch.zeros_like(h_emb)

		para_value = self._coalesce_para(para, self.para)
		num_candidates = all_t_embs.size(0)
		batch_size = h_emb.size(0)
		embed_dim = all_t_embs.size(-1)
		chunk_size = self._entity_chunk_size(batch_size, embed_dim)

		if num_candidates <= chunk_size:
			return self._score_hr_candidate_chunk(h_emb, r_emb, all_t_embs, dr_emb, para_value)

		scores = h_emb.new_empty(batch_size, num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			scores[:, start:end] = self._score_hr_candidate_chunk(
				h_emb, r_emb, all_t_embs[start:end], dr_emb, para_value,
			)
		return scores

	def score_rt_(
		self,
		all_h_embs: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		para: float | torch.Tensor | None = None,
	) -> torch.Tensor:
		"""1-vs-all head prediction scores ``[B, |E|]``."""

		if dr_emb is None:
			dr_emb = torch.zeros_like(t_emb)

		para_value = self._coalesce_para(para, self.para)
		num_heads = all_h_embs.size(0)
		batch_size = r_emb.size(0)
		embed_dim = all_h_embs.size(-1)
		chunk_size = self._entity_chunk_size(batch_size, embed_dim)

		if num_heads <= chunk_size:
			return self._score_rt_candidate_chunk(all_h_embs, r_emb, t_emb, dr_emb, para_value)

		scores = r_emb.new_empty(batch_size, num_heads)
		for start in range(0, num_heads, chunk_size):
			end = min(start + chunk_size, num_heads)
			scores[:, start:end] = self._score_rt_candidate_chunk(
				all_h_embs[start:end], r_emb, t_emb, dr_emb, para_value,
			)
		return scores

	# ------------------------------------------------------------------
	# Normalized (cosine) and distance LP over AU blocks
	# ------------------------------------------------------------------

	@staticmethod
	def _split_au_blocks(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Split ``cat(semantic, additive)`` AU vectors at the midpoint."""

		mid = vectors.size(-1) // 2
		return vectors[..., :mid], vectors[..., mid:]

	@classmethod
	def _normalized_pair_score(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
		"""Cosine similarity along the last dimension."""

		left = F.normalize(left, p=2, dim=-1)
		right = F.normalize(right, p=2, dim=-1)
		return torch.sum(left * right, dim=-1)

	@classmethod
	def _normalized_block_pair_score(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
		"""Sum of per-block cosines (semantic + additive), matching DaBR ``φ`` structure."""

		l_sem, l_add = self._split_au_blocks(left)
		r_sem, r_add = self._split_au_blocks(right)
		return self._normalized_pair_score(l_sem, r_sem) + self._normalized_pair_score(l_add, r_add)

	def normalized_score_hr(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""1-to-1 block-cosine score for tail prediction."""

		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		return self._normalized_block_pair_score(
			self.build_query(h_emb, r_emb, dr_emb=dr_emb),
			self._au_tail_vector(t_emb, r_emb),
		)

	def normalized_score_rt(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""1-to-1 block-cosine score for head prediction."""

		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		return self._normalized_block_pair_score(
			self._au_head_vector(h_emb, r_emb, dr_emb),
			self.build_inv_query(r_emb, t_emb),
		)

	@staticmethod
	def _raw_from_entity_au(entity_au: torch.Tensor) -> torch.Tensor:
		"""Recover raw entity rows from ``au_entity_embeddings`` (``cat(e, e)``)."""

		return entity_au[..., : entity_au.size(-1) // 2]

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

	def _au_head_vectors_batch(
		self,
		entity_emb: torch.Tensor,
		r_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Relation-aware head AU candidates ``[B, C, 2D]`` for 1-vs-all LP."""

		num_ent = entity_emb.size(0)
		if r_emb.dim() == 1:
			r_emb = r_emb.unsqueeze(0)
		batch_size = r_emb.size(0)
		ent_exp = entity_emb.unsqueeze(0).expand(batch_size, num_ent, -1)
		dr_emb = self._coalesce_dr(ent_exp[:, 0], dr_emb)
		r_exp = r_emb.unsqueeze(1).expand(batch_size, num_ent, -1)
		dr_exp = dr_emb.unsqueeze(1).expand(batch_size, num_ent, -1)
		flat_ent = ent_exp.reshape(batch_size * num_ent, -1)
		flat_r = r_exp.reshape(batch_size * num_ent, -1)
		q_mult = self._vec_vec_wise_multiplication(flat_ent, flat_r).view(batch_size, num_ent, -1)
		return torch.cat([q_mult, ent_exp + dr_exp], dim=-1)

	def _au_tail_vectors_batch(
		self,
		entity_emb: torch.Tensor,
		r_emb: torch.Tensor,
	) -> torch.Tensor:
		"""Relation-aware tail AU candidates ``[B, C, 2D]``: ``cat(t⊗r⁻¹, t)``."""

		num_ent = entity_emb.size(0)
		if r_emb.dim() == 1:
			r_emb = r_emb.unsqueeze(0)
		batch_size = r_emb.size(0)
		ent_exp = entity_emb.unsqueeze(0).expand(batch_size, num_ent, -1)
		r_inv_exp = self._quat_inv(r_emb).unsqueeze(1).expand(batch_size, num_ent, -1)
		flat_ent = ent_exp.reshape(batch_size * num_ent, -1)
		flat_r_inv = r_inv_exp.reshape(batch_size * num_ent, -1)
		t_mult = self._vec_vec_wise_multiplication(flat_ent, flat_r_inv).view(batch_size, num_ent, -1)
		return torch.cat([t_mult, ent_exp], dim=-1)

	@classmethod
	def _distance_1vsall_score(
		self,
		query: torch.Tensor,
		candidates: torch.Tensor,
		degree: float,
	) -> torch.Tensor:
		"""Negative block-wise Lp distance (higher is better)."""

		q_sem, q_add = self._split_au_blocks(query)
		c_sem, c_add = self._split_au_blocks(candidates)
		dist_sem = torch.norm(q_sem.unsqueeze(1) - c_sem, p=degree, dim=-1)
		dist_add = torch.norm(q_add.unsqueeze(1) - c_add, p=degree, dim=-1)
		return -(dist_sem + dist_add)

	@classmethod
	def _normalized_1vsall_score(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
		"""1-vs-all sum of per-block cosines between query and candidates."""

		q_sem, q_add = self._split_au_blocks(query)
		c_sem, c_add = self._split_au_blocks(candidates)
		q_sem = F.normalize(q_sem, p=2, dim=-1)
		q_add = F.normalize(q_add, p=2, dim=-1)
		c_sem = F.normalize(c_sem, p=2, dim=-1)
		c_add = F.normalize(c_add, p=2, dim=-1)
		return (q_sem.unsqueeze(1) * c_sem).sum(dim=-1) + (q_add.unsqueeze(1) * c_add).sum(dim=-1)

	def _au_hr_scores_chunked(
		self,
		query: torch.Tensor,
		all_entity_embs: torch.Tensor,
		r_emb: torch.Tensor,
		predict_head: bool,
		dr_emb: torch.Tensor | None = None,
		use_distance: bool = False,
		distance_degree: float = 2.0,
	) -> torch.Tensor:
		"""Chunked 1-vs-all AU scores with relation-aware candidates."""

		num_candidates = all_entity_embs.size(0)
		batch_size = query.size(0)
		embed_dim = all_entity_embs.size(-1)
		chunk_size = self._entity_chunk_size(max(batch_size, 1), embed_dim * 2)

		scores = query.new_empty(batch_size, num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			entity_chunk = all_entity_embs[start:end]

			if predict_head:
				targets = self._au_head_vectors_batch(entity_chunk, r_emb, dr_emb)
			else:
				targets = self._au_tail_vectors_batch(entity_chunk, r_emb)

			if targets.size(0) == 1 and batch_size > 1:
				targets = targets.expand(batch_size, -1, -1)

			if use_distance:
				scores[:, start:end] = self._distance_1vsall_score(
					query, targets, degree=distance_degree,
				)
			else:
				scores[:, start:end] = self._normalized_1vsall_score(query, targets)
		return scores

	def normalized_score_hr_(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		all_t_embs: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""1-vs-all block-cosine tail scores (candidates via ``cat(t⊗r⁻¹, t)``)."""

		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		all_t_embs = self._raw_from_entity_au(all_t_embs)
		batch_size = h_emb.size(0)
		num_candidates = all_t_embs.size(0)

		scores = h_emb.new_empty(batch_size, num_candidates)
		for row_indices, r_row, (h_sub, dr_sub) in self._group_batch_by_relation(r_emb, h_emb, dr_emb):
			r_sub = r_row.expand(h_sub.size(0), -1)
			query = self.build_query(h_sub, r_sub, dr_emb=dr_sub)
			group_scores = self._au_hr_scores_chunked(
				query, all_t_embs, r_row, predict_head=False,
			)
			scores.index_copy_(0, row_indices, group_scores)
		return scores

	def normalized_score_rt_(
		self,
		all_h_embs: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""1-vs-all block-cosine head scores (candidates via ``cat(h⊗r, h+dr)``)."""

		dr_emb = self._coalesce_dr(t_emb, dr_emb)
		all_h_embs = self._raw_from_entity_au(all_h_embs)
		batch_size = t_emb.size(0)
		num_candidates = all_h_embs.size(0)

		scores = t_emb.new_empty(batch_size, num_candidates)
		for row_indices, r_row, (t_sub, dr_sub) in self._group_batch_by_relation(r_emb, t_emb, dr_emb):
			r_sub = r_row.expand(t_sub.size(0), -1)
			query = self.build_inv_query(r_sub, t_sub)
			group_scores = self._au_hr_scores_chunked(
				query,
				all_h_embs,
				r_row,
				predict_head=True,
				dr_emb=dr_sub[:1] if dr_sub is not None else None,
			)
			scores.index_copy_(0, row_indices, group_scores)
		return scores

	def _coalesce_lp_distance_degree(self, kwargs: dict) -> float:
		"""Resolve Lp degree for distance-based AU link prediction."""

		degree = kwargs.pop('lp_distance_degree', None)
		if degree is None and self.args is not None:
			degree = getattr(self.args, 'lp_distance_degree', None)
		return float(degree if degree is not None else 2.0)

	def distance_score_hr_(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		all_t_embs: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""1-vs-all negative block-Lp tail scores."""

		distance_degree = self._coalesce_lp_distance_degree(kwargs)
		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		all_t_embs = self._raw_from_entity_au(all_t_embs)
		batch_size = h_emb.size(0)
		num_candidates = all_t_embs.size(0)

		scores = h_emb.new_empty(batch_size, num_candidates)
		for row_indices, r_row, (h_sub, dr_sub) in self._group_batch_by_relation(r_emb, h_emb, dr_emb):
			r_sub = r_row.expand(h_sub.size(0), -1)
			query = self.build_query(h_sub, r_sub, dr_emb=dr_sub)
			group_scores = self._au_hr_scores_chunked(
				query,
				all_t_embs,
				r_row,
				predict_head=False,
				use_distance=True,
				distance_degree=distance_degree,
			)
			scores.index_copy_(0, row_indices, group_scores)
		return scores

	def distance_score_rt_(
		self,
		all_h_embs: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""1-vs-all negative block-Lp head scores."""

		distance_degree = self._coalesce_lp_distance_degree(kwargs)
		dr_emb = self._coalesce_dr(t_emb, dr_emb)
		all_h_embs = self._raw_from_entity_au(all_h_embs)
		batch_size = t_emb.size(0)
		num_candidates = all_h_embs.size(0)

		scores = t_emb.new_empty(batch_size, num_candidates)
		for row_indices, r_row, (t_sub, dr_sub) in self._group_batch_by_relation(r_emb, t_emb, dr_emb):
			r_sub = r_row.expand(t_sub.size(0), -1)
			query = self.build_inv_query(r_sub, t_sub)
			group_scores = self._au_hr_scores_chunked(
				query,
				all_h_embs,
				r_row,
				predict_head=True,
				dr_emb=dr_sub[:1] if dr_sub is not None else None,
				use_distance=True,
				distance_degree=distance_degree,
			)
			scores.index_copy_(0, row_indices, group_scores)
		return scores
