"""Pure TransERR scorer operating on raw tensors only."""

from __future__ import annotations

import torch

from base.kge_scorer import KGEScorer


def build_scorer(args) -> TransERRScorer:
	return TransERRScorer(args)


class TransERRScorer(KGEScorer):
	"""TransERR score function with explicit 1-to-1 and 1-vs-All tensor paths.

	Matches ``KnowledgeGraphEmbedding`` TransERR (Sun et al.) when
	``triple_relation_embedding`` is enabled: higher score is better,
	``gamma - ||h⊗wh + r - t⊗wt||_1`` with normalized quaternion relation parts.
	"""

	bidirectional_score_batch = True

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		self.dim = int(getattr(args, 'dim', 0) or 0)
		margin_value = getattr(args, 'margin', None)
		if margin_value is None:
			margin_value = getattr(args, 'gamma', 12.0)
		self.gamma = float(margin_value)

	@staticmethod
	def _q_norm(relation: torch.Tensor) -> torch.Tensor:
		s_b, x_b, y_b, z_b = torch.chunk(relation, 4, dim=-1)
		denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
		s_b = s_b / denominator_b
		x_b = x_b / denominator_b
		y_b = y_b / denominator_b
		z_b = z_b / denominator_b
		return torch.cat([s_b, x_b, y_b, z_b], dim=-1)

	@staticmethod
	def _calc(head: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
		s_a, x_a, y_a, z_a = torch.chunk(head, 4, dim=-1)
		s_b, x_b, y_b, z_b = torch.chunk(relation, 4, dim=-1)

		a = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
		b = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
		c = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
		d = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
		return torch.cat([a, b, c, d], dim=-1)

	def _align_for_candidates(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Broadcast entity/relation tensors for optional candidate dimensions."""

		if h_emb.dim() == t_emb.dim():
			return h_emb, r_emb, t_emb

		if t_emb.dim() == h_emb.dim() + 1:
			return h_emb.unsqueeze(1), r_emb.unsqueeze(1), t_emb

		if h_emb.dim() == t_emb.dim() + 1:
			return h_emb, r_emb.unsqueeze(1), t_emb.unsqueeze(1)

		raise ValueError(
			f'Unsupported TransERR tensor ranks: h={h_emb.dim()}, r={r_emb.dim()}, t={t_emb.dim()}'
		)

	def _score_tensor(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		if h_emb.dim() == 2 and r_emb.dim() == 2 and t_emb.dim() == 2:
			if h_emb.size(0) == r_emb.size(0) == t_emb.size(0):
				h_aligned, r_aligned, t_aligned = h_emb, r_emb, t_emb
			elif h_emb.size(0) == r_emb.size(0):
				batch_size = h_emb.size(0)
				num_candidates = t_emb.size(0)
				h_aligned = h_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
				r_aligned = r_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
				t_aligned = t_emb.unsqueeze(0).expand(batch_size, num_candidates, -1)
			elif t_emb.size(0) == r_emb.size(0):
				batch_size = r_emb.size(0)
				num_candidates = h_emb.size(0)
				h_aligned = h_emb.unsqueeze(0).expand(batch_size, num_candidates, -1)
				r_aligned = r_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
				t_aligned = t_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
			else:
				raise ValueError(
					f'Unsupported TransERR batch layout: h={tuple(h_emb.shape)}, '
					f'r={tuple(r_emb.shape)}, t={tuple(t_emb.shape)}'
				)
		else:
			h_aligned, r_aligned, t_aligned = self._align_for_candidates(h_emb, r_emb, t_emb)

		wh, r_mid, wt = torch.chunk(r_aligned, 3, dim=-1)
		wh = self._q_norm(wh)
		wt = self._q_norm(wt)
		diff = self._calc(h_aligned, wh) + r_mid - self._calc(t_aligned, wt)
		return self.gamma - torch.norm(diff, p=1, dim=-1)

	@staticmethod
	def embedding_regularization(model) -> torch.Tensor:
		"""Mean squared L2 norm over all entity and relation embeddings (official TransERR)."""

		ent_w = model.ent_embedder.embedding.weight
		rel_w = model.rel_embedder.embedding.weight
		ent_reg = torch.sum(ent_w ** 2, dim=-1)
		rel_reg = torch.sum(rel_w ** 2, dim=-1)
		return torch.cat([ent_reg, rel_reg]).mean()

	def _entity_chunk_size(self, batch_size: int) -> int:
		configured = int(getattr(self.args, 'eval_entity_chunk_size', 256) or 256)
		bytes_budget = int(getattr(self.args, 'eval_entity_chunk_bytes', 128 * 1024 * 1024) or 128 * 1024 * 1024)
		per_candidate = max(1, batch_size * self.dim * 4 * 4)
		memory_limit = max(1, bytes_budget // per_candidate)
		return max(1, min(configured, memory_limit))

	def score_spo(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		return self._score_tensor(h_emb, r_emb, t_emb)

	def score_po(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		return self._score_tensor(h_emb, r_emb, t_emb)

	def score_spo_candidates(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		return self._score_tensor(h_emb, r_emb, t_emb)

	def score_po_candidates(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		return self._score_tensor(h_emb, r_emb, t_emb)

	def score_sp_(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		all_t_embs: torch.Tensor,
	) -> torch.Tensor:
		num_candidates = all_t_embs.size(0)
		batch_size = h_emb.size(0)
		chunk_size = self._entity_chunk_size(batch_size)
		scores = h_emb.new_empty(batch_size, num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			scores[:, start:end] = self._score_tensor(h_emb, r_emb, all_t_embs[start:end])
		return scores

	def score_po_(
		self,
		all_h_embs: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		num_candidates = all_h_embs.size(0)
		batch_size = t_emb.size(0)
		chunk_size = self._entity_chunk_size(batch_size)
		scores = t_emb.new_empty(batch_size, num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			scores[:, start:end] = self._score_tensor(all_h_embs[start:end], r_emb, t_emb)
		return scores
