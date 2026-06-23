"""Pure TransE scorer operating on raw tensors only."""

from __future__ import annotations

import torch

from base.kge_scorer import KGEScorer


def build_scorer(args) -> TransEScorer:
	return TransEScorer(args)


class TransEScorer(KGEScorer):
	"""TransE score function with explicit 1-to-1 and 1-vs-All tensor paths.

	Matches ``KnowledgeGraphEmbedding`` (Sun et al.): higher score is better,
	``gamma - ||h + r - t||_1`` for tail prediction and
	``gamma - ||h + (r - t)||_1`` for head prediction.
	"""

	bidirectional_score_batch = True

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		self.dim = int(getattr(args, 'dim', 0) or 0)
		margin_value = getattr(args, 'margin', None)
		if margin_value is None:
			margin_value = getattr(args, 'gamma', 6.0)
		self.gamma = float(margin_value)
		epsilon = float(getattr(args, 'epsilon', 2.0))
		self.embedding_range = float((self.gamma + epsilon) / max(self.dim, 1))

	def _entity_chunk_size(self, batch_size: int) -> int:
		"""Candidate chunk size for 1-vs-all scoring (controls peak GPU memory)."""

		configured = int(getattr(self.args, 'eval_entity_chunk_size', 256) or 256)
		bytes_budget = int(getattr(self.args, 'eval_entity_chunk_bytes', 128 * 1024 * 1024) or 128 * 1024 * 1024)
		per_candidate = max(1, batch_size * self.dim * 4 * 2)
		memory_limit = max(1, bytes_budget // per_candidate)
		return max(1, min(configured, memory_limit))

	def _score_distance(self, diff: torch.Tensor) -> torch.Tensor:
		return self.gamma - torch.norm(diff, p=1, dim=-1)

	def score_spo(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return standard TransE tail scores for matching batches of triples."""

		return self._score_distance((h_emb + r_emb) - t_emb)

	def score_po(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return standard TransE head scores for matching batches of triples."""

		return self._score_distance(h_emb + (r_emb - t_emb))

	def score_spo_candidates(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		"""Score many tail candidates per row: ``h_emb,r_emb`` are [B, D], ``t_emb`` is [B, C, D]."""

		diff = (h_emb.unsqueeze(1) + r_emb.unsqueeze(1)) - t_emb
		return self._score_distance(diff)

	def score_po_candidates(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		"""Score many head candidates per row: ``h_emb`` is [B, C, D], ``r_emb,t_emb`` are [B, D]."""

		diff = h_emb + (r_emb.unsqueeze(1) - t_emb.unsqueeze(1))
		return self._score_distance(diff)

	def _score_1vsall(
		self,
		query: torch.Tensor,
		candidates: torch.Tensor,
	) -> torch.Tensor:
		num_candidates = candidates.size(0)
		batch_size = query.size(0)
		chunk_size = self._entity_chunk_size(batch_size)
		scores = query.new_empty(batch_size, num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			diff = query.unsqueeze(1) - candidates[start:end].unsqueeze(0)
			scores[:, start:end] = self._score_distance(diff)
		return scores

	def score_sp_(self, h_emb: torch.Tensor, r_emb: torch.Tensor, all_t_embs: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all TransE tail scores using LibKGE-style sp_ broadcasting."""

		return self._score_1vsall(h_emb + r_emb, all_t_embs)

	def score_po_(self, all_h_embs: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all TransE head scores for each (relation, tail) query."""

		query = r_emb - t_emb
		return self._score_1vsall(query, all_h_embs)

	def build_query(self, h_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
		return h_emb + r_emb

	def build_po_query(self, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		return r_emb - t_emb
