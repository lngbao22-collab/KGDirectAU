"""Pure pRotatE scorer operating on raw tensors only."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from base.kge_scorer import KGEScorer


def build_scorer(args) -> pRotatEScorer:
	return pRotatEScorer(args)


@torch.no_grad()
def normalize_protate_phases(model) -> None:
	"""Wrap entity/relation embeddings so pRotatE phases stay in [-pi, pi].

	Matches the original RotatE/pRotatE reference implementation, which maps
	raw tables through ``embedding / (embedding_range / pi)`` before scoring.
	Without wrapping, Adagrad can drift embeddings to large values; ``sin(phase)``
	then oscillates and 1-vs-all ranks stay near random even while negsamp loss falls.
	"""

	from utils.device import get_model_obj

	model_obj = get_model_obj(model)
	scorer = getattr(model_obj, 'scorer', None)
	embedding_range = float(getattr(scorer, 'embedding_range', 0.0) or 0.0)
	if embedding_range <= 0.0:
		return
	phase_scale = embedding_range / math.pi

	for attr in ('ent_embedder', 'rel_embedder'):
		embedder = getattr(model_obj, attr, None)
		if embedder is None or not hasattr(embedder, 'weight'):
			continue
		embeddings = embedder.weight.data
		phases = embeddings / phase_scale
		phases = phases + math.pi
		phases = torch.remainder(phases, 2.0 * math.pi)
		phases = phases - math.pi
		embedder.weight.data[:] = phases * phase_scale


class pRotatEScorer(KGEScorer):
	"""pRotatE score function with explicit 1-to-1 and 1-vs-All tensor paths."""

	bidirectional_score_batch = True
	kgau_alignment_mode = 'sin_phase'

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		self.dim = int(getattr(args, "dim", 0) or 0)
		margin_value = getattr(args, "margin", None)
		self.margin = float(6.0 if margin_value is None else margin_value)
		epsilon = float(getattr(args, "epsilon", 2.0))
		self.embedding_range = float((self.margin + epsilon) / max(self.dim, 1))
		modulus_init = getattr(args, 'modulus', None)
		if modulus_init is None:
			modulus_init = 0.5 * self.embedding_range
		self.modulus = nn.Parameter(torch.tensor(float(modulus_init)))

	def _phase(self, embeddings: torch.Tensor) -> torch.Tensor:
		"""Map raw tensors into the phase space used by pRotatE."""

		return embeddings / (self.embedding_range / math.pi)

	def _score_phase(self, phase: torch.Tensor) -> torch.Tensor:
		return self.margin - torch.abs(torch.sin(phase)).sum(dim=-1) * self.modulus

	def score_spo(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return standard pRotatE tail scores for matching batches of triples."""

		phase = self._phase(h_emb) + self._phase(r_emb) - self._phase(t_emb)
		return self._score_phase(phase)

	def score_po(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return standard pRotatE head scores for matching batches of triples."""

		phase = self._phase(h_emb) + (self._phase(r_emb) - self._phase(t_emb))
		return self._score_phase(phase)

	def score_spo_candidates(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		"""Score many tail candidates per row: ``h_emb,r_emb`` are [B, D], ``t_emb`` is [B, C, D]."""

		phase = (
			self._phase(h_emb).unsqueeze(1)
			+ self._phase(r_emb).unsqueeze(1)
			- self._phase(t_emb)
		)
		return self._score_phase(phase)

	def score_po_candidates(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		"""Score many head candidates per row: ``h_emb`` is [B, C, D], ``r_emb,t_emb`` are [B, D]."""

		phase = self._phase(h_emb) + (
			self._phase(r_emb).unsqueeze(1) - self._phase(t_emb).unsqueeze(1)
		)
		return self._score_phase(phase)

	def score_sp_(self, h_emb: torch.Tensor, r_emb: torch.Tensor, all_t_embs: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all pRotatE tail scores using raw tensor broadcasting."""

		phase = (
			self._phase(h_emb).unsqueeze(1)
			+ self._phase(r_emb).unsqueeze(1)
			- self._phase(all_t_embs).unsqueeze(0)
		)
		return self._score_phase(phase)

	def score_po_(self, all_h_embs: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all pRotatE head scores (LibKGE ``_po`` combine)."""

		phase = (
			self._phase(all_h_embs).unsqueeze(0)
			+ (self._phase(r_emb).unsqueeze(1) - self._phase(t_emb).unsqueeze(1))
		)
		return self._score_phase(phase)

	def build_query(self, h_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
		"""Tail-prediction query vectors for cosine / Lp-distance link prediction."""

		return self._phase(h_emb) + self._phase(r_emb)

	def build_po_query(self, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Head-prediction query vectors for cosine / Lp-distance link prediction."""

		return self._phase(r_emb) - self._phase(t_emb)

	def au_representations(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		**kwargs,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		phase_head = self._phase(h_emb)
		phase_relation = self._phase(r_emb)
		phase_tail = self._phase(t_emb)
		return phase_head + phase_relation, phase_tail, phase_head

	def au_entity_embeddings(self, entity_emb: torch.Tensor) -> torch.Tensor:
		return self._phase(entity_emb)
