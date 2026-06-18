"""Abstract and unified model interfaces for KGDirectAU."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.embeddings import load_relation_to_idx
from data.dict_hub import get_entity_dict


class BaseModel(nn.Module, ABC):
	"""Abstract base for text encoders (e.g. SimKGC) with dict-based forward passes."""

	@abstractmethod
	def forward(self, *args, **kwargs) -> dict:
		"""Run a forward pass and return model-specific outputs."""

	@abstractmethod
	def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
		"""Convert model outputs into logits/labels for the training objective."""


class KGEModel(nn.Module):
	"""LibKGE-style binder that ties entity/relation embedders to a pure relational scorer.

	The scorer owns all tensor math (scoring, ``build_query``, ``au_representations``).
	This class owns index lookups, optional auxiliary relation embedders, normalization
	flags, and 1-vs-all delegation.
	"""

	def __init__(
		self,
		ent_embedder: nn.Module,
		rel_embedder: nn.Module,
		scorer: nn.Module,
		args: Any | None = None,
		aux_embedders: Mapping[str, nn.Module] | None = None,
	):
		super().__init__()
		self.ent_embedder = ent_embedder
		self.rel_embedder = rel_embedder
		self.scorer = scorer
		self.args = args
		self.aux_embedders = nn.ModuleDict(dict(aux_embedders or {}))
		self.rel_to_idx = load_relation_to_idx(args) if args is not None else {}
		self.normalize_lp_scores = _normalize_lp_flag(args) if args is not None else False
		self.normalize_au_vectors = _normalize_au_flag(args) if args is not None else False

	@property
	def bidirectional_score_batch(self) -> bool:
		return bool(getattr(self.scorer, 'bidirectional_score_batch', False))

	@property
	def kgau_alignment_mode(self) -> str | None:
		return getattr(self.scorer, 'kgau_alignment_mode', None)

	def get_s_embedder(self) -> nn.Module:
		return self.ent_embedder

	def get_o_embedder(self) -> nn.Module:
		return self.ent_embedder

	def get_p_embedder(self) -> nn.Module:
		return self.rel_embedder

	def get_scorer(self) -> nn.Module:
		return self.scorer

	def embed_s(self, indices: torch.Tensor) -> torch.Tensor:
		return self._embed(self.ent_embedder, indices)

	def embed_p(self, indices: torch.Tensor) -> torch.Tensor:
		return self._embed(self.rel_embedder, indices)

	def embed_o(self, indices: torch.Tensor) -> torch.Tensor:
		return self._embed(self.ent_embedder, indices)

	def embed_all_entities(self) -> torch.Tensor:
		return self._embed_all(self.ent_embedder)

	@staticmethod
	def _embed(embedder: nn.Module, indices: torch.Tensor) -> torch.Tensor:
		if hasattr(embedder, 'embed'):
			return embedder.embed(indices)
		return embedder(indices)

	@staticmethod
	def _embed_all(embedder: nn.Module) -> torch.Tensor:
		if hasattr(embedder, 'embed_all'):
			return embedder.embed_all()
		if hasattr(embedder, 'get_all'):
			return embedder.get_all()
		raise AttributeError(f'{type(embedder).__name__} has no embed_all/get_all method')

	def _scorer_kwargs(self, p: torch.Tensor | None = None, **extra: Any) -> dict[str, Any]:
		"""Forward auxiliary relation embeddings and caller overrides to the scorer."""

		kwargs = dict(extra)
		if p is not None:
			for key, embedder in self.aux_embedders.items():
				kwargs[f'{key}_emb'] = self._embed(embedder, p)
		return kwargs

	def _normalize_lp_vector(self, vectors: torch.Tensor) -> torch.Tensor:
		if not self.normalize_lp_scores:
			return vectors
		return F.normalize(vectors, p=2, dim=-1)

	def _normalize_au_vector(self, vectors: torch.Tensor) -> torch.Tensor:
		if not self.normalize_au_vectors:
			return vectors
		return F.normalize(vectors, p=2, dim=-1)

	def _cosine_similarity_scores(
		self,
		query_vectors: torch.Tensor,
		candidate_vectors: torch.Tensor,
	) -> torch.Tensor:
		"""Dot-product scores after optional L2 normalization (= cosine similarity)."""

		query_vectors = self._normalize_lp_vector(query_vectors)
		candidate_vectors = self._normalize_lp_vector(candidate_vectors)
		return torch.mm(query_vectors, candidate_vectors.t())

	def _tail_query_vectors(self, s: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
		return self.scorer.build_query(self.embed_s(s), self.embed_p(p))

	def _head_query_vectors(self, p: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
		return self.scorer.build_po_query(self.embed_p(p), self.embed_o(o))

	def score_spo(
		self,
		s: torch.Tensor,
		p: torch.Tensor,
		o: torch.Tensor,
		**kwargs: Any,
	) -> torch.Tensor:
		if self.normalize_lp_scores:
			query = self._tail_query_vectors(s, p)
			tail = self.embed_o(o)
			return self._cosine_similarity_scores(query, tail).diag()
		scorer_kwargs = {**self._scorer_kwargs(p), **kwargs}
		return self.scorer.score_spo(
			self.embed_s(s),
			self.embed_p(p),
			self.embed_o(o),
			**scorer_kwargs,
		)

	def score_sp_(
		self,
		s: torch.Tensor,
		p: torch.Tensor,
		all_o_embs: torch.Tensor | None = None,
		**kwargs: Any,
	) -> torch.Tensor:
		if all_o_embs is None:
			all_o_embs = self.embed_all_entities()
		if self.normalize_lp_scores:
			return self._cosine_similarity_scores(self._tail_query_vectors(s, p), all_o_embs)
		scorer_kwargs = {**self._scorer_kwargs(p), **kwargs}
		return self.scorer.score_sp_(
			self.embed_s(s),
			self.embed_p(p),
			all_o_embs,
			**scorer_kwargs,
		)

	def score_sp(
		self,
		s: torch.Tensor,
		p: torch.Tensor,
		o: torch.Tensor | None = None,
		**kwargs: Any,
	) -> torch.Tensor:
		if o is None:
			return self.score_sp_(s, p, **kwargs)
		scorer_kwargs = {**self._scorer_kwargs(p), **kwargs}
		return self.scorer.score_sp_(
			self.embed_s(s),
			self.embed_p(p),
			self._embed(self.ent_embedder, o),
			**scorer_kwargs,
		)

	def score_po_(
		self,
		p: torch.Tensor,
		o: torch.Tensor,
		all_s_embs: torch.Tensor | None = None,
		**kwargs: Any,
	) -> torch.Tensor:
		if not hasattr(self.scorer, 'score_po_'):
			raise NotImplementedError(f'{type(self.scorer).__name__} does not implement score_po_')
		if all_s_embs is None:
			all_s_embs = self.embed_all_entities()
		if self.normalize_lp_scores:
			return self._cosine_similarity_scores(self._head_query_vectors(p, o), all_s_embs)
		scorer_kwargs = {**self._scorer_kwargs(p), **kwargs}
		return self.scorer.score_po_(
			all_s_embs,
			self.embed_p(p),
			self.embed_o(o),
			**scorer_kwargs,
		)

	def score_po(
		self,
		p: torch.Tensor,
		o: torch.Tensor,
		s: torch.Tensor | None = None,
		**kwargs: Any,
	) -> torch.Tensor:
		if s is None:
			return self.score_po_(p, o, **kwargs)
		if hasattr(self.scorer, 'score_po'):
			scorer_kwargs = {**self._scorer_kwargs(p), **kwargs}
			return self.scorer.score_po(
				self._embed(self.ent_embedder, s),
				self.embed_p(p),
				self.embed_o(o),
				**scorer_kwargs,
			)
		raise NotImplementedError(f'{type(self.scorer).__name__} does not implement score_po')

	def query_all_entities_scores(self, s: torch.Tensor, p: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		return self.score_sp_(s, p, **kwargs)

	def predict_tail_sp_(self, h_idx: torch.Tensor, r_idx: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		return self.score_sp_(h_idx, r_idx, **kwargs)

	def predict_head_po_(self, r_idx: torch.Tensor, t_idx: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		return self.score_po_(r_idx, t_idx, **kwargs)

	@property
	def device(self) -> torch.device:
		return next(self.parameters()).device

	def entity_embeddings(
		self,
		device: torch.device | None = None,
		max_samples: int | None = None,
	) -> torch.Tensor:
		vectors = self.embed_all_entities()
		if max_samples is not None and int(max_samples) > 0 and vectors.size(0) > int(max_samples):
			indices = torch.randperm(vectors.size(0), device=vectors.device)[: int(max_samples)]
			vectors = vectors.index_select(0, indices)
		return vectors.to(device) if device is not None else vectors

	def au_entity_embeddings(self, device: torch.device | None = None, **kwargs: Any) -> torch.Tensor:
		if hasattr(self.scorer, 'au_entity_embeddings'):
			vectors = self.scorer.au_entity_embeddings(self.embed_all_entities())
			return vectors.to(device) if device is not None else vectors
		return self.entity_embeddings(device=device, **kwargs)

	def get_queries_targets(self, s: torch.Tensor, p: torch.Tensor, o: torch.Tensor):
		"""AU (query, tail, head) vectors — delegated to the scorer."""

		h = self.embed_s(s)
		r = self.embed_p(p)
		t = self.embed_o(o)
		query, tail, head = self.scorer.au_representations(h, r, t, **self._scorer_kwargs(p))
		if self.normalize_au_vectors:
			query = self._normalize_au_vector(query)
			tail = self._normalize_au_vector(tail)
			head = self._normalize_au_vector(head)
		return query, tail, head

	def score_batch(self, head_ids, relations, tail_entity_ids) -> torch.Tensor:
		entity_dict = get_entity_dict()
		device = self.device
		rel_lookup = lambda relation: resolve_relation_index(relation, self.rel_to_idx)
		head_indices = as_index_tensor(head_ids, entity_dict.entity_to_idx, device)
		relation_indices = as_index_tensor(relations, rel_lookup, device)
		tail_indices = as_index_tensor(tail_entity_ids, entity_dict.entity_to_idx, device)
		return self.score_spo(head_indices, relation_indices, tail_indices)

	def forward(self, src: torch.Tensor, rel: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
		return self.score_spo(src, rel, dst)


def resolve_relation_index(relation: str, relation_to_idx: dict[str, int]) -> int:
	if relation in relation_to_idx:
		return relation_to_idx[relation]
	normalized = ' '.join(relation.split())
	if normalized in relation_to_idx:
		return relation_to_idx[normalized]
	if relation.startswith('inverse '):
		base_relation = relation[len('inverse '):]
		inverse_relation = f'inverse {base_relation}'
		if inverse_relation in relation_to_idx:
			return relation_to_idx[inverse_relation]
		if base_relation in relation_to_idx:
			return relation_to_idx[base_relation]
	raise KeyError(relation)


def as_index_tensor(values, lookup, device: torch.device) -> torch.Tensor:
	if torch.is_tensor(values):
		return values.to(device=device, dtype=torch.long)
	return torch.tensor([lookup(value) for value in values], dtype=torch.long, device=device)


def _normalize_lp_flag(args) -> bool:
	value = getattr(args, 'normalize_lp_scores', None)
	if value is not None:
		return bool(value)
	return False


def _normalize_au_flag(args) -> bool:
	value = getattr(args, 'normalize_au_vectors', None)
	if value is not None:
		return bool(value)
	model = str(getattr(args, 'model', '') or '')
	if not model.endswith('-AU'):
		return False
	if 'protate' in model.lower():
		return False
	return True
