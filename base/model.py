"""Abstract and unified model interfaces for KGDirectAU."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.embeddings import ParameterEmbedder, _scaled_init, load_relation_to_idx
from data.dataset import Example, load_data
from data.dict_hub import get_entity_dict
from models.embedders.lookup_embedder import LookupEmbedder
from models.scorers.complex_scorer import ComplExScorer
from models.scorers.dabr_scorer import DaBRScorer
from models.scorers.distmult_scorer import DistMultScorer
from models.scorers.protate_scorer import pRotatEScorer
from models.scorers.rotate_scorer import RotatEScorer


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

	The scorer receives raw embedding tensors only; this class owns all index lookups and
	1-vs-all broadcasting (``score_sp_`` / ``score_po_``).
	"""

	def __init__(
		self,
		ent_embedder: nn.Module,
		rel_embedder: nn.Module,
		scorer: nn.Module,
		args: Any | None = None,
	):
		super().__init__()
		self.ent_embedder = ent_embedder
		self.rel_embedder = rel_embedder
		self.scorer = scorer
		self.args = args
		self.rel_to_idx = load_relation_to_idx(args) if args is not None else {}
		self.normalize_lp_scores = _normalize_lp_flag(args) if args is not None else False

	def get_s_embedder(self) -> nn.Module:
		"""Return the subject (head) entity embedder."""

		return self.ent_embedder

	def get_o_embedder(self) -> nn.Module:
		"""Return the object (tail) entity embedder."""

		return self.ent_embedder

	def get_p_embedder(self) -> nn.Module:
		"""Return the relation embedder."""

		return self.rel_embedder

	def get_scorer(self) -> nn.Module:
		"""Return the relational score function."""

		return self.scorer

	def embed_s(self, indices: torch.Tensor) -> torch.Tensor:
		"""Look up subject entity embeddings."""

		return self._embed(self.ent_embedder, indices)

	def embed_p(self, indices: torch.Tensor) -> torch.Tensor:
		"""Look up relation embeddings."""

		return self._embed(self.rel_embedder, indices)

	def embed_o(self, indices: torch.Tensor) -> torch.Tensor:
		"""Look up object entity embeddings."""

		return self._embed(self.ent_embedder, indices)

	def embed_all_entities(self) -> torch.Tensor:
		"""Return the full entity embedding matrix for 1-vs-all scoring."""

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

	def _scorer_kwargs(self) -> dict:
		"""Extra keyword arguments forwarded to the scorer (override in subclasses)."""

		return {}

	def _normalize_lp_vector(self, vectors: torch.Tensor) -> torch.Tensor:
		"""L2-normalize vectors when ``normalize_lp_scores`` is enabled."""

		if not self.normalize_lp_scores:
			return vectors
		return F.normalize(vectors, p=2, dim=-1)

	def _link_prediction_scores(
		self,
		query_vectors: torch.Tensor,
		candidate_vectors: torch.Tensor,
	) -> torch.Tensor:
		"""Dot-product link-prediction scores with optional L2 normalization."""

		query_vectors = self._normalize_lp_vector(query_vectors)
		candidate_vectors = self._normalize_lp_vector(candidate_vectors)
		return torch.mm(query_vectors, candidate_vectors.t())

	def _lp_query_vectors(self, s: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
		"""Build tail-prediction query vectors (ComplEx ``build_query`` or DistMult ``h * r``)."""

		head = self.embed_s(s)
		relation = self.embed_p(p)
		if hasattr(self.scorer, 'build_query'):
			return self.scorer.build_query(head, relation)
		return head * relation

	def _lp_po_query_vectors(self, p: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
		"""Build head-prediction query vectors (DistMult-style ``t * r``)."""

		tail = self.embed_o(o)
		relation = self.embed_p(p)
		return tail * relation

	def score_spo(
		self,
		s: torch.Tensor,
		p: torch.Tensor,
		o: torch.Tensor,
		**kwargs: Any,
	) -> torch.Tensor:
		"""Score aligned (subject, predicate, object) triples by index."""

		if self.normalize_lp_scores:
			query = self._lp_query_vectors(s, p)
			tail = self.embed_o(o)
			return self._link_prediction_scores(query, tail).diag()
		scorer_kwargs = {**self._scorer_kwargs(), **kwargs}
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
		"""Score each (s, p) pair against all (or supplied) tail entities."""

		if all_o_embs is None:
			all_o_embs = self.embed_all_entities()
		if self.normalize_lp_scores:
			return self._link_prediction_scores(self._lp_query_vectors(s, p), all_o_embs)
		scorer_kwargs = {**self._scorer_kwargs(), **kwargs}
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
		"""LibKGE alias: 1-vs-all tail prediction, or restricted candidate set when ``o`` is set."""

		if o is None:
			return self.score_sp_(s, p, **kwargs)
		return self.scorer.score_sp_(
			self.embed_s(s),
			self.embed_p(p),
			self._embed(self.ent_embedder, o),
			**{**self._scorer_kwargs(), **kwargs},
		)

	def score_po_(
		self,
		p: torch.Tensor,
		o: torch.Tensor,
		all_s_embs: torch.Tensor | None = None,
		**kwargs: Any,
	) -> torch.Tensor:
		"""Score each (p, o) pair against all (or supplied) head entities."""

		if not hasattr(self.scorer, 'score_po_'):
			raise NotImplementedError(f'{type(self.scorer).__name__} does not implement score_po_')
		if all_s_embs is None:
			all_s_embs = self.embed_all_entities()
		if self.normalize_lp_scores:
			return self._link_prediction_scores(self._lp_po_query_vectors(p, o), all_s_embs)
		scorer_kwargs = {**self._scorer_kwargs(), **kwargs}
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
		"""LibKGE alias: 1-vs-all head prediction, or restricted candidate set when ``s`` is set."""

		if s is None:
			return self.score_po_(p, o, **kwargs)
		if hasattr(self.scorer, 'score_po'):
			scorer_kwargs = {**self._scorer_kwargs(), **kwargs}
			return self.scorer.score_po(
				self._embed(self.ent_embedder, s),
				self.embed_p(p),
				self.embed_o(o),
				**scorer_kwargs,
			)
		raise NotImplementedError(f'{type(self.scorer).__name__} does not implement score_po')

	def query_all_entities_scores(self, s: torch.Tensor, p: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		"""Training/eval helper used by softmax and KGAU strategies."""

		return self.score_sp_(s, p, **kwargs)

	def predict_tail_sp_(self, h_idx: torch.Tensor, r_idx: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		"""LibKGE ``sp_`` mode: score all tail entities for each (head, relation) pair."""

		return self.score_sp_(h_idx, r_idx, **kwargs)

	def predict_head_po_(self, r_idx: torch.Tensor, t_idx: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		"""LibKGE ``_po`` mode: score all head entities for each (relation, tail) pair."""

		return self.score_po_(r_idx, t_idx, **kwargs)

	@property
	def device(self) -> torch.device:
		"""Device of the first model parameter."""

		return next(self.parameters()).device

	def entity_embeddings(self, device: torch.device | None = None) -> torch.Tensor:
		"""Return the full entity table (optionally moved to ``device``)."""

		vectors = self.embed_all_entities()
		return vectors.to(device) if device is not None else vectors

	def get_queries_targets(self, s: torch.Tensor, p: torch.Tensor, o: torch.Tensor):
		"""Default AU query/target/head vectors using embedders and optional scorer query builder."""

		query = self._lp_query_vectors(s, p)
		head = self.embed_s(s)
		tail = self.embed_o(o)
		if self.normalize_lp_scores:
			query = self._normalize_lp_vector(query)
			tail = self._normalize_lp_vector(tail)
			head = self._normalize_lp_vector(head)
		return query, tail, head

	def score_batch(self, head_ids, relations, tail_entity_ids) -> torch.Tensor:
		"""Score a batch of (head, relation, tail) triples by entity/relation id."""

		entity_dict = get_entity_dict()
		device = self.device
		rel_lookup = lambda relation: resolve_relation_index(relation, self.rel_to_idx)
		head_indices = as_index_tensor(head_ids, entity_dict.entity_to_idx, device)
		relation_indices = as_index_tensor(relations, rel_lookup, device)
		tail_indices = as_index_tensor(tail_entity_ids, entity_dict.entity_to_idx, device)
		return self.score_spo(head_indices, relation_indices, tail_indices)

	def forward(self, src: torch.Tensor, rel: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
		"""Default training forward: score a batch of triple indices."""

		return self.score_spo(src, rel, dst)


def resolve_relation_index(relation: str, relation_to_idx: dict[str, int]) -> int:
	"""Map a relation string to its embedding index."""

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
	return str(getattr(args, 'model', '')).endswith('-AU')


class _RelationLookupMixin:
	rel_to_idx: dict[str, int]
	entity_dict = None

	def _relation_to_idx(self, relation: str) -> int:
		if relation in self.rel_to_idx:
			return self.rel_to_idx[relation]
		if relation.startswith('inverse '):
			base_relation = relation[len('inverse '):]
			inverse_relation = f'inverse {base_relation}'
			if inverse_relation not in self.rel_to_idx and base_relation in self.rel_to_idx:
				return self.rel_to_idx[base_relation]
		if relation.startswith('inverse_'):
			base_relation = relation[len('inverse_'):]
			candidate = '_' + base_relation if not base_relation.startswith('_') else base_relation
			if candidate in self.rel_to_idx:
				return self.rel_to_idx[candidate]
		normalized = ' '.join(relation.split())
		if normalized in self.rel_to_idx:
			return self.rel_to_idx[normalized]
		raise KeyError(relation)


class DistMultModel(_RelationLookupMixin, KGEModel):
	"""DistMult bound through the unified ``KGEModel`` binder."""

	def __init__(self, args):
		self.entity_dict = get_entity_dict()
		self.rel_to_idx = load_relation_to_idx(args)
		self.dim = int(args.dim)
		self.normalize_lp_scores = _normalize_lp_flag(args)
		self.adversarial_training = bool(getattr(args, 'adversarial_training', False))

		n_ent = len(self.entity_dict)
		n_rel = max(len(self.rel_to_idx), 1)
		ent_embedder = LookupEmbedder(n_ent, self.dim, args)
		rel_embedder = LookupEmbedder(n_rel, self.dim, args)
		if self.adversarial_training:
			epsilon = 2.0
			margin = float(getattr(args, 'margin', 200.0))
			embedding_range = (margin + epsilon) / self.dim
			nn.init.uniform_(ent_embedder.embedding.weight, a=-embedding_range, b=embedding_range)
			nn.init.uniform_(rel_embedder.embedding.weight, a=-embedding_range, b=embedding_range)
		else:
			_scaled_init(ent_embedder, self.dim)
			_scaled_init(rel_embedder, self.dim)

		super().__init__(ent_embedder, rel_embedder, DistMultScorer(args), args)
		self.ent_embed = ent_embedder
		self.rel_embed = rel_embedder

	def _link_prediction_scores(self, query_vectors: torch.Tensor, candidate_vectors: torch.Tensor) -> torch.Tensor:
		if self.normalize_lp_scores:
			query_vectors = F.normalize(query_vectors, p=2, dim=-1)
			candidate_vectors = F.normalize(candidate_vectors, p=2, dim=-1)
		return torch.mm(query_vectors, candidate_vectors.t())

	def forward(self, src, rel=None, dst=None):
		if torch.is_tensor(src) and src.dim() == 2 and src.size(-1) == 3 and isinstance(dst, str):
			return self._adversarial_forward(src, rel, dst)
		if self.normalize_lp_scores:
			h, r, t = self.embed_s(src), self.embed_p(rel), self.embed_o(dst)
			return self._link_prediction_scores(h * r, t).diag()
		return self.score_spo(src, rel, dst)

	def get_queries_targets(self, src, rel, dst):
		h, r, t = self.embed_s(src), self.embed_p(rel), self.embed_o(dst)
		return h * r, t, h

	def query_all_entities_scores(self, src: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
		if self.normalize_lp_scores:
			h, r = self.embed_s(src), self.embed_p(rel)
			return self._link_prediction_scores(h * r, self.embed_all_entities())
		return self.score_sp_(src, rel)

	def predict_tail_sp_(self, h_idx: torch.Tensor, r_idx: torch.Tensor, **kwargs) -> torch.Tensor:
		if self.normalize_lp_scores:
			h, r = self.embed_s(h_idx), self.embed_p(r_idx)
			return self._link_prediction_scores(h * r, self.embed_all_entities())
		return super().predict_tail_sp_(h_idx, r_idx, **kwargs)

	def hr_embeddings(self, examples: Sequence[Example], device: torch.device | None = None) -> torch.Tensor:
		device = device or self.ent_embedder.embedding.weight.device
		head_indices = as_index_tensor([ex.head_id for ex in examples], self.entity_dict.entity_to_idx, device)
		relation_indices = as_index_tensor([ex.relation for ex in examples], self._relation_to_idx, device)
		return self.embed_s(head_indices) * self.embed_p(relation_indices)

	def score_batch(self, head_ids, relations, tail_entity_ids) -> torch.Tensor:
		device = self.ent_embedder.embedding.weight.device
		head_indices = as_index_tensor(head_ids, self.entity_dict.entity_to_idx, device)
		relation_indices = as_index_tensor(relations, self._relation_to_idx, device)
		candidate_indices = as_index_tensor(tail_entity_ids, self.entity_dict.entity_to_idx, device)
		h, r = self.embed_s(head_indices), self.embed_p(relation_indices)
		return self._link_prediction_scores(h * r, self.embed_o(candidate_indices))

	def adversarial_l3_regularization(self) -> torch.Tensor:
		return self.ent_embedder.embedding.weight.norm(p=3) ** 3 + self.rel_embedder.embedding.weight.norm(p=3) ** 3

	def _distmult_score(self, head, relation, tail, mode: str) -> torch.Tensor:
		if mode == 'head-batch':
			score = head * (relation * tail)
		else:
			score = (head * relation) * tail
		return score.sum(dim=2)

	def _adversarial_score(self, positive_sample, negative_sample=None, mode: str = 'single') -> torch.Tensor:
		if mode == 'single':
			head = self.embed_s(positive_sample[:, 0]).unsqueeze(1)
			relation = self.embed_p(positive_sample[:, 1]).unsqueeze(1)
			tail = self.embed_o(positive_sample[:, 2]).unsqueeze(1)
		elif mode == 'head-batch':
			batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
			head = self.embed_s(negative_sample.reshape(-1)).reshape(batch_size, negative_sample_size, -1)
			relation = self.embed_p(positive_sample[:, 1]).unsqueeze(1)
			tail = self.embed_o(positive_sample[:, 2]).unsqueeze(1)
		elif mode == 'tail-batch':
			batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
			head = self.embed_s(positive_sample[:, 0]).unsqueeze(1)
			relation = self.embed_p(positive_sample[:, 1]).unsqueeze(1)
			tail = self.embed_o(negative_sample.reshape(-1)).reshape(batch_size, negative_sample_size, -1)
		else:
			raise ValueError(f'mode {mode} not supported')
		return self._distmult_score(head, relation, tail, mode)

	def _adversarial_forward(self, positive_sample, negative_sample=None, mode: str = 'single') -> dict:
		pos_scores = self._adversarial_score(positive_sample, mode='single')
		neg_scores = None
		if negative_sample is not None:
			neg_scores = self._adversarial_score(positive_sample, negative_sample=negative_sample, mode=mode)
		return {'positive_scores': pos_scores, 'negative_scores': neg_scores}


class ComplExModel(_RelationLookupMixin, KGEModel):
	"""ComplEx with split real/imag lookup tables bound through ``KGEModel``."""

	def __init__(self, args):
		self.entity_dict = get_entity_dict()
		self.rel_to_idx = load_relation_to_idx(args)
		self.dim = int(args.dim)
		self.normalize_lp_scores = _normalize_lp_flag(args)

		n_ent = len(self.entity_dict)
		n_rel = max(len(self.rel_to_idx), 1)
		self.ent_re_embed = LookupEmbedder(n_ent, self.dim, args)
		self.ent_im_embed = LookupEmbedder(n_ent, self.dim, args)
		self.rel_re_embed = LookupEmbedder(n_rel, self.dim, args)
		self.rel_im_embed = LookupEmbedder(n_rel, self.dim, args)
		if getattr(args, 'init_scaled', True):
			for module in (self.ent_re_embed, self.ent_im_embed, self.rel_re_embed, self.rel_im_embed):
				_scaled_init(module, self.dim)

		super().__init__(self.ent_re_embed, self.rel_re_embed, ComplExScorer(args), args)

	def embed_s(self, indices: torch.Tensor) -> torch.Tensor:
		return self._embed_entity(indices)

	def embed_o(self, indices: torch.Tensor) -> torch.Tensor:
		return self._embed_entity(indices)

	def embed_p(self, indices: torch.Tensor) -> torch.Tensor:
		return self._embed_relation(indices)

	def embed_all_entities(self) -> torch.Tensor:
		device = self.ent_re_embed.embedding.weight.device
		return self._embed_entity(torch.arange(len(self.entity_dict), device=device))

	def _needs_relation_conjugate(self, relation: str) -> bool:
		if not str(relation).startswith('inverse '):
			return False
		base_relation = str(relation)[len('inverse '):]
		return f'inverse {base_relation}' not in self.rel_to_idx

	def _embed_entity(self, indices: torch.Tensor) -> torch.Tensor:
		return torch.cat([self.ent_re_embed(indices), self.ent_im_embed(indices)], dim=-1)

	def _embed_relation(self, relation_indices: torch.Tensor, relations: Sequence[str] | None = None) -> torch.Tensor:
		r_re = self.rel_re_embed(relation_indices)
		r_im = self.rel_im_embed(relation_indices)
		if relations is not None:
			mask = torch.tensor(
				[self._needs_relation_conjugate(relation) for relation in relations],
				dtype=torch.bool,
				device=r_re.device,
			)
			if mask.any():
				r_im = torch.where(mask.unsqueeze(-1), -r_im, r_im)
		return torch.cat([r_re, r_im], dim=-1)

	def _query_vectors(self, head_indices: torch.Tensor, relation_indices: torch.Tensor) -> torch.Tensor:
		h_re, h_im = self.ent_re_embed(head_indices), self.ent_im_embed(head_indices)
		r_re, r_im = self.rel_re_embed(relation_indices), self.rel_im_embed(relation_indices)
		return torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)

	def _link_prediction_scores(self, query_vectors: torch.Tensor, candidate_vectors: torch.Tensor) -> torch.Tensor:
		if self.normalize_lp_scores:
			query_vectors = F.normalize(query_vectors, p=2, dim=-1)
			candidate_vectors = F.normalize(candidate_vectors, p=2, dim=-1)
		return torch.mm(query_vectors, candidate_vectors.t())

	def forward(self, src, rel, dst):
		if self.normalize_lp_scores:
			return self._link_prediction_scores(self._query_vectors(src, rel), self.embed_o(dst)).diag()
		return self.score_spo(src, rel, dst)

	def get_queries_targets(self, src, rel, dst):
		return self._query_vectors(src, rel), self.embed_o(dst), self.embed_s(src)

	def query_all_entities_scores(self, src: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
		if self.normalize_lp_scores:
			return self._link_prediction_scores(self._query_vectors(src, rel), self.embed_all_entities())
		return self.score_sp_(src, rel)

	def predict_tail_sp_(self, h_idx: torch.Tensor, r_idx: torch.Tensor, **kwargs) -> torch.Tensor:
		if self.normalize_lp_scores:
			return self._link_prediction_scores(self._query_vectors(h_idx, r_idx), self.embed_all_entities())
		return super().predict_tail_sp_(h_idx, r_idx, **kwargs)

	def hr_embeddings(self, examples: Sequence[Example], device: torch.device | None = None) -> torch.Tensor:
		device = device or self.ent_re_embed.embedding.weight.device
		head_indices = as_index_tensor([ex.head_id for ex in examples], self.entity_dict.entity_to_idx, device)
		relation_indices = as_index_tensor([ex.relation for ex in examples], self._relation_to_idx, device)
		return self._query_vectors(head_indices, relation_indices)

	def score_batch(self, head_ids, relations, tail_entity_ids) -> torch.Tensor:
		device = self.ent_re_embed.embedding.weight.device
		head_indices = as_index_tensor(head_ids, self.entity_dict.entity_to_idx, device)
		relation_indices = as_index_tensor(relations, self._relation_to_idx, device)
		candidate_indices = as_index_tensor(tail_entity_ids, self.entity_dict.entity_to_idx, device)
		return self._link_prediction_scores(
			self._query_vectors(head_indices, relation_indices),
			self.embed_o(candidate_indices),
		)


class RotatEModel(_RelationLookupMixin, KGEModel):
	"""RotatE with phase relations bound through ``KGEModel``."""

	bidirectional_score_batch = True

	def __init__(self, args):
		self.entity_dict = get_entity_dict()
		self.rel_to_idx = load_relation_to_idx(args)
		self.hidden_dim = int(getattr(args, 'dim', 500))
		margin = float(getattr(args, 'margin', 6.0))
		epsilon = 2.0
		embedding_range = (margin + epsilon) / self.hidden_dim

		n_ent = len(self.entity_dict)
		n_rel = max(len(self.rel_to_idx), 1)
		self.entity_embedding = nn.Parameter(torch.zeros(n_ent, self.hidden_dim * 2))
		self.relation_embedding = nn.Parameter(torch.zeros(n_rel, self.hidden_dim))
		nn.init.uniform_(self.entity_embedding, a=-embedding_range, b=embedding_range)
		nn.init.uniform_(self.relation_embedding, a=-embedding_range, b=embedding_range)

		ent_embedder = ParameterEmbedder(self.entity_embedding)
		rel_embedder = ParameterEmbedder(self.relation_embedding)
		super().__init__(ent_embedder, rel_embedder, RotatEScorer(args), args)

	@property
	def entity_dim(self):
		return self.hidden_dim * 2

	def forward(self, positive_sample, negative_sample=None, mode: str = 'single') -> dict:
		pos_scores = self._score(positive_sample, mode='single')
		neg_scores = None
		if negative_sample is not None:
			neg_scores = self._score(positive_sample, negative_sample=negative_sample, mode=mode)
		return {'positive_scores': pos_scores, 'negative_scores': neg_scores}

	def _score(self, positive_sample, negative_sample=None, mode: str = 'single') -> torch.Tensor:
		if mode == 'single':
			return self.score_spo(positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2]).unsqueeze(1)
		if mode == 'head-batch':
			batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
			h = self.embed_s(negative_sample.reshape(-1)).reshape(batch_size, negative_sample_size, -1)
			r = self.embed_p(positive_sample[:, 1]).unsqueeze(1)
			t = self.embed_o(positive_sample[:, 2]).unsqueeze(1)
			h_flat = h.reshape(-1, h.size(-1))
			r_flat = r.expand(-1, negative_sample_size, -1).reshape(-1, r.size(-1))
			t_flat = t.expand(-1, negative_sample_size, -1).reshape(-1, t.size(-1))
			return self.scorer.score_po(h_flat, r_flat, t_flat).reshape(batch_size, negative_sample_size, 1)
		if mode == 'tail-batch':
			batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
			h = self.embed_s(positive_sample[:, 0]).unsqueeze(1)
			r = self.embed_p(positive_sample[:, 1]).unsqueeze(1)
			t = self.embed_o(negative_sample.reshape(-1)).reshape(batch_size, negative_sample_size, -1)
			h_flat = h.expand(-1, negative_sample_size, -1).reshape(-1, h.size(-1))
			r_flat = r.expand(-1, negative_sample_size, -1).reshape(-1, r.size(-1))
			t_flat = t.reshape(-1, t.size(-1))
			return self.scorer.score_spo(h_flat, r_flat, t_flat).reshape(batch_size, negative_sample_size, 1)
		raise ValueError(f'mode {mode} not supported')

	def get_queries_targets(self, src, rel, dst):
		h, r, t = self.embed_s(src), self.embed_p(rel), self.embed_o(dst)
		h_re, h_im = torch.chunk(h, 2, dim=-1)
		r_phase = self.scorer._phase(r)
		q = torch.cat([
			h_re * torch.cos(r_phase) - h_im * torch.sin(r_phase),
			h_re * torch.sin(r_phase) + h_im * torch.cos(r_phase),
		], dim=-1)
		return q, t, h


class pRotatEModel(_RelationLookupMixin, KGEModel):
	"""pRotatE with sin-phase KGAU alignment bound through ``KGEModel``."""

	kga_u_alignment_mode = 'sin_phase'
	bidirectional_score_batch = True

	def __init__(self, args):
		self.entity_dict = get_entity_dict()
		self.rel_to_idx = load_relation_to_idx(args)
		self.hidden_dim = int(getattr(args, 'dim', 500))
		margin = float(getattr(args, 'margin', 6.0))
		epsilon = 2.0
		embedding_range = (margin + epsilon) / self.hidden_dim

		n_ent = len(self.entity_dict)
		n_rel = max(len(self.rel_to_idx), 1)
		self.entity_embedding = nn.Parameter(torch.zeros(n_ent, self.hidden_dim))
		self.relation_embedding = nn.Parameter(torch.zeros(n_rel, self.hidden_dim))
		nn.init.uniform_(self.entity_embedding, a=-embedding_range, b=embedding_range)
		nn.init.uniform_(self.relation_embedding, a=-embedding_range, b=embedding_range)

		ent_embedder = ParameterEmbedder(self.entity_embedding)
		rel_embedder = ParameterEmbedder(self.relation_embedding)
		super().__init__(ent_embedder, rel_embedder, pRotatEScorer(args), args)

	def forward(self, positive_sample, negative_sample=None, mode: str = 'single') -> dict:
		pos_scores = self._score(positive_sample, mode='single')
		neg_scores = None
		if negative_sample is not None:
			neg_scores = self._score(positive_sample, negative_sample=negative_sample, mode=mode)
		return {'positive_scores': pos_scores, 'negative_scores': neg_scores}

	def _score(self, positive_sample, negative_sample=None, mode: str = 'single') -> torch.Tensor:
		if mode == 'single':
			return self.score_spo(positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2]).unsqueeze(1)
		if mode == 'head-batch':
			batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
			h = self.embed_s(negative_sample.reshape(-1)).reshape(batch_size, negative_sample_size, -1)
			r = self.embed_p(positive_sample[:, 1]).unsqueeze(1)
			t = self.embed_o(positive_sample[:, 2]).unsqueeze(1)
			h_flat = h.reshape(-1, h.size(-1))
			r_flat = r.expand(-1, negative_sample_size, -1).reshape(-1, r.size(-1))
			t_flat = t.expand(-1, negative_sample_size, -1).reshape(-1, t.size(-1))
			return self.scorer.score_po(h_flat, r_flat, t_flat).reshape(batch_size, negative_sample_size, 1)
		if mode == 'tail-batch':
			batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
			h = self.embed_s(positive_sample[:, 0]).unsqueeze(1)
			r = self.embed_p(positive_sample[:, 1]).unsqueeze(1)
			t = self.embed_o(negative_sample.reshape(-1)).reshape(batch_size, negative_sample_size, -1)
			h_flat = h.expand(-1, negative_sample_size, -1).reshape(-1, h.size(-1))
			r_flat = r.expand(-1, negative_sample_size, -1).reshape(-1, r.size(-1))
			t_flat = t.reshape(-1, t.size(-1))
			return self.scorer.score_spo(h_flat, r_flat, t_flat).reshape(batch_size, negative_sample_size, 1)
		raise ValueError(f'mode {mode} not supported')

	def get_queries_targets(self, src, rel, dst):
		head = self.embed_s(src)
		relation = self.embed_p(rel)
		tail = self.embed_o(dst)
		phase_head = self.scorer._phase(head)
		phase_relation = self.scorer._phase(relation)
		phase_tail = self.scorer._phase(tail)
		return phase_head + phase_relation, phase_tail, phase_head

	def au_entity_embeddings(self, device: torch.device | None = None) -> torch.Tensor:
		vectors = self.scorer._phase(self.entity_embedding)
		return vectors.to(device) if device is not None else vectors


class DaBRModel(_RelationLookupMixin, KGEModel):
	"""DaBR with quaternion lookup tables bound through ``KGEModel``."""

	bidirectional_score_batch = True

	def __init__(
		self,
		args,
		ent_embedder=None,
		rel_embedder=None,
		scorer=None,
		dr_embedder=None,
	):
		self.entity_dict = get_entity_dict()
		self.rel_to_idx = load_relation_to_idx(args)
		dim = int(getattr(args, 'dim', getattr(args, 'hidden_size', 100)))
		emb_dim = 4 * dim

		n_ent = len(self.entity_dict)
		n_rel = max(len(self.rel_to_idx), 1)
		if ent_embedder is None:
			self.ent_embeddings = LookupEmbedder(n_ent, emb_dim, args)
			self.rel_embeddings = LookupEmbedder(n_rel, emb_dim, args)
			self.Dr = LookupEmbedder(n_rel, emb_dim, args)
			for module in (self.ent_embeddings, self.rel_embeddings, self.Dr):
				nn.init.xavier_uniform_(module.embedding.weight)
		else:
			self.ent_embeddings = ent_embedder
			self.rel_embeddings = rel_embedder
			self.Dr = dr_embedder if dr_embedder is not None else LookupEmbedder(n_rel, emb_dim, args)
			if dr_embedder is None:
				nn.init.xavier_uniform_(self.Dr.embedding.weight)

		self.para = nn.Parameter(torch.tensor([float(getattr(args, 'para', 0.1))]), requires_grad=True)
		super().__init__(
			self.ent_embeddings,
			self.rel_embeddings,
			scorer if scorer is not None else DaBRScorer(args),
			args,
		)

	def _scorer_kwargs(self) -> dict:
		return {'para': self.para}

	def embed_s(self, indices: torch.Tensor) -> torch.Tensor:
		return self.ent_embeddings(indices)

	def embed_o(self, indices: torch.Tensor) -> torch.Tensor:
		return self.ent_embeddings(indices)

	def embed_p(self, indices: torch.Tensor) -> torch.Tensor:
		return self.rel_embeddings(indices)

	def embed_all_entities(self) -> torch.Tensor:
		return self.ent_embeddings.get_all()

	def score_spo(self, s, p, o, **kwargs):
		dr = self.Dr(p)
		return super().score_spo(s, p, o, dr_emb=dr, **kwargs)

	def score_sp_(self, s, p, all_o_embs=None, **kwargs):
		dr = self.Dr(p)
		return super().score_sp_(s, p, all_o_embs, dr_emb=dr, **kwargs)

	def forward(self, batch_dict: dict) -> dict:
		score = self.score_spo(batch_dict['head_id'], batch_dict['relation'], batch_dict['tail_id'])
		return {'score': score}

	def get_queries_targets(self, src, rel, dst):
		h, r, t = self.embed_s(src), self.embed_p(rel), self.embed_o(dst)
		dr = self.Dr(rel)
		q_mult = DaBRScorer._quat_mul(h, r)
		t_mult = -DaBRScorer._quat_mul(t, DaBRScorer._quat_inv(r))
		return torch.cat([q_mult, h + dr], dim=-1), torch.cat([t_mult, t], dim=-1), h

	def entity_embeddings(self, device: torch.device | None = None, max_samples: int | None = None) -> torch.Tensor:
		vectors = self.embed_all_entities()
		if max_samples is not None and int(max_samples) > 0 and vectors.size(0) > int(max_samples):
			indices = torch.randperm(vectors.size(0), device=vectors.device)[: int(max_samples)]
			vectors = vectors.index_select(0, indices)
		return vectors.to(device) if device is not None else vectors
