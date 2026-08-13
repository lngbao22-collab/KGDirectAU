"""

* ``KGEBase`` — shared base for models, scorers, and embedders
* ``KGEScorer`` — pure-tensor relational score functions
* ``KGEEmbedder`` — embedders for a fixed vocabulary of objects
* ``KGEModel`` — binder that ties entity/relation embedders to one or more scorers

Relation maps / reciprocal helpers live in ``utils.relations``; head-eval routing
in ``utils.eval_modes``; lookup init in ``LookupEmbedder``.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.dict_hub import get_entity_dict
from utils.relations import as_index_tensor, load_relation_to_idx, resolve_relation_index


class KGEBase(nn.Module):
	r"""Base class for all KGE models, scorers, and embedders."""

	def __init__(self, args: Any | None = None):
		super().__init__()
		self.args = args
		self.meta: dict[str, Any] = {}

	@staticmethod
	def _initialize(what: Tensor, initialize: str, initialize_args: dict | None = None) -> None:
		initialize_args = dict(initialize_args or {})
		try:
			getattr(torch.nn.init, initialize)(what, **initialize_args)
		except Exception as exc:
			raise ValueError(
				f'invalid initialization options: {initialize} with args {initialize_args}'
			) from exc

	def penalty(self, **kwargs) -> List[Tensor]:
		r"""Optional extra penalty terms added to the training loss."""

		return []



class KGEScorer(KGEBase):
	"""Pure tensor KGE score functions — no embedding index lookups.

	**Scorer contract:** subclasses implement ``score_emb(h, r, t, combine)``.

	``combine`` modes:

	* ``hrt`` — row-wise triples → ``[n, 1]``
	* ``hr_`` — each ``(h, r)`` vs all tails → ``[n, n_t]``
	* ``_rt`` — each ``(r, t)`` vs all heads → ``[n, n_h]``
	* ``hr_c`` — many tail candidates per row (``t`` is ``[B, C, D]``) → ``[B, C]``
	* ``_rt_c`` — many head candidates per row (``h`` is ``[B, C, D]``) → ``[B, C]``

	Query / target composition for KGAU and cosine LP lives on ``KGEModel``.

	Optional hook:

	* ``au_entity_embeddings`` — map entity rows into the AU / alternate LP space.
	"""

	bidirectional_score_batch: bool = False
	kgau_alignment_mode: str | None = None

	def score_emb(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		combine: str,
		**kwargs: Any,
	) -> torch.Tensor:
		"""Score embeddings under ``combine`` (see class docstring)."""

		raise NotImplementedError(f'{type(self).__name__} must implement score_emb')

	def supports_candidate_scoring(self) -> bool:
		"""Whether ``combine`` modes ``hr_c`` / ``_rt_c`` are available."""

		return False


class KGEEmbedder(KGEBase):
	r"""Base class for embedders of a fixed number of objects (entities, relations, ...)."""

	def __init__(self, args: Any | None = None, *, dim: int | None = None):
		super().__init__(args)
		self.dim = dim

	def forward(self, indexes: Tensor) -> Tensor:
		return self.embed(indexes)

	def embed(self, indexes: Tensor) -> Tensor:
		"""Compute embeddings for the given indexes."""

		raise NotImplementedError

	def embed_all(self) -> Tensor:
		"""Return embeddings for all objects in the vocabulary."""

		raise NotImplementedError


class KGEModel(KGEBase):
	"""Binder that ties entity/relation embedders to relational scorers.

	``scorers`` is a non-empty list of ``KGEScorer`` modules. ComplEx uses a
	single Hermitian scorer. Default LP / AU math delegates to ``scorers[0]``.
	"""

	def __init__(
		self,
		ent_embedder: nn.Module,
		rel_embedder: nn.Module,
		scorers: Sequence[nn.Module] | nn.Module,
		args: Any | None = None,
		aux_embedders: Mapping[str, nn.Module] | None = None,
	):
		super().__init__(args)
		self.ent_embedder = ent_embedder
		self.rel_embedder = rel_embedder
		self.scorers = self._as_scorer_list(scorers)
		self.aux_embedders = nn.ModuleDict(dict(aux_embedders or {}))
		self.rel_to_idx = load_relation_to_idx(args) if args is not None else {}
		self.normalize_lp_scores = self.normalize_lp_flag(args) if args is not None else False
		self.lp_score_mode = self.resolve_lp_score_mode(args) if args is not None else 'original'
		self.lp_distance_degree = self.resolve_lp_distance_degree(args) if args is not None else 2.0
		self.normalize_au_vectors = self.normalize_au_flag(args) if args is not None else False

	@staticmethod
	def normalize_lp_flag(args: Any | None) -> bool:
		if args is None:
			return False
		value = getattr(args, 'normalize_lp_scores', None)
		if value is not None:
			return bool(value)
		return False

	@staticmethod
	def resolve_lp_score_mode(args: Any | None) -> str:
		if args is None:
			return 'original'
		value = getattr(args, 'lp_score_mode', None)
		if value is None:
			return 'cosine' if KGEModel.normalize_lp_flag(args) else 'original'
		mode = str(value).lower().replace('-', '_')
		aliases = {
			'native': 'original',
			'raw': 'original',
			'lp': 'lp_distance',
			'l_distance': 'lp_distance',
			'distance': 'lp_distance',
		}
		mode = aliases.get(mode, mode)
		if mode not in {'original', 'cosine', 'lp_distance'}:
			raise ValueError(f'Unsupported lp_score_mode: {value}')
		return mode

	@staticmethod
	def resolve_lp_distance_degree(args: Any | None) -> float:
		if args is None:
			return 2.0
		value = getattr(args, 'lp_distance_degree', None)
		if value is None:
			value = getattr(args, 'distance_degree_l', None)
		if value is None:
			return 2.0
		degree = float(value)
		if degree <= 0:
			raise ValueError('lp_distance_degree must be > 0')
		return degree

	@staticmethod
	def normalize_au_flag(args: Any | None) -> bool:
		if args is None:
			return False
		value = getattr(args, 'normalize_au_vectors', None)
		if value is not None:
			return bool(value)
		model = str(getattr(args, 'model', '') or '')
		if not model.endswith('-AU'):
			return False
		return True

	@staticmethod
	def _as_scorer_list(scorers: Sequence[nn.Module] | nn.Module) -> nn.ModuleList:
		if isinstance(scorers, nn.ModuleList):
			modules = list(scorers)
		elif isinstance(scorers, nn.Module):
			modules = [scorers]
		else:
			modules = list(scorers)
		if not modules:
			raise ValueError('scorers must contain at least one KGEScorer')
		return nn.ModuleList(modules)

	@property
	def bidirectional_score_batch(self) -> bool:
		return bool(getattr(self.get_scorer(), 'bidirectional_score_batch', False))

	@property
	def kgau_alignment_mode(self) -> str | None:
		return getattr(self.get_scorer(), 'kgau_alignment_mode', None)

	def get_h_embedder(self) -> nn.Module:
		return self.ent_embedder

	def get_t_embedder(self) -> nn.Module:
		return self.ent_embedder

	def get_r_embedder(self) -> nn.Module:
		return self.rel_embedder

	def get_scorer(self, index: int = 0) -> nn.Module:
		"""Return the primary scorer (``index=0``) or another component scorer."""

		return self.scorers[index]

	def get_scorers(self) -> nn.ModuleList:
		return self.scorers

	# --- Regularization (``penalty`` / embedder Lp terms) ---

	@staticmethod
	def _regularize_p(args: Any | None) -> int:
		"""Lp order: prefer ``regularization_p`` (GB-Magic), then ``regularize_p``, else 3."""

		if args is None:
			return 3
		for key in ('regularization_p', 'regularize_p'):
			raw = getattr(args, key, None)
			if raw is not None:
				return int(raw)
		return 3

	@staticmethod
	def _regularize_weighted(args: Any | None, role: str) -> bool:
		if args is None:
			return True
		if role == 'entity':
			raw = getattr(args, 'entity_regularize_weighted', None)
		else:
			raw = getattr(args, 'relation_regularize_weighted', None)
		if raw is None:
			return True
		return bool(raw)

	@staticmethod
	def _uses_kbc_n3_regularization(args: Any | None) -> bool:
		regularizer = str(getattr(args, 'regularizer', '') or '').lower() if args is not None else ''
		return regularizer in {'n3_kbc', 'kbc_n3'}

	@staticmethod
	def _lookup_embedding_rows(embedder: nn.Module, indexes: torch.Tensor) -> torch.Tensor | None:
		if hasattr(embedder, 'embedding'):
			return embedder.embedding(indexes.long())
		if hasattr(embedder, 'ent_re') and hasattr(embedder, 'ent_im'):
			return torch.cat(
				[embedder.ent_re.embedding(indexes.long()), embedder.ent_im.embedding(indexes.long())],
				dim=-1,
			)
		if hasattr(embedder, 'rel_re') and hasattr(embedder, 'rel_im'):
			return torch.cat(
				[embedder.rel_re.embedding(indexes.long()), embedder.rel_im.embedding(indexes.long())],
				dim=-1,
			)
		if hasattr(embedder, 'weight'):
			return embedder.weight.index_select(0, indexes.long())
		return None

	@staticmethod
	def _embedding_table_l3(embedder: nn.Module, p: int = 3) -> torch.Tensor | None:
		if hasattr(embedder, 'embedding'):
			return embedder.embedding.weight.norm(p=p) ** p
		if hasattr(embedder, 'ent_re') and hasattr(embedder, 'ent_im'):
			return (
				embedder.ent_re.embedding.weight.norm(p=p) ** p
				+ embedder.ent_im.embedding.weight.norm(p=p) ** p
			)
		if hasattr(embedder, 'rel_re') and hasattr(embedder, 'rel_im'):
			return (
				embedder.rel_re.embedding.weight.norm(p=p) ** p
				+ embedder.rel_im.embedding.weight.norm(p=p) ** p
			)
		if hasattr(embedder, 'weight'):
			return embedder.weight.norm(p=p) ** p
		return None

	@staticmethod
	def _weighted_lp_penalty(
		embedder: nn.Module,
		indexes: torch.Tensor,
		*,
		weight: float,
		p: int,
		num_indexes: int,
	) -> torch.Tensor | None:
		if weight == 0.0 or indexes.numel() == 0:
			return None
		flat_indexes = indexes.reshape(-1).long()
		unique_indexes, counts = torch.unique(flat_indexes, return_counts=True)
		parameters = KGEModel._lookup_embedding_rows(embedder, unique_indexes)
		if parameters is None:
			return None
		if p % 2 == 1:
			parameters = torch.abs(parameters)
		return (
			weight
			/ p
			* (parameters ** p * counts.float().view(-1, 1)).sum()
			/ max(int(num_indexes), 1)
		)

	@staticmethod
	def _complex_batch_factor_norms(
		ent_embedder: nn.Module,
		rel_embedder: nn.Module,
		h_idx: torch.Tensor,
		r_idx: torch.Tensor,
		t_idx: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
		"""kbc ComplEx N3 factors: L2 norms of complex components per rank."""

		if not (hasattr(ent_embedder, 'ent_re') and hasattr(ent_embedder, 'ent_im')):
			return None
		if not (hasattr(rel_embedder, 'rel_re') and hasattr(rel_embedder, 'rel_im')):
			return None

		def _entity_norms(indexes: torch.Tensor) -> torch.Tensor:
			re = ent_embedder.ent_re.embedding(indexes.long())
			im = ent_embedder.ent_im.embedding(indexes.long())
			return torch.sqrt(re ** 2 + im ** 2)

		def _relation_norms(indexes: torch.Tensor) -> torch.Tensor:
			re = rel_embedder.rel_re.embedding(indexes.long())
			im = rel_embedder.rel_im.embedding(indexes.long())
			return torch.sqrt(re ** 2 + im ** 2)

		return _entity_norms(h_idx), _relation_norms(r_idx), _entity_norms(t_idx)

	@staticmethod
	def _complex_table_l3_penalty(embedder: nn.Module, p: int = 3) -> torch.Tensor | None:
		"""Sum |w|^p over ComplEx re/im tables without concatenating weights."""

		if hasattr(embedder, 'ent_re') and hasattr(embedder, 'ent_im'):
			return (
				embedder.ent_re.embedding.weight.abs().pow(p).sum()
				+ embedder.ent_im.embedding.weight.abs().pow(p).sum()
			)
		if hasattr(embedder, 'rel_re') and hasattr(embedder, 'rel_im'):
			return (
				embedder.rel_re.embedding.weight.abs().pow(p).sum()
				+ embedder.rel_im.embedding.weight.abs().pow(p).sum()
			)
		return None

	def _kbc_n3_regularization(self, batch_triples: torch.Tensor) -> torch.Tensor | None:
		"""kbc-style N3 on batch ComplEx factors: sum_i w |f_i|^3 / batch_size."""

		args = self.args
		weight = float(getattr(args, 'regularize_weight', None) or 0.0) if args is not None else 0.0
		if weight == 0.0 and args is not None:
			weight = float(getattr(args, 'entity_regularize_weight', 0.0) or 0.0)
		if weight == 0.0:
			return None
		if self.ent_embedder is None or self.rel_embedder is None:
			return None

		factors = self._complex_batch_factor_norms(
			self.ent_embedder,
			self.rel_embedder,
			batch_triples[:, 0],
			batch_triples[:, 1],
			batch_triples[:, 2],
		)
		if factors is None:
			return None

		batch_size = max(int(factors[0].shape[0]), 1)
		norm = torch.zeros((), device=factors[0].device, dtype=factors[0].dtype)
		for factor in factors:
			norm = norm + weight * torch.sum(torch.abs(factor) ** 3)
		return norm / batch_size

	def _lp_regularization(self, batch_triples: torch.Tensor | None = None) -> torch.Tensor | None:
		"""Lp embedding regularization with optional batch weighting."""

		args = self.args
		ent_weight = float(getattr(args, 'entity_regularize_weight', 0.0) or 0.0) if args is not None else 0.0
		rel_weight = float(getattr(args, 'relation_regularize_weight', 0.0) or 0.0) if args is not None else 0.0
		if ent_weight == 0.0 and rel_weight == 0.0:
			return None

		p = self._regularize_p(args)
		terms: list[torch.Tensor] = []

		if ent_weight > 0.0 and self.ent_embedder is not None:
			if batch_triples is not None and self._regularize_weighted(args, 'entity'):
				entity_indexes = torch.cat((batch_triples[:, 0], batch_triples[:, 2]))
				ent_term = self._weighted_lp_penalty(
					self.ent_embedder,
					entity_indexes,
					weight=ent_weight,
					p=p,
					num_indexes=batch_triples.size(0),
				)
			else:
				ent_term = self._embedding_table_l3(self.ent_embedder, p=p)
				if ent_term is not None:
					ent_term = ent_weight * ent_term / p
			if ent_term is not None:
				terms.append(ent_term)

		if rel_weight > 0.0 and self.rel_embedder is not None:
			if batch_triples is not None and self._regularize_weighted(args, 'relation'):
				rel_term = self._weighted_lp_penalty(
					self.rel_embedder,
					batch_triples[:, 1],
					weight=rel_weight,
					p=p,
					num_indexes=batch_triples.size(0),
				)
			else:
				rel_term = self._embedding_table_l3(self.rel_embedder, p=p)
				if rel_term is not None:
					rel_term = rel_weight * rel_term / p
			if rel_term is not None:
				terms.append(rel_term)

		if not terms:
			return None
		return sum(terms)

	def penalty(self, **kwargs) -> List[Tensor]:
		"""Extra penalty terms added to the training loss."""

		batch_triples = kwargs.get('batch_triples')
		if self._uses_kbc_n3_regularization(self.args):
			if batch_triples is None:
				return []
			term = self._kbc_n3_regularization(batch_triples)
			return [term] if term is not None else []
		term = self._lp_regularization(batch_triples=batch_triples)
		return [term] if term is not None else []

	def regularization_term(self, batch_triples: torch.Tensor | None = None) -> torch.Tensor | None:
		"""Sum of ``penalty`` terms, or ``None`` when regularization is inactive."""

		terms = self.penalty(batch_triples=batch_triples)
		if not terms:
			return None
		return sum(terms)

	def embedding_l3_penalty(self, p: int = 3) -> torch.Tensor | None:
		"""Unweighted Lp penalty over entity and relation embedding tables."""

		terms: list[torch.Tensor] = []
		for embedder in (self.ent_embedder, self.rel_embedder):
			if embedder is None:
				continue
			term = self._embedding_table_l3(embedder, p=p)
			if term is not None:
				terms.append(term)
		if not terms:
			return None
		return sum(terms)

	def rotate_style_embedding_l3_penalty(self, p: int = 3) -> torch.Tensor | None:
		"""RotatE/OpenKE global L3: ``||E||_p^p + ||R||_p^p`` over full weight matrices."""

		terms: list[torch.Tensor] = []
		for embedder in (self.ent_embedder, self.rel_embedder):
			if embedder is None:
				continue
			term = self._complex_table_l3_penalty(embedder, p=p)
			if term is None and hasattr(embedder, 'embedding'):
				term = embedder.embedding.weight.abs().pow(p).sum()
			elif term is None and hasattr(embedder, 'weight'):
				term = embedder.weight.abs().pow(p).sum()
			if term is not None:
				terms.append(term)
		if not terms:
			return None
		return sum(terms)

	def embed_h(self, indices: torch.Tensor) -> torch.Tensor:
		return self._embed(self.ent_embedder, indices)

	def embed_r(self, indices: torch.Tensor) -> torch.Tensor:
		return self._embed(self.rel_embedder, indices)

	def embed_t(self, indices: torch.Tensor) -> torch.Tensor:
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

	def _scorer_kwargs(self, r: torch.Tensor | None = None, **extra: Any) -> dict[str, Any]:
		"""Forward auxiliary relation embeddings and caller overrides to the scorer."""

		kwargs = dict(extra)
		if r is not None:
			for key, embedder in self.aux_embedders.items():
				kwargs[f'{key}_emb'] = self._embed(embedder, r)
		return kwargs

	def _normalize_lp_vector(self, vectors: torch.Tensor) -> torch.Tensor:
		if not self.normalize_lp_scores:
			return vectors
		return F.normalize(vectors, p=2, dim=-1)

	def _uses_cosine_lp_scores(self) -> bool:
		mode = getattr(self, 'lp_score_mode', None)
		if mode is None:
			return bool(self.normalize_lp_scores)
		return mode == 'cosine'

	def _uses_distance_lp_scores(self) -> bool:
		return getattr(self, 'lp_score_mode', None) in {'distance', 'lp_distance'}

	def _uses_alternate_lp_entity_space(self) -> bool:
		return self._uses_cosine_lp_scores() or self._uses_distance_lp_scores()

	def _lp_entity_vectors(self, entity_emb: torch.Tensor) -> torch.Tensor:
		"""Map entity embeddings into the LP vector space used by cosine / Lp-distance scoring."""

		if self._uses_alternate_lp_entity_space() and hasattr(self.get_scorer(), 'au_entity_embeddings'):
			return self.get_scorer().au_entity_embeddings(entity_emb)
		return entity_emb

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

	def _distance_scores(
		self,
		query_vectors: torch.Tensor,
		candidate_vectors: torch.Tensor,
	) -> torch.Tensor:
		"""Negative Lp distance scores; larger scores mean closer candidates."""

		p = float(getattr(self, 'lp_distance_degree', 2.0) or 2.0)
		return -torch.cdist(query_vectors, candidate_vectors, p=p)

	def _tail_query_vectors(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
		return self.query_encoder(h, r)

	def _head_query_vectors(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		return self.inverse_query_encoder(r, t)

	def score_hrt(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		**kwargs: Any,
	) -> torch.Tensor:
		if self._uses_distance_lp_scores():
			query = self._tail_query_vectors(h, r)
			tail = self._lp_entity_vectors(self.embed_t(t))
			distance_degree = float(getattr(self, 'lp_distance_degree', 2.0) or 2.0)
			return -torch.linalg.vector_norm(query - tail, ord=distance_degree, dim=-1)
		if self._uses_cosine_lp_scores():
			if hasattr(self.get_scorer(), 'normalized_score_hr'):
				scorer_kwargs = {**self._scorer_kwargs(r), **kwargs}
				return self.get_scorer().normalized_score_hr(
					self.embed_h(h),
					self.embed_r(r),
					self.embed_t(t),
					**scorer_kwargs,
				)
			query = self._tail_query_vectors(h, r)
			tail = self._lp_entity_vectors(self.embed_t(t))
			return self._cosine_similarity_scores(query, tail).diag()
		scorer_kwargs = {**self._scorer_kwargs(r), **kwargs}
		return self.get_scorer().score_emb(
			self.embed_h(h),
			self.embed_r(r),
			self.embed_t(t),
			'hrt',
			**scorer_kwargs,
		).view(-1)

	def score_hr_(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		all_t_embs: torch.Tensor | None = None,
		**kwargs: Any,
	) -> torch.Tensor:
		if all_t_embs is None:
			all_t_embs = self.embed_all_entities()
		lp_entity_embs = self._lp_entity_vectors(all_t_embs)
		if self._uses_distance_lp_scores():
			if hasattr(self.get_scorer(), 'distance_score_hr_'):
				scorer_kwargs = {
					**self._scorer_kwargs(r),
					**kwargs,
					'lp_distance_degree': float(getattr(self, 'lp_distance_degree', 2.0) or 2.0),
				}
				return self.get_scorer().distance_score_hr_(
					self.embed_h(h),
					self.embed_r(r),
					lp_entity_embs,
					**scorer_kwargs,
				)
			return self._distance_scores(self._tail_query_vectors(h, r), lp_entity_embs)
		if self._uses_cosine_lp_scores():
			if hasattr(self.get_scorer(), 'normalized_score_hr_'):
				scorer_kwargs = {**self._scorer_kwargs(r), **kwargs}
				return self.get_scorer().normalized_score_hr_(
					self.embed_h(h),
					self.embed_r(r),
					lp_entity_embs,
					**scorer_kwargs,
				)
			return self._cosine_similarity_scores(self._tail_query_vectors(h, r), lp_entity_embs)
		scorer_kwargs = {**self._scorer_kwargs(r), **kwargs}
		return self.get_scorer().score_emb(
			self.embed_h(h),
			self.embed_r(r),
			all_t_embs,
			'hr_',
			**scorer_kwargs,
		)

	def score_hr(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor | None = None,
		**kwargs: Any,
	) -> torch.Tensor:
		if t is None:
			return self.score_hr_(h, r, **kwargs)
		scorer_kwargs = {**self._scorer_kwargs(r), **kwargs}
		return self.get_scorer().score_emb(
			self.embed_h(h),
			self.embed_r(r),
			self._embed(self.ent_embedder, t),
			'hr_',
			**scorer_kwargs,
		)

	def score_rt_(
		self,
		r: torch.Tensor,
		t: torch.Tensor,
		all_h_embs: torch.Tensor | None = None,
		**kwargs: Any,
	) -> torch.Tensor:
		if all_h_embs is None:
			all_h_embs = self.embed_all_entities()
		lp_entity_embs = self._lp_entity_vectors(all_h_embs)
		if self._uses_distance_lp_scores():
			if hasattr(self.get_scorer(), 'distance_score_rt_'):
				scorer_kwargs = {
					**self._scorer_kwargs(r),
					**kwargs,
					'lp_distance_degree': float(getattr(self, 'lp_distance_degree', 2.0) or 2.0),
				}
				return self.get_scorer().distance_score_rt_(
					lp_entity_embs,
					self.embed_r(r),
					self.embed_t(t),
					**scorer_kwargs,
				)
			return self._distance_scores(self._head_query_vectors(r, t), lp_entity_embs)
		if self._uses_cosine_lp_scores():
			if hasattr(self.get_scorer(), 'normalized_score_rt_'):
				scorer_kwargs = {**self._scorer_kwargs(r), **kwargs}
				return self.get_scorer().normalized_score_rt_(
					lp_entity_embs,
					self.embed_r(r),
					self.embed_t(t),
					**scorer_kwargs,
				)
			return self._cosine_similarity_scores(self._head_query_vectors(r, t), lp_entity_embs)
		scorer_kwargs = {**self._scorer_kwargs(r), **kwargs}
		return self.get_scorer().score_emb(
			all_h_embs,
			self.embed_r(r),
			self.embed_t(t),
			'_rt',
			**scorer_kwargs,
		)

	def score_rt(
		self,
		r: torch.Tensor,
		t: torch.Tensor,
		h: torch.Tensor | None = None,
		**kwargs: Any,
	) -> torch.Tensor:
		if h is None:
			return self.score_rt_(r, t, **kwargs)
		if self.normalize_lp_scores and hasattr(self.get_scorer(), 'normalized_score_rt'):
			scorer_kwargs = {**self._scorer_kwargs(r), **kwargs}
			return self.get_scorer().normalized_score_rt(
				self._embed(self.ent_embedder, h),
				self.embed_r(r),
				self.embed_t(t),
				**scorer_kwargs,
			)
		scorer_kwargs = {**self._scorer_kwargs(r), **kwargs}
		return self.get_scorer().score_emb(
			self._embed(self.ent_embedder, h),
			self.embed_r(r),
			self.embed_t(t),
			'hrt',
			**scorer_kwargs,
		).view(-1)

	def query_all_entities_scores(self, h: torch.Tensor, r: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		return self.score_hr_(h, r, **kwargs)

	def predict_tail_hr_(self, h_idx: torch.Tensor, r_idx: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		return self.score_hr_(h_idx, r_idx, **kwargs)

	def predict_head_rt_(self, r_idx: torch.Tensor, t_idx: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		return self.score_rt_(r_idx, t_idx, **kwargs)

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
		if hasattr(self.get_scorer(), 'au_entity_embeddings'):
			vectors = self.get_scorer().au_entity_embeddings(self.embed_all_entities())
			return vectors.to(device) if device is not None else vectors
		return self.entity_embeddings(device=device, **kwargs)

	def get_queries_targets(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		*,
		predict_head: bool = False,
	):
		"""AU (query, align_target, head_entity) vectors via model encoders."""

		if predict_head:
			query = self.inverse_query_encoder(r, t)
		else:
			query = self.query_encoder(h, r)
		target = self.target_encoder(h, r, t, predict_head=predict_head)
		head = self.uniformity_head_encoder(h, r)
		if self.normalize_au_vectors:
			query = self._normalize_au_vector(query)
			target = self._normalize_au_vector(target)
			head = self._normalize_au_vector(head)
		return query, target, head

	def query_encoder(self, h: torch.Tensor, r: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		"""Encode the tail-prediction query from head and relation indices.

		Default (ComplEx-style): ``embed(h) * embed(r)``. Subclasses override.
		"""

		del kwargs
		return self.embed_h(h) * self.embed_r(r)

	def inverse_query_encoder(self, r: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		"""Encode the head-prediction query from relation and tail indices.

		Default (ComplEx-style): ``embed(t) * embed(r)``. Subclasses override.
		"""

		del kwargs
		return self.embed_t(t) * self.embed_r(r)

	def target_encoder(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		*,
		predict_head: bool = False,
		**kwargs: Any,
	) -> torch.Tensor:
		"""Encode the AU alignment target for the current prediction mode.

		Default (most bilinear models): raw entity embedding — tail when predicting
		tails, head when predicting heads.  Subclasses whose score function aligns
		against a relation-conditioned target should override
		this and set ``target_uses_relation = True``.
		"""

		del r, kwargs  # unused in the entity-only default
		if predict_head:
			return self.embed_h(h)
		return self.embed_t(t)

	#: When True, ``target_encoder`` consumes the relation (and possibly aux) embeddings.
	target_uses_relation: bool = False

	def uniformity_head_encoder(self, h: torch.Tensor, r: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		"""Head entity vector for AU head/entity uniformity terms."""

		del r, kwargs
		return self.embed_h(h)

	def score_batch(self, head_ids, relations, tail_entity_ids) -> torch.Tensor:
		entity_dict = get_entity_dict()
		device = self.device
		rel_lookup = lambda relation: resolve_relation_index(relation, self.rel_to_idx)
		head_indices = as_index_tensor(head_ids, entity_dict.entity_to_idx, device)
		relation_indices = as_index_tensor(relations, rel_lookup, device)
		tail_indices = as_index_tensor(tail_entity_ids, entity_dict.entity_to_idx, device)
		return self.score_hrt(head_indices, relation_indices, tail_indices)

	def forward(self, src: torch.Tensor, rel: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
		return self.score_hrt(src, rel, dst)
