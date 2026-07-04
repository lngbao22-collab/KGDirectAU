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
		self.lp_score_mode = _lp_score_mode(args) if args is not None else 'original'
		self.lp_distance_degree = _lp_distance_degree(args) if args is not None else 2.0
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

		if self._uses_alternate_lp_entity_space() and hasattr(self.scorer, 'au_entity_embeddings'):
			return self.scorer.au_entity_embeddings(entity_emb)
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
		if self._uses_distance_lp_scores():
			query = self._tail_query_vectors(s, p)
			tail = self._lp_entity_vectors(self.embed_o(o))
			distance_degree = float(getattr(self, 'lp_distance_degree', 2.0) or 2.0)
			return -torch.linalg.vector_norm(query - tail, ord=distance_degree, dim=-1)
		if self._uses_cosine_lp_scores():
			if hasattr(self.scorer, 'normalized_score_spo'):
				scorer_kwargs = {**self._scorer_kwargs(p), **kwargs}
				return self.scorer.normalized_score_spo(
					self.embed_s(s),
					self.embed_p(p),
					self.embed_o(o),
					**scorer_kwargs,
				)
			query = self._tail_query_vectors(s, p)
			tail = self._lp_entity_vectors(self.embed_o(o))
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
		lp_entity_embs = self._lp_entity_vectors(all_o_embs)
		if self._uses_distance_lp_scores():
			return self._distance_scores(self._tail_query_vectors(s, p), lp_entity_embs)
		if self._uses_cosine_lp_scores():
			if hasattr(self.scorer, 'normalized_score_sp_'):
				scorer_kwargs = {**self._scorer_kwargs(p), **kwargs}
				return self.scorer.normalized_score_sp_(
					self.embed_s(s),
					self.embed_p(p),
					lp_entity_embs,
					**scorer_kwargs,
				)
			return self._cosine_similarity_scores(self._tail_query_vectors(s, p), lp_entity_embs)
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
		lp_entity_embs = self._lp_entity_vectors(all_s_embs)
		if self._uses_distance_lp_scores():
			return self._distance_scores(self._head_query_vectors(p, o), lp_entity_embs)
		if self._uses_cosine_lp_scores():
			if hasattr(self.scorer, 'normalized_score_po_'):
				scorer_kwargs = {**self._scorer_kwargs(p), **kwargs}
				return self.scorer.normalized_score_po_(
					lp_entity_embs,
					self.embed_p(p),
					self.embed_o(o),
					**scorer_kwargs,
				)
			return self._cosine_similarity_scores(self._head_query_vectors(p, o), lp_entity_embs)
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
		if self.normalize_lp_scores and hasattr(self.scorer, 'normalized_score_po'):
			scorer_kwargs = {**self._scorer_kwargs(p), **kwargs}
			return self.scorer.normalized_score_po(
				self._embed(self.ent_embedder, s),
				self.embed_p(p),
				self.embed_o(o),
				**scorer_kwargs,
			)
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

	def get_queries_targets(
		self,
		s: torch.Tensor,
		p: torch.Tensor,
		o: torch.Tensor,
		*,
		predict_head: bool = False,
	):
		"""AU (query, align_target, head_entity) vectors — delegated to the scorer."""

		h = self.embed_s(s)
		r = self.embed_p(p)
		t = self.embed_o(o)
		scorer_kwargs = {**self._scorer_kwargs(p), 'predict_head': predict_head}
		query, tail, head = self.scorer.au_representations(h, r, t, **scorer_kwargs)
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


class TextKGEModel(KGEModel):
	"""KGEModel binder for token-input encoders with joint (head, relation) queries."""

	training_input_mode = 'tokens'

	def __init__(
		self,
		ent_embedder: nn.Module,
		query_embedder: nn.Module,
		scorer: nn.Module,
		args: Any | None = None,
		contrastive_state: nn.Module | None = None,
	):
		super().__init__(ent_embedder, query_embedder, scorer, args)
		if contrastive_state is not None:
			self.contrastive_state = contrastive_state
		elif args is not None:
			from models.scorers.simkgc_scorer import build_contrastive_state

			hidden_size = int(getattr(getattr(ent_embedder, 'config', None), 'hidden_size', getattr(args, 'dim', 768)))
			self.contrastive_state = build_contrastive_state(args, hidden_size)
		else:
			self.contrastive_state = None

	@property
	def query_embedder(self) -> nn.Module:
		return self.rel_embedder

	@property
	def log_inv_t(self) -> torch.Tensor:
		return self.contrastive_state.log_inv_t

	@property
	def add_margin(self) -> float:
		return self.contrastive_state.add_margin

	@property
	def batch_size(self) -> int:
		return self.contrastive_state.batch_size

	@property
	def pre_batch(self) -> int:
		return self.contrastive_state.pre_batch

	@property
	def pre_batch_vectors(self) -> torch.Tensor:
		return self.contrastive_state.pre_batch_vectors

	@property
	def pre_batch_exs(self) -> list:
		return self.contrastive_state.pre_batch_exs

	@property
	def offset(self) -> int:
		return self.contrastive_state.offset

	@offset.setter
	def offset(self, value: int) -> None:
		self.contrastive_state.offset = value

	def _tail_query_vectors(self, s: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
		return self.query_embedder.embed_sp(s, p)

	def score_spo(
		self,
		s: torch.Tensor,
		p: torch.Tensor,
		o: torch.Tensor,
		**kwargs: Any,
	) -> torch.Tensor:
		query = self._tail_query_vectors(s, p)
		tail = self._lp_entity_vectors(self.embed_o(o))
		if self._uses_distance_lp_scores():
			distance_degree = float(getattr(self, 'lp_distance_degree', 2.0) or 2.0)
			return -torch.linalg.vector_norm(query - tail, ord=distance_degree, dim=-1)
		if self._uses_cosine_lp_scores():
			return self._cosine_similarity_scores(query, tail).diag()
		return self.scorer.score_spo(query, p, tail, **{**self._scorer_kwargs(p), **kwargs})

	def score_sp_(
		self,
		s: torch.Tensor,
		p: torch.Tensor,
		all_o_embs: torch.Tensor | None = None,
		**kwargs: Any,
	) -> torch.Tensor:
		if all_o_embs is None:
			all_o_embs = self.embed_all_entities()
		query = self._tail_query_vectors(s, p)
		lp_entity_embs = self._lp_entity_vectors(all_o_embs)
		if self._uses_distance_lp_scores():
			return self._distance_scores(query, lp_entity_embs)
		if self._uses_cosine_lp_scores():
			return self._cosine_similarity_scores(query, lp_entity_embs)
		scorer_kwargs = {**self._scorer_kwargs(p), **kwargs}
		return self.scorer.score_sp_(query, p, all_o_embs, **scorer_kwargs)

	def get_queries_targets(self, s: torch.Tensor, p: torch.Tensor, o: torch.Tensor):
		from data.dataloader import collate
		from data.dataset import Example
		from data.dict_hub import get_relation_id_map
		from utils.device import move_to_cuda

		entity_dict = get_entity_dict()
		relation_id_map = get_relation_id_map() or {}
		idx_to_relation = {int(value): key for key, value in relation_id_map.items()}

		examples = []
		for head_idx, relation_idx, tail_idx in zip(s.tolist(), p.tolist(), o.tolist()):
			head_entity = entity_dict.get_entity_by_idx(int(head_idx))
			tail_entity = entity_dict.get_entity_by_idx(int(tail_idx))
			relation = idx_to_relation.get(int(relation_idx), str(int(relation_idx)))
			examples.append(Example(head_id=head_entity.entity_id, relation=relation, tail_id=tail_entity.entity_id))

		batch_dict = collate([example.vectorize() for example in examples])
		if torch.cuda.is_available():
			batch_dict = move_to_cuda(batch_dict)
		outputs = self.forward(**batch_dict)
		query, tail, head = outputs['hr_vector'], outputs['tail_vector'], outputs['head_vector']
		if self.normalize_au_vectors:
			query = self._normalize_au_vector(query)
			tail = self._normalize_au_vector(tail)
			head = self._normalize_au_vector(head)
		return query, tail, head

	def _au_needs_head_vectors(self) -> bool:
		"""True when AU head/entity uniformity (``gamma_h`` / ``gamma_ent``) is active.

		Text encoders only encode head vectors on demand, so this decides whether the
		training forward must produce them for the uniformity terms.
		"""

		gamma_h = float(getattr(self.args, 'gamma_h', 0.0) or 0.0)
		gamma_ent = float(getattr(self.args, 'gamma_ent', 0.0) or 0.0)
		return gamma_h > 0.0 or gamma_ent > 0.0

	def forward(
		self,
		hr_token_ids=None,
		hr_mask=None,
		hr_token_type_ids=None,
		tail_token_ids=None,
		tail_mask=None,
		tail_token_type_ids=None,
		head_token_ids=None,
		head_mask=None,
		head_token_type_ids=None,
		only_ent_embedding=False,
		encode_hr_only=False,
		src: torch.Tensor | None = None,
		rel: torch.Tensor | None = None,
		dst: torch.Tensor | None = None,
		**kwargs,
	):
		if src is not None and rel is not None and dst is not None:
			return self.score_spo(src, rel, dst)

		if only_ent_embedding:
			return self.predict_ent_embedding(
				tail_token_ids,
				tail_mask,
				tail_token_type_ids,
			)

		if encode_hr_only:
			hr_vector = self.query_embedder.encode(hr_token_ids, hr_mask, hr_token_type_ids)
			return {'hr_vector': hr_vector}

		hr_vector = self.query_embedder.encode(hr_token_ids, hr_mask, hr_token_type_ids)
		use_self_negative = self.training and bool(getattr(self.args, 'use_self_negative', False))
		# Head vectors are also needed when the AU loss uses head/entity uniformity
		# (``gamma_h`` / ``gamma_ent``); otherwise they would be ``None`` and crash uniformity.
		need_head_vector = head_token_ids is not None and (
			use_self_negative or (self.training and self._au_needs_head_vectors())
		)
		if need_head_vector:
			batch_size = tail_token_ids.size(0)
			combined_ids = torch.cat([tail_token_ids, head_token_ids], dim=0)
			combined_mask = torch.cat([tail_mask, head_mask], dim=0)
			combined_type_ids = torch.cat([tail_token_type_ids, head_token_type_ids], dim=0)
			combined = self.ent_embedder.encode(combined_ids, combined_mask, combined_type_ids)
			tail_vector = combined[:batch_size]
			head_vector = combined[batch_size:]
		else:
			tail_vector = self.ent_embedder.encode(tail_token_ids, tail_mask, tail_token_type_ids)
			head_vector = None
		return {
			'hr_vector': hr_vector,
			'tail_vector': tail_vector,
			'head_vector': head_vector,
		}

	@torch.no_grad()
	def predict_ent_embedding(
		self,
		tail_token_ids,
		tail_mask,
		tail_token_type_ids,
		**kwargs,
	) -> dict:
		ent_vectors = self.ent_embedder.encode(
			tail_token_ids,
			tail_mask,
			tail_token_type_ids,
		)
		return {'ent_vectors': ent_vectors.detach()}

	def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
		"""InfoNCE logits with masking, pre-batch, and self-negative terms (SimKGC-style)."""

		hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
		batch_size = hr_vector.size(0)
		labels = torch.arange(batch_size, device=hr_vector.device)

		logits = hr_vector.mm(tail_vector.t())
		if self.training and self.add_margin:
			logits.diagonal().sub_(self.add_margin)
		logits = logits * self.log_inv_t.exp()

		triplet_mask = batch_dict.get('triplet_mask', None)
		if triplet_mask is not None:
			logits.masked_fill_(~triplet_mask.to(hr_vector.device), -1e4)

		if self.pre_batch > 0 and self.training:
			pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
			logits = torch.cat([logits, pre_batch_logits], dim=-1)

		if getattr(self.args, 'use_self_negative', False) and self.training:
			head_vector = output_dict['head_vector']
			self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
			self_negative_mask = batch_dict.get('self_negative_mask', None)
			if self_negative_mask is None:
				self_negative_mask = torch.ones(batch_size, dtype=torch.bool, device=hr_vector.device)
			else:
				self_negative_mask = self_negative_mask.to(hr_vector.device).bool()
			self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
			logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

		return {
			'logits': logits,
			'labels': labels,
			'inv_t': self.log_inv_t.detach().exp(),
			'hr_vector': hr_vector.detach(),
			'tail_vector': tail_vector.detach(),
			'head_vector': output_dict['head_vector'].detach() if output_dict.get('head_vector') is not None else None,
		}

	def _compute_pre_batch_logits(
		self,
		hr_vector: torch.Tensor,
		tail_vector: torch.Tensor,
		batch_dict: dict,
	) -> torch.Tensor:
		from models.samplers.masking_sampler import construct_mask

		assert tail_vector.size(0) == self.batch_size
		batch_exs = batch_dict['batch_data']
		pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
		pre_batch_logits = pre_batch_logits * self.log_inv_t.exp() * float(getattr(self.args, 'pre_batch_weight', 0.5))
		if self.pre_batch_exs[-1] is not None:
			pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
			pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

		self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
		self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
		self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

		return pre_batch_logits

	def entity_embeddings(
		self,
		device: torch.device | None = None,
		batch_size: int | None = None,
		num_workers: int | None = None,
		max_samples: int | None = None,
	) -> torch.Tensor:
		entity_exs = get_entity_dict().entity_exs
		if max_samples is not None and int(max_samples) > 0 and len(entity_exs) > int(max_samples):
			indices = torch.randperm(len(entity_exs))[: int(max_samples)].tolist()
			entity_exs = [entity_exs[i] for i in indices]

		loader_workers = self.ent_embedder._resolve_entity_loader_workers(num_workers, len(entity_exs))
		vectors = self.ent_embedder._encode_entity_exs(
			entity_exs,
			batch_size=batch_size,
			num_workers=loader_workers,
			show_progress=False,
		)
		return vectors.to(device) if device is not None else vectors

	def predict_by_examples(self, examples, batch_size=None, num_workers: int = 1):
		"""Deprecated: use ``score_sp_`` / index-based LP eval."""

		from data.dataset import Dataset
		from utils.device import move_to_cuda

		if batch_size is None:
			batch_size = max(int(getattr(self.args, 'batch_size', 512)), 512)
		else:
			batch_size = max(int(batch_size), 512)

		data_loader = torch.utils.data.DataLoader(
			Dataset(path='', examples=examples, task=self.args.dataset),
			num_workers=num_workers,
			batch_size=batch_size,
			collate_fn=__import__('data.dataloader', fromlist=['collate']).collate,
			shuffle=False,
		)

		hr_tensor_list, tail_tensor_list = [], []
		use_cuda = torch.cuda.is_available()
		for batch_dict in data_loader:
			if use_cuda:
				batch_dict = move_to_cuda(batch_dict)
			outputs = self(**batch_dict)
			hr_tensor_list.append(outputs['hr_vector'])
			tail_tensor_list.append(outputs['tail_vector'])
		return torch.cat(hr_tensor_list, dim=0), torch.cat(tail_tensor_list, dim=0)

	def predict_by_entities(self, entity_exs, batch_size=None, num_workers=None, show_progress=None):
		"""Deprecated: use ``embed_all_entities``."""

		if batch_size is None:
			batch_size = max(int(getattr(self.args, 'batch_size', 512)), 1024)
		else:
			batch_size = max(int(batch_size), 512)
		if show_progress is None:
			show_progress = not self.training
		loader_workers = self.ent_embedder._resolve_entity_loader_workers(num_workers, len(entity_exs))
		return self.ent_embedder._encode_entity_exs(
			entity_exs,
			batch_size=batch_size,
			num_workers=loader_workers,
			show_progress=show_progress,
		)


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


def _lp_score_mode(args) -> str:
	value = getattr(args, 'lp_score_mode', None)
	if value is None:
		return 'cosine' if _normalize_lp_flag(args) else 'original'
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


def _lp_distance_degree(args) -> float:
	value = getattr(args, 'lp_distance_degree', None)
	if value is None:
		value = getattr(args, 'distance_degree_l', None)
	if value is None:
		return 2.0
	degree = float(value)
	if degree <= 0:
		raise ValueError('lp_distance_degree must be > 0')
	return degree


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
