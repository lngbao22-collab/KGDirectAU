"""LibKGE-style model bases and shared embedding utilities for KGDirectAU.

Class hierarchy (aligned with libkge ``kge.model.kge_model``):

* ``KGEBase`` — shared base for models, scorers, and embedders
* ``KGEScorer`` — pure-tensor relational score functions (libkge ``RelationalScorer``)
* ``KGEEmbedder`` — embedders for a fixed vocabulary of objects
* ``KGEModel`` — binder that ties entity/relation embedders to one or more scorers
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.dataset import load_data
from data.dict_hub import get_entity_dict


# ---------------------------------------------------------------------------
# Relation / embedding utilities (formerly base/embeddings.py)
# ---------------------------------------------------------------------------


def _relation_path_candidates(args) -> list[str]:
	paths = []
	for source_path in [getattr(args, 'train_path', ''), getattr(args, 'valid_path', ''), getattr(args, 'test_path', '')]:
		if not source_path:
			continue
		paths.append(os.path.join(os.path.dirname(source_path), 'relation2id.json'))
		paths.append(os.path.join(os.path.dirname(source_path), 'relations.json'))
		paths.append(os.path.join(os.path.dirname(source_path), 'relation2idx.json'))
	paths.append(os.path.join('data', getattr(args, 'dataset', ''), 'relation2id.json'))
	paths.append(os.path.join('data', getattr(args, 'dataset', ''), 'preprocessed', 'relation2id.json'))
	return paths


def use_reciprocal_relations(args: Any | None) -> bool:
	return bool(getattr(args, 'add_reciprocal_relations', False)) or use_kbc_reciprocal_relations(args)


def use_kbc_reciprocal_relations(args: Any | None) -> bool:
	return bool(getattr(args, 'kbc_reciprocal_relations', False))


def kbc_forward_relation_count(relation_to_idx: dict[str, int]) -> int:
	"""Number of forward relation slots (kbc ``n_predicates // 2`` after doubling)."""

	forward_values = [
		int(value)
		for key, value in relation_to_idx.items()
		if not str(key).startswith('inverse ')
	]
	if not forward_values:
		return 0
	return max(forward_values) + 1


def add_kbc_inverse_relations(relation_to_idx: dict[str, int]) -> dict[str, int]:
	"""Assign inverse relation id = forward_id + n_forward (kbc reciprocal layout)."""

	updated = dict(relation_to_idx)
	n_forward = kbc_forward_relation_count(updated)
	for relation, idx in list(updated.items()):
		if str(relation).startswith('inverse '):
			continue
		inverse_relation = f'inverse {relation}'
		if inverse_relation not in updated:
			updated[inverse_relation] = int(idx) + n_forward
	return updated


def add_inverse_relations(relation_to_idx: dict[str, int]) -> dict[str, int]:
	"""Assign each forward relation a distinct inverse-relation ID (LibKGE reciprocal_relations_model)."""

	updated = dict(relation_to_idx)
	next_idx = max(updated.values(), default=-1) + 1
	for relation in list(updated.keys()):
		if relation.startswith('inverse '):
			continue
		inverse_relation = f'inverse {relation}'
		if inverse_relation not in updated:
			updated[inverse_relation] = next_idx
			next_idx += 1
	return updated


def build_forward_to_inverse_index_tensor(relation_to_idx: dict[str, int]) -> torch.Tensor | None:
	"""Map forward relation indices to inverse indices (KvsAll ``_rt`` / kbc CE head eval)."""

	if not relation_to_idx:
		return None
	max_idx = max(int(value) for value in relation_to_idx.values())
	mapping = torch.full((max_idx + 1,), -1, dtype=torch.long)
	for relation, fwd_idx in relation_to_idx.items():
		if str(relation).startswith('inverse '):
			continue
		inv_idx = relation_to_idx.get(f'inverse {relation}')
		if inv_idx is not None:
			mapping[int(fwd_idx)] = int(inv_idx)
	n_forward = kbc_forward_relation_count(relation_to_idx)
	if n_forward > 0:
		for fwd_idx in range(n_forward):
			if int(mapping[fwd_idx]) < 0:
				inv_idx = int(fwd_idx) + n_forward
				if inv_idx <= max_idx:
					mapping[fwd_idx] = inv_idx
	if int(mapping.ge(0).sum()) == 0:
		return None
	return mapping


HEAD_EVAL_MODES = frozenset({'rt_forward', 'rt_inverse', 'hr_inverse'})


def resolve_head_eval_mode(args: Any | None, *, eval_forward: bool) -> str:
	"""Choose backward link-prediction scoring to match the training recipe.

	Returns one of ``tail``, ``rt_forward``, ``rt_inverse``, or ``hr_inverse``.

	When ``args.head_eval_mode`` is set in JSON/CLI, it overrides strategy-based
	inference for the backward (head) pass. Use ``rt_forward`` for adversarial
	BCE / direct head prediction with the forward relation (no inverse relation).
	KGAU with reciprocal relations trains inverse triplets ``(t, r^{-1}, h)`` as
	tail prediction, so backward eval defaults to ``hr_inverse``.
	"""

	if eval_forward:
		return 'tail'
	if args is None:
		return 'rt_forward'

	explicit = getattr(args, 'head_eval_mode', None)
	if explicit is not None:
		mode = str(explicit).strip().lower()
		if mode in {'auto', 'infer', 'default', ''}:
			pass
		elif mode in HEAD_EVAL_MODES:
			return mode
		else:
			raise ValueError(
				f'Unsupported head_eval_mode={explicit!r}; '
				f'expected one of {sorted(HEAD_EVAL_MODES)} or auto'
			)

	strategy = (getattr(args, 'model_strategy_path', '') or '').replace('\\', '/').lower()
	loss_path = (getattr(args, 'model_loss_path', '') or '').replace('\\', '/').lower()

	if 'negsamp' in strategy or 'adversarial_bce' in loss_path:
		return 'rt_forward'

	if 'kvsall' in strategy:
		from models.strategies.kvsall_strategy import kvsall_uses_rt_training

		if kvsall_uses_rt_training(args) and use_reciprocal_relations(args):
			return 'rt_inverse'
		return 'rt_forward'

	if '1vsall' in strategy:
		if use_kbc_reciprocal_relations(args) and not bool(getattr(args, 'bidirectional_1vsall', True)):
			return 'hr_inverse'
		return 'rt_forward'

	if 'kgau' in strategy:
		if bool(getattr(args, 'kgau_bidirectional', False)):
			return 'rt_forward'
		if use_reciprocal_relations(args):
			return 'hr_inverse'
		return 'rt_forward'

	return 'rt_forward'


def uses_forward_examples_for_backward_eval(args: Any | None) -> bool:
	"""Backward eval always starts from forward (h, r, t) test triples for index KGE."""

	return resolve_head_eval_mode(args, eval_forward=False) != 'tail'


def _apply_relation_display_aliases(relation_to_idx: dict[str, int], args) -> dict[str, int]:
	"""Add human-readable relation aliases (FB15k-237 path IDs -> display strings)."""

	dataset = str(getattr(args, 'dataset', '') or '').lower()
	if dataset != 'fb15k237':
		return relation_to_idx

	from data.preprocess import _normalize_fb15k237_relation

	updated = dict(relation_to_idx)
	for relation, idx in relation_to_idx.items():
		relation_str = str(relation)
		if relation_str.startswith('inverse '):
			base_id = relation_str[len('inverse '):]
			if not base_id.startswith('/'):
				continue
			display = _normalize_fb15k237_relation(base_id)
			updated[f'inverse {display}'] = idx
			continue
		if not relation_str.startswith('/'):
			continue
		display = _normalize_fb15k237_relation(relation_str)
		updated[display] = idx
		normalized = ' '.join(display.split())
		if normalized != display:
			updated[normalized] = idx
	return updated


def load_relation_to_idx(args) -> dict[str, int]:
	for path in _relation_path_candidates(args):
		if not path or not os.path.exists(path):
			continue
		with open(path, 'r', encoding='utf-8') as handle:
			mapping = json.load(handle)
		if isinstance(mapping, dict):
			relation_to_idx = {str(key): int(value) for key, value in mapping.items()}
			if use_kbc_reciprocal_relations(args):
				relation_to_idx = add_kbc_inverse_relations(relation_to_idx)
			elif use_reciprocal_relations(args):
				relation_to_idx = add_inverse_relations(relation_to_idx)
			return _apply_relation_display_aliases(relation_to_idx, args)

	relations: list[str] = []
	seen: set[str] = set()
	for example in load_data(getattr(args, 'train_path', ''), add_forward_triplet=False, add_backward_triplet=False):
		if example.relation not in seen:
			seen.add(example.relation)
			relations.append(example.relation)
	relation_to_idx = {relation: idx for idx, relation in enumerate(relations)}
	if use_kbc_reciprocal_relations(args):
		relation_to_idx = add_kbc_inverse_relations(relation_to_idx)
	elif use_reciprocal_relations(args):
		relation_to_idx = add_inverse_relations(relation_to_idx)
	return _apply_relation_display_aliases(relation_to_idx, args)


def _scaled_init(module: nn.Module, dim: int, sigma: float = 0.2) -> None:
	scale = (dim / sigma ** 2) ** (1 / 6)
	for param in module.parameters():
		param.data.div_(scale)


def init_lookup_embedding(module: nn.Module, args: Any | None, dim: int, role: str = 'entity') -> None:
	"""Initialize a lookup embedding table (LibKGE ``lookup_embedder.initialize``)."""

	init_method = str(getattr(args, 'init_method', '') or '').lower()
	if not init_method:
		model_name = str(getattr(args, 'model', '') or '').lower()
		if any(name in model_name for name in ('rotate', 'protate', 'transe')):
			margin_raw = getattr(args, 'margin', None)
			margin = 6.0 if margin_raw is None else float(margin_raw)
			epsilon_raw = getattr(args, 'epsilon', None)
			epsilon = 2.0 if epsilon_raw is None else float(epsilon_raw)
			bound = (margin + epsilon) / max(1, dim)
			nn.init.uniform_(module.embedding.weight, a=-bound, b=bound)
			return
		if model_name in {'distmult', 'distmult-au', 'complex', 'complex-au'}:
			_scaled_init(module, dim)
			return
		nn.init.xavier_uniform_(module.embedding.weight)
		return

	weight = module.embedding.weight
	if init_method == 'uniform_':
		low = float(getattr(args, 'init_uniform_a', -0.05))
		high = getattr(args, 'init_uniform_b', None)
		high = float(-low if high is None else high)
		if low > high:
			low, high = high, low
		nn.init.uniform_(weight, a=low, b=high)
	elif init_method in {'xavier_uniform_', 'xavier_uniform'}:
		gain = float(getattr(args, 'init_xavier_gain', 1.0))
		nn.init.xavier_uniform_(weight, gain=gain)
	elif init_method in {'xavier_normal_', 'xavier_normal'}:
		gain = float(getattr(args, 'init_xavier_gain', 1.0))
		nn.init.xavier_normal_(weight, gain=gain)
	elif init_method == 'normal_':
		mean = float(getattr(args, 'init_normal_mean', 0.0))
		std = float(getattr(args, 'init_normal_std', 1.0))
		nn.init.normal_(weight, mean=mean, std=std)
	elif init_method == 'scaled':
		_scaled_init(module, dim)
	elif init_method == 'kbc':
		scale = float(getattr(args, 'init_scale', 1e-3))
		weight.data.mul_(scale)
	else:
		nn.init.xavier_uniform_(weight)


def _lookup_dropout_rate(args: Any | None, role: str) -> float:
	if role == 'entity':
		raw = getattr(args, 'entity_dropout', None)
		if raw is None:
			raw = getattr(args, 'dropout', 0.0)
	else:
		raw = getattr(args, 'relation_dropout', None)
		if raw is None:
			raw = getattr(args, 'dropout', 0.0)
	return float(raw or 0.0)


def _regularize_p(args: Any | None) -> int:
	raw = getattr(args, 'regularize_p', None)
	if raw is not None:
		return int(raw)
	return 3


def _regularize_weighted(args: Any | None, role: str) -> bool:
	if role == 'entity':
		raw = getattr(args, 'entity_regularize_weighted', None)
	else:
		raw = getattr(args, 'relation_regularize_weighted', None)
	if raw is None:
		return True
	return bool(raw)


def _lookup_embedding_rows(embedder: nn.Module, indexes: torch.Tensor) -> torch.Tensor | None:
	if hasattr(embedder, 'embedding'):
		return embedder.embedding(indexes.long())
	if hasattr(embedder, 'ent_re') and hasattr(embedder, 'ent_im'):
		return torch.cat([embedder.ent_re.embedding(indexes.long()), embedder.ent_im.embedding(indexes.long())], dim=-1)
	if hasattr(embedder, 'rel_re') and hasattr(embedder, 'rel_im'):
		return torch.cat([embedder.rel_re.embedding(indexes.long()), embedder.rel_im.embedding(indexes.long())], dim=-1)
	if hasattr(embedder, 'weight'):
		return embedder.weight.index_select(0, indexes.long())
	return None


def _embedding_table_l3(embedder: nn.Module, p: int = 3) -> torch.Tensor | None:
	if hasattr(embedder, 'embedding'):
		return embedder.embedding.weight.norm(p=p) ** p
	if hasattr(embedder, 'ent_re') and hasattr(embedder, 'ent_im'):
		return embedder.ent_re.embedding.weight.norm(p=p) ** p + embedder.ent_im.embedding.weight.norm(p=p) ** p
	if hasattr(embedder, 'rel_re') and hasattr(embedder, 'rel_im'):
		return embedder.rel_re.embedding.weight.norm(p=p) ** p + embedder.rel_im.embedding.weight.norm(p=p) ** p
	if hasattr(embedder, 'weight'):
		return embedder.weight.norm(p=p) ** p
	return None


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
	parameters = _lookup_embedding_rows(embedder, unique_indexes)
	if parameters is None:
		return None
	if (p % 2 == 1):
		parameters = torch.abs(parameters)
	return (
		weight
		/ p
		* (parameters ** p * counts.float().view(-1, 1)).sum()
		/ max(int(num_indexes), 1)
	)


def _complex_batch_factor_norms(
	ent_embedder: nn.Module,
	rel_embedder: nn.Module,
	h_idx: torch.Tensor,
	r_idx: torch.Tensor,
	t_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
	"""Return kbc ComplEx N3 factors: L2 norms of complex components per rank."""

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


def compute_kbc_n3_regularization(
	model: nn.Module,
	args: Any | None,
	*,
	batch_triples: torch.Tensor,
) -> torch.Tensor | None:
	"""kbc-style N3 on batch ComplEx factors: sum_i w |f_i|^3 / batch_size."""

	weight = float(getattr(args, 'regularize_weight', None) or 0.0)
	if weight == 0.0:
		weight = float(getattr(args, 'entity_regularize_weight', 0.0) or 0.0)
	if weight == 0.0:
		return None

	ent_embedder = getattr(model, 'ent_embedder', None)
	rel_embedder = getattr(model, 'rel_embedder', None)
	if ent_embedder is None or rel_embedder is None:
		return None

	h_idx = batch_triples[:, 0]
	r_idx = batch_triples[:, 1]
	t_idx = batch_triples[:, 2]
	factors = _complex_batch_factor_norms(ent_embedder, rel_embedder, h_idx, r_idx, t_idx)
	if factors is None:
		return None

	batch_size = max(int(factors[0].shape[0]), 1)
	norm = torch.zeros((), device=factors[0].device, dtype=factors[0].dtype)
	for factor in factors:
		norm = norm + weight * torch.sum(torch.abs(factor) ** 3)
	return norm / batch_size


def _uses_kbc_n3_regularization(args: Any | None) -> bool:
	regularizer = str(getattr(args, 'regularizer', '') or '').lower()
	return regularizer in {'n3_kbc', 'kbc_n3'}


def compute_kge_regularization(
	model: nn.Module,
	args: Any | None,
	*,
	batch_triples: torch.Tensor | None = None,
) -> torch.Tensor | None:
	"""Dispatch LibKGE Lp or kbc N3 regularization based on ``args.regularizer``."""

	if _uses_kbc_n3_regularization(args):
		if batch_triples is None:
			return None
		return compute_kbc_n3_regularization(model, args, batch_triples=batch_triples)
	return compute_kge_l3_regularization(model, args, batch_triples=batch_triples)


def compute_kge_l3_regularization(
	model: nn.Module,
	args: Any | None,
	*,
	batch_triples: torch.Tensor | None = None,
) -> torch.Tensor | None:
	"""LibKGE-style Lp embedding regularization with optional batch weighting."""

	ent_weight = float(getattr(args, 'entity_regularize_weight', 0.0) or 0.0)
	rel_weight = float(getattr(args, 'relation_regularize_weight', 0.0) or 0.0)
	if ent_weight == 0.0 and rel_weight == 0.0:
		return None

	p = _regularize_p(args)
	terms: list[torch.Tensor] = []
	ent_embedder = getattr(model, 'ent_embedder', None)
	rel_embedder = getattr(model, 'rel_embedder', None)

	if ent_weight > 0.0 and ent_embedder is not None:
		if batch_triples is not None and _regularize_weighted(args, 'entity'):
			entity_indexes = torch.cat((batch_triples[:, 0], batch_triples[:, 2]))
			ent_term = _weighted_lp_penalty(
				ent_embedder,
				entity_indexes,
				weight=ent_weight,
				p=p,
				num_indexes=batch_triples.size(0),
			)
		else:
			ent_term = _embedding_table_l3(ent_embedder, p=p)
			if ent_term is not None:
				ent_term = ent_weight * ent_term / p
		if ent_term is not None:
			terms.append(ent_term)

	if rel_weight > 0.0 and rel_embedder is not None:
		if batch_triples is not None and _regularize_weighted(args, 'relation'):
			rel_term = _weighted_lp_penalty(
				rel_embedder,
				batch_triples[:, 1],
				weight=rel_weight,
				p=p,
				num_indexes=batch_triples.size(0),
			)
		else:
			rel_term = _embedding_table_l3(rel_embedder, p=p)
			if rel_term is not None:
				rel_term = rel_weight * rel_term / p
		if rel_term is not None:
			terms.append(rel_term)

	if not terms:
		return None
	return sum(terms)


def embedding_l3_penalty(model: nn.Module, p: int = 3) -> torch.Tensor | None:
	"""Unweighted Lp penalty over entity and relation embedding tables."""

	terms: list[torch.Tensor] = []
	for embedder in (getattr(model, 'ent_embedder', None), getattr(model, 'rel_embedder', None)):
		if embedder is None:
			continue
		term = _embedding_table_l3(embedder, p=p)
		if term is not None:
			terms.append(term)
	if not terms:
		return None
	return sum(terms)


def _concat_complex_table_weight(embedder: nn.Module) -> torch.Tensor | None:
	"""Return one weight matrix for ComplEx re/im tables (RotatE ``-de`` / ``-dr`` layout)."""

	if hasattr(embedder, 'ent_re') and hasattr(embedder, 'ent_im'):
		return torch.cat([embedder.ent_re.embedding.weight, embedder.ent_im.embedding.weight], dim=1)
	if hasattr(embedder, 'rel_re') and hasattr(embedder, 'rel_im'):
		return torch.cat([embedder.rel_re.embedding.weight, embedder.rel_im.embedding.weight], dim=1)
	return None


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


def rotate_style_embedding_l3_penalty(model: nn.Module, p: int = 3) -> torch.Tensor | None:
	"""RotatE global L3: ``||E||_p^p + ||R||_p^p`` over full entity/relation weight matrices."""

	terms: list[torch.Tensor] = []
	for embedder in (getattr(model, 'ent_embedder', None), getattr(model, 'rel_embedder', None)):
		if embedder is None:
			continue
		term = _complex_table_l3_penalty(embedder, p=p)
		if term is None and hasattr(embedder, 'embedding'):
			term = embedder.embedding.weight.abs().pow(p).sum()
		elif term is None and hasattr(embedder, 'weight'):
			term = embedder.weight.abs().pow(p).sum()
		if term is not None:
			terms.append(term)
	if not terms:
		return None
	return sum(terms)


# ---------------------------------------------------------------------------
# LibKGE-style class hierarchy
# ---------------------------------------------------------------------------


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


class ParameterEmbedder(KGEEmbedder):
	"""Wrap a shared ``nn.Parameter`` matrix as a LibKGE-style embedder."""

	def __init__(self, weight: nn.Parameter):
		dim = int(weight.size(-1)) if weight.dim() >= 1 else None
		super().__init__(dim=dim)
		self.register_parameter('weight', weight)

	def forward(self, indices: torch.Tensor) -> torch.Tensor:
		return self.weight.index_select(0, indices.long())

	def embed(self, indices: torch.Tensor) -> torch.Tensor:
		return self.forward(indices)

	def get_all(self) -> torch.Tensor:
		return self.weight

	def embed_all(self) -> torch.Tensor:
		return self.weight


class BaseModel(nn.Module, ABC):
	"""Abstract base for text encoders (e.g. SimKGC) with dict-based forward passes."""

	@abstractmethod
	def forward(self, *args, **kwargs) -> dict:
		"""Run a forward pass and return model-specific outputs."""

	@abstractmethod
	def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
		"""Convert model outputs into logits/labels for the training objective."""


class KGEModel(KGEBase):
	"""LibKGE-style binder that ties entity/relation embedders to relational scorers.

	``scorers`` is a non-empty list of ``KGEScorer`` modules. Single-score models use
	length 1; hybrid models (e.g. DaBR) may register multiple component scorers.
	Default LP / AU math delegates to ``scorers[0]`` (the primary scorer); subclasses
	may combine several entries.
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
		self.normalize_lp_scores = _normalize_lp_flag(args) if args is not None else False
		self.lp_score_mode = _lp_score_mode(args) if args is not None else 'original'
		self.lp_distance_degree = _lp_distance_degree(args) if args is not None else 2.0
		self.normalize_au_vectors = _normalize_au_flag(args) if args is not None else False

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

		Default (DistMult-style): ``embed(h) * embed(r)``. Subclasses override.
		"""

		del kwargs
		return self.embed_h(h) * self.embed_r(r)

	def inverse_query_encoder(self, r: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
		"""Encode the head-prediction query from relation and tail indices.

		Default (DistMult-style): ``embed(t) * embed(r)``. Subclasses override.
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
		against a relation-conditioned target (DaBR, TransERR, …) should override
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


class TextKGEModel(KGEModel):
	"""KGEModel binder for token-input encoders with joint (head, relation) queries."""

	training_input_mode = 'tokens'

	def __init__(
		self,
		ent_embedder: nn.Module,
		query_embedder: nn.Module,
		scorers: Sequence[nn.Module] | nn.Module,
		args: Any | None = None,
		contrastive_state: nn.Module | None = None,
	):
		super().__init__(ent_embedder, query_embedder, scorers, args)
		if contrastive_state is not None:
			self.contrastive_state = contrastive_state
		elif args is not None:
			from models.simkgc import build_contrastive_state

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

	def _tail_query_vectors(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
		return self.query_embedder.embed_hr(h, r)

	def score_hrt(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		**kwargs: Any,
	) -> torch.Tensor:
		query = self._tail_query_vectors(h, r)
		tail = self._lp_entity_vectors(self.embed_t(t))
		if self._uses_distance_lp_scores():
			distance_degree = float(getattr(self, 'lp_distance_degree', 2.0) or 2.0)
			return -torch.linalg.vector_norm(query - tail, ord=distance_degree, dim=-1)
		if self._uses_cosine_lp_scores():
			return self._cosine_similarity_scores(query, tail).diag()
		return self.get_scorer().score_emb(
			query, r, tail, 'hrt', **{**self._scorer_kwargs(r), **kwargs}
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
		query = self._tail_query_vectors(h, r)
		lp_entity_embs = self._lp_entity_vectors(all_t_embs)
		if self._uses_distance_lp_scores():
			return self._distance_scores(query, lp_entity_embs)
		if self._uses_cosine_lp_scores():
			return self._cosine_similarity_scores(query, lp_entity_embs)
		scorer_kwargs = {**self._scorer_kwargs(r), **kwargs}
		return self.get_scorer().score_emb(query, r, all_t_embs, 'hr_', **scorer_kwargs)

	def get_queries_targets(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor):
		from data.dataloader import collate
		from data.dataset import Example
		from data.dict_hub import get_relation_id_map
		from utils.device import move_to_cuda

		entity_dict = get_entity_dict()
		relation_id_map = get_relation_id_map() or {}
		idx_to_relation = {int(value): key for key, value in relation_id_map.items()}

		examples = []
		for head_idx, relation_idx, tail_idx in zip(h.tolist(), r.tolist(), t.tolist()):
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
			return self.score_hrt(src, rel, dst)

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
		"""Deprecated: use ``score_hr_`` / index-based LP eval."""

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
