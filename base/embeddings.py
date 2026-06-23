"""Shared embedding utilities used by model binders and lookup embedders."""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn as nn

from data.dataset import load_data


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
	"""Map forward relation indices to inverse indices (KvsAll ``_po`` / kbc CE head eval)."""

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


HEAD_EVAL_MODES = frozenset({'po_forward', 'po_inverse', 'sp_inverse'})


def resolve_head_eval_mode(args: Any | None, *, eval_forward: bool) -> str:
	"""Choose backward link-prediction scoring to match the training recipe.

	Returns one of ``tail``, ``po_forward``, ``po_inverse``, or ``sp_inverse``.

	When ``args.head_eval_mode`` is set in JSON/CLI, it overrides strategy-based
	inference for the backward (head) pass. Use ``po_forward`` for adversarial
	BCE / direct head prediction with the forward relation (no inverse relation).
	KGAU with reciprocal relations trains inverse triplets ``(t, r^{-1}, h)`` as
	tail prediction, so backward eval defaults to ``sp_inverse``.
	"""

	if eval_forward:
		return 'tail'
	if args is None:
		return 'po_forward'

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
		return 'po_forward'

	if 'kvsall' in strategy:
		from models.strategies.kvsall_strategy import kvsall_uses_po_training

		if kvsall_uses_po_training(args) and use_reciprocal_relations(args):
			return 'po_inverse'
		return 'po_forward'

	if '1vsall' in strategy:
		if use_kbc_reciprocal_relations(args) and not bool(getattr(args, 'bidirectional_1vsall', True)):
			return 'sp_inverse'
		return 'po_forward'

	if 'kgau' in strategy:
		if use_reciprocal_relations(args):
			return 'sp_inverse'
		return 'po_forward'

	return 'po_forward'


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
			margin = float(getattr(args, 'margin', 6.0))
			epsilon = float(getattr(args, 'epsilon', 2.0))
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


class ParameterEmbedder(nn.Module):
	"""Wrap a shared ``nn.Parameter`` matrix as a LibKGE-style embedder."""

	def __init__(self, weight: nn.Parameter):
		super().__init__()
		self.register_parameter('weight', weight)

	def forward(self, indices: torch.Tensor) -> torch.Tensor:
		return self.weight.index_select(0, indices.long())

	def embed(self, indices: torch.Tensor) -> torch.Tensor:
		return self.forward(indices)

	def get_all(self) -> torch.Tensor:
		return self.weight

	def embed_all(self) -> torch.Tensor:
		return self.weight
