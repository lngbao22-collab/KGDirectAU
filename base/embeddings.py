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
	return bool(getattr(args, 'add_reciprocal_relations', False))


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


def load_relation_to_idx(args) -> dict[str, int]:
	for path in _relation_path_candidates(args):
		if not path or not os.path.exists(path):
			continue
		with open(path, 'r', encoding='utf-8') as handle:
			mapping = json.load(handle)
		if isinstance(mapping, dict):
			relation_to_idx = {str(key): int(value) for key, value in mapping.items()}
			if use_reciprocal_relations(args):
				relation_to_idx = add_inverse_relations(relation_to_idx)
			return relation_to_idx

	relations: list[str] = []
	seen: set[str] = set()
	for example in load_data(getattr(args, 'train_path', ''), add_forward_triplet=False, add_backward_triplet=False):
		if example.relation not in seen:
			seen.add(example.relation)
			relations.append(example.relation)
	relation_to_idx = {relation: idx for idx, relation in enumerate(relations)}
	if use_reciprocal_relations(args):
		relation_to_idx = add_inverse_relations(relation_to_idx)
	return relation_to_idx


def _scaled_init(module: nn.Module, dim: int, sigma: float = 0.2) -> None:
	scale = (dim / sigma ** 2) ** (1 / 6)
	for param in module.parameters():
		param.data.div_(scale)


def init_lookup_embedding(module: nn.Module, args: Any | None, dim: int, role: str = 'entity') -> None:
	"""Initialize a lookup embedding table (LibKGE ``lookup_embedder.initialize``)."""

	init_method = str(getattr(args, 'init_method', '') or '').lower()
	if not init_method:
		model_name = str(getattr(args, 'model', '') or '').lower()
		if any(name in model_name for name in ('rotate', 'protate')):
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


def _embedding_table_l3(embedder: nn.Module) -> torch.Tensor | None:
	if hasattr(embedder, 'embedding'):
		return embedder.embedding.weight.norm(p=3) ** 3
	if hasattr(embedder, 'ent_re') and hasattr(embedder, 'ent_im'):
		return embedder.ent_re.embedding.weight.norm(p=3) ** 3 + embedder.ent_im.embedding.weight.norm(p=3) ** 3
	if hasattr(embedder, 'rel_re') and hasattr(embedder, 'rel_im'):
		return embedder.rel_re.embedding.weight.norm(p=3) ** 3 + embedder.rel_im.embedding.weight.norm(p=3) ** 3
	if hasattr(embedder, 'weight'):
		return embedder.weight.norm(p=3) ** 3
	return None


def compute_kge_l3_regularization(model: nn.Module, args: Any | None) -> torch.Tensor | None:
	"""LibKGE-style L3 embedding regularization with per-table weights."""

	ent_weight = float(getattr(args, 'entity_regularize_weight', 0.0) or 0.0)
	rel_weight = float(getattr(args, 'relation_regularize_weight', 0.0) or 0.0)
	if ent_weight == 0.0 and rel_weight == 0.0:
		return None

	terms: list[torch.Tensor] = []
	ent_embedder = getattr(model, 'ent_embedder', None)
	rel_embedder = getattr(model, 'rel_embedder', None)
	if ent_weight > 0.0 and ent_embedder is not None:
		ent_term = _embedding_table_l3(ent_embedder)
		if ent_term is not None:
			terms.append(ent_weight * ent_term)
	if rel_weight > 0.0 and rel_embedder is not None:
		rel_term = _embedding_table_l3(rel_embedder)
		if rel_term is not None:
			terms.append(rel_weight * rel_term)
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
