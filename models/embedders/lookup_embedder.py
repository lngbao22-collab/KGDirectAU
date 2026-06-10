"""Lookup-table embedders and factories for all index-based KGE models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from base.model import ParameterEmbedder, _scaled_init, load_relation_to_idx
from data.dict_hub import get_entity_dict


class LookupEmbedder(nn.Module):
	"""Lightweight embedding table with explicit initialization and retrieval helpers."""

	def __init__(self, num_items: int, dim: int, args: Any | None = None):
		super().__init__()
		self.num_items = int(num_items)
		self.dim = int(dim)
		self.args = args
		self.embedding = nn.Embedding(self.num_items, self.dim)
		self._reset_parameters()

	def _reset_parameters(self) -> None:
		model_name = str(getattr(self.args, 'model', '')).lower()
		if any(name in model_name for name in ('rotate', 'protate')):
			margin = float(getattr(self.args, 'margin', 6.0))
			epsilon = float(getattr(self.args, 'epsilon', 2.0))
			bound = (margin + epsilon) / max(1, self.dim)
			nn.init.uniform_(self.embedding.weight, a=-bound, b=bound)
		else:
			nn.init.xavier_uniform_(self.embedding.weight)

	def forward(self, indices: torch.Tensor) -> torch.Tensor:
		return self.embedding(indices.long())

	def get_all(self) -> torch.Tensor:
		return self.embedding.weight

	def embed(self, indices: torch.Tensor) -> torch.Tensor:
		return self.forward(indices)

	def embed_all(self) -> torch.Tensor:
		return self.get_all()


class ComplExEntityEmbedder(nn.Module):
	"""Concatenate real/imaginary entity lookup tables for ComplEx."""

	def __init__(self, ent_re: LookupEmbedder, ent_im: LookupEmbedder):
		super().__init__()
		self.ent_re = ent_re
		self.ent_im = ent_im

	def forward(self, indices: torch.Tensor) -> torch.Tensor:
		return torch.cat([self.ent_re(indices), self.ent_im(indices)], dim=-1)

	def embed(self, indices: torch.Tensor) -> torch.Tensor:
		return self.forward(indices)

	def embed_all(self) -> torch.Tensor:
		device = self.ent_re.embedding.weight.device
		return self.forward(torch.arange(self.ent_re.num_items, device=device))


class ComplExRelationEmbedder(nn.Module):
	"""Concatenate real/imaginary relation lookup tables for ComplEx."""

	def __init__(self, rel_re: LookupEmbedder, rel_im: LookupEmbedder):
		super().__init__()
		self.rel_re = rel_re
		self.rel_im = rel_im

	def forward(self, indices: torch.Tensor) -> torch.Tensor:
		return torch.cat([self.rel_re(indices), self.rel_im(indices)], dim=-1)

	def embed(self, indices: torch.Tensor) -> torch.Tensor:
		return self.forward(indices)


def _counts(args) -> tuple[int, int]:
	entity_dict = get_entity_dict()
	rel_to_idx = load_relation_to_idx(args)
	return len(entity_dict), max(len(rel_to_idx), 1)


def _model_name(args) -> str:
	return str(getattr(args, 'model', '') or '').lower()


def _is_complex(args) -> bool:
	return 'complex' in _model_name(args)


def _is_rotate(args) -> bool:
	name = _model_name(args)
	return 'rotate' in name and 'protate' not in name


def _is_protate(args) -> bool:
	return 'protate' in _model_name(args)


def _is_dabr(args) -> bool:
	return 'dabr' in _model_name(args)


def _embedding_range(args, dim: int) -> float:
	margin = float(getattr(args, 'margin', 6.0))
	return (margin + 2.0) / max(1, dim)


def _init_lookup_table(embedder: LookupEmbedder, args, dim: int) -> None:
	if bool(getattr(args, 'adversarial_training', False)):
		epsilon = 2.0
		margin = float(getattr(args, 'margin', 200.0))
		embedding_range = (margin + epsilon) / dim
		nn.init.uniform_(embedder.embedding.weight, a=-embedding_range, b=embedding_range)
	elif _model_name(args) in {'distmult', 'distmult-au', 'complex', 'complex-au'}:
		_scaled_init(embedder, dim)
	else:
		embedder._reset_parameters()


def build_entity_embedder(args) -> nn.Module:
	n_ent, n_rel = _counts(args)
	dim = int(getattr(args, 'dim', 200))

	if _is_complex(args):
		ent_re = LookupEmbedder(n_ent, dim, args)
		ent_im = LookupEmbedder(n_ent, dim, args)
		if getattr(args, 'init_scaled', True):
			for module in (ent_re, ent_im):
				_scaled_init(module, dim)
		return ComplExEntityEmbedder(ent_re, ent_im)

	if _is_rotate(args):
		hidden_dim = dim
		weight = nn.Parameter(torch.zeros(n_ent, hidden_dim * 2))
		nn.init.uniform_(weight, a=-_embedding_range(args, hidden_dim), b=_embedding_range(args, hidden_dim))
		return ParameterEmbedder(weight)

	if _is_protate(args):
		hidden_dim = dim
		weight = nn.Parameter(torch.zeros(n_ent, hidden_dim))
		nn.init.uniform_(weight, a=-_embedding_range(args, hidden_dim), b=_embedding_range(args, hidden_dim))
		return ParameterEmbedder(weight)

	if _is_dabr(args):
		emb_dim = 4 * dim
		embedder = LookupEmbedder(n_ent, emb_dim, args)
		nn.init.xavier_uniform_(embedder.embedding.weight)
		return embedder

	embedder = LookupEmbedder(n_ent, dim, args)
	_init_lookup_table(embedder, args, dim)
	return embedder


def build_relation_embedder(args) -> nn.Module:
	_, n_rel = _counts(args)
	dim = int(getattr(args, 'dim', 200))

	if _is_complex(args):
		rel_re = LookupEmbedder(n_rel, dim, args)
		rel_im = LookupEmbedder(n_rel, dim, args)
		if getattr(args, 'init_scaled', True):
			for module in (rel_re, rel_im):
				_scaled_init(module, dim)
		return ComplExRelationEmbedder(rel_re, rel_im)

	if _is_rotate(args) or _is_protate(args):
		hidden_dim = dim
		weight = nn.Parameter(torch.zeros(n_rel, hidden_dim))
		nn.init.uniform_(weight, a=-_embedding_range(args, hidden_dim), b=_embedding_range(args, hidden_dim))
		return ParameterEmbedder(weight)

	if _is_dabr(args):
		emb_dim = 4 * int(getattr(args, 'dim', getattr(args, 'hidden_size', 100)))
		embedder = LookupEmbedder(n_rel, emb_dim, args)
		nn.init.xavier_uniform_(embedder.embedding.weight)
		return embedder

	embedder = LookupEmbedder(n_rel, dim, args)
	_init_lookup_table(embedder, args, dim)
	return embedder


def build_dr_embedder(args) -> LookupEmbedder:
	"""DaBR-specific relation drift table (used when binding ``DaBRModel``)."""

	_, n_rel = _counts(args)
	emb_dim = 4 * int(getattr(args, 'dim', getattr(args, 'hidden_size', 100)))
	embedder = LookupEmbedder(n_rel, emb_dim, args)
	nn.init.xavier_uniform_(embedder.embedding.weight)
	return embedder
