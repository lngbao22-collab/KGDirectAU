"""Lookup-table embedders and factories for ComplEx / ComplEx-AU."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.model import KGEEmbedder
from data.dict_hub import get_entity_dict
from utils.relations import load_relation_to_idx


class LookupEmbedder(KGEEmbedder):
	"""Lightweight embedding table with configurable init and optional dropout."""

	input_mode = 'indices'

	def __init__(
		self,
		num_items: int,
		dim: int,
		args: Any | None = None,
		*,
		role: str = 'entity',
	):
		super().__init__(args, dim=int(dim))
		self.num_items = int(num_items)
		self.role = role
		self.dropout_rate = self.dropout_rate_from_args(args, role)
		sparse = bool(getattr(args, 'sparse_embeddings', False))
		self.embedding = nn.Embedding(self.num_items, self.dim, sparse=sparse)
		self._reset_parameters()

	@staticmethod
	def dropout_rate_from_args(args: Any | None, role: str) -> float:
		if role == 'entity':
			raw = getattr(args, 'entity_dropout', None) if args is not None else None
			if raw is None:
				raw = getattr(args, 'dropout', 0.0) if args is not None else 0.0
		else:
			raw = getattr(args, 'relation_dropout', None) if args is not None else None
			if raw is None:
				raw = getattr(args, 'dropout', 0.0) if args is not None else 0.0
		return float(raw or 0.0)

	@staticmethod
	def scaled_init(module: nn.Module, dim: int, sigma: float = 0.2) -> None:
		"""ComplEx-style scale: ``(dim / σ²)^(1/6)``."""

		scale = (dim / sigma ** 2) ** (1 / 6)
		for param in module.parameters():
			param.data.div_(scale)

	@staticmethod
	def initialize(module: nn.Module, args: Any | None, dim: int, role: str = 'entity') -> None:
		"""Initialize a lookup embedding table (``lookup_embedder.initialize``)."""

		del role  # reserved for role-specific defaults
		init_method = str(getattr(args, 'init_method', '') or '').lower() if args is not None else ''
		if not init_method:
			LookupEmbedder.scaled_init(module, dim)
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
			LookupEmbedder.scaled_init(module, dim)
		elif init_method == 'kbc':
			scale = float(getattr(args, 'init_scale', 1e-3))
			weight.data.mul_(scale)
		else:
			nn.init.xavier_uniform_(weight)

	def _reset_parameters(self) -> None:
		if getattr(self.args, 'init_method', None):
			self.initialize(self, self.args, self.dim, role=self.role)
			return
		nn.init.xavier_uniform_(self.embedding.weight)

	def forward(self, indices: torch.Tensor) -> torch.Tensor:
		vectors = self.embedding(indices.long())
		if self.training and self.dropout_rate > 0.0:
			vectors = F.dropout(vectors, p=self.dropout_rate, training=True)
		return vectors

	def get_all(self) -> torch.Tensor:
		vectors = self.embedding.weight
		if self.training and self.dropout_rate > 0.0:
			vectors = F.dropout(vectors, p=self.dropout_rate, training=True)
		return vectors

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


def _float_arg(args, name: str, default: float) -> float:
	"""Read a float hyperparameter; treat JSON/CLI ``null`` like unset."""

	raw = getattr(args, name, None)
	return default if raw is None else float(raw)


def _is_complex(args) -> bool:
	return 'complex' in _model_name(args)


def _adversarial_gamma(args) -> float:
	"""Adversarial embedding-init gamma/margin (default 200)."""

	raw = getattr(args, 'margin', None)
	if raw is None:
		raw = getattr(args, 'gamma', 200.0)
	return float(raw)


def _adversarial_uniform_init(embedder: LookupEmbedder, args, dim: int) -> None:
	"""Uniform init with range ``(gamma + 2) / dim``."""

	epsilon = _float_arg(args, 'epsilon', 2.0)
	embedding_range = (_adversarial_gamma(args) + epsilon) / max(dim, 1)
	nn.init.uniform_(embedder.embedding.weight, a=-embedding_range, b=embedding_range)


def _complex_part_dim(args, dim: int, *, role: str) -> int:
	"""Per-component ComplEx table width; default ON preserves existing ComplEx."""

	flag = 'double_entity_embedding' if role == 'entity' else 'double_relation_embedding'
	if bool(getattr(args, flag, True)):
		return dim
	return max(dim // 2, 1)


def _init_complex_pair(first: LookupEmbedder, second: LookupEmbedder, args, dim: int, part_dim: int) -> None:
	if bool(getattr(args, 'adversarial_training', False)):
		for module in (first, second):
			_adversarial_uniform_init(module, args, dim)
	elif getattr(args, 'init_method', None):
		for module in (first, second):
			LookupEmbedder.initialize(module, args, part_dim, role=module.role)
	elif getattr(args, 'init_scaled', True):
		for module in (first, second):
			LookupEmbedder.scaled_init(module, part_dim)


def build_entity_embedder(args) -> nn.Module:
	if not _is_complex(args):
		raise ValueError(f'lookup_embedder only supports ComplEx / ComplEx-AU, got {getattr(args, "model", None)!r}')
	n_ent, _n_rel = _counts(args)
	dim = int(getattr(args, 'dim', 200))
	entity_dim = _complex_part_dim(args, dim, role='entity')
	ent_re = LookupEmbedder(n_ent, entity_dim, args, role='entity')
	ent_im = LookupEmbedder(n_ent, entity_dim, args, role='entity')
	_init_complex_pair(ent_re, ent_im, args, dim, entity_dim)
	return ComplExEntityEmbedder(ent_re, ent_im)


def build_relation_embedder(args) -> nn.Module:
	if not _is_complex(args):
		raise ValueError(f'lookup_embedder only supports ComplEx / ComplEx-AU, got {getattr(args, "model", None)!r}')
	_, n_rel = _counts(args)
	dim = int(getattr(args, 'dim', 200))
	relation_dim = _complex_part_dim(args, dim, role='relation')
	rel_re = LookupEmbedder(n_rel, relation_dim, args, role='relation')
	rel_im = LookupEmbedder(n_rel, relation_dim, args, role='relation')
	_init_complex_pair(rel_re, rel_im, args, dim, relation_dim)
	return ComplExRelationEmbedder(rel_re, rel_im)
