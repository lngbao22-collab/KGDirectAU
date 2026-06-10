"""Standalone lookup embedder for turning item IDs into dense tensors."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


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
		"""Initialize embedding weights using the model family convention when available."""

		model_name = str(getattr(self.args, "model", "")).lower()
		if any(name in model_name for name in ("rotate", "protate")):
			margin = float(getattr(self.args, "margin", 6.0))
			epsilon = float(getattr(self.args, "epsilon", 2.0))
			bound = (margin + epsilon) / max(1, self.dim)
			nn.init.uniform_(self.embedding.weight, a=-bound, b=bound)
		else:
			nn.init.xavier_uniform_(self.embedding.weight)

	def forward(self, indices: torch.Tensor) -> torch.Tensor:
		"""Return dense embeddings for a batch of item indices."""

		return self.embedding(indices.long())

	def get_all(self) -> torch.Tensor:
		"""Return the full embedding matrix for 1-vs-all broadcasting."""

		return self.embedding.weight

	def embed(self, indices: torch.Tensor) -> torch.Tensor:
		"""Alias for forward to ease reuse from existing code paths."""

		return self.forward(indices)

	def embed_all(self) -> torch.Tensor:
		"""Alias for get_all to mirror the libkge-style embedder API."""

		return self.get_all()

