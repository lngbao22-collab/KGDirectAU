"""Alignment and uniformity loss for KGAU."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def distinct_first_indices(keys: torch.Tensor) -> torch.Tensor:
	"""Return row indices of the first occurrence of each unique key in the batch."""

	if keys.numel() == 0:
		return keys.new_empty(0, dtype=torch.long)
	if keys.dim() == 1:
		_, inverse = torch.unique(keys, sorted=False, return_inverse=True)
	else:
		_, inverse = torch.unique(keys, dim=0, sorted=False, return_inverse=True)
	num_unique = int(inverse.max().item()) + 1
	indices = torch.full((num_unique,), keys.size(0), dtype=torch.long, device=keys.device)
	positions = torch.arange(keys.size(0), device=keys.device)
	indices.scatter_reduce_(0, inverse, positions, reduce='amin')
	return indices


def _coalesce_float(value, default: float) -> float:
	"""Treat missing or JSON-null hyperparameters as the default."""

	return default if value is None else float(value)


def select_distinct_rows(vectors: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
	"""Keep one embedding row per unique key (first occurrence in the batch)."""

	if vectors.size(0) == 0:
		return vectors
	keys = keys.to(device=vectors.device)
	indices = distinct_first_indices(keys)
	return vectors.index_select(0, indices)


class KGAULoss(nn.Module):
	"""Alignment and uniformity loss for knowledge graph embeddings."""

	def __init__(
		self,
		gamma_q=1.0,
		gamma_t=1.0,
		gamma_h=0.0,
		gamma_ent=0.0,
		tuni=2.0,
		max_uniformity_samples: int = 1024,
		additive_margin: float = 0.0,
	):
		super().__init__()
		self.gamma_q = _coalesce_float(gamma_q, 1.0)
		self.gamma_t = _coalesce_float(gamma_t, 1.0)
		self.gamma_h = _coalesce_float(gamma_h, 0.0)
		self.gamma_ent = _coalesce_float(gamma_ent, 0.0)
		# `tuni` is the uniformity temperature/scaling factor
		self.tuni = _coalesce_float(tuni, 2.0)
		self.max_uniformity_samples = max_uniformity_samples
		# InfoNCE additive margin gamma; geometric threshold m = 2 * gamma on squared L2.
		self.additive_margin = _coalesce_float(additive_margin, 0.0)

	def alignment_loss(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		"""Expected squared L2 distance between paired positive query and target embeddings."""

		q = F.normalize(q, p=2, dim=-1)
		t = F.normalize(t, p=2, dim=-1)
		return (q - t).pow(2).sum(dim=-1).mean()

	def _prepare_uniformity_pairs(self, x: torch.Tensor) -> torch.Tensor | None:
		"""Normalize and subsample embeddings; return squared pairwise L2 distances."""

		if x is None:
			return None
		if x.size(0) < 2:
			return None
		max_samples = int(getattr(self, 'max_uniformity_samples', 0) or 0)
		if max_samples > 0 and x.size(0) > max_samples:
			indices = torch.randperm(x.size(0), device=x.device)[:max_samples]
			x = x.index_select(0, indices)
			if x.size(0) < 2:
				return None
		x = F.normalize(x, p=2, dim=-1)
		pairwise = torch.pdist(x, p=2)
		if pairwise.numel() == 0:
			return None
		return pairwise.pow(2)

	def uniformity_loss(self, x: torch.Tensor) -> torch.Tensor:
		"""Uniformity on the unit hypersphere.

		When additive_margin is 0, uses the original Gaussian-potential AU term.
		When additive_margin > 0, uses margin-aware repulsion with m = 2 * additive_margin.
		"""

		loss, _ = self.uniformity_loss_with_stats(x)
		return loss

	def uniformity_loss_with_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
		"""Return uniformity loss and the fraction of pairs inside the margin buffer (margin mode only)."""

		if x is None:
			return torch.tensor(0.0), 0.0
		if x.size(0) < 2:
			return x.new_zeros(()), 0.0
		dist_sq = self._prepare_uniformity_pairs(x)
		if dist_sq is None:
			return x.new_zeros(()), 0.0
		margin = float(self.additive_margin)
		if margin <= 0.0:
			potential = torch.exp(-self.tuni * dist_sq)
			return potential.mean().log(), 1.0
		geom_margin = 2.0 * margin
		active_frac = float((dist_sq < geom_margin).float().mean().item())
		potential = torch.exp(self.tuni * F.relu(geom_margin - dist_sq))
		return potential.mean().log(), active_frac

	def forward(
		self,
		q: torch.Tensor,
		t: torch.Tensor,
		h: torch.Tensor | None = None,
		ent: torch.Tensor | None = None,
		q_uni: torch.Tensor | None = None,
		t_uni: torch.Tensor | None = None,
		h_uni: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Return the total AU loss together with alignment and uniformity terms."""

		q_norm = F.normalize(q, p=2, dim=-1)
		t_norm = F.normalize(t, p=2, dim=-1)
		l_align = self.alignment_loss(q_norm, t_norm)

		l_unif = q_norm.new_zeros(())

		if self.gamma_q > 0:
			q_uniformity = q_uni if q_uni is not None else q_norm
			l_unif = l_unif + self.gamma_q * self.uniformity_loss(q_uniformity)
		if self.gamma_t > 0:
			t_uniformity = t_uni if t_uni is not None else t_norm
			l_unif = l_unif + self.gamma_t * self.uniformity_loss(t_uniformity)
		if h is not None and self.gamma_h > 0:
			h_uniformity = h_uni if h_uni is not None else F.normalize(h, p=2, dim=-1)
			l_unif = l_unif + self.gamma_h * self.uniformity_loss(h_uniformity)
		if ent is not None and self.gamma_ent > 0:
			ent_norm = F.normalize(ent, p=2, dim=-1)
			l_unif = l_unif + self.gamma_ent * self.uniformity_loss(ent_norm)

		total_loss = l_align + l_unif
		return total_loss, l_align, l_unif
