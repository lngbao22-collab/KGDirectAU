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
		alignment_mode: str = 'cosine',
		normalize_uniformity: bool = True,
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
		# `cosine`: L2-normalize paired vectors (DistMult/ComplEx/SimKGC).
		# `phase_residual`: element-wise squared phase residual without global normalization (RotatE family).
		self.alignment_mode = alignment_mode or 'cosine'
		self.normalize_uniformity = normalize_uniformity

	def alignment_loss(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		"""Expected squared L2 distance between paired positive query and target embeddings."""

		q = F.normalize(q, p=2, dim=-1)
		t = F.normalize(t, p=2, dim=-1)
		return (q - t).pow(2).sum(dim=-1).mean()

	def _subsample_uniformity_rows(self, x: torch.Tensor) -> torch.Tensor | None:
		"""Cap row count before uniformity (entity table or large batches)."""

		if x is None or x.size(0) < 2:
			return None
		max_samples = int(getattr(self, 'max_uniformity_samples', 0) or 0)
		if max_samples > 0 and x.size(0) > max_samples:
			indices = torch.randperm(x.size(0), device=x.device)[:max_samples]
			x = x.index_select(0, indices)
		return x if x.size(0) >= 2 else None

	def _max_uniformity_pair_count(self, num_rows: int, dim: int) -> int:
		"""Choose how many pairwise distances to estimate without O(n^2) autograd memory."""

		full_pairs = num_rows * (num_rows - 1) // 2
		if full_pairs <= 0:
			return 0
		max_samples = int(getattr(self, 'max_uniformity_samples', 0) or 0)
		# `pdist` backward requires O(n^2 * dim) intermediate storage; budget 32 MiB to stay safe
		# across high-dim AU vectors (e.g. DaBR concatenates two 2000-D vectors → 4000-D).
		pdist_budget = 32 * 1024 * 1024
		if num_rows * num_rows * max(dim, 1) * 2 <= pdist_budget:  # assume fp16 (×2 bytes)
			return full_pairs
		pair_cap = int(getattr(self, 'max_uniformity_pairs', 0) or 0)
		if pair_cap <= 0:
			pair_cap = max(4096, max_samples * 8)
		return min(full_pairs, pair_cap)

	@staticmethod
	def _random_pairwise_dist_sq(x: torch.Tensor, num_pairs: int) -> torch.Tensor:
		"""Monte Carlo squared L2 distances between random row pairs (memory-safe)."""

		n = x.size(0)
		i = torch.randint(0, n, (num_pairs,), device=x.device)
		j = torch.randint(0, n, (num_pairs,), device=x.device)
		same = i == j
		if same.any():
			j = torch.where(same, (j + 1) % n, j)
		return (x[i] - x[j]).pow(2).sum(dim=-1)

	def _prepare_uniformity_pairs(self, x: torch.Tensor) -> torch.Tensor | None:
		"""Normalize and subsample embeddings; return squared pairwise L2 distances."""

		x = self._subsample_uniformity_rows(x)
		if x is None:
			return None
		if self.normalize_uniformity:
			x = F.normalize(x, p=2, dim=-1)
		num_pairs = self._max_uniformity_pair_count(x.size(0), x.size(-1))
		if num_pairs <= 0:
			return None
		full_pairs = x.size(0) * (x.size(0) - 1) // 2
		if num_pairs >= full_pairs:
			pairwise = torch.pdist(x, p=2)
			if pairwise.numel() == 0:
				return None
			return pairwise.pow(2)
		return self._random_pairwise_dist_sq(x, num_pairs)

	def uniformity_loss(self, x: torch.Tensor) -> torch.Tensor:
		"""Uniformity on the unit hypersphere.

		When additive_margin is 0, uses the original Gaussian-potential AU term.
		When additive_margin > 0, uses margin-aware repulsion with m = 2 * additive_margin.
		"""

		loss, _ = self.uniformity_loss_with_stats(x)
		return loss

	def _margin_uniformity_fraction(self, margin: float) -> float:
		"""Map InfoNCE-style additive margin to a closest-pair fraction for uniformity."""

		# gamma=0.02 -> penalize the closest ~20% of pairs; clamp for stability.
		return min(0.5, max(0.05, margin * 10.0))

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
		# Fixed m = 2*gamma is far too small in high dimensions (random pairs have d^2 ~ 2).
		# Use an adaptive buffer from batch geometry: repel the closest fraction of pairs.
		target_frac = self._margin_uniformity_fraction(margin)
		geom_margin = torch.quantile(dist_sq, target_frac)
		buffer_penalty = torch.exp(self.tuni * F.relu(geom_margin - dist_sq))
		# Keep classic AU spread so early epochs still have strong uniformity signal.
		spread = torch.exp(-self.tuni * dist_sq).mean().log()
		buffer = buffer_penalty.mean().log()
		active_frac = float((dist_sq < geom_margin).float().mean().item())
		return spread + buffer, active_frac

	def forward(
		self,
		q: torch.Tensor,
		t: torch.Tensor,
		h: torch.Tensor | None = None,
		ent: torch.Tensor | None = None,
		q_uni: torch.Tensor | None = None,
		t_uni: torch.Tensor | None = None,
		h_uni: torch.Tensor | None = None,
		external_align: torch.Tensor | None = None,
		return_stats: bool = False,
	):
		"""Return the total AU loss together with alignment and uniformity terms.

		Each uniformity term is computed exactly once. When ``return_stats`` is
		True, also return the gamma-weighted fraction of query/target pairs that
		fall inside the margin buffer (only meaningful when ``additive_margin`` > 0).
		"""

		if external_align is not None:
			l_align = external_align
		elif self.alignment_mode == 'phase_residual':
			l_align = (q - t).pow(2).sum(dim=-1).mean()
		else:
			l_align = self.alignment_loss(q, t)

		l_unif = q.new_zeros(())
		active_sum = 0.0
		active_weight = 0.0

		if self.gamma_q > 0:
			q_uniformity = q_uni if q_uni is not None else q
			term, frac = self.uniformity_loss_with_stats(q_uniformity)
			l_unif = l_unif + self.gamma_q * term
			active_sum += self.gamma_q * frac
			active_weight += self.gamma_q
		if self.gamma_t > 0:
			t_uniformity = t_uni if t_uni is not None else t
			term, frac = self.uniformity_loss_with_stats(t_uniformity)
			l_unif = l_unif + self.gamma_t * term
			active_sum += self.gamma_t * frac
			active_weight += self.gamma_t
		if h is not None and self.gamma_h > 0:
			h_uniformity = h_uni if h_uni is not None else h
			term, _ = self.uniformity_loss_with_stats(h_uniformity)
			l_unif = l_unif + self.gamma_h * term
		if ent is not None and self.gamma_ent > 0:
			ent_rows = self._subsample_uniformity_rows(ent)
			if ent_rows is not None:
				term, _ = self.uniformity_loss_with_stats(ent_rows)
				l_unif = l_unif + self.gamma_ent * term

		total_loss = l_align + l_unif
		if return_stats:
			if float(self.additive_margin) > 0.0 and active_weight > 0:
				margin_active_frac = active_sum / active_weight
			else:
				margin_active_frac = 0.0
			return total_loss, l_align, l_unif, margin_active_frac
		return total_loss, l_align, l_unif
