"""Alignment and uniformity loss for KGAU."""

from __future__ import annotations

import math

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


_GAMMA_NAMES = ('q', 't', 'h', 'ent', 'cross')


class KGAULoss(nn.Module):
	"""Alignment and uniformity loss for knowledge graph embeddings."""

	def __init__(
		self,
		gamma_q=1.0,
		gamma_t=1.0,
		gamma_h=0.0,
		gamma_ent=0.0,
		gamma_cross=0.0,
		tuni=2.0,
		learnable_tuni: bool = False,
		learnable_au_gammas: bool = False,
		learnable_au_weights: bool = False,
		alpha: float = 1.0,
		max_uniformity_samples: int = 1024,
		additive_margin: float = 0.0,
		alignment_mode: str = 'cosine',
		normalize_uniformity: bool = True,
	):
		super().__init__()
		self.learnable_au_weights = bool(learnable_au_weights)
		self.learnable_au_gammas = bool(learnable_au_gammas) and not self.learnable_au_weights
		gamma_learnable = self.learnable_au_weights or self.learnable_au_gammas
		gamma_inits = {
			'q': _coalesce_float(gamma_q, 1.0),
			't': _coalesce_float(gamma_t, 1.0),
			'h': _coalesce_float(gamma_h, 0.0),
			'ent': _coalesce_float(gamma_ent, 0.0),
			'cross': _coalesce_float(gamma_cross, 0.0),
		}
		for name, value in gamma_inits.items():
			init = float(value)
			# init <= 0 disables a term (fixed or learnable); only positive inits are scheduled/learned.
			self.register_buffer(f'gamma_init_{name}', torch.tensor(init))
			if not gamma_learnable:
				setattr(self, f'_gamma_{name}', init)
			elif init > 0.0:
				# Bounded downward adjustment only: exp(adj) in (0, 1], init at 0.
				# Unconstrained log-scale gammas rise under AU loss because uniformity is negative.
				setattr(self, f'log_gamma_adj_{name}', nn.Parameter(torch.zeros(())))
		self.register_buffer('gamma_schedule_mult', torch.tensor(1.0))
		if self.learnable_au_weights:
			alpha_init = _coalesce_float(alpha, 1.0)
			self.register_buffer('alpha_init', torch.tensor(alpha_init))
			# Increase-only: alpha = alpha_init * exp(clamp(adj, min=0)).
			self.log_alpha_adj = nn.Parameter(torch.zeros(()))
		# `tuni` is the uniformity temperature/scaling factor; optionally learnable via log-scale.
		tuni_val = _coalesce_float(tuni, 2.0)
		if learnable_tuni:
			self.log_tuni = nn.Parameter(torch.tensor(math.log(tuni_val)))
		else:
			self._tuni = tuni_val
		self.max_uniformity_samples = max_uniformity_samples
		# InfoNCE additive margin gamma; geometric threshold m = 2 * gamma on squared L2.
		self.additive_margin = _coalesce_float(additive_margin, 0.0)
		# `cosine`: L2-normalize paired vectors (DistMult/ComplEx/SimKGC).
		# `phase_residual`: element-wise squared phase residual without global normalization.
		# `sin_phase`: pRotatE link-pred term sum_i |sin(theta_q,i - theta_t,i)| (no global normalize).
		self.alignment_mode = alignment_mode or 'cosine'
		self.normalize_uniformity = normalize_uniformity

	def _gamma_init_value(self, name: str) -> float:
		if self.learnable_au_weights or self.learnable_au_gammas:
			return float(getattr(self, f'gamma_init_{name}').detach().cpu().item())
		return float(getattr(self, f'_gamma_{name}'))

	def alpha_value(self) -> float:
		"""Scalar effective alignment weight for logging."""

		value = self._effective_alpha()
		if torch.is_tensor(value):
			return float(value.detach().cpu().item())
		return float(value)

	def _effective_alpha(self) -> torch.Tensor | float:
		if not self.learnable_au_weights:
			return 1.0
		return self.alpha_init * torch.exp(torch.clamp(self.log_alpha_adj, min=0.0))

	def gamma_schedule_mult_value(self) -> float:
		return float(self.gamma_schedule_mult.detach().cpu().item())

	def _learnable_gamma_factor(self, name: str) -> torch.Tensor:
		adj = getattr(self, f'log_gamma_adj_{name}')
		return torch.exp(torch.clamp(adj, max=0.0))

	def _learnable_gamma_enabled(self, name: str) -> bool:
		return hasattr(self, f'log_gamma_adj_{name}')

	def set_gamma_schedule_mult(self, mult: float) -> None:
		self.gamma_schedule_mult.fill_(float(mult))

	def clamp_learnable_gamma_adj(self) -> None:
		"""Keep learnable AU auxiliary params within their bounded ranges."""

		if self.learnable_au_weights or self.learnable_au_gammas:
			for name in _GAMMA_NAMES:
				if not hasattr(self, f'log_gamma_adj_{name}'):
					continue
				with torch.no_grad():
					getattr(self, f'log_gamma_adj_{name}').clamp_(max=0.0)
		if self.learnable_au_weights:
			with torch.no_grad():
				self.log_alpha_adj.clamp_(min=0.0)

	def _effective_gamma(self, name: str) -> torch.Tensor | float:
		if self.learnable_au_weights:
			init = getattr(self, f'gamma_init_{name}')
			if not self._learnable_gamma_enabled(name):
				return init
			return init * self._learnable_gamma_factor(name)
		mult = self.gamma_schedule_mult
		if self.learnable_au_gammas:
			init = getattr(self, f'gamma_init_{name}')
			if not self._learnable_gamma_enabled(name):
				return init * mult
			return init * mult * self._learnable_gamma_factor(name)
		return float(getattr(self, f'_gamma_{name}')) * float(mult)

	def gamma_value(self, name: str) -> float:
		"""Scalar effective gamma for logging and control flow."""

		value = self._effective_gamma(name)
		if torch.is_tensor(value):
			return float(value.detach().cpu().item())
		return float(value)

	def gamma_active(self, name: str, eps: float = 1e-6) -> bool:
		return self.gamma_value(name) > eps

	@property
	def gamma_q(self) -> float:
		return self.gamma_value('q')

	@property
	def gamma_t(self) -> float:
		return self.gamma_value('t')

	@property
	def gamma_h(self) -> float:
		return self.gamma_value('h')

	@property
	def gamma_ent(self) -> float:
		return self.gamma_value('ent')

	@property
	def gamma_cross(self) -> float:
		return self.gamma_value('cross')

	@property
	def tuni(self):
		if hasattr(self, 'log_tuni'):
			return torch.exp(self.log_tuni)
		return self._tuni

	@tuni.setter
	def tuni(self, value):
		if hasattr(self, 'log_tuni'):
			with torch.no_grad():
				self.log_tuni.data.fill_(math.log(float(value)))
		else:
			self._tuni = float(value)

	def alignment_loss(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		"""Expected squared L2 distance between paired positive query and target embeddings."""

		q = F.normalize(q, p=2, dim=-1)
		t = F.normalize(t, p=2, dim=-1)
		return (q - t).pow(2).sum(dim=-1).mean()

	def sin_phase_alignment_loss(self, phase_query: torch.Tensor, phase_target: torch.Tensor) -> torch.Tensor:
		"""Alignment in native pRotatE geometry: mean sum of |sin(phase residual)| per dimension.

		Matches the penalty inside ``pRotatEEncoder._rotate_score`` (before margin/modulus):
		minimizing this term raises positive link-prediction scores without cosine normalization.
		"""

		residual = phase_query - phase_target
		return torch.abs(torch.sin(residual)).sum(dim=-1).mean()

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

	def _scaled_uniformity_dist_sq(self, dist_sq: torch.Tensor) -> torch.Tensor:
		"""Map pairwise distances to a stable scale for the Gaussian potential."""

		if self.normalize_uniformity:
			return dist_sq
		# Raw phase vectors have O(dim) squared L2 distance; normalize by batch geometry so
		# `tuni` stays comparable to the unit-sphere case (typically 2-4).
		scale = dist_sq.median().clamp_min(1e-6)
		return dist_sq / scale

	@staticmethod
	def _log_mean_potential(neg_scaled_dist_sq: torch.Tensor) -> torch.Tensor:
		"""Compute log(mean(exp(-x))) without log(0) underflow or 0/0 NaN gradients."""

		if neg_scaled_dist_sq.numel() == 0:
			return neg_scaled_dist_sq.new_zeros(())
		return torch.logsumexp(-neg_scaled_dist_sq, dim=0) - torch.log(
			neg_scaled_dist_sq.new_tensor(float(neg_scaled_dist_sq.numel()))
		)

	def uniformity_loss_with_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
		"""Return uniformity loss and the fraction of pairs inside the margin buffer (margin mode only)."""

		if x is None:
			return torch.tensor(0.0), 0.0
		if x.size(0) < 2:
			return x.new_zeros(()), 0.0
		dist_sq = self._prepare_uniformity_pairs(x)
		if dist_sq is None:
			return x.new_zeros(()), 0.0
		scaled_dist_sq = self._scaled_uniformity_dist_sq(dist_sq)
		margin = float(self.additive_margin)
		if margin <= 0.0:
			return self._log_mean_potential(self.tuni * scaled_dist_sq), 1.0
		# Fixed m = 2*gamma is far too small in high dimensions (random pairs have d^2 ~ 2).
		# Use an adaptive buffer from batch geometry: repel the closest fraction of pairs.
		target_frac = self._margin_uniformity_fraction(margin)
		geom_margin = torch.quantile(scaled_dist_sq, target_frac)
		buffer_penalty = torch.exp(self.tuni * F.relu(geom_margin - scaled_dist_sq))
		# Keep classic AU spread so early epochs still have strong uniformity signal.
		spread = self._log_mean_potential(self.tuni * scaled_dist_sq)
		buffer = buffer_penalty.mean().clamp_min(1e-12).log()
		active_frac = float((scaled_dist_sq < geom_margin).float().mean().item())
		return spread + buffer, active_frac

	def forward_alignment(
		self,
		q: torch.Tensor,
		t: torch.Tensor,
		external_align: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Alignment term only (mean over rows of ``q`` and ``t``)."""

		if external_align is not None:
			return external_align
		if self.alignment_mode == 'sin_phase':
			return self.sin_phase_alignment_loss(q, t)
		if self.alignment_mode == 'phase_residual':
			return (q - t).pow(2).sum(dim=-1).mean()
		return self.alignment_loss(q, t)

	def forward_uniformity(
		self,
		q: torch.Tensor,
		t: torch.Tensor,
		q_uni: torch.Tensor | None = None,
		t_uni: torch.Tensor | None = None,
		h: torch.Tensor | None = None,
		h_uni: torch.Tensor | None = None,
		ent: torch.Tensor | None = None,
		cross_uni: torch.Tensor | None = None,
		return_components: bool = False,
	) -> tuple[torch.Tensor, float] | tuple[torch.Tensor, float, dict[str, dict]]:
		"""Uniformity terms only; returns (loss, margin-active fraction for logging)."""

		l_unif = q.new_zeros(())
		active_sum = 0.0
		active_weight = 0.0
		components: dict[str, dict] = {}

		def _record(name: str, scale_key: str, raw_term: torch.Tensor, scale) -> None:
			nonlocal l_unif
			weighted = raw_term * scale
			l_unif = l_unif + weighted
			if return_components:
				components[f'unif_{name}'] = {
					'raw': raw_term,
					'scale': scale,
					'scale_name': scale_key,
					'weighted': weighted,
				}

		gamma_q = self._effective_gamma('q')
		if self.gamma_active('q'):
			q_uniformity = q_uni if q_uni is not None else q
			term, frac = self.uniformity_loss_with_stats(q_uniformity)
			_record('q', 'gamma_q', term, gamma_q)
			active_sum += self.gamma_value('q') * frac
			active_weight += self.gamma_value('q')
		gamma_t = self._effective_gamma('t')
		if self.gamma_active('t'):
			t_uniformity = t_uni if t_uni is not None else t
			term, frac = self.uniformity_loss_with_stats(t_uniformity)
			_record('t', 'gamma_t', term, gamma_t)
			active_sum += self.gamma_value('t') * frac
			active_weight += self.gamma_value('t')
		gamma_h = self._effective_gamma('h')
		if h is not None and self.gamma_active('h'):
			h_uniformity = h_uni if h_uni is not None else h
			term, _ = self.uniformity_loss_with_stats(h_uniformity)
			_record('h', 'gamma_h', term, gamma_h)
		gamma_ent = self._effective_gamma('ent')
		if ent is not None and self.gamma_active('ent'):
			ent_rows = self._subsample_uniformity_rows(ent)
			if ent_rows is not None:
				term, _ = self.uniformity_loss_with_stats(ent_rows)
				_record('ent', 'gamma_ent', term, gamma_ent)
		gamma_cross = self._effective_gamma('cross')
		if cross_uni is not None and self.gamma_active('cross'):
			term, _ = self.uniformity_loss_with_stats(cross_uni)
			_record('cross', 'gamma_cross', term, gamma_cross)

		if float(self.additive_margin) > 0.0 and active_weight > 0:
			margin_active_frac = active_sum / active_weight
		else:
			margin_active_frac = 0.0
		if return_components:
			return l_unif, margin_active_frac, components
		return l_unif, margin_active_frac

	def forward_components(
		self,
		q: torch.Tensor,
		t: torch.Tensor,
		h: torch.Tensor | None = None,
		ent: torch.Tensor | None = None,
		q_uni: torch.Tensor | None = None,
		t_uni: torch.Tensor | None = None,
		h_uni: torch.Tensor | None = None,
		cross_uni: torch.Tensor | None = None,
		external_align: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, dict[str, dict]]:
		"""Return total/align/unif losses, margin stats, and per-term component breakdown."""

		l_align = self.forward_alignment(q, t, external_align=external_align)
		l_unif, margin_active_frac, components = self.forward_uniformity(
			q, t, q_uni=q_uni, t_uni=t_uni, h=h, h_uni=h_uni, ent=ent, cross_uni=cross_uni,
			return_components=True,
		)
		alpha = self._effective_alpha()
		l_align_weighted = l_align * alpha if torch.is_tensor(alpha) else l_align * float(alpha)
		components['align'] = {
			'raw': l_align,
			'scale': alpha,
			'scale_name': 'alpha',
			'weighted': l_align_weighted,
		}
		total_loss = l_align_weighted + l_unif
		return total_loss, l_align, l_unif, margin_active_frac, components

	def forward(
		self,
		q: torch.Tensor,
		t: torch.Tensor,
		h: torch.Tensor | None = None,
		ent: torch.Tensor | None = None,
		q_uni: torch.Tensor | None = None,
		t_uni: torch.Tensor | None = None,
		h_uni: torch.Tensor | None = None,
		cross_uni: torch.Tensor | None = None,
		external_align: torch.Tensor | None = None,
		return_stats: bool = False,
	):
		"""Return the total AU loss together with alignment and uniformity terms.

		Each uniformity term is computed exactly once. When ``return_stats`` is
		True, also return the gamma-weighted fraction of query/target pairs that
		fall inside the margin buffer (only meaningful when ``additive_margin`` > 0).
		"""

		l_align = self.forward_alignment(q, t, external_align=external_align)
		l_unif, margin_active_frac = self.forward_uniformity(
			q, t, q_uni=q_uni, t_uni=t_uni, h=h, h_uni=h_uni, ent=ent, cross_uni=cross_uni,
		)

		alpha = self._effective_alpha()
		l_align_weighted = l_align * alpha if torch.is_tensor(alpha) else l_align * float(alpha)
		total_loss = l_align_weighted + l_unif
		if return_stats:
			return total_loss, l_align, l_unif, margin_active_frac
		return total_loss, l_align, l_unif


def compute_loss(args):
	"""KGAU builds ``KGAULoss`` inside the strategy; loss pillar is optional."""
	del args
	return None
