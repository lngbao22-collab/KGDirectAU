"""Alignment and uniformity loss for KGAU."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared with NegSamp / AllNeg chunk heuristics: cap peak pair-block memory.
_UNIFORM_PAIR_CHUNK_BYTES_BUDGET = 512 * 1024 * 1024


def resolve_uniform_pair_chunk_size(n: int, dim: int, explicit: int = 0) -> int:
	"""Pair-block width C for chunked uniformity.

	explicit > 0: use min(explicit, n).
	Else (0 = auto): choose C so a [C, C] float32 block stays near the shared
	~512MiB budget, with a soft cap of 256 (reference: B=512 entity term n=1024).
	"""

	if n <= 1:
		return max(n, 1)
	explicit = int(explicit or 0)
	if explicit > 0:
		return min(explicit, n)
	max_by_pair = int((_UNIFORM_PAIR_CHUNK_BYTES_BUDGET / 8.0) ** 0.5)
	max_by_dim = max(_UNIFORM_PAIR_CHUNK_BYTES_BUDGET // max(dim * 4 * 4, 1), 32)
	auto = max(32, min(max_by_pair, max_by_dim, n, 256))
	return int(auto)


def _normalized_sqdist_block(xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
	"""||a-b||^2 for L2-normalized rows via 2 - 2 a·b. Shape [Ci, Cj]."""

	return (2.0 - 2.0 * (xi @ xj.transpose(0, 1))).clamp_min(0)


def chunked_pairwise_uniformity(
	x: torch.Tensor,
	uniform_t: float | torch.Tensor = 4,
	pair_chunk_size: int = 0,
	*,
	already_normalized: bool = False,
) -> torch.Tensor:
	"""Exact Wang-Isola uniformity: log(mean_{i<j} exp(-t ||x_i-x_j||^2)).

	Same pairs as ``torch.pdist``, but accumulates over [C,C] blocks so peak
	memory stays O(C^2 + n·D) instead of pdist's backward spike ~O(n^2·D).
	"""

	if not already_normalized:
		x = F.normalize(x, dim=-1)
	n = x.size(0)
	if n < 2:
		return (x * 0).sum()

	chunk = resolve_uniform_pair_chunk_size(n, x.size(1), pair_chunk_size)
	sum_exp = x.new_zeros(())
	count = 0

	for i0 in range(0, n, chunk):
		i1 = min(i0 + chunk, n)
		xi = x[i0:i1]
		for j0 in range(i0, n, chunk):
			j1 = min(j0 + chunk, n)
			xj = x[j0:j1]
			sq = _normalized_sqdist_block(xi, xj)
			if i0 == j0:
				tri = torch.triu(
					torch.ones(i1 - i0, j1 - j0, device=x.device, dtype=torch.bool),
					diagonal=1,
				)
				vals = sq.masked_select(tri)
			else:
				vals = sq.reshape(-1)
			sum_exp = sum_exp + (-uniform_t * vals).exp().sum()
			count += vals.numel()

	return (sum_exp / max(count, 1)).log()


def chunked_pairwise_margin_uniformity(
	x: torch.Tensor,
	uniform_margin: float = 2.0,
	uniform_t: float | torch.Tensor = 4,
	pair_chunk_size: int = 0,
	*,
	already_normalized: bool = False,
) -> torch.Tensor:
	"""Exact soft-margin uniformity: log(mean_{i<j} exp(t * ReLU(m - ||x_i-x_j||^2)))."""

	if not already_normalized:
		x = F.normalize(x, dim=-1)
	n = x.size(0)
	if n < 2:
		return (x * 0).sum()

	chunk = resolve_uniform_pair_chunk_size(n, x.size(1), pair_chunk_size)
	sum_exp = x.new_zeros(())
	count = 0

	for i0 in range(0, n, chunk):
		i1 = min(i0 + chunk, n)
		xi = x[i0:i1]
		for j0 in range(i0, n, chunk):
			j1 = min(j0 + chunk, n)
			xj = x[j0:j1]
			sq = _normalized_sqdist_block(xi, xj)
			if i0 == j0:
				tri = torch.triu(
					torch.ones(i1 - i0, j1 - j0, device=x.device, dtype=torch.bool),
					diagonal=1,
				)
				vals = sq.masked_select(tri)
			else:
				vals = sq.reshape(-1)
			sum_exp = sum_exp + F.relu(uniform_margin - vals).mul(uniform_t).exp().sum()
			count += vals.numel()

	return (sum_exp / max(count, 1)).log()


def chunked_pairwise_sqdist(
	x: torch.Tensor,
	pair_chunk_size: int = 0,
	*,
	already_normalized: bool = False,
) -> torch.Tensor:
	"""Collect exact i<j squared distances via [C,C] blocks (no ``torch.pdist``)."""

	if not already_normalized:
		x = F.normalize(x, dim=-1)
	n = x.size(0)
	if n < 2:
		return x.new_empty(0)

	chunk = resolve_uniform_pair_chunk_size(n, x.size(1), pair_chunk_size)
	parts: list[torch.Tensor] = []
	for i0 in range(0, n, chunk):
		i1 = min(i0 + chunk, n)
		xi = x[i0:i1]
		for j0 in range(i0, n, chunk):
			j1 = min(j0 + chunk, n)
			xj = x[j0:j1]
			sq = _normalized_sqdist_block(xi, xj)
			if i0 == j0:
				tri = torch.triu(
					torch.ones(i1 - i0, j1 - j0, device=x.device, dtype=torch.bool),
					diagonal=1,
				)
				parts.append(sq.masked_select(tri))
			else:
				parts.append(sq.reshape(-1))
	return torch.cat(parts, dim=0) if parts else x.new_empty(0)


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
		theta: float = 1.0,
		alpha: float = 2.0,
		tuni=2.0,
		learnable_tuni: bool = False,
		learnable_au_theta: bool = False,
		learnable_au_gammas: bool = False,
		tuni_as_theta: bool = False,
		max_uniformity_samples: int = 1024,
		additive_margin: float = 0.0,
		alignment_mode: str = 'cosine',
		normalize_uniformity: bool = True,
		assume_unit_norm: bool = False,
		average_uniformity_terms: bool = False,
		uniformity_full_pdist: bool = False,
		uniformity_pdist_gb: float | None = None,
		uniform_pair_chunk_size: int = 0,
	):
		super().__init__()
		self.average_uniformity_terms = bool(average_uniformity_terms)
		self.uniformity_full_pdist = bool(uniformity_full_pdist)
		self.uniformity_pdist_gb = uniformity_pdist_gb
		# 0 = auto (~512MiB / soft-cap 256); >0 forces pair-block width. Always used
		# instead of torch.pdist when computing full i<j uniformity.
		self.uniform_pair_chunk_size = int(uniform_pair_chunk_size or 0)
		self.tuni_as_theta = bool(tuni_as_theta)
		self.learnable_au_theta = bool(learnable_au_theta)
		self.learnable_au_gammas = bool(learnable_au_gammas)
		# Alignment degree: E[sum_i |q_i - t_i|^alpha]; alpha=2 recovers squared L2.
		self.alpha = float(_coalesce_float(alpha, 2.0))
		theta_init = _coalesce_float(theta, 1.0)
		self.register_buffer('theta_init', torch.tensor(theta_init))
		if not self.learnable_au_theta:
			self._theta = theta_init
		else:
			# Bounded upward adjustment only: exp(adj) in [1, inf), init at 0.
			# Unconstrained log-scale theta falls under alignment loss minimization.
			self.log_theta_adj = nn.Parameter(torch.zeros(()))
		self.register_buffer('theta_schedule_mult', torch.tensor(1.0))
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
			if not self.learnable_au_gammas:
				setattr(self, f'_gamma_{name}', init)
			elif init > 0.0:
				# Bounded downward adjustment only: exp(adj) in (0, 1], init at 0.
				# Unconstrained log-scale gammas rise under AU loss because uniformity is negative.
				setattr(self, f'log_gamma_adj_{name}', nn.Parameter(torch.zeros(())))
		self.register_buffer('gamma_schedule_mult', torch.tensor(1.0))
		# `tuni` is the uniformity temperature; with ``tuni_as_theta`` it also replaces theta.
		tuni_val = _coalesce_float(tuni, 2.0)
		self.register_buffer('tuni_init_log', torch.tensor(math.log(tuni_val)))
		if learnable_tuni:
			self.log_tuni = nn.Parameter(torch.tensor(math.log(tuni_val)))
		else:
			self._tuni = tuni_val
		self.max_uniformity_samples = max_uniformity_samples
		# InfoNCE additive margin gamma; geometric threshold m = 2 * gamma on squared L2.
		self.additive_margin = _coalesce_float(additive_margin, 0.0)
		# `cosine`: L2-normalize paired vectors (DistMult/ComplEx/SimKGC).
		# `phase_residual`: element-wise |phase residual|^alpha without global normalization.
		# `sin_phase`: pRotatE link-pred term sum_i |sin(theta_q,i - theta_t,i)| (no global normalize).
		self.alignment_mode = alignment_mode or 'cosine'
		self.normalize_uniformity = normalize_uniformity
		# When True, q/t/h/ent inputs are already L2-normalized (``normalize_au_vectors`` in the model).
		self.assume_unit_norm = bool(assume_unit_norm)

	def _l2_normalize_if_needed(self, x: torch.Tensor) -> torch.Tensor:
		if self.assume_unit_norm:
			return x
		return F.normalize(x, p=2, dim=-1)

	def _gamma_init_value(self, name: str) -> float:
		if self.learnable_au_gammas:
			return float(getattr(self, f'gamma_init_{name}').detach().cpu().item())
		return float(getattr(self, f'_gamma_{name}'))

	def gamma_schedule_mult_value(self) -> float:
		return float(self.gamma_schedule_mult.detach().cpu().item())

	def theta_schedule_mult_value(self) -> float:
		return float(self.theta_schedule_mult.detach().cpu().item())

	def _learnable_gamma_factor(self, name: str) -> torch.Tensor:
		adj = getattr(self, f'log_gamma_adj_{name}')
		return torch.exp(torch.clamp(adj, max=0.0))

	def _learnable_gamma_enabled(self, name: str) -> bool:
		return hasattr(self, f'log_gamma_adj_{name}')

	def set_gamma_schedule_mult(self, mult: float) -> None:
		self.gamma_schedule_mult.fill_(float(mult))

	def set_theta_schedule_mult(self, mult: float) -> None:
		self.theta_schedule_mult.fill_(float(mult))

	def _learnable_theta_factor(self) -> torch.Tensor:
		return torch.exp(torch.clamp(self.log_theta_adj, min=0.0))

	def clamp_learnable_theta_adj(self) -> None:
		"""Keep learnable theta adjustments at or above 0 (effective factor in [1, inf))."""

		if not self.learnable_au_theta:
			return
		with torch.no_grad():
			self.log_theta_adj.clamp_(min=0.0)

	def clamp_learnable_gamma_adj(self) -> None:
		"""Keep learnable gamma adjustments at or below 0 (effective factor in (0, 1])."""

		if not self.learnable_au_gammas:
			return
		for name in ('q', 't', 'h', 'ent', 'cross'):
			if not hasattr(self, f'log_gamma_adj_{name}'):
				continue
			with torch.no_grad():
				getattr(self, f'log_gamma_adj_{name}').clamp_(max=0.0)

	def clamp_learnable_tuni(self) -> None:
		"""When ``tuni_as_theta``, keep learnable tuni at or above its initial value."""

		if not self.tuni_as_theta or not hasattr(self, 'log_tuni'):
			return
		with torch.no_grad():
			self.log_tuni.clamp_(min=float(self.tuni_init_log.item()))

	def _effective_gamma(self, name: str) -> torch.Tensor | float:
		mult = self.gamma_schedule_mult
		if self.learnable_au_gammas:
			init = getattr(self, f'gamma_init_{name}')
			if not self._learnable_gamma_enabled(name):
				return init * mult
			return init * mult * self._learnable_gamma_factor(name)
		return float(getattr(self, f'_gamma_{name}')) * float(mult)

	def _effective_theta(self) -> torch.Tensor | float:
		if self.tuni_as_theta:
			return self.tuni
		mult = self.theta_schedule_mult
		if self.learnable_au_theta:
			return self.theta_init * mult * self._learnable_theta_factor()
		return float(self._theta) * float(mult)

	def theta_value(self) -> float:
		"""Scalar effective alignment scale for logging and control flow."""

		value = self._effective_theta()
		if torch.is_tensor(value):
			return float(value.detach().cpu().item())
		return float(value)

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

	def _feature_alignment(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		"""Mean feature-wise |q - t|^alpha over positive pairs (alpha=2 => squared L2)."""

		# abs() keeps fractional alpha well-defined for negative residuals.
		return (q - t).abs().pow(self.alpha).sum(dim=-1).mean()

	def alignment_loss(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		"""Expected |q - t|^alpha feature distance between paired positive embeddings."""

		q = self._l2_normalize_if_needed(q)
		t = self._l2_normalize_if_needed(t)
		return self._feature_alignment(q, t)

	def sin_phase_alignment_loss(self, phase_query: torch.Tensor, phase_target: torch.Tensor) -> torch.Tensor:
		"""Alignment in native pRotatE geometry: mean sum of |sin(phase residual)| per dimension.

		Matches the penalty inside ``pRotatEEncoder._rotate_score`` (before margin/modulus):
		minimizing this term raises positive link-prediction scores without cosine normalization.
		"""

		residual = phase_query - phase_target
		return torch.abs(torch.sin(residual)).sum(dim=-1).mean()

	def _uniformity_row_cap(self, dim: int) -> int | None:
		"""Optional explicit row cap via ``max_uniformity_samples`` (full exact mode skips GB budgets)."""

		del dim
		max_samples = int(getattr(self, 'max_uniformity_samples', 0) or 0)
		if max_samples > 0:
			return max_samples
		# Chunked i<j uniformity replaces ``torch.pdist``; no pdist-backward row budget.
		return None

	def _subsample_uniformity_rows(self, x: torch.Tensor) -> torch.Tensor | None:
		"""Optionally cap row count before uniformity (entity table or large batches)."""

		if x is None or x.size(0) < 2:
			return None
		if self.uniformity_full_pdist:
			# Exact full-batch path: only honor an explicit max_uniformity_samples > 0.
			row_cap = self._uniformity_row_cap(x.size(-1))
			if row_cap is not None and x.size(0) > row_cap:
				indices = torch.randperm(x.size(0), device=x.device)[:row_cap]
				x = x.index_select(0, indices)
		else:
			max_samples = int(getattr(self, 'max_uniformity_samples', 0) or 0)
			if max_samples > 0 and x.size(0) > max_samples:
				indices = torch.randperm(x.size(0), device=x.device)[:max_samples]
				x = x.index_select(0, indices)
		return x if x.size(0) >= 2 else None

	def _max_uniformity_pair_count(self, num_rows: int, dim: int) -> int:
		"""Choose how many pairwise distances to estimate (Monte Carlo path only)."""

		full_pairs = num_rows * (num_rows - 1) // 2
		if full_pairs <= 0:
			return 0
		if self.uniformity_full_pdist:
			return full_pairs
		max_samples = int(getattr(self, 'max_uniformity_samples', 0) or 0)
		# Prefer exact chunked i<j when the pair matrix is small enough that a full
		# reduction is cheap; otherwise Monte Carlo with a bounded pair count.
		pair_budget = 32 * 1024 * 1024
		if num_rows * num_rows * 4 <= pair_budget:
			return full_pairs
		pair_cap = int(getattr(self, 'max_uniformity_pairs', 0) or 0)
		if pair_cap <= 0:
			pair_cap = max(4096, max_samples * 8)
		del dim
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

	def _normalize_uniformity_rows(self, x: torch.Tensor) -> torch.Tensor:
		if self.normalize_uniformity:
			return self._l2_normalize_if_needed(x)
		return x

	def _prepare_uniformity_pairs(self, x: torch.Tensor) -> torch.Tensor | None:
		"""Normalize and subsample embeddings; return squared pairwise L2 distances.

		Full i<j distances use chunked blocks (never ``torch.pdist``).
		"""

		x = self._subsample_uniformity_rows(x)
		if x is None:
			return None
		x = self._normalize_uniformity_rows(x)
		num_pairs = self._max_uniformity_pair_count(x.size(0), x.size(-1))
		if num_pairs <= 0:
			return None
		full_pairs = x.size(0) * (x.size(0) - 1) // 2
		if num_pairs >= full_pairs:
			# Unit-sphere identity ``||a-b||^2 = 2-2a·b`` requires normalized rows.
			if self.normalize_uniformity or self.assume_unit_norm:
				dists = chunked_pairwise_sqdist(
					x,
					pair_chunk_size=self.uniform_pair_chunk_size,
					already_normalized=True,
				)
			else:
				# Non-normalized vectors: fall back to explicit row differences in chunks.
				dists = self._chunked_raw_pairwise_sqdist(x)
			return dists if dists.numel() > 0 else None
		return self._random_pairwise_dist_sq(x, num_pairs)

	def _chunked_raw_pairwise_sqdist(self, x: torch.Tensor) -> torch.Tensor:
		"""Exact i<j ||a-b||^2 for non-normalized rows via [C,C] blocks."""

		n = x.size(0)
		if n < 2:
			return x.new_empty(0)
		chunk = resolve_uniform_pair_chunk_size(n, x.size(1), self.uniform_pair_chunk_size)
		parts: list[torch.Tensor] = []
		for i0 in range(0, n, chunk):
			i1 = min(i0 + chunk, n)
			xi = x[i0:i1]
			for j0 in range(i0, n, chunk):
				j1 = min(j0 + chunk, n)
				xj = x[j0:j1]
				# (xi - xj)^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj
				sq_i = xi.pow(2).sum(dim=-1, keepdim=True)
				sq_j = xj.pow(2).sum(dim=-1, keepdim=True).transpose(0, 1)
				sq = (sq_i + sq_j - 2.0 * (xi @ xj.transpose(0, 1))).clamp_min(0)
				if i0 == j0:
					tri = torch.triu(
						torch.ones(i1 - i0, j1 - j0, device=x.device, dtype=torch.bool),
						diagonal=1,
					)
					parts.append(sq.masked_select(tri))
				else:
					parts.append(sq.reshape(-1))
		return torch.cat(parts, dim=0) if parts else x.new_empty(0)

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

		margin = float(self.additive_margin)
		uniformity_temp = self.tuni

		# Classic AU + full i<j: exact chunked reduction (same formula as pdist, lower peak).
		if margin <= 0.0 and (
			self.uniformity_full_pdist
			or self._uses_exact_chunked_uniformity(x)
		):
			x_prep = self._subsample_uniformity_rows(x)
			if x_prep is None:
				return x.new_zeros(()), 0.0
			normalized = self.normalize_uniformity or self.assume_unit_norm
			x_prep = self._normalize_uniformity_rows(x_prep)
			if normalized:
				return chunked_pairwise_uniformity(
					x_prep,
					uniform_t=uniformity_temp,
					pair_chunk_size=self.uniform_pair_chunk_size,
					already_normalized=True,
				), 1.0
			# Non-normalized: materialize chunked distances then log-mean potential.
			dist_sq = self._chunked_raw_pairwise_sqdist(x_prep)
			if dist_sq.numel() == 0:
				return x.new_zeros(()), 0.0
			scaled_dist_sq = self._scaled_uniformity_dist_sq(dist_sq)
			return self._log_mean_potential(uniformity_temp * scaled_dist_sq), 1.0

		dist_sq = self._prepare_uniformity_pairs(x)
		if dist_sq is None:
			return x.new_zeros(()), 0.0
		scaled_dist_sq = self._scaled_uniformity_dist_sq(dist_sq)
		if margin <= 0.0:
			return self._log_mean_potential(uniformity_temp * scaled_dist_sq), 1.0
		# Fixed m = 2*gamma is far too small in high dimensions (random pairs have d^2 ~ 2).
		# Use an adaptive buffer from batch geometry: repel the closest fraction of pairs.
		target_frac = self._margin_uniformity_fraction(margin)
		geom_margin = torch.quantile(scaled_dist_sq, target_frac)
		buffer_penalty = torch.exp(uniformity_temp * F.relu(geom_margin - scaled_dist_sq))
		# Keep classic AU spread so early epochs still have strong uniformity signal.
		spread = self._log_mean_potential(uniformity_temp * scaled_dist_sq)
		buffer = buffer_penalty.mean().clamp_min(1e-12).log()
		active_frac = float((scaled_dist_sq < geom_margin).float().mean().item())
		return spread + buffer, active_frac

	def _uses_exact_chunked_uniformity(self, x: torch.Tensor) -> bool:
		"""True when Monte Carlo would still request the full i<j pair set."""

		x_probe = x if x.size(0) < 2 else x
		num_rows = x_probe.size(0)
		full_pairs = num_rows * (num_rows - 1) // 2
		if full_pairs <= 0:
			return False
		return self._max_uniformity_pair_count(num_rows, x_probe.size(-1)) >= full_pairs

	def _block_alignment_loss(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		"""|q - t|^alpha alignment on per-block L2-normalized vectors (eval-consistent).

		Unlike ``alignment_loss``, this always normalizes each block even when
		``assume_unit_norm`` is set, because a global AU normalization does not make
		individual ``dabr_blocks`` sub-vectors unit norm.
		"""

		q = F.normalize(q, p=2, dim=-1)
		t = F.normalize(t, p=2, dim=-1)
		return self._feature_alignment(q, t)

	def _raw_alignment_loss(
		self,
		q: torch.Tensor,
		t: torch.Tensor,
		external_align: torch.Tensor | None = None,
	) -> torch.Tensor:
		if external_align is not None:
			return external_align
		if self.alignment_mode == 'sin_phase':
			return self.sin_phase_alignment_loss(q, t)
		if self.alignment_mode == 'phase_residual':
			return self._feature_alignment(q, t)
		if self.alignment_mode == 'dabr_blocks':
			mid = q.size(-1) // 2
			if mid <= 0 or mid * 2 != q.size(-1):
				return self.alignment_loss(q, t)
			# Each block must be L2-normalized independently so training matches the
			# per-block cosine used at eval (``_normalized_block_pair_score``). A single
			# global ``normalize_au_vectors`` leaves the blocks with unequal norms, which
			# over-weights the additive branch and diverges from the scored geometry.
			return (
				self._block_alignment_loss(q[..., :mid], t[..., :mid])
				+ self._block_alignment_loss(q[..., mid:], t[..., mid:])
			)
		return self.alignment_loss(q, t)

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

		l_align = self._effective_theta() * self._raw_alignment_loss(q, t, external_align=external_align)

		l_unif = q.new_zeros(())
		uniform_count = 0
		active_sum = 0.0
		active_weight = 0.0

		gamma_q = self._effective_gamma('q')
		if self.gamma_active('q'):
			q_uniformity = q_uni if q_uni is not None else q
			term, frac = self.uniformity_loss_with_stats(q_uniformity)
			l_unif = l_unif + gamma_q * term
			uniform_count += 1
			active_sum += self.gamma_value('q') * frac
			active_weight += self.gamma_value('q')
		gamma_t = self._effective_gamma('t')
		if self.gamma_active('t'):
			t_uniformity = t_uni if t_uni is not None else t
			term, frac = self.uniformity_loss_with_stats(t_uniformity)
			l_unif = l_unif + gamma_t * term
			uniform_count += 1
			active_sum += self.gamma_value('t') * frac
			active_weight += self.gamma_value('t')
		gamma_h = self._effective_gamma('h')
		if h is not None and self.gamma_active('h'):
			h_uniformity = h_uni if h_uni is not None else h
			term, _ = self.uniformity_loss_with_stats(h_uniformity)
			l_unif = l_unif + gamma_h * term
			uniform_count += 1
		gamma_ent = self._effective_gamma('ent')
		if ent is not None and self.gamma_active('ent'):
			ent_rows = self._subsample_uniformity_rows(ent)
			if ent_rows is not None:
				term, _ = self.uniformity_loss_with_stats(ent_rows)
				l_unif = l_unif + gamma_ent * term
				uniform_count += 1
		gamma_cross = self._effective_gamma('cross')
		if cross_uni is not None and self.gamma_active('cross'):
			term, _ = self.uniformity_loss_with_stats(cross_uni)
			l_unif = l_unif + gamma_cross * term
			uniform_count += 1

		if self.average_uniformity_terms and uniform_count > 0:
			l_unif = l_unif / uniform_count

		if float(self.additive_margin) > 0.0 and active_weight > 0:
			margin_active_frac = active_sum / active_weight
		else:
			margin_active_frac = 0.0

		total_loss = l_align + l_unif
		if return_stats:
			return total_loss, l_align, l_unif, margin_active_frac
		return total_loss, l_align, l_unif


def compute_loss(args):
	"""KGAU builds ``KGAULoss`` inside the strategy; loss pillar is optional."""
	del args
	return None
