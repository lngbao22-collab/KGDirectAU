"""Alignment and uniformity loss for KGAU."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGAULoss(nn.Module):
	"""Alignment and uniformity loss for knowledge graph embeddings."""

	def __init__(self, gamma_q=1.0, gamma_t=1.0, gamma_h=0.0, gamma_ent=0.0, tuni=2.0):
		super().__init__()
		self.gamma_q = gamma_q
		self.gamma_t = gamma_t
		self.gamma_h = gamma_h
		self.gamma_ent = gamma_ent
		# `tuni` is the uniformity temperature/scaling factor
		self.tuni = tuni

	def alignment_loss(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		"""Expected squared L2 distance between paired positive query and target embeddings."""

		q = F.normalize(q, p=2, dim=-1)
		t = F.normalize(t, p=2, dim=-1)
		return (q - t).pow(2).sum(dim=-1).mean()

	def uniformity_loss(self, x: torch.Tensor) -> torch.Tensor:
		"""Gaussian potential based uniformity loss on the hypersphere."""

		if x is None:
			return torch.tensor(0.0)
		if x.size(0) < 2:
			return x.new_zeros(())
		x = F.normalize(x, p=2, dim=-1)
		pairwise = torch.pdist(x, p=2)
		if pairwise.numel() == 0:
			return x.new_zeros(())
		potential = torch.exp(-self.tuni * pairwise.pow(2))
		return potential.mean().log()

	def forward(
		self,
		q: torch.Tensor,
		t: torch.Tensor,
		h: torch.Tensor | None = None,
		ent: torch.Tensor | None = None
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Return the total AU loss together with alignment and uniformity terms."""

		# 1. Normalize Query and Target
		q_norm = F.normalize(q, p=2, dim=-1)
		t_norm = F.normalize(t, p=2, dim=-1)

		# 2. Calculate Alignment Loss
		l_align = self.alignment_loss(q_norm, t_norm)

		# 3. Initialize Uniformity Loss Tensor
		l_unif = q_norm.new_zeros(())

		# 4. Calculate Core Uniformity Losses
		if self.gamma_q > 0:
			l_unif = l_unif + self.gamma_q * self.uniformity_loss(q_norm)
		if self.gamma_t > 0:
			l_unif = l_unif + self.gamma_t * self.uniformity_loss(t_norm)

		# 5. Calculate Bonus Head Uniformity Loss
		if h is not None and self.gamma_h > 0:
			h_norm = F.normalize(h, p=2, dim=-1)
			l_unif = l_unif + self.gamma_h * self.uniformity_loss(h_norm)

		# 6. Calculate Bonus Entity Uniformity Loss
		if ent is not None and self.gamma_ent > 0:
			ent_norm = F.normalize(ent, p=2, dim=-1)
			l_unif = l_unif + self.gamma_ent * self.uniformity_loss(ent_norm)

		total_loss = l_align + l_unif
		return total_loss, l_align, l_unif