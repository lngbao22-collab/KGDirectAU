"""DistMult encoder module."""

import torch
import torch.nn as nn


class DistMultEncoder(nn.Module):
	"""DistMult encoder that returns raw compatibility scores for triples."""

	def __init__(self, n_ent: int, n_rel: int, args):
		super().__init__()
		sigma = 0.2
		self.dim = args.dim
		self.rel_embed = nn.Embedding(n_rel, args.dim)
		self.ent_embed = nn.Embedding(n_ent, args.dim)
		scale = (args.dim / sigma ** 2) ** (1 / 6)
		for param in self.parameters():
			param.data.div_(scale)

	def forward(self, src, rel, dst):
		"""Return raw DistMult scores for the provided triples."""

		return torch.sum(self.ent_embed(src) * self.rel_embed(rel) * self.ent_embed(dst), dim=-1)

	def score(self, src, rel, dst):
		return self.forward(src, rel, dst)

	def dist(self, src, rel, dst):
		return -self.forward(src, rel, dst)

	def prob_logit(self, src, rel, dst):
		return self.forward(src, rel, dst)