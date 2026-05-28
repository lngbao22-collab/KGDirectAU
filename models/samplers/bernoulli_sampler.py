"""Bernoulli listwise sampler for DistMult and ComplEx training."""

from collections import defaultdict

import numpy as np
import torch
from numpy.random import choice


def get_bern_prob(data, n_ent, n_rel):
	"""Compute relation-specific Bernoulli corruption probabilities."""

	src, rel, dst = data
	edges = defaultdict(lambda: defaultdict(set))
	rev_edges = defaultdict(lambda: defaultdict(set))
	for s, r, t in zip(src, rel, dst):
		edges[int(r)][int(s)].add(int(t))
		rev_edges[int(r)][int(t)].add(int(s))

	bern_prob = torch.zeros(n_rel)
	for r in edges.keys():
		tph = sum(len(tails) for tails in edges[r].values()) / max(len(edges[r]), 1)
		htp = sum(len(heads) for heads in rev_edges[r].values()) / max(len(rev_edges[r]), 1)
		bern_prob[r] = tph / (tph + htp) if (tph + htp) > 0 else 0.5
	return bern_prob


class BernoulliListwiseSampler(object):
	"""Generate listwise candidate triples with the true triple in column 0."""

	def __init__(self, data, n_ent, n_rel, n_sample):
		self.bern_prob = get_bern_prob(data, n_ent, n_rel)
		self.n_ent = n_ent
		self.n_sample = n_sample

	def corrupt(self, src, rel, dst, keep_truth=True):
		n = len(src)
		prob = self.bern_prob[rel]
		selection = torch.bernoulli(prob).cpu().numpy().astype('bool')
		src_np = src.cpu().numpy()
		dst_np = dst.cpu().numpy()

		src_out = np.tile(src_np, (self.n_sample, 1)).transpose()
		dst_out = np.tile(dst_np, (self.n_sample, 1)).transpose()
		rel_out = rel.unsqueeze(1).expand(n, self.n_sample)

		if keep_truth:
			ent_random = choice(self.n_ent, (n, self.n_sample - 1))
			src_out[selection, 1:] = ent_random[selection]
			dst_out[~selection, 1:] = ent_random[~selection]
		else:
			ent_random = choice(self.n_ent, (n, self.n_sample))
			src_out[selection, :] = ent_random[selection]
			dst_out[~selection, :] = ent_random[~selection]

		return torch.from_numpy(src_out).long(), rel_out.long(), torch.from_numpy(dst_out).long()