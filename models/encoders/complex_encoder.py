"""ComplEx encoder module."""

import torch
import torch.nn as nn

from data.dict_hub import get_entity_dict, get_relation_id_map


def build_model(args) -> nn.Module:
	"""Factory function to build a ComplEx encoder."""

	entity_dict = get_entity_dict()
	relation_id_map = get_relation_id_map()
	return ComplExEncoder(n_ent=len(entity_dict), n_rel=max(len(relation_id_map), 1), args=args)


class ComplExEncoder(nn.Module):
	"""ComplEx encoder that returns raw compatibility scores for triples."""

	def __init__(self, n_ent: int, n_rel: int, args):
		super().__init__()
		sigma = 0.2
		self.dim = args.dim
		self.rel_re_embed = nn.Embedding(n_rel, args.dim)
		self.rel_im_embed = nn.Embedding(n_rel, args.dim)
		self.ent_re_embed = nn.Embedding(n_ent, args.dim)
		self.ent_im_embed = nn.Embedding(n_ent, args.dim)
		scale = (args.dim / sigma ** 2) ** (1 / 6)
		for param in self.parameters():
			param.data.div_(scale)

	def forward(self, src, rel, dst):
		"""Return raw ComplEx scores for the provided triples."""

		return (
			torch.sum(self.rel_re_embed(rel) * self.ent_re_embed(src) * self.ent_re_embed(dst), dim=-1)
			+ torch.sum(self.rel_re_embed(rel) * self.ent_im_embed(src) * self.ent_im_embed(dst), dim=-1)
			+ torch.sum(self.rel_im_embed(rel) * self.ent_re_embed(src) * self.ent_im_embed(dst), dim=-1)
			- torch.sum(self.rel_im_embed(rel) * self.ent_im_embed(src) * self.ent_re_embed(dst), dim=-1)
		)

	def score(self, src, rel, dst):
		return -self.forward(src, rel, dst)

	def dist(self, src, rel, dst):
		return -self.forward(src, rel, dst)

	def prob_logit(self, src, rel, dst):
		return self.forward(src, rel, dst)

	def _relation_to_index(self, relation):
		relation_id_map = get_relation_id_map()
		if isinstance(relation, int):
			return relation
		if relation in relation_id_map:
			return relation_id_map[relation]
		if isinstance(relation, str) and relation.startswith('inverse '):
			base_relation = relation[len('inverse '):]
			if base_relation in relation_id_map:
				return relation_id_map[base_relation]
		raise KeyError(relation)

	def score_batch(self, head_ids, relation_ids, tail_ids):
		"""Score a batch of candidate tails for the provided head-relation pairs."""

		entity_dict = get_entity_dict()
		head_idx = torch.tensor([entity_dict.entity_to_idx(h) if not isinstance(h, int) else h for h in head_ids], device=self.rel_re_embed.weight.device)
		rel_idx = torch.tensor([self._relation_to_index(r) for r in relation_ids], device=self.rel_re_embed.weight.device)
		if len(tail_ids) == 0:
			return torch.empty((len(head_ids), 0), device=self.rel_re_embed.weight.device)
		if isinstance(tail_ids[0], str):
			tail_idx = torch.tensor([entity_dict.entity_to_idx(t) for t in tail_ids], device=self.rel_re_embed.weight.device)
		else:
			tail_idx = torch.tensor(tail_ids, device=self.rel_re_embed.weight.device)

		head_re = self.ent_re_embed(head_idx)
		head_im = self.ent_im_embed(head_idx)
		rel_re = self.rel_re_embed(rel_idx)
		rel_im = self.rel_im_embed(rel_idx)
		tail_re = self.ent_re_embed(tail_idx)
		tail_im = self.ent_im_embed(tail_idx)

		return (
			torch.einsum('bd,bd,cd->bc', rel_re, head_re, tail_re)
			+ torch.einsum('bd,bd,cd->bc', rel_re, head_im, tail_im)
			+ torch.einsum('bd,bd,cd->bc', rel_im, head_re, tail_im)
			- torch.einsum('bd,bd,cd->bc', rel_im, head_im, tail_re)
		)