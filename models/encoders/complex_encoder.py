"""ComplEx encoder module."""

from __future__ import annotations

import json
import os
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import Example, load_data
from data.dict_hub import get_entity_dict


def _relation_path_candidates(args) -> list[str]:
	paths = []
	for source_path in [getattr(args, 'train_path', ''), getattr(args, 'valid_path', ''), getattr(args, 'test_path', '')]:
		if not source_path:
			continue
		paths.append(os.path.join(os.path.dirname(source_path), 'relation2id.json'))
		paths.append(os.path.join(os.path.dirname(source_path), 'relations.json'))
		paths.append(os.path.join(os.path.dirname(source_path), 'relation2idx.json'))
	paths.append(os.path.join('data', getattr(args, 'dataset', ''), 'relation2id.json'))
	paths.append(os.path.join('data', getattr(args, 'dataset', ''), 'preprocessed', 'relation2id.json'))
	return paths


def _load_relation_to_idx(args) -> dict[str, int]:
	for path in _relation_path_candidates(args):
		if not path or not os.path.exists(path):
			continue
		with open(path, 'r', encoding='utf-8') as handle:
			mapping = json.load(handle)
		if isinstance(mapping, dict):
			return {str(key): int(value) for key, value in mapping.items()}

	relations = []
	seen = set()
	for example in load_data(getattr(args, 'train_path', ''), add_forward_triplet=False, add_backward_triplet=False):
		if example.relation not in seen:
			seen.add(example.relation)
			relations.append(example.relation)
	return {relation: idx for idx, relation in enumerate(relations)}


def _as_index_tensor(values, lookup, device: torch.device) -> torch.Tensor:
	if torch.is_tensor(values):
		return values.to(device=device, dtype=torch.long)
	return torch.tensor([lookup(value) for value in values], dtype=torch.long, device=device)


class ComplExEncoder(nn.Module):
	"""ComplEx encoder that exposes AU-friendly query and entity embeddings."""

	def __init__(self, n_ent: int, n_rel: int, args):
		super().__init__()
		sigma = 0.2
		self.dim = args.dim
		self.rel_re_embed = nn.Embedding(n_rel, args.dim)
		self.rel_im_embed = nn.Embedding(n_rel, args.dim)
		self.ent_re_embed = nn.Embedding(n_ent, args.dim)
		self.ent_im_embed = nn.Embedding(n_ent, args.dim)
		self.entity_dict = get_entity_dict()
		self.rel_to_idx = _load_relation_to_idx(args)
		scale = (args.dim / sigma ** 2) ** (1 / 6)
		for param in self.parameters():
			param.data.div_(scale)

	def forward(self, src, rel, dst) -> torch.Tensor:
		"""Return raw ComplEx scores for the provided triples."""

		return (
			torch.sum(self.rel_re_embed(rel) * self.ent_re_embed(src) * self.ent_re_embed(dst), dim=-1)
			+ torch.sum(self.rel_re_embed(rel) * self.ent_im_embed(src) * self.ent_im_embed(dst), dim=-1)
			+ torch.sum(self.rel_im_embed(rel) * self.ent_re_embed(src) * self.ent_im_embed(dst), dim=-1)
			- torch.sum(self.rel_im_embed(rel) * self.ent_im_embed(src) * self.ent_re_embed(dst), dim=-1)
		)

	def score(self, src, rel, dst) -> torch.Tensor:
		"""Alias for forward to compute scores for triples."""

		return self.forward(src, rel, dst)

	def dist(self, src, rel, dst) -> torch.Tensor:
		"""Return negative scores as distances for the provided triples."""

		return -self.forward(src, rel, dst)

	def prob_logit(self, src, rel, dst) -> torch.Tensor:
		"""Return probabilities as logits for the provided triples."""

		return self.forward(src, rel, dst)

	def get_queries_targets(self, src, rel, dst):
		"""Return query, target, and head embeddings for AU training."""

		h_re = self.ent_re_embed(src)
		h_im = self.ent_im_embed(src)
		r_re = self.rel_re_embed(rel)
		r_im = self.rel_im_embed(rel)
		t_re = self.ent_re_embed(dst)
		t_im = self.ent_im_embed(dst)
		q = torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)
		t = torch.cat([t_re, t_im], dim=-1)
		h = torch.cat([h_re, h_im], dim=-1)
		return q, t, h

	def entity_embeddings(self, device: torch.device | None = None) -> torch.Tensor:
		"""Return L2-normalized entity embeddings for retrieval."""

		entity_vectors = torch.cat([self.ent_re_embed.weight, self.ent_im_embed.weight], dim=-1)
		entity_vectors = F.normalize(entity_vectors, p=2, dim=-1)
		if device is not None:
			entity_vectors = entity_vectors.to(device)
		return entity_vectors

	def hr_embeddings(self, examples: Sequence[Example], device: torch.device | None = None) -> torch.Tensor:
		"""Return L2-normalized query embeddings for a list of examples."""

		if device is None:
			device = self.ent_re_embed.weight.device
		head_indices = _as_index_tensor([example.head_id for example in examples], self.entity_dict.entity_to_idx, device)
		relation_indices = _as_index_tensor([example.relation for example in examples], self.rel_to_idx.__getitem__, device)
		h_re = self.ent_re_embed(head_indices)
		h_im = self.ent_im_embed(head_indices)
		r_re = self.rel_re_embed(relation_indices)
		r_im = self.rel_im_embed(relation_indices)
		query_vectors = torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)
		return F.normalize(query_vectors, p=2, dim=-1)

	def predict_by_examples(self, examples: Sequence[Example], batch_size: int | None = None, num_workers: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return query and target embeddings for link prediction evaluation."""

		device = self.ent_re_embed.weight.device
		head_indices = _as_index_tensor([example.head_id for example in examples], self.entity_dict.entity_to_idx, device)
		relation_indices = _as_index_tensor([example.relation for example in examples], self.rel_to_idx.__getitem__, device)
		tail_indices = _as_index_tensor([example.tail_id for example in examples], self.entity_dict.entity_to_idx, device)
		h_re = self.ent_re_embed(head_indices)
		h_im = self.ent_im_embed(head_indices)
		r_re = self.rel_re_embed(relation_indices)
		r_im = self.rel_im_embed(relation_indices)
		query_vectors = F.normalize(torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1), p=2, dim=-1)
		tail_vectors = F.normalize(torch.cat([self.ent_re_embed(tail_indices), self.ent_im_embed(tail_indices)], dim=-1), p=2, dim=-1)
		return query_vectors, tail_vectors

	def predict_by_entities(self, entity_exs, batch_size: int | None = None, num_workers: int = 2) -> torch.Tensor:
		"""Return entity embeddings for a list of entity examples."""

		device = self.ent_re_embed.weight.device
		entity_ids = [getattr(entity_ex, 'entity_id', getattr(entity_ex, 'tail_id', '')) for entity_ex in entity_exs]
		entity_indices = _as_index_tensor(entity_ids, self.entity_dict.entity_to_idx, device)
		return self.entity_embeddings(device=device)[entity_indices]

	def score_batch(self, head_ids, relations, tail_entity_ids) -> torch.Tensor:
		"""Score a batch of heads/relations against a candidate tail set."""

		device = self.ent_re_embed.weight.device
		head_indices = _as_index_tensor(head_ids, self.entity_dict.entity_to_idx, device)
		relation_indices = _as_index_tensor(relations, self.rel_to_idx.__getitem__, device)
		candidate_indices = _as_index_tensor(tail_entity_ids, self.entity_dict.entity_to_idx, device)
		h_re = self.ent_re_embed(head_indices)
		h_im = self.ent_im_embed(head_indices)
		r_re = self.rel_re_embed(relation_indices)
		r_im = self.rel_im_embed(relation_indices)
		query_vectors = F.normalize(torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1), p=2, dim=-1)
		candidate_vectors = self.entity_embeddings(device=device)[candidate_indices]
		return torch.mm(query_vectors, candidate_vectors.t())

	def compute_logits(self, output_dict: dict | torch.Tensor, batch_dict: dict) -> dict:
		"""Convert a forward pass into logits for triple classification."""

		if torch.is_tensor(output_dict):
			if output_dict.dim() == 1:
				logits = torch.diag(output_dict)
			else:
				logits = output_dict
			return {'logits': logits}

		if isinstance(output_dict, dict):
			if 'logits' in output_dict:
				return output_dict
			query_vectors = output_dict.get('q')
			if query_vectors is None:
				query_vectors = output_dict.get('hr_vector')
			target_vectors = output_dict.get('t')
			if target_vectors is None:
				target_vectors = output_dict.get('tail_vector')
			if query_vectors is None or target_vectors is None:
				raise KeyError('Output dict must contain query and target vectors')
			query_vectors = F.normalize(query_vectors, p=2, dim=-1)
			target_vectors = F.normalize(target_vectors, p=2, dim=-1)
			return {'logits': torch.mm(query_vectors, target_vectors.t())}

		raise TypeError('Unsupported model output type for logits computation')


def build_model(args):
	"""Factory helper used by the evaluator to rebuild the model from checkpoints."""

	entity_dict = get_entity_dict()
	relation_to_idx = _load_relation_to_idx(args)
	model = ComplExEncoder(len(entity_dict), len(relation_to_idx), args)
	model.rel_to_idx = relation_to_idx
	return model