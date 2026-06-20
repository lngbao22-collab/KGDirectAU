"""KvsAll training: LibKGE-style sp_ and _po queries with KL/BCE multi-hot loss."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader

from base.embeddings import load_relation_to_idx, use_reciprocal_relations
from models.builder import (
	_resolve_nentity,
	apply_kge_regularization,
	build_lr_scheduler,
	build_optimizer,
	config_float,
	init_index_kge_trainer,
	load_loss_fn_for_paradigm,
	run_index_kge_train_loop,
)
from utils.device import get_model_obj
from utils.logger import logger


def resolve_kvsall_query_types(args) -> list[str]:
	"""Return enabled KvsAll query types (LibKGE default: sp_ and _po)."""

	raw = getattr(args, 'kvsall_query_types', None)
	if raw is None:
		return ['sp_', '_po']
	if isinstance(raw, str):
		return [part.strip() for part in raw.split(',') if part.strip()]
	return [str(part) for part in raw]


def kvsall_uses_po_training(args) -> bool:
	return '_po' in resolve_kvsall_query_types(args)


def _iter_triple_rows(triples) -> list[tuple[int, int, int]]:
	if torch.is_tensor(triples):
		return [tuple(int(v) for v in row) for row in triples.detach().cpu().tolist()]
	return [(int(h), int(r), int(t)) for h, r, t in triples]


def build_kvsall_sp_index(triples) -> list[dict[str, Any]]:
	"""Group triples into unique (h, r) queries with all true tail answers."""

	query_to_tails: dict[tuple[int, int], set[int]] = defaultdict(set)
	for h, r, t in _iter_triple_rows(triples):
		query_to_tails[(h, r)].add(t)

	return [
		{
			'query_type': 'sp_',
			'head_id': h,
			'relation': r,
			'target_ids': sorted(tails),
		}
		for (h, r), tails in query_to_tails.items()
	]


def build_kvsall_po_index(triples) -> list[dict[str, Any]]:
	"""Group triples into unique (r, t) queries with all true head answers."""

	query_to_heads: dict[tuple[int, int], set[int]] = defaultdict(set)
	for h, r, t in _iter_triple_rows(triples):
		query_to_heads[(r, t)].add(h)

	return [
		{
			'query_type': '_po',
			'relation': r,
			'tail_id': t,
			'target_ids': sorted(heads),
		}
		for (r, t), heads in query_to_heads.items()
	]


def build_kvsall_index(triples, query_types: list[str] | None = None) -> list[dict[str, Any]]:
	"""Build LibKGE-style KvsAll examples for the requested query types."""

	if query_types is None:
		query_types = ['sp_']
	grouped_data: list[dict[str, Any]] = []
	if 'sp_' in query_types:
		grouped_data.extend(build_kvsall_sp_index(triples))
	if '_po' in query_types:
		grouped_data.extend(build_kvsall_po_index(triples))
	return grouped_data


def _build_forward_to_inverse_map(args, model) -> torch.Tensor | None:
	if not use_reciprocal_relations(args):
		return None
	model_obj = get_model_obj(model)
	rel_to_idx = getattr(model_obj, 'rel_to_idx', None) or load_relation_to_idx(args)
	if not rel_to_idx:
		return None
	max_idx = max(rel_to_idx.values())
	mapping = torch.full((max_idx + 1,), -1, dtype=torch.long)
	for relation, fwd_idx in rel_to_idx.items():
		if relation.startswith('inverse '):
			continue
		inv_idx = rel_to_idx.get(f'inverse {relation}')
		if inv_idx is not None:
			mapping[int(fwd_idx)] = int(inv_idx)
	if int(mapping.ge(0).sum()) == 0:
		return None
	return mapping


def _collate_kvsall_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
	query_type_indexes = torch.tensor(
		[0 if item['query_type'] == 'sp_' else 1 for item in batch],
		dtype=torch.long,
	)
	sp_items = [item for item in batch if item['query_type'] == 'sp_']
	po_items = [item for item in batch if item['query_type'] == '_po']

	collated: dict[str, Any] = {
		'batch_size': len(batch),
		'query_type_indexes': query_type_indexes,
	}

	if sp_items:
		collated['sp_head_id'] = torch.tensor([item['head_id'] for item in sp_items], dtype=torch.long)
		collated['sp_relation'] = torch.tensor([item['relation'] for item in sp_items], dtype=torch.long)
		collated['sp_target_ids'] = [item['target_ids'] for item in sp_items]
	else:
		collated['sp_head_id'] = torch.empty(0, dtype=torch.long)
		collated['sp_relation'] = torch.empty(0, dtype=torch.long)
		collated['sp_target_ids'] = []

	if po_items:
		collated['po_relation'] = torch.tensor([item['relation'] for item in po_items], dtype=torch.long)
		collated['po_tail_id'] = torch.tensor([item['tail_id'] for item in po_items], dtype=torch.long)
		collated['po_target_ids'] = [item['target_ids'] for item in po_items]
	else:
		collated['po_relation'] = torch.empty(0, dtype=torch.long)
		collated['po_tail_id'] = torch.empty(0, dtype=torch.long)
		collated['po_target_ids'] = []

	return collated


class KvsAllStrategy:
	"""Train with LibKGE KvsAll queries (sp_ and/or _po) and multi-hot labels."""

	def __init__(
		self,
		model,
		sampler,
		loss_fn,
		args,
		train_triples: torch.Tensor | None = None,
		grouped_train_data: list[dict[str, Any]] | None = None,
	):
		del sampler
		init_index_kge_trainer(self, model, args)

		self.query_types = resolve_kvsall_query_types(args)
		if grouped_train_data is None:
			if train_triples is None:
				raise ValueError('KvsAllStrategy requires train_triples or grouped_train_data from build_pipeline')
			grouped_train_data = build_kvsall_index(train_triples, self.query_types)

		self.grouped_train_data = grouped_train_data
		self.num_entities = _resolve_nentity(args, model)
		self.label_smoothing = config_float(args, 'label_smoothing', 0.0)
		if self.label_smoothing < 0.0:
			raise ValueError(f'label_smoothing must be >= 0, got {self.label_smoothing}')
		if self.label_smoothing > 0.0 and self.label_smoothing <= (1.0 / self.num_entities):
			self.label_smoothing = 1.0 / self.num_entities

		self.loss_fn = loss_fn if loss_fn is not None else load_loss_fn_for_paradigm(args, 'kvsall')
		self._forward_to_inverse_rel = _build_forward_to_inverse_map(args, model)
		weight_decay = config_float(args, 'weight_decay', 0.0)
		self.optimizer = build_optimizer(args, self.model.parameters(), weight_decay)
		self.lr_scheduler = build_lr_scheduler(args, self.optimizer)

		batch_size = max(int(getattr(args, 'batch_size', 1)), 1)
		self.train_loader = DataLoader(
			self.grouped_train_data,
			batch_size=batch_size,
			shuffle=True,
			collate_fn=_collate_kvsall_batch,
			num_workers=getattr(args, 'workers', 0),
			pin_memory=torch.cuda.is_available(),
			drop_last=False,
		)
		sp_count = sum(1 for item in grouped_train_data if item['query_type'] == 'sp_')
		po_count = sum(1 for item in grouped_train_data if item['query_type'] == '_po')
		logger.info(
			'KvsAll: %d examples (sp_=%d, _po=%d) from training triples '
			'(num_entities=%d, label_smoothing=%.6f, query_types=%s)',
			len(self.grouped_train_data),
			sp_count,
			po_count,
			self.num_entities,
			self.label_smoothing,
			self.query_types,
		)

	def _build_labels(self, target_ids_list: list[list[int]], batch_size: int) -> torch.Tensor:
		labels = torch.zeros((batch_size, self.num_entities), device=self.device)
		for row_idx, true_ids in enumerate(target_ids_list):
			if true_ids:
				labels[row_idx, true_ids] = 1.0

		if self.label_smoothing > 0.0:
			labels = (1.0 - self.label_smoothing) * labels + (1.0 / self.num_entities)
		return labels

	def _po_relation_indices(self, relation_indices: torch.Tensor) -> torch.Tensor:
		if self._forward_to_inverse_rel is None:
			return relation_indices
		mapping = self._forward_to_inverse_rel.to(self.device)
		return mapping[relation_indices.to(self.device)]

	def _regularization_triples(self, batch: dict[str, Any]) -> torch.Tensor | None:
		rows: list[list[int]] = []

		sp_head_ids = batch['sp_head_id']
		sp_relations = batch['sp_relation']
		for row_idx, target_ids in enumerate(batch['sp_target_ids']):
			head_id = int(sp_head_ids[row_idx])
			relation = int(sp_relations[row_idx])
			for tail_id in target_ids:
				rows.append([head_id, relation, int(tail_id)])

		po_relations = batch['po_relation']
		po_tails = batch['po_tail_id']
		if po_relations.numel() > 0:
			mapped_relations = self._po_relation_indices(po_relations.to(self.device)).cpu()
			for row_idx, target_ids in enumerate(batch['po_target_ids']):
				relation = int(mapped_relations[row_idx])
				tail_id = int(po_tails[row_idx])
				for head_id in target_ids:
					rows.append([int(head_id), relation, tail_id])

		if not rows:
			return None
		return torch.tensor(rows, dtype=torch.long, device=self.device)

	def iter_training_batches(self, epoch: int, dataloader=None):
		del epoch, dataloader
		for batch in self.train_loader:
			yield batch

	def train_batch(self, batch, epoch: int) -> float:
		del epoch
		self.model.train()
		batch_size = int(batch['batch_size'])
		self.optimizer.zero_grad()
		loss = torch.zeros((), device=self.device)

		sp_head_ids = batch['sp_head_id']
		if sp_head_ids.numel() > 0:
			sp_head_ids = sp_head_ids.to(self.device)
			sp_relations = batch['sp_relation'].to(self.device)
			scores_sp = self.model.score_sp_(sp_head_ids, sp_relations)
			labels_sp = self._build_labels(batch['sp_target_ids'], sp_head_ids.size(0))
			loss = loss + self.loss_fn(scores_sp, labels_sp) / batch_size

		po_relations = batch['po_relation']
		if po_relations.numel() > 0:
			po_relations = po_relations.to(self.device)
			po_tails = batch['po_tail_id'].to(self.device)
			po_relations_for_score = self._po_relation_indices(po_relations)
			scores_po = self.model.score_po_(po_relations_for_score, po_tails)
			labels_po = self._build_labels(batch['po_target_ids'], po_relations.size(0))
			loss = loss + self.loss_fn(scores_po, labels_po) / batch_size

		loss = apply_kge_regularization(
			loss,
			self.model,
			self.args,
			batch_triples=self._regularization_triples(batch),
		)
		loss.backward()
		self.optimizer.step()
		return float(loss.item())

	def train_epoch(self, dataloader, epoch: int) -> float:
		del dataloader
		self.model.train()
		total_loss = 0.0
		num_queries = 0

		for batch in self.iter_training_batches(epoch):
			batch_size = int(batch['batch_size'])
			loss = self.train_batch(batch, epoch)
			total_loss += loss * batch_size
			num_queries += batch_size

		avg_loss = total_loss / max(num_queries, 1)
		logger.info('[EPOCH %s] Train | Loss: %.4f', epoch + 1, avg_loss)
		return avg_loss

	def train_loop(self, dataloader=None) -> dict:
		return run_index_kge_train_loop(self, dataloader)


Strategy = KvsAllStrategy
