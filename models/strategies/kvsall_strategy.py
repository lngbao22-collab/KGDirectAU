"""KvsAll training: group triples by (h, r) and train with multi-hot BCE (LibKGE flow)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader

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
from utils.logger import logger


def build_kvsall_index(triples) -> list[dict[str, Any]]:
	"""Group triples into unique (h, r) queries with all true tail answers."""

	query_to_tails: dict[tuple[int, int], set[int]] = defaultdict(set)
	if torch.is_tensor(triples):
		rows = triples.detach().cpu().tolist()
	else:
		rows = triples

	for h, r, t in rows:
		query_to_tails[(int(h), int(r))].add(int(t))

	grouped_data = []
	for (h, r), tails in query_to_tails.items():
		grouped_data.append({
			'head_id': h,
			'relation': r,
			'tail_ids': sorted(tails),
		})
	return grouped_data


def _collate_kvsall_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
	head_ids = torch.tensor([item['head_id'] for item in batch], dtype=torch.long)
	relations = torch.tensor([item['relation'] for item in batch], dtype=torch.long)
	tail_ids = [item['tail_ids'] for item in batch]
	return {'head_id': head_ids, 'relation': relations, 'tail_ids': tail_ids}


class KvsAllStrategy:
	"""Train by grouping (h, r) queries and scoring all entities with multi-hot labels."""

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

		if grouped_train_data is None:
			if train_triples is None:
				raise ValueError('KvsAllStrategy requires train_triples or grouped_train_data from build_pipeline')
			grouped_train_data = build_kvsall_index(train_triples)

		self.grouped_train_data = grouped_train_data
		self.num_entities = _resolve_nentity(args, model)
		self.label_smoothing = config_float(args, 'label_smoothing', 0.0)
		if self.label_smoothing < 0.0:
			raise ValueError(f'label_smoothing must be >= 0, got {self.label_smoothing}')
		if self.label_smoothing > 0.0 and self.label_smoothing <= (1.0 / self.num_entities):
			self.label_smoothing = 1.0 / self.num_entities

		self.loss_fn = loss_fn if loss_fn is not None else load_loss_fn_for_paradigm(args, 'kvsall')
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
		logger.info(
			'KvsAll: %d unique (h, r) queries from training triples (num_entities=%d, label_smoothing=%.6f)',
			len(self.grouped_train_data),
			self.num_entities,
			self.label_smoothing,
		)

	def _build_labels(self, tail_ids_list: list[list[int]], batch_size: int) -> torch.Tensor:
		labels = torch.zeros((batch_size, self.num_entities), device=self.device)
		for row_idx, true_tails in enumerate(tail_ids_list):
			if true_tails:
				labels[row_idx, true_tails] = 1.0

		if self.label_smoothing > 0.0:
			labels = (1.0 - self.label_smoothing) * labels + (1.0 / self.num_entities)
		return labels

	def train_epoch(self, dataloader, epoch: int) -> float:
		del dataloader
		self.model.train()
		total_loss = 0.0
		num_queries = 0

		for batch in self.train_loader:
			head_ids = batch['head_id'].to(self.device)
			relations = batch['relation'].to(self.device)
			tail_ids_list = batch['tail_ids']
			batch_size = head_ids.size(0)

			self.optimizer.zero_grad()
			scores_sp = self.model.score_sp_(head_ids, relations)
			labels = self._build_labels(tail_ids_list, batch_size)
			loss = self.loss_fn(scores_sp, labels)
			loss = apply_kge_regularization(loss, self.model, self.args)
			loss.backward()
			self.optimizer.step()

			total_loss += float(loss.item()) * batch_size
			num_queries += batch_size

		avg_loss = total_loss / max(num_queries, 1)
		logger.info('[EPOCH %s] Train | Loss: %.4f', epoch + 1, avg_loss)
		return avg_loss

	def train_loop(self, dataloader=None) -> dict:
		return run_index_kge_train_loop(self, dataloader)


Strategy = KvsAllStrategy
