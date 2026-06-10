"""1-vs-all broadcasting training paradigm (LibKGE ``TrainingJob1vsAll`` flow)."""

from __future__ import annotations

import math

import torch

from models.builder import (
	build_optimizer,
	init_index_kge_trainer,
	load_loss_fn_for_paradigm,
	run_index_kge_train_loop,
)
from utils.device import get_model_obj
from utils.logger import logger


class OneVsAllStrategy:
	"""Train by broadcasting (h, r) and (r, t) against all entities with cross-entropy."""

	def __init__(self, model, sampler, loss_fn, args, train_data=None):
		del sampler
		init_index_kge_trainer(self, model, args)
		if train_data is None:
			raise ValueError('OneVsAllStrategy requires train_data from build_pipeline')
		self.train_src, self.train_rel, self.train_dst = train_data
		self.loss_fn = loss_fn if loss_fn is not None else load_loss_fn_for_paradigm(args, '1vsall')

		if torch.cuda.is_available():
			self.train_src = self.train_src.to(self.device)
			self.train_rel = self.train_rel.to(self.device)
			self.train_dst = self.train_dst.to(self.device)

		batch_size = max(getattr(args, 'batch_size', 1), 1)
		num_batches = max(math.ceil(self.train_src.size(0) / batch_size), 1)
		weight_decay = float(getattr(args, 'weight_decay', None) or 0.0)
		self.weight_decay = weight_decay
		self.optimizer = build_optimizer(args, self.model.parameters(), self.weight_decay)
		self.bidirectional = bool(getattr(args, 'bidirectional_1vsall', True))

	def _iter_batches(self, batch_size: int):
		src, rel, dst = self.train_src, self.train_rel, self.train_dst
		for start in range(0, len(src), batch_size):
			end = start + batch_size
			yield src[start:end], rel[start:end], dst[start:end]

	def train_epoch(self, dataloader, epoch: int) -> float:
		del dataloader
		self.model.train()
		batch_size = max(getattr(self.args, 'batch_size', 1), 1)
		model_obj = get_model_obj(self.model)

		perm = torch.randperm(self.train_src.size(0), device=self.train_src.device)
		src = self.train_src.index_select(0, perm).to(self.device)
		rel = self.train_rel.index_select(0, perm).to(self.device)
		dst = self.train_dst.index_select(0, perm).to(self.device)

		epoch_loss = 0.0
		num_examples = src.size(0)
		for h_idx, r_idx, t_idx in self._iter_batches(batch_size):
			h_idx = h_idx.to(self.device)
			r_idx = r_idx.to(self.device)
			t_idx = t_idx.to(self.device)

			self.optimizer.zero_grad()
			scores_sp = self.model.score_sp_(h_idx, r_idx)
			loss_sp = self.loss_fn(scores_sp, t_idx)

			if self.bidirectional and hasattr(model_obj.scorer, 'score_po_'):
				scores_po = self.model.score_po_(r_idx, t_idx)
				loss_po = self.loss_fn(scores_po, h_idx)
				loss = (loss_sp + loss_po) / 2.0
			else:
				loss = loss_sp

			loss.backward()
			self.optimizer.step()
			epoch_loss += loss.item() * h_idx.size(0)

		avg_loss = epoch_loss / max(num_examples, 1)
		logger.info('[EPOCH %s] Train | Loss: %.4f', epoch + 1, avg_loss)
		return avg_loss

	def train_loop(self, dataloader=None) -> dict:
		return run_index_kge_train_loop(self, dataloader)


Strategy = OneVsAllStrategy
SoftmaxStrategy = OneVsAllStrategy
