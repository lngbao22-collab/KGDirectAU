"""Negative-sampling training paradigm (LibKGE ``TrainingJobNegativeSampling`` flow)."""

from __future__ import annotations

from typing import Iterable

import torch

from models.builder import (
	apply_kge_regularization,
	build_lr_scheduler,
	build_optimizer,
	config_bool,
	config_float,
	config_int,
	init_index_kge_trainer,
	load_loss_fn_for_paradigm,
	load_sampler,
	run_index_kge_train_loop,
)
from models.scorers.rotate_scorer import normalize_rotate_phases
from utils.device import get_model_obj
from utils.logger import logger


class NegSampStrategy:
	"""Train with a sampler + 1-to-1 ``score_spo`` scoring and an injected loss."""

	def __init__(
		self,
		model,
		sampler,
		loss_fn,
		args,
		train_triples: torch.Tensor | None = None,
		train_dataloader=None,
		ngpus_per_node: int = 1,
	):
		del ngpus_per_node
		init_index_kge_trainer(self, model, args)
		self.train_triples = train_triples.long() if train_triples is not None else None
		self.train_dataloader = train_dataloader
		self.sampler = sampler if sampler is not None else load_sampler(args, self.train_triples, model)
		self.loss_fn = loss_fn if loss_fn is not None else load_loss_fn_for_paradigm(args, 'negsamp')

		self.base_lr = config_float(args, 'lr', config_float(args, 'learning_rate', 5e-5))
		weight_decay = config_float(args, 'weight_decay', 0.0)
		self.optimizer = build_optimizer(args, self.model.parameters(), weight_decay)
		self.lr_scheduler = build_lr_scheduler(args, self.optimizer)
		self.global_step = 0
		self.next_lr_decay_step = config_int(args, 'warm_up_steps', None)
		self.lr_decay_factor = config_float(args, 'lr_decay_factor', 0.1)
		self.max_steps = config_int(args, 'max_steps', None)
		self.shuffle_train = config_bool(args, 'shuffle_train', False)
		self.use_amp = bool(getattr(args, 'use_amp', False)) and torch.cuda.is_available()
		self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
		self._slot_modes = ['head-batch', 'tail-batch']
		self._slot_step = 0
		self._pointwise_mode = hasattr(self.sampler, 'num_entities') or type(self.sampler).__name__ == 'PointwiseNegSampler'
		self._normalize_rotate_phases = (
			config_bool(args, 'normalize_phases', False)
			and str(getattr(args, 'model', '') or '').lower() == 'rotate'
		)
		if self._normalize_rotate_phases:
			normalize_rotate_phases(self.model)

	def _maybe_decay_learning_rate(self) -> None:
		if self.next_lr_decay_step is None or self.global_step < self.next_lr_decay_step:
			return
		decay_factor = max(self.lr_decay_factor, 0.0)
		new_lr = self.optimizer.param_groups[0]['lr'] * decay_factor
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = new_lr
		logger.info('Change learning rate to %.8f at step %d', new_lr, self.global_step)
		self.next_lr_decay_step = int(self.next_lr_decay_step * 3)

	def _iter_train_batches(self, epoch: int):
		if self.train_dataloader is not None:
			yield from self.train_dataloader
			return

		batch_size = max(getattr(self.args, 'batch_size', 1024), 1)
		triples = self.train_triples
		if self.shuffle_train:
			generator = torch.Generator()
			generator.manual_seed(int(getattr(self.args, 'seed', 0) or 0) + int(epoch))
			permutation = torch.randperm(triples.size(0), generator=generator)
			triples = triples[permutation]
		for start in range(0, len(triples), batch_size):
			yield triples[start:start + batch_size]

	def _score_triple(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		*,
		predict_head: bool,
	) -> torch.Tensor:
		if predict_head and hasattr(get_model_obj(self.model).scorer, 'score_po'):
			return self.model.score_po(r, t, s=h)
		return self.model.score_spo(h, r, t)

	def _score_filtered_negatives(
		self,
		pos_triples: torch.Tensor,
		neg_entity_ids: torch.Tensor,
		mode: str,
	) -> tuple[torch.Tensor, torch.Tensor]:
		h = pos_triples[:, 0]
		r = pos_triples[:, 1]
		t = pos_triples[:, 2]
		predict_head = mode == 'head-batch'
		pos_scores = self._score_triple(h, r, t, predict_head=predict_head)

		batch_size, num_neg = neg_entity_ids.shape
		if mode == 'tail-batch':
			h_exp = h.unsqueeze(1).expand(-1, num_neg).reshape(-1)
			r_exp = r.unsqueeze(1).expand(-1, num_neg).reshape(-1)
			t_neg = neg_entity_ids.reshape(-1)
			neg_scores = self.model.score_spo(h_exp, r_exp, t_neg).view(batch_size, num_neg)
		elif mode == 'head-batch':
			h_neg = neg_entity_ids.reshape(-1)
			r_exp = r.unsqueeze(1).expand(-1, num_neg).reshape(-1)
			t_exp = t.unsqueeze(1).expand(-1, num_neg).reshape(-1)
			neg_scores = self.model.score_po(r_exp, t_exp, s=h_neg).view(batch_size, num_neg)
		else:
			raise ValueError(f'Unsupported negative-sampling mode: {mode}')

		return pos_scores, neg_scores

	def _score_pointwise_batch(
		self,
		pos_triples: torch.Tensor,
		neg_triples: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor]:
		pos_scores = self.model.score_spo(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
		neg_scores = self.model.score_spo(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
		return pos_scores, neg_scores

	def _compute_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, weights) -> torch.Tensor:
		"""Dispatch to negsamp-style loss or row-wise softmax fallback."""

		try:
			return self.loss_fn(pos_scores, neg_scores, weights)
		except TypeError:
			pass
		try:
			return self.loss_fn(pos_scores, neg_scores)
		except TypeError:
			pass

		pos_scores = pos_scores.reshape(-1)
		if neg_scores.dim() == 3:
			neg_scores = neg_scores.squeeze(-1)
		if neg_scores.dim() == 1:
			neg_scores = neg_scores.unsqueeze(1)
		row_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
		targets = torch.zeros(row_scores.size(0), dtype=torch.long, device=row_scores.device)
		return self.loss_fn(row_scores, targets)

	def _regularization_term(self) -> torch.Tensor | None:
		model_obj = get_model_obj(self.model)
		reg_coef = config_float(self.args, 'regularization', 0.0)
		if reg_coef <= 0.0:
			return None
		if hasattr(model_obj, 'adversarial_l3_regularization'):
			return reg_coef * model_obj.adversarial_l3_regularization()
		if hasattr(model_obj, 'entity_embedding') and hasattr(model_obj, 'relation_embedding'):
			return reg_coef * (
				model_obj.entity_embedding.norm(p=3) ** 3
				+ model_obj.relation_embedding.norm(p=3) ** 3
			)
		return None

	def _dabr_regularization(self, pos_triples: torch.Tensor) -> torch.Tensor | None:
		model_obj = get_model_obj(self.model)
		if not hasattr(model_obj, 'regularization'):
			return None
		entity_reg_weight = float(getattr(self.args, 'entity_reg_weight', 0.0))
		relation_reg_weight = float(getattr(self.args, 'relation_reg_weight', 0.0))
		if entity_reg_weight <= 0.0 and relation_reg_weight <= 0.0:
			return None

		h = model_obj.embed_s(pos_triples[:, 0])
		r = model_obj.embed_p(pos_triples[:, 1])
		t = model_obj.embed_o(pos_triples[:, 2])
		dr = model_obj.Dr(pos_triples[:, 1])
		reg_ent = model_obj.regularization(h) + model_obj.regularization(t)
		reg_rel = model_obj.regularization(r) + model_obj.regularization(dr)
		return (entity_reg_weight * reg_ent) + (relation_reg_weight * reg_rel)

	def train_epoch(self, dataloader: Iterable | None, epoch: int) -> float:
		self.model.train()
		total_loss = 0.0
		step = 0
		data_source = dataloader if dataloader is not None else self._iter_train_batches(epoch)

		for batch in data_source:
			if self.max_steps is not None and self.global_step >= self.max_steps:
				break

			self.optimizer.zero_grad()
			current_mode = None if self._pointwise_mode else self._slot_modes[self._slot_step % 2]
			if not self._pointwise_mode:
				self._slot_step += 1

			sample_result = self.sampler.sample(batch, current_mode)
			if len(sample_result) == 4:
				pos_batch, neg_batch, weights, mode = sample_result
			else:
				pos_batch, neg_batch, weights = sample_result
				mode = current_mode

			if isinstance(batch, dict):
				for key in batch:
					if torch.is_tensor(batch[key]):
						batch[key] = batch[key].to(self.device)

			if torch.is_tensor(pos_batch):
				pos_batch = pos_batch.to(self.device)
			if torch.is_tensor(neg_batch):
				neg_batch = neg_batch.to(self.device)
			if weights is not None and torch.is_tensor(weights):
				weights = weights.to(self.device)

			with torch.autocast(device_type='cuda', enabled=self.use_amp):
				if self._pointwise_mode:
					pos_scores, neg_scores = self._score_pointwise_batch(pos_batch, neg_batch)
					loss = self._compute_loss(pos_scores, neg_scores, weights)
					dabr_reg = self._dabr_regularization(pos_batch)
					if dabr_reg is not None:
						loss = loss + dabr_reg
				else:
					pos_scores, neg_scores = self._score_filtered_negatives(pos_batch, neg_batch, mode)
					loss = self._compute_loss(pos_scores, neg_scores, weights)
					loss = apply_kge_regularization(
						loss,
						self.model,
						self.args,
						batch_triples=pos_batch,
					)
					reg_term = self._regularization_term()
					if reg_term is not None:
						loss = loss + reg_term

			if self.use_amp:
				self.scaler.scale(loss).backward()
				if self._pointwise_mode:
					self.scaler.unscale_(self.optimizer)
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
				self.scaler.step(self.optimizer)
				self.scaler.update()
			else:
				loss.backward()
				if self._pointwise_mode:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
				self.optimizer.step()

			if self._normalize_rotate_phases:
				normalize_rotate_phases(self.model)

			self.global_step += 1
			self._maybe_decay_learning_rate()
			total_loss += float(loss.item())
			step += 1

		avg_loss = total_loss / max(step, 1)
		logger.info('[EPOCH %s] Train | Loss: %.4f', epoch + 1, avg_loss)
		return avg_loss

	def train_loop(self, dataloader=None) -> dict:
		return run_index_kge_train_loop(self, dataloader)


Strategy = NegSampStrategy
AdversarialStrategy = NegSampStrategy
PointwiseStrategy = NegSampStrategy
