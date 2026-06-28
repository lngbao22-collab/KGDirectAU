"""Negative-sampling training paradigm (LibKGE ``TrainingJobNegativeSampling`` flow)."""

from __future__ import annotations

import math
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
from base.embeddings import rotate_style_embedding_l3_penalty
from models.losses.bce_loss import bce_logit_offset, uses_bce_logit_offset
from models.losses.loss_utilities import (
	compute_adversarial_negsamp_losses_chunked,
	compute_standard_negsamp_losses_chunked,
)
from models.scorers.protate_scorer import normalize_protate_phases
from models.scorers.rotate_scorer import normalize_rotate_phases
from utils.device import get_model_obj
from utils.logger import logger
from utils.training_cadence import init_step_cadence_state, maybe_decay_lr_at_step, uses_step_cadence


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
		init_step_cadence_state(self)
		self.shuffle_train = config_bool(args, 'shuffle_train', False)
		self.use_amp = bool(getattr(args, 'use_amp', False)) and torch.cuda.is_available()
		device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.scaler = torch.amp.GradScaler(device_type, enabled=self.use_amp)
		self._pointwise_mode = hasattr(self.sampler, 'num_entities') or type(self.sampler).__name__ == 'PointwiseNegSampler'
		model_name = str(getattr(args, 'model', '') or '').lower()
		self._normalize_phases_fn = None
		if config_bool(args, 'normalize_phases', False):
			if model_name == 'rotate':
				self._normalize_phases_fn = normalize_rotate_phases
			elif 'protate' in model_name:
				self._normalize_phases_fn = normalize_protate_phases
		if self._normalize_phases_fn is not None:
			self._normalize_phases_fn(self.model)
		if self.train_triples is not None and torch.cuda.is_available():
			self.train_triples = self.train_triples.pin_memory()

	def _neg_score_chunk_size(self, num_neg: int) -> int | None:
		raw = config_int(self.args, 'neg_score_chunk_size', None)
		if raw is not None and int(raw) <= 0:
			return None
		model_obj = get_model_obj(self.model)
		uses_batched = self._supports_batched_candidate_scoring(model_obj)
		if raw is None:
			return None if uses_batched else 128
		chunk_size = int(raw)
		if num_neg <= chunk_size:
			return None
		return chunk_size

	def _uses_adversarial_bce(self) -> bool:
		path = str(getattr(self.args, 'model_loss_path', '') or '').lower()
		return 'adversarial_bce' in path

	def _supports_batched_candidate_scoring(self, model_obj) -> bool:
		scorer = getattr(model_obj, 'scorer', None)
		return scorer is not None and hasattr(scorer, 'score_spo_candidates') and hasattr(scorer, 'score_po_candidates')

	def _positive_scores(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		*,
		mode: str = 'tail-batch',
		model_obj=None,
		h_emb: torch.Tensor | None = None,
		r_emb: torch.Tensor | None = None,
		t_emb: torch.Tensor | None = None,
	):
		"""Positive triple scores aligned with the active head/tail corruption mode."""

		model_obj = model_obj or get_model_obj(self.model)
		if h_emb is None:
			h_emb = model_obj.embed_s(h)
		if r_emb is None:
			r_emb = model_obj.embed_p(r)
		if t_emb is None:
			t_emb = model_obj.embed_o(t)
		scorer = model_obj.scorer
		if mode == 'head-batch' and hasattr(scorer, 'score_po'):
			return scorer.score_po(h_emb, r_emb, t_emb)
		if hasattr(scorer, 'score_spo'):
			return scorer.score_spo(h_emb, r_emb, t_emb)
		return self.model.score_spo(h, r, t)

	def _candidate_scoring_context(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		mode: str,
	):
		model_obj = get_model_obj(self.model)
		if not self._supports_batched_candidate_scoring(model_obj):
			return None

		h_emb = model_obj.embed_s(h)
		r_emb = model_obj.embed_p(r)
		t_emb = model_obj.embed_o(t)
		pos_scores = self._positive_scores(
			h,
			r,
			t,
			mode=mode,
			model_obj=model_obj,
			h_emb=h_emb,
			r_emb=r_emb,
			t_emb=t_emb,
		)
		return {
			'mode': mode,
			'h_emb': h_emb,
			'r_emb': r_emb,
			't_emb': t_emb,
			'pos_scores': pos_scores,
		}

	def _adversarial_temperature(self) -> float:
		return float(
			getattr(self.args, 'adversarial_temp', None)
			or getattr(self.args, 'adversarial_temperature', 1.0)
			or 1.0
		)

	def _backward_loss(self, loss: torch.Tensor, *, retain_graph: bool = False) -> None:
		if self.use_amp:
			self.scaler.scale(loss).backward(retain_graph=retain_graph)
		else:
			loss.backward(retain_graph=retain_graph)

	def _optimizer_step(self) -> None:
		if self.use_amp:
			if self._pointwise_mode:
				self.scaler.unscale_(self.optimizer)
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
			self.scaler.step(self.optimizer)
			self.scaler.update()
		else:
			if self._pointwise_mode:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
			self.optimizer.step()

	def _maybe_decay_learning_rate(self) -> None:
		maybe_decay_lr_at_step(self, self.global_step)

	def iter_training_batches(self, epoch: int, dataloader=None):
		if dataloader is not None:
			yield from dataloader
			return
		if self._pointwise_mode or self.train_triples is None:
			yield from self._iter_train_batches(epoch)
			return
		yield from self._iter_bidirectional_train_batches(epoch)

	def train_batch(self, batch, epoch: int, *, mode: str | None = None) -> float:
		del epoch
		batch, mode = self._unpack_training_batch(batch, mode)
		if not self._pointwise_mode and mode is None:
			raise ValueError('Filtered negative-sampling requires an explicit head-batch or tail-batch mode')
		self.model.train()
		self.optimizer.zero_grad(set_to_none=True)

		sample_result = self.sampler.sample(batch, mode)
		if len(sample_result) == 4:
			pos_batch, neg_batch, weights, sampled_mode = sample_result
			mode = sampled_mode
		else:
			pos_batch, neg_batch, weights = sample_result

		if isinstance(batch, dict):
			for key in batch:
				if torch.is_tensor(batch[key]):
					batch[key] = batch[key].to(self.device)

		if torch.is_tensor(pos_batch):
			pos_batch = pos_batch.to(self.device, non_blocking=True)
		if torch.is_tensor(neg_batch):
			neg_batch = neg_batch.to(self.device, non_blocking=True)
		if weights is not None and torch.is_tensor(weights):
			weights = weights.to(self.device, non_blocking=True)

		with torch.autocast(device_type='cuda', enabled=self.use_amp):
			if self._pointwise_mode:
				pos_scores, neg_scores = self._score_pointwise_batch(pos_batch, neg_batch)
				loss = self._compute_loss(pos_scores, neg_scores, weights)
				dabr_reg = self._dabr_regularization(pos_batch)
				if dabr_reg is not None:
					loss = loss + dabr_reg
				self._backward_loss(loss)
			else:
				chunk_size = self._neg_score_chunk_size(int(neg_batch.size(1)))
				if chunk_size is None:
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
					self._backward_loss(loss)
				else:
					loss_parts = self._filtered_negsamp_loss_parts_chunked(
						pos_batch,
						neg_batch,
						mode,
						weights,
						chunk_size,
					)
					reg_term = self._regularization_term()
					if reg_term is not None:
						loss_parts[-1] = loss_parts[-1] + reg_term
					for idx, loss_part in enumerate(loss_parts):
						self._backward_loss(
							loss_part,
							retain_graph=idx < len(loss_parts) - 1,
						)
					anchor = loss_parts[0].new_zeros(())
					kge_reg = apply_kge_regularization(
						anchor,
						self.model,
						self.args,
						batch_triples=pos_batch,
					)
					if kge_reg is not anchor:
						self._backward_loss(kge_reg)
					loss = sum((part.detach() for part in loss_parts), start=anchor)
					if kge_reg is not anchor:
						loss = loss + kge_reg.detach()

		self._optimizer_step()

		if self._normalize_phases_fn is not None:
			self._normalize_phases_fn(self.model)

		loss_value = float(loss.detach().item())
		if not math.isfinite(loss_value):
			raise FloatingPointError(
				f'Non-finite training loss ({loss_value}) in {mode} batch; '
				'check embeddings/scores for NaN/Inf'
			)
		return loss_value

	def _unpack_training_batch(self, batch, mode: str | None) -> tuple[torch.Tensor | dict, str | None]:
		if (
			mode is None
			and isinstance(batch, (tuple, list))
			and len(batch) == 2
			and isinstance(batch[1], str)
		):
			return batch[0], batch[1]
		return batch, mode

	def _shuffled_triples(self, epoch: int, *, stream: str) -> torch.Tensor:
		triples = self.train_triples
		if not self.shuffle_train:
			return triples
		generator = torch.Generator()
		seed = int(getattr(self.args, 'seed', 0) or 0) + int(epoch)
		stream_offsets = {'head': 1_000_003, 'tail': 2_000_003}
		generator.manual_seed(seed + stream_offsets.get(stream, 0))
		return triples[torch.randperm(triples.size(0), generator=generator)]

	@staticmethod
	def _batch_chunks(triples: torch.Tensor, batch_size: int):
		for start in range(0, len(triples), batch_size):
			yield triples[start:start + batch_size]

	def _iter_bidirectional_train_batches(self, epoch: int):
		"""Alternate tail/head batches from independent shuffles (KGE ``BidirectionalOneShotIterator``)."""

		batch_size = max(getattr(self.args, 'batch_size', 1024), 1)
		head_batches = list(
			self._batch_chunks(self._shuffled_triples(epoch, stream='head'), batch_size)
		)
		tail_batches = list(
			self._batch_chunks(self._shuffled_triples(epoch, stream='tail'), batch_size)
		)
		head_idx = 0
		tail_idx = 0
		step = 0
		while head_idx < len(head_batches) or tail_idx < len(tail_batches):
			step += 1
			if step % 2 == 0:
				if head_idx < len(head_batches):
					yield head_batches[head_idx], 'head-batch'
					head_idx += 1
				elif tail_idx < len(tail_batches):
					yield tail_batches[tail_idx], 'tail-batch'
					tail_idx += 1
				else:
					break
			elif tail_idx < len(tail_batches):
				yield tail_batches[tail_idx], 'tail-batch'
				tail_idx += 1
			elif head_idx < len(head_batches):
				yield head_batches[head_idx], 'head-batch'
				head_idx += 1
			else:
				break

	def _iter_train_batches(self, epoch: int):
		if self.train_dataloader is not None:
			yield from self.train_dataloader
			return

		batch_size = max(getattr(self.args, 'batch_size', 1024), 1)
		for chunk in self._batch_chunks(self._shuffled_triples(epoch, stream='tail'), batch_size):
			yield chunk

	def _score_filtered_negatives(
		self,
		pos_triples: torch.Tensor,
		neg_entity_ids: torch.Tensor,
		mode: str,
	) -> tuple[torch.Tensor, torch.Tensor]:
		h = pos_triples[:, 0]
		r = pos_triples[:, 1]
		t = pos_triples[:, 2]
		context = self._candidate_scoring_context(h, r, t, mode)
		if context is not None:
			batch_size, num_neg = neg_entity_ids.shape
			model_obj = get_model_obj(self.model)
			scorer = model_obj.scorer
			if mode == 'tail-batch':
				t_emb = model_obj.embed_o(neg_entity_ids.reshape(-1)).view(batch_size, num_neg, -1)
				neg_scores = scorer.score_spo_candidates(context['h_emb'], context['r_emb'], t_emb)
			elif mode == 'head-batch':
				h_emb = model_obj.embed_s(neg_entity_ids.reshape(-1)).view(batch_size, num_neg, -1)
				neg_scores = scorer.score_po_candidates(h_emb, context['r_emb'], context['t_emb'])
			else:
				raise ValueError(f'Unsupported negative-sampling mode: {mode}')
			return context['pos_scores'], neg_scores

		pos_scores = self._positive_scores(h, r, t, mode=mode)

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

	def _score_negatives_slice(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		neg_entity_ids: torch.Tensor,
		mode: str,
		col_start: int,
		col_end: int,
		*,
		context=None,
	) -> torch.Tensor:
		neg_slice = neg_entity_ids[:, col_start:col_end]
		chunk_neg = neg_slice.size(1)
		if context is not None:
			model_obj = get_model_obj(self.model)
			scorer = model_obj.scorer
			batch_size = h.size(0)
			if mode == 'tail-batch':
				t_emb = model_obj.embed_o(neg_slice.reshape(-1)).view(batch_size, chunk_neg, -1)
				return scorer.score_spo_candidates(context['h_emb'], context['r_emb'], t_emb)
			if mode == 'head-batch':
				h_emb = model_obj.embed_s(neg_slice.reshape(-1)).view(batch_size, chunk_neg, -1)
				return scorer.score_po_candidates(h_emb, context['r_emb'], context['t_emb'])
			raise ValueError(f'Unsupported negative-sampling mode: {mode}')

		if mode == 'tail-batch':
			h_exp = h.unsqueeze(1).expand(-1, chunk_neg).reshape(-1)
			r_exp = r.unsqueeze(1).expand(-1, chunk_neg).reshape(-1)
			t_neg = neg_slice.reshape(-1)
			return self.model.score_spo(h_exp, r_exp, t_neg).view(h.size(0), chunk_neg)
		if mode == 'head-batch':
			h_neg = neg_slice.reshape(-1)
			r_exp = r.unsqueeze(1).expand(-1, chunk_neg).reshape(-1)
			t_exp = t.unsqueeze(1).expand(-1, chunk_neg).reshape(-1)
			return self.model.score_po(r_exp, t_exp, s=h_neg).view(h.size(0), chunk_neg)
		raise ValueError(f'Unsupported negative-sampling mode: {mode}')

	def _filtered_negsamp_loss_parts_chunked(
		self,
		pos_triples: torch.Tensor,
		neg_entity_ids: torch.Tensor,
		mode: str,
		weights,
		chunk_size: int,
	) -> list[torch.Tensor]:
		h = pos_triples[:, 0]
		r = pos_triples[:, 1]
		t = pos_triples[:, 2]
		context = self._candidate_scoring_context(h, r, t, mode)
		if context is not None:
			pos_scores = context['pos_scores']
		else:
			pos_scores = self._positive_scores(h, r, t, mode=mode)
		num_neg = int(neg_entity_ids.size(1))

		def score_neg_columns(start: int, end: int) -> torch.Tensor:
			return self._score_negatives_slice(
				h,
				r,
				t,
				neg_entity_ids,
				mode,
				start,
				end,
				context=context,
			)

		if self._uses_adversarial_bce():
			weight_chunk_size = config_int(self.args, 'neg_weight_chunk_size', None)
			if weight_chunk_size is not None and int(weight_chunk_size) <= 0:
				weight_chunk_size = None
			return compute_adversarial_negsamp_losses_chunked(
				pos_scores,
				score_neg_columns,
				num_neg=num_neg,
				adversarial_temp=self._adversarial_temperature(),
				weights=weights,
				chunk_size=chunk_size,
				weight_chunk_size=weight_chunk_size,
			)

		offset = bce_logit_offset(self.args) if uses_bce_logit_offset(self.args) else 0.0
		return compute_standard_negsamp_losses_chunked(
			pos_scores,
			score_neg_columns,
			num_neg=num_neg,
			weights=weights,
			chunk_size=chunk_size,
			offset=offset,
		)

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

		if self._pointwise_mode:
			# Pointwise DaBR uses one loss term per scored triple; subsampling weights
			# are not defined at positive-row granularity.
			weights = None

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
		reg_coef = config_float(self.args, 'regularization', 0.0)
		if reg_coef <= 0.0:
			return None
		model_obj = get_model_obj(self.model)
		l3_term = rotate_style_embedding_l3_penalty(model_obj)
		if l3_term is not None:
			return reg_coef * l3_term
		return None

	def _aux_relation_embedding(self, model_obj, relation_indices: torch.Tensor):
		aux = getattr(model_obj, 'aux_embedders', None)
		if aux is not None and 'dr' in aux:
			return aux['dr'](relation_indices)
		dr = getattr(model_obj, 'Dr', None)
		if dr is not None:
			return dr(relation_indices)
		return None

	def _dabr_regularization(self, pos_triples: torch.Tensor) -> torch.Tensor | None:
		model_obj = get_model_obj(self.model)
		scorer = getattr(model_obj, 'scorer', None)
		reg_fn = getattr(model_obj, 'regularization', None) or getattr(scorer, 'regularization', None)
		if reg_fn is None:
			return None

		entity_reg_weight = getattr(self.args, 'entity_reg_weight', None)
		if entity_reg_weight is None:
			entity_reg_weight = getattr(self.args, 'lmbda', 0.0)
		relation_reg_weight = getattr(self.args, 'relation_reg_weight', None)
		if relation_reg_weight is None:
			relation_reg_weight = getattr(self.args, 'lmbda_two', 0.0)
		entity_reg_weight = float(entity_reg_weight or 0.0)
		relation_reg_weight = float(relation_reg_weight or 0.0)
		if entity_reg_weight <= 0.0 and relation_reg_weight <= 0.0:
			return None

		h = model_obj.embed_s(pos_triples[:, 0])
		r = model_obj.embed_p(pos_triples[:, 1])
		t = model_obj.embed_o(pos_triples[:, 2])
		dr_embedder = self._aux_relation_embedding(model_obj, pos_triples[:, 1])
		if dr_embedder is None:
			return None
		dr = dr_embedder
		reg_ent = reg_fn(h) + reg_fn(t)
		reg_rel = reg_fn(r) + reg_fn(dr)
		return (entity_reg_weight * reg_ent) + (relation_reg_weight * reg_rel)

	def train_epoch(self, dataloader: Iterable | None, epoch: int) -> float:
		if uses_step_cadence(self.args):
			raise RuntimeError('train_epoch should not be called directly under step-based training cadence')

		self.model.train()
		total_loss = 0.0
		step = 0
		for batch in self.iter_training_batches(epoch, dataloader):
			loss = self.train_batch(batch, epoch)
			self.global_step += 1
			self._maybe_decay_learning_rate()
			total_loss += loss
			step += 1

		avg_loss = total_loss / max(step, 1)
		logger.info('[EPOCH %s] Train | Loss: %.4f', epoch + 1, avg_loss)
		return avg_loss

	def train_loop(self, dataloader=None) -> dict:
		return run_index_kge_train_loop(self, dataloader)


Strategy = NegSampStrategy
AdversarialStrategy = NegSampStrategy
PointwiseStrategy = NegSampStrategy
