"""Training strategy for KGAU."""

from __future__ import annotations

import os
import math
import time
from typing import Iterator

import torch
from torch import optim
from torch.optim import Adam

from base.evaluator import Evaluator
from data.dataloader import collate
from data.dataset import Dataset, load_data
from data.dict_hub import build_tokenizer, get_entity_dict, get_relation_id_map
from models.builder import load_attr_from_path
from utils.checkpoint import best_model_path, checkpoint_path, delete_old_ckt, save_checkpoint
from utils.device import get_model_obj, move_to_cuda, report_num_trainable_parameters
from utils.logger import AverageMeter, ProgressMeter, logger
from models.losses.au_loss import KGAULoss, distinct_first_indices, select_distinct_rows


def _load_encoder(args) -> torch.nn.Module:
	"""Factory helper used by the evaluator to rebuild the model from checkpoints."""

	encoder_path = getattr(args, 'model_encoder_path', '') or 'models/encoders/distmult_encoder.py'
	build_model = load_attr_from_path(encoder_path, 'build_model')
	return build_model(args)


def _uses_text_inputs(args) -> bool:
	"""Return True when the configured encoder consumes tokenized text inputs."""

	encoder_path = getattr(args, 'model_encoder_path', '') or ''
	return os.path.basename(encoder_path) == 'bert_encoder.py'


def _config_float(args, name: str, default: float) -> float:
	"""Read a float hyperparameter from args, treating JSON null as unset."""

	value = getattr(args, name, None)
	return default if value is None else float(value)


def _is_dabr_encoder(args) -> bool:
	"""Return True when the configured encoder is DaBR (or DaBR-AU)."""

	encoder_path = str(getattr(args, 'model_encoder_path', '') or '').lower()
	model_name = str(getattr(args, 'model', '') or '').lower()
	return 'dabr' in encoder_path or 'dabr' in model_name


def _add_inverse_relations(relation_to_idx: dict[str, int]) -> dict[str, int]:
	"""Ensure every forward relation has its own distinct inverse-relation ID.

	Backward triplets carry the relation string ``"inverse {relation}"`` (see
	``data.dataset.reverse_triplet``). Assigning each inverse relation a fresh
	index keeps forward and inverse relations distinct for uniformity dedup keys.
	"""

	updated = dict(relation_to_idx)
	next_idx = max(updated.values(), default=-1) + 1
	for relation in list(updated.keys()):
		if relation.startswith('inverse '):
			continue
		inverse_relation = f'inverse {relation}'
		if inverse_relation not in updated:
			updated[inverse_relation] = next_idx
			next_idx += 1
	return updated


def _build_relation_to_idx() -> dict[str, int]:
	"""Build the relation->index map with distinct IDs for inverse relations."""

	base = {str(key): int(value) for key, value in get_relation_id_map().items()}
	return _add_inverse_relations(base)


def _build_optimizer(args, parameters, weight_decay: float):
	"""Build the training optimizer; respects ``optim`` in config (adam/adagrad/sgd)."""

	lr = float(getattr(args, 'lr', getattr(args, 'learning_rate', 2e-5)))
	optim_name = str(getattr(args, 'optim', 'adam')).lower()
	if optim_name == 'adagrad':
		return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
	if optim_name == 'sgd':
		return optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
	return Adam(parameters, lr=lr, weight_decay=weight_decay)


class KGAUStrategy(Evaluator):
	"""Knowledge Graph Alignment and Uniformity training loop for KG encoders."""

	def __init__(self, args, ngpus_per_node):
		super().__init__(args)
		self.ngpus_per_node = ngpus_per_node
		self.uses_text_inputs = _uses_text_inputs(args)
		if self.uses_text_inputs:
			build_tokenizer(args)
		# Inverse triplets (reverse_triplet) are only used by text encoders (e.g. SimKGC/BERT).
		add_backward_triplet = self.uses_text_inputs
		self.train_examples = load_data(
			args.train_path, add_forward_triplet=True, add_backward_triplet=add_backward_triplet)
		logger.info(
			'Training examples: %d (backward triplets=%s)',
			len(self.train_examples), add_backward_triplet,
		)
		self.entity_dict = get_entity_dict()
		self.model = _load_encoder(args)
		logger.info(self.model)
		self.relation_to_idx = _build_relation_to_idx()
		if not self.uses_text_inputs:
			self.train_src, self.train_rel, self.train_dst = self._examples_to_tensors(self.train_examples)
		else:
			self.train_loader = torch.utils.data.DataLoader(
				Dataset(path='', examples=self.train_examples, task=args.dataset),
				batch_size=max(getattr(args, 'batch_size', 1), 1),
				shuffle=True,
				collate_fn=collate,
				num_workers=getattr(args, 'workers', 0),
				pin_memory=True,
				drop_last=True,
			)

		if torch.cuda.device_count() > 1:
			self.model = torch.nn.DataParallel(self.model).cuda()
		elif torch.cuda.is_available():
			self.model.cuda()
		self.device = next(self.model.parameters()).device
		if not self.uses_text_inputs and self.train_src.device != self.device:
			self.train_src = self.train_src.to(self.device)
			self.train_rel = self.train_rel.to(self.device)
			self.train_dst = self.train_dst.to(self.device)

		report_num_trainable_parameters(get_model_obj(self.model))

		weight_decay = getattr(args, 'weight_decay', None)
		if weight_decay is None:
			weight_decay = 0.0
		batch_size = max(getattr(args, 'batch_size', 1), 1)
		num_batches = max(math.ceil(len(self.train_examples) / batch_size), 1)
		self.weight_decay = float(weight_decay) / num_batches
		self.optimizer = _build_optimizer(args, self.model.parameters(), self.weight_decay)

		tuni_val = _config_float(args, 'tuni', _config_float(args, 'temperature', _config_float(args, 't', 2.0)))

		self.criterion = KGAULoss(
			gamma_q=_config_float(args, 'gamma_q', 1.0),
			gamma_t=_config_float(args, 'gamma_t', 1.0),
			gamma_h=_config_float(args, 'gamma_h', 0.0),
			gamma_ent=_config_float(args, 'gamma_ent', 0.0),
			tuni=tuni_val,
			max_uniformity_samples=int(_config_float(args, 'max_uniformity_samples', 1024)),
			additive_margin=_config_float(args, 'additive_margin', 0.0),
		).to(self.device)
		self.best_metric = None
		self.best_checkpoint_path = None
		self.train_time = 0.0
		self.valid_time = 0.0
		self.total_time = 0.0

	def _resolve_relation_index(self, relation: str) -> int:
		"""Resolve a relation string to its index.

		Forward and inverse relations have distinct IDs (see
		``_add_inverse_relations``); inverse relations are looked up directly
		rather than collapsed onto their forward counterpart.
		"""

		if relation in self.relation_to_idx:
			return self.relation_to_idx[relation]
		normalized = ' '.join(relation.split())
		if normalized in self.relation_to_idx:
			return self.relation_to_idx[normalized]
		raise KeyError(relation)

	def _examples_to_tensors(self, examples) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Convert a list of examples into tensors of head, relation, and tail indices."""

		head_indices = torch.tensor([self.entity_dict.entity_to_idx(example.head_id) for example in examples], dtype=torch.long)
		relation_indices = torch.tensor([self._resolve_relation_index(example.relation) for example in examples], dtype=torch.long)
		tail_indices = torch.tensor([self.entity_dict.entity_to_idx(example.tail_id) for example in examples], dtype=torch.long)
		return head_indices, relation_indices, tail_indices

	def _iter_batches(
		self,
		src,
		rel,
		dst,
		batch_size,
		shuffle: bool = False,
	) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
		"""Iterate over batches of examples; optionally shuffle index order each epoch."""

		num_examples = len(src)
		if shuffle and num_examples > 1:
			order = torch.randperm(num_examples, device=src.device)
			src = src.index_select(0, order)
			rel = rel.index_select(0, order)
			dst = dst.index_select(0, order)

		for start in range(0, num_examples, batch_size):
			end = start + batch_size
			yield src[start:end], rel[start:end], dst[start:end]

	def _validation_interval(self) -> int:
		"""Epochs between full link-prediction validation runs.

		KGAU validation is epoch-based, so it is driven by ``epoch_per_eval``
		(``0`` or unset means validate every epoch). The step-based
		``eval_every_n_step`` knob is intentionally not consulted here.
		"""

		raw = getattr(self.args, 'epoch_per_eval', None)
		interval = int(raw) if raw is not None else 0
		if interval <= 0 or interval > int(self.args.epochs):
			return 1
		return interval

	def _should_validate(self, epoch: int) -> bool:
		"""Return True when link-prediction validation should run after this epoch."""

		interval = self._validation_interval()
		epoch_number = epoch + 1
		return epoch_number % interval == 0 or epoch_number >= int(self.args.epochs)

	def _uniformity_keys(
		self,
		head_indices: torch.Tensor,
		relation_indices: torch.Tensor,
		tail_indices: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Build deduplication keys for query (head, relation), tail, and head uniformity terms."""

		device = head_indices.device
		q_keys = torch.stack(
			[
				head_indices.to(device=device, dtype=torch.long),
				relation_indices.to(device=device, dtype=torch.long),
			],
			dim=1,
		)
		t_keys = tail_indices.to(device=device, dtype=torch.long)
		h_keys = head_indices.to(device=device, dtype=torch.long)
		return q_keys, t_keys, h_keys

	def _uniformity_keys_from_examples(self, examples) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Build deduplication keys from a collated batch of training examples."""

		head_indices, relation_indices, tail_indices = self._examples_to_tensors(examples)
		return self._uniformity_keys(
			head_indices.to(self.device),
			relation_indices.to(self.device),
			tail_indices.to(self.device),
		)

	def _distinct_uniformity_inputs(
		self,
		q_raw: torch.Tensor,
		t_raw: torch.Tensor,
		h_raw: torch.Tensor,
		q_keys: torch.Tensor,
		t_keys: torch.Tensor,
		h_keys: torch.Tensor,
	) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int, int]:
		"""Deduplicate embeddings per uniformity type before AU uniformity loss."""

		q_uni = select_distinct_rows(q_raw, q_keys) if self.criterion.gamma_q > 0 else None
		t_uni = select_distinct_rows(t_raw, t_keys) if self.criterion.gamma_t > 0 else None
		h_uni = select_distinct_rows(h_raw, h_keys) if self.criterion.gamma_h > 0 else None
		n_unique_q = q_uni.size(0) if q_uni is not None else 0
		n_unique_t = t_uni.size(0) if t_uni is not None else 0
		return q_uni, t_uni, h_uni, n_unique_q, n_unique_t

	def _count_unique_uniformity_keys(
		self,
		head_indices: torch.Tensor,
		relation_indices: torch.Tensor,
		tail_indices: torch.Tensor,
	) -> tuple[int, int]:
		"""Count unique query/tail keys in a full training batch (logging only; cheap)."""

		q_keys, t_keys, _ = self._uniformity_keys(head_indices, relation_indices, tail_indices)
		n_unique_q = int(distinct_first_indices(q_keys).numel()) if self.criterion.gamma_q > 0 else 0
		n_unique_t = int(distinct_first_indices(t_keys).numel()) if self.criterion.gamma_t > 0 else 0
		return n_unique_q, n_unique_t

	def _au_loss_with_distinct_keys(
		self,
		q_raw: torch.Tensor,
		t_raw: torch.Tensor,
		h_raw: torch.Tensor,
		ent_raw: torch.Tensor | None,
		q_keys: torch.Tensor,
		t_keys: torch.Tensor,
		h_keys: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, float]:
		"""KGAU loss with deduplicated uniformity inputs (by entity/relation id keys)."""

		q_uni, t_uni, h_uni, n_unique_q, n_unique_t = self._distinct_uniformity_inputs(
			q_raw, t_raw, h_raw, q_keys, t_keys, h_keys)
		loss, l_align, l_unif, margin_active_frac = self.criterion(
			q_raw, t_raw, h_raw, ent_raw, q_uni=q_uni, t_uni=t_uni, h_uni=h_uni, return_stats=True)
		return loss, l_align, l_unif, n_unique_q, n_unique_t, margin_active_frac

	def _entity_uniformity_vectors(self, model) -> torch.Tensor | None:
		"""Optional entity embeddings for uniformity; never materialize the full table on GPU."""

		if self.criterion.gamma_ent <= 0:
			return None
		max_samples = int(getattr(self.criterion, 'max_uniformity_samples', 0) or 0)
		if hasattr(model, 'entity_embeddings'):
			return model.entity_embeddings(device=self.device, max_samples=max_samples or None)
		return None

	def _train_micro_batch_size(self, batch_size: int) -> int:
		"""Split training batches only for DaBR (high memory AU vectors). Other encoders use full ``batch_size``."""

		if not _is_dabr_encoder(self.args):
			return batch_size
		explicit = getattr(self.args, 'train_micro_batch_size', None)
		if explicit is not None:
			return max(int(explicit), 1)
		# Default DaBR cap: 64 rows per forward/backward chunk on ~15 GiB GPUs.
		if batch_size > 64:
			return 64
		return batch_size

	def _backward_au_loss(
		self,
		loss: torch.Tensor,
		batch_fraction: float,
		use_amp: bool,
	) -> None:
		"""Backprop a weighted AU loss fragment (for micro-batching)."""

		scaled_loss = loss * batch_fraction
		if use_amp:
			self.scaler.scale(scaled_loss).backward()
		else:
			scaled_loss.backward()

	def _train_au_tensor_batch(
		self,
		model,
		ss: torch.Tensor,
		rs: torch.Tensor,
		ts: torch.Tensor,
		use_amp: bool,
	) -> tuple[float, float, float, int, int, float, int]:
		"""Run one optimizer step on a head/relation/tail batch.

		Non-DaBR encoders use a single forward/backward over the full batch (unchanged).
		DaBR may split into smaller chunks when ``train_micro_batch_size`` or the default cap applies.
		"""

		total = ss.size(0)
		n_uq_log, n_ut_log = self._count_unique_uniformity_keys(ss, rs, ts)
		micro_batch = min(self._train_micro_batch_size(total), total)

		if micro_batch >= total:
			return self._train_au_tensor_batch_single(
				model, ss, rs, ts, use_amp, n_uq_log, n_ut_log, total,
			)
		return self._train_au_tensor_batch_micro(
			model, ss, rs, ts, use_amp, n_uq_log, n_ut_log, total, micro_batch,
		)

	def _train_au_tensor_batch_single(
		self,
		model,
		ss: torch.Tensor,
		rs: torch.Tensor,
		ts: torch.Tensor,
		use_amp: bool,
		n_uq_log: int,
		n_ut_log: int,
		total: int,
	) -> tuple[float, float, float, int, int, float, int]:
		"""Full-batch training step (DistMult-AU, ComplEx-AU, RotatE-AU, etc.)."""

		self.optimizer.zero_grad()
		q_keys, t_keys, h_keys = self._uniformity_keys(ss, rs, ts)
		if use_amp:
			with torch.amp.autocast(device_type='cuda'):
				q_raw, t_raw, h_raw = model.get_queries_targets(ss, rs, ts)
				ent_raw = self._entity_uniformity_vectors(model)
				loss, l_align, l_unif, _, _, margin_active = self._au_loss_with_distinct_keys(
					q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys)
			self.scaler.scale(loss).backward()
			self.scaler.unscale_(self.optimizer)
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
			self.scaler.step(self.optimizer)
			self.scaler.update()
		else:
			q_raw, t_raw, h_raw = model.get_queries_targets(ss, rs, ts)
			ent_raw = self._entity_uniformity_vectors(model)
			loss, l_align, l_unif, _, _, margin_active = self._au_loss_with_distinct_keys(
				q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
			self.optimizer.step()
		return loss.item(), l_align.item(), l_unif.item(), n_uq_log, n_ut_log, margin_active, total

	def _train_au_tensor_batch_micro(
		self,
		model,
		ss: torch.Tensor,
		rs: torch.Tensor,
		ts: torch.Tensor,
		use_amp: bool,
		n_uq_log: int,
		n_ut_log: int,
		total: int,
		micro_batch: int,
	) -> tuple[float, float, float, int, int, float, int]:
		"""DaBR-only: gradient accumulation over micro-batches to avoid OOM."""

		loss_sum = 0.0
		align_sum = 0.0
		unif_sum = 0.0
		margin_acc = 0.0
		margin_batches = 0

		self.optimizer.zero_grad()
		for start in range(0, total, micro_batch):
			end = min(start + micro_batch, total)
			fraction = (end - start) / total
			q_keys, t_keys, h_keys = self._uniformity_keys(ss[start:end], rs[start:end], ts[start:end])
			if use_amp:
				with torch.amp.autocast(device_type='cuda'):
					q_raw, t_raw, h_raw = model.get_queries_targets(ss[start:end], rs[start:end], ts[start:end])
					ent_raw = self._entity_uniformity_vectors(model)
					loss, l_align, l_unif, _, _, margin_active = self._au_loss_with_distinct_keys(
						q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys)
				self._backward_au_loss(loss, fraction, use_amp=True)
			else:
				q_raw, t_raw, h_raw = model.get_queries_targets(ss[start:end], rs[start:end], ts[start:end])
				ent_raw = self._entity_uniformity_vectors(model)
				loss, l_align, l_unif, _, _, margin_active = self._au_loss_with_distinct_keys(
					q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys)
				self._backward_au_loss(loss, fraction, use_amp=False)
			chunk = end - start
			loss_sum += loss.item() * chunk
			align_sum += l_align.item() * chunk
			unif_sum += l_unif.item() * chunk
			margin_acc += margin_active
			margin_batches += 1

		if use_amp:
			self.scaler.unscale_(self.optimizer)
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
			self.scaler.step(self.optimizer)
			self.scaler.update()
		else:
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
			self.optimizer.step()

		avg_margin = (margin_acc / margin_batches) if margin_batches > 0 else 0.0
		return loss_sum / total, align_sum / total, unif_sum / total, n_uq_log, n_ut_log, avg_margin, total

	def _extract_monitor_value(self, metric_dict, valid_metric='mrr') -> float | None:
		"""Extract the value to monitor for checkpointing decisions from the metric dictionary."""

		if not metric_dict:
			return None
		if valid_metric in metric_dict:
			return metric_dict[valid_metric]
		if 'loss' in metric_dict:
			return -metric_dict['loss']
		for value in metric_dict.values():
			if isinstance(value, (int, float)):
				return value
		return None

	def _resolve_link_prediction_path(self, path: str) -> str:
		"""Resolve a raw link-prediction split from a labeled validation/test path."""

		if not path:
			return ''

		candidates = [path]
		base_dir = os.path.dirname(path)
		parent_dir = os.path.dirname(base_dir)
		basename = os.path.basename(path)

		if '_w_label' in basename:
			stripped = basename.replace('_w_label', '')
			candidates.extend([
				os.path.join(base_dir, stripped),
				os.path.join(parent_dir, stripped),
			])
			if stripped.endswith('.json'):
				stripped_txt = stripped[:-5]
				candidates.extend([
					os.path.join(base_dir, stripped_txt),
					os.path.join(parent_dir, stripped_txt),
				])

		for candidate in candidates:
			if candidate and os.path.exists(candidate):
				return candidate
		return ''

	def _validation_eval_path(self) -> str:
		"""Determine the path to use for validation link prediction."""

		for candidate in [
			self._resolve_link_prediction_path(getattr(self.args, 'valid_path', '')),
			getattr(self.args, 'valid_path', ''),
		]:
			if candidate and os.path.exists(candidate):
				return candidate
		return ''

	def _average_metric_dict(self, forward_metrics: dict, backward_metrics: dict) -> dict:
		"""Average matching numeric metrics from forward and backward evaluation."""

		if not forward_metrics or not backward_metrics:
			return forward_metrics or backward_metrics or {}

		averaged_metrics = {}
		for key in forward_metrics.keys() & backward_metrics.keys():
			forward_value = forward_metrics[key]
			backward_value = backward_metrics[key]
			if isinstance(forward_value, (int, float)) and isinstance(backward_value, (int, float)):
				averaged_metrics[key] = (forward_value + backward_value) / 2
		return averaged_metrics

	def _format_link_metrics(self, epoch: int, direction: str, metrics: dict) -> str:
		"""Format link prediction metrics for a single direction."""

		return (
			f'[EPOCH {epoch}] Valid ({direction}) | '
			f'MR: {metrics.get("mr", metrics.get("mean_rank", 0.0)):.4f} | '
			f'MRR: {metrics.get("mrr", 0.0):.4f} | '
			f'H@1: {metrics.get("hit@1", metrics.get("hits@1", 0.0)):.4f} | '
			f'H@3: {metrics.get("hit@3", metrics.get("hits@3", 0.0)):.4f} | '
			f'H@10: {metrics.get("hit@10", metrics.get("hits@10", 0.0)):.4f}'
		)

	def _format_avg_link_metrics(self, epoch: int, metrics: dict) -> str:
		"""Format averaged validation link prediction metrics."""

		return (
			f'[EPOCH {epoch}] Valid (Avg) | '
			f'MR: {metrics.get("mr", metrics.get("mean_rank", 0.0)):.4f} | '
			f'MRR: {metrics.get("mrr", 0.0):.4f} | '
			f'H@1: {metrics.get("hit@1", metrics.get("hits@1", 0.0)):.4f} | '
			f'H@3: {metrics.get("hit@3", metrics.get("hits@3", 0.0)):.4f} | '
			f'H@10: {metrics.get("hit@10", metrics.get("hits@10", 0.0)):.4f}'
		)

	def train_epoch(self, epoch) -> float:
		"""Train the model for one epoch and return the average training loss."""

		self.model.train()
		epoch_loss = 0.0
		epoch_align_loss = 0.0
		epoch_unif_loss = 0.0
		epoch_unique_q = 0.0
		epoch_unique_t = 0.0
		epoch_margin_active = 0.0
		epoch_batches = 0
		batch_size = max(getattr(self.args, 'batch_size', 1024), 1)
		model = get_model_obj(self.model)
		use_amp = bool(getattr(self.args, 'use_amp', False))

		if self.uses_text_inputs:
			losses = AverageMeter('Loss', ':.4')
			progress = ProgressMeter(len(self.train_loader), [losses], prefix='Epoch: [{}]'.format(epoch))
			for i, batch_dict in enumerate(self.train_loader):
				self.model.train()
				if torch.cuda.is_available():
					batch_dict = move_to_cuda(batch_dict)
				self.optimizer.zero_grad()
				q_keys, t_keys, h_keys = self._uniformity_keys_from_examples(batch_dict['batch_data'])
				if use_amp:
					with torch.amp.autocast(device_type='cuda'):
						outputs = self.model(**batch_dict)
						q_raw = outputs['hr_vector']
						t_raw = outputs['tail_vector']
						h_raw = outputs['head_vector']
						ent_raw = self._entity_uniformity_vectors(model)
						loss, l_align, l_unif, n_uq, n_ut, margin_active = self._au_loss_with_distinct_keys(
							q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys)
					self.scaler.scale(loss).backward()
					self.scaler.unscale_(self.optimizer)
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
					self.scaler.step(self.optimizer)
					self.scaler.update()
				else:
					outputs = self.model(**batch_dict)
					q_raw = outputs['hr_vector']
					t_raw = outputs['tail_vector']
					h_raw = outputs['head_vector']
					ent_raw = self._entity_uniformity_vectors(model)
					loss, l_align, l_unif, n_uq, n_ut, margin_active = self._au_loss_with_distinct_keys(
						q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys)
					loss.backward()
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
					self.optimizer.step()
				batch_examples = len(batch_dict['batch_data'])
				losses.update(loss.item(), batch_examples)
				epoch_align_loss += l_align.item() * batch_examples
				epoch_unif_loss += l_unif.item() * batch_examples
				epoch_loss += loss.item() * batch_examples
				epoch_unique_q += n_uq
				epoch_unique_t += n_ut
				epoch_margin_active += margin_active
				epoch_batches += 1
				if i % self.args.print_freq == 0:
					progress.display(i)
		else:
			for batch_idx, (ss, rs, ts) in enumerate(
				self._iter_batches(self.train_src, self.train_rel, self.train_dst, batch_size, shuffle=True),
			):
				loss, l_align, l_unif, n_uq, n_ut, margin_active, n_examples = self._train_au_tensor_batch(
					model, ss, rs, ts, use_amp,
				)
				epoch_align_loss += l_align * n_examples
				epoch_unif_loss += l_unif * n_examples
				epoch_loss += loss * n_examples
				epoch_unique_q += n_uq
				epoch_unique_t += n_ut
				epoch_margin_active += margin_active
				epoch_batches += 1

		avg_count = max(len(self.train_examples), 1)
		avg_loss = epoch_loss / avg_count
		avg_align_loss = epoch_align_loss / avg_count
		avg_unif_loss = epoch_unif_loss / avg_count
		display_epoch = epoch + 1
		if epoch_batches > 0:
			avg_unique_q = epoch_unique_q / epoch_batches
			avg_unique_t = epoch_unique_t / epoch_batches
			avg_margin_active = epoch_margin_active / epoch_batches
		else:
			avg_unique_q = avg_unique_t = avg_margin_active = 0.0
		if float(self.criterion.additive_margin) > 0.0:
			logger.info(
				'[EPOCH %s] train loss: %.6f | align: %.6f | uniformity: %.6f | '
				'avg unique q/t per batch: %.0f/%.0f (of %d) | margin-buffer pairs: %.2f%%',
				display_epoch, avg_loss, avg_align_loss, avg_unif_loss,
				avg_unique_q, avg_unique_t, batch_size, 100.0 * avg_margin_active,
			)
		else:
			logger.info(
				'[EPOCH %s] train loss: %.6f | align: %.6f | uniformity: %.6f | '
				'avg unique q/t per batch: %.0f/%.0f (of %d)',
				display_epoch, avg_loss, avg_align_loss, avg_unif_loss,
				avg_unique_q, avg_unique_t, batch_size,
			)
		self.train_component_losses = {
			'loss': avg_loss,
			'align': avg_align_loss,
			'uniformity': avg_unif_loss,
			'avg_unique_q': avg_unique_q,
			'avg_unique_t': avg_unique_t,
		}
		if float(self.criterion.additive_margin) > 0.0:
			self.train_component_losses['margin_buffer_pair_frac'] = avg_margin_active
		return avg_loss

	@torch.no_grad()
	def eval_epoch(self, epoch, train_loss=None) -> dict:
		"""Evaluate the model on the validation set and return a dictionary of metrics."""

		metric_dict = {}
		valid_eval_path = self._validation_eval_path()
		display_epoch = epoch + 1
		if valid_eval_path:
			valid_entity_dict = get_entity_dict()
			valid_output_path = os.path.join(self.args.output_dir, 'valid_link_prediction.log')
			forward_metrics = self.evaluate_link_prediction_inplace(
				self.model, valid_eval_path, valid_entity_dict, valid_output_path, eval_forward=True)
			backward_metrics = self.evaluate_link_prediction_inplace(
				self.model, valid_eval_path, valid_entity_dict, valid_output_path, eval_forward=False)
			if forward_metrics:
				logger.info(self._format_link_metrics(display_epoch, 'Fwd', forward_metrics))
			if backward_metrics:
				logger.info(self._format_link_metrics(display_epoch, 'Bwd', backward_metrics))
			if forward_metrics and backward_metrics:
				avg_metrics = self._average_metric_dict(forward_metrics, backward_metrics)
				logger.info(self._format_avg_link_metrics(display_epoch, avg_metrics))
				for key, value in avg_metrics.items():
					if isinstance(value, (int, float)):
						metric_dict[key] = round(value, 4)
			elif forward_metrics:
				metric_dict.update(forward_metrics)
			elif backward_metrics:
				metric_dict.update(backward_metrics)
		else:
			logger.warning('[EPOCH %s] No validation link-prediction split found; skipping valid LP metrics', display_epoch)
			if train_loss is not None:
				metric_dict['loss'] = round(train_loss, 4)
		return metric_dict

	def train_loop(self) -> dict:
		"""Execute the full training loop over multiple epochs, including checkpointing and timing."""

		if self.args.use_amp:
			self.scaler = torch.amp.GradScaler('cuda')

		validation_interval = self._validation_interval()
		logger.info('KGAU validation interval: every %d epoch(s)', validation_interval)

		total_start_time = time.time()
		for epoch in range(self.args.epochs):
			epoch_train_start = time.time()
			train_loss = self.train_epoch(epoch)
			self.train_time += time.time() - epoch_train_start

			if self._should_validate(epoch):
				eval_start = time.time()
				metric_dict = self.eval_epoch(epoch, train_loss=train_loss)
				self.valid_time += time.time() - eval_start
			else:
				metric_dict = {'loss': round(train_loss, 4)}

			if not metric_dict:
				metric_dict = {'loss': round(train_loss, 4)}

			monitor_value = self._extract_monitor_value(metric_dict)
			is_best = monitor_value is not None and (self.best_metric is None or monitor_value > self.best_metric.get('score', float('-inf')))
			if is_best:
				self.best_metric = {'score': monitor_value, 'metrics': metric_dict, 'epoch': epoch}

			filename = checkpoint_path(self.args.output_dir, epoch)
			saved_checkpoint_path = save_checkpoint({
				'epoch': epoch,
				'best_epoch': epoch if is_best else None,
				'best_metric': self.best_metric,
				'args': self.args.__dict__,
				'state_dict': get_model_obj(self.model).state_dict(),
			}, is_best=is_best, filename=filename)
			if is_best:
				self.best_checkpoint_path = best_model_path(self.args.output_dir)
			elif self.best_checkpoint_path is None:
				self.best_checkpoint_path = saved_checkpoint_path
			delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.output_dir), keep=self.args.max_to_keep)

		self.total_time = time.time() - total_start_time
		logger.info('[Timing] Training time (s): %.2f', round(self.train_time, 2))
		logger.info('[Timing] Valid time (s): %.2f', round(self.valid_time, 2))
		logger.info('[Timing] Total run time (s): %.2f', round(self.total_time, 2))

		return {
			'best_epoch': None if self.best_metric is None else self.best_metric.get('epoch'),
			'best_mrr': None if self.best_metric is None else self.best_metric.get('score'),
			'train_time': self.train_time,
			'valid_time': self.valid_time,
			'total_time': self.total_time,
		}

Strategy = KGAUStrategy
