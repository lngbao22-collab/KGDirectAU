"""Training strategy for KGAU."""

from __future__ import annotations

import os
import math
import time
from typing import Iterator

import torch
from torch.optim import Adam

from base.evaluator import Evaluator
from data.dataloader import collate
from data.dataset import Dataset, load_data
from data.dict_hub import build_tokenizer, get_entity_dict, get_relation_id_map
from models.builder import load_attr_from_path
from utils.checkpoint import best_model_path, checkpoint_path, delete_old_ckt, save_checkpoint
from utils.device import get_model_obj, move_to_cuda, report_num_trainable_parameters
from utils.logger import AverageMeter, ProgressMeter, logger
from models.losses.au_loss import KGAULoss, select_distinct_rows


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
		self.entity_dict = get_entity_dict()
		self.model = _load_encoder(args)
		logger.info(self.model)
		if not self.uses_text_inputs:
			self.relation_to_idx = {str(k): int(v) for k, v in get_relation_id_map().items()}
			self.train_src, self.train_rel, self.train_dst = self._examples_to_tensors(self.train_examples)
		else:
			self.relation_to_idx = {str(k): int(v) for k, v in get_relation_id_map().items()}
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

		report_num_trainable_parameters(get_model_obj(self.model))

		weight_decay = getattr(args, 'weight_decay', None)
		if weight_decay is None:
			weight_decay = 0.0
		batch_size = max(getattr(args, 'batch_size', 1), 1)
		num_batches = max(math.ceil(len(self.train_examples) / batch_size), 1)
		self.weight_decay = float(weight_decay) / num_batches
		self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=self.weight_decay)

		tuni_val = _config_float(args, 'tuni', _config_float(args, 'temperature', _config_float(args, 't', 2.0)))

		self.criterion = KGAULoss(
			gamma_q=_config_float(args, 'gamma_q', 1.0),
			gamma_t=_config_float(args, 'gamma_t', 1.0),
			gamma_h=_config_float(args, 'gamma_h', 0.0),
			gamma_ent=_config_float(args, 'gamma_ent', 0.0),
			tuni=tuni_val,
			max_uniformity_samples=int(_config_float(args, 'max_uniformity_samples', 1024)),
		).to(self.device)
		self.best_metric = None
		self.best_checkpoint_path = None
		self.train_time = 0.0
		self.valid_time = 0.0
		self.total_time = 0.0

	def _examples_to_tensors(self, examples) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Convert a list of examples into tensors of head, relation, and tail indices."""

		head_indices = torch.tensor([self.entity_dict.entity_to_idx(example.head_id) for example in examples], dtype=torch.long)
		relation_indices = torch.tensor([self.relation_to_idx[example.relation] for example in examples], dtype=torch.long)
		tail_indices = torch.tensor([self.entity_dict.entity_to_idx(example.tail_id) for example in examples], dtype=torch.long)
		return head_indices, relation_indices, tail_indices

	def _iter_batches(self, src, rel, dst, batch_size) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
		"""Iterate over batches of examples."""

		for start in range(0, len(src), batch_size):
			end = start + batch_size
			yield src[start:end], rel[start:end], dst[start:end]

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
		return self._uniformity_keys(head_indices, relation_indices, tail_indices)

	def _distinct_uniformity_inputs(
		self,
		q_raw: torch.Tensor,
		t_raw: torch.Tensor,
		h_raw: torch.Tensor,
		q_keys: torch.Tensor,
		t_keys: torch.Tensor,
		h_keys: torch.Tensor,
	) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
		"""Deduplicate embeddings per uniformity type before AU uniformity loss."""

		q_uni = select_distinct_rows(q_raw, q_keys) if self.criterion.gamma_q > 0 else None
		t_uni = select_distinct_rows(t_raw, t_keys) if self.criterion.gamma_t > 0 else None
		h_uni = select_distinct_rows(h_raw, h_keys) if self.criterion.gamma_h > 0 else None
		return q_uni, t_uni, h_uni

	def _au_loss(
		self,
		q_raw: torch.Tensor,
		t_raw: torch.Tensor,
		h_raw: torch.Tensor,
		ent_raw: torch.Tensor | None,
		q_keys: torch.Tensor,
		t_keys: torch.Tensor,
		h_keys: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Compute KGAU loss; alignment uses the full batch, uniformity uses distinct keys."""

		q_uni, t_uni, h_uni = self._distinct_uniformity_inputs(q_raw, t_raw, h_raw, q_keys, t_keys, h_keys)
		return self.criterion(q_raw, t_raw, h_raw, ent_raw, q_uni=q_uni, t_uni=t_uni, h_uni=h_uni)

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
						ent_raw = model.entity_embeddings(device=self.device) if self.criterion.gamma_ent > 0 else None
						loss, l_align, l_unif = self._au_loss(q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys)
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
					ent_raw = model.entity_embeddings(device=self.device) if self.criterion.gamma_ent > 0 else None
					loss, l_align, l_unif = self._au_loss(q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys)
					loss.backward()
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
					self.optimizer.step()
				losses.update(loss.item(), len(batch_dict['batch_data']))
				epoch_align_loss += l_align.item() * len(batch_dict['batch_data'])
				epoch_unif_loss += l_unif.item() * len(batch_dict['batch_data'])
				epoch_loss += loss.item() * len(batch_dict['batch_data'])
				if i % self.args.print_freq == 0:
					progress.display(i)
		else:
			for ss, rs, ts in self._iter_batches(self.train_src, self.train_rel, self.train_dst, batch_size):
				ss = ss.to(self.device)
				rs = rs.to(self.device)
				ts = ts.to(self.device)
				self.optimizer.zero_grad()
				q_keys, t_keys, h_keys = self._uniformity_keys(ss, rs, ts)
				if use_amp:
					with torch.amp.autocast(device_type='cuda'):
						q_raw, t_raw, h_raw = model.get_queries_targets(ss, rs, ts)
						ent_raw = model.entity_embeddings(device=self.device) if self.criterion.gamma_ent > 0 else None
						loss, l_align, l_unif = self._au_loss(q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys)
					self.scaler.scale(loss).backward()
					self.scaler.unscale_(self.optimizer)
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
					self.scaler.step(self.optimizer)
					self.scaler.update()
				else:
					q_raw, t_raw, h_raw = model.get_queries_targets(ss, rs, ts)
					ent_raw = model.entity_embeddings(device=self.device) if self.criterion.gamma_ent > 0 else None
					loss, l_align, l_unif = self._au_loss(q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys)
					loss.backward()
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
					self.optimizer.step()
				epoch_align_loss += l_align.item() * ss.size(0)
				epoch_unif_loss += l_unif.item() * ss.size(0)
				epoch_loss += loss.item() * ss.size(0)

		avg_count = max(len(self.train_examples), 1)
		avg_loss = epoch_loss / avg_count
		avg_align_loss = epoch_align_loss / avg_count
		avg_unif_loss = epoch_unif_loss / avg_count
		display_epoch = epoch + 1
		logger.info('[EPOCH %s] train loss: %.6f | align: %.6f | uniformity: %.6f', display_epoch, avg_loss, avg_align_loss, avg_unif_loss)
		self.train_component_losses = {
			'loss': avg_loss,
			'align': avg_align_loss,
			'uniformity': avg_unif_loss,
		}
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

		total_start_time = time.time()
		for epoch in range(self.args.epochs):
			epoch_train_start = time.time()
			train_loss = self.train_epoch(epoch)
			self.train_time += time.time() - epoch_train_start

			eval_start = time.time()
			metric_dict = self.eval_epoch(epoch, train_loss=train_loss)
			self.valid_time += time.time() - eval_start

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
			'best_checkpoint_path': self.best_checkpoint_path,
		}

Strategy = KGAUStrategy
