"""Training strategy for KGAU."""

from __future__ import annotations

import json
import os
import time

import torch
from torch.optim import Adam

from base.evaluator import Evaluator
from data.dataset import load_data
from data.dict_hub import get_entity_dict
from models.builder import load_attr_from_path
from utils.checkpoint import best_model_path, checkpoint_path, delete_old_ckt, save_checkpoint
from utils.device import get_model_obj, report_num_trainable_parameters
from utils.logger import logger


def _relation_path_candidates(args) -> list[str]:
	"""Return a list of candidate paths for loading the relation-to-index mapping."""

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
	"""Load the relation-to-index mapping from candidate paths or construct it from training data."""

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


def _load_encoder(args) -> torch.nn.Module:
	"""Factory helper used by the evaluator to rebuild the model from checkpoints."""

	encoder_path = getattr(args, 'model_encoder_path', '') or 'models/encoders/distmult_encoder.py'
	build_model = load_attr_from_path(encoder_path, 'build_model')
	return build_model(args)


class KGAUStrategy(Evaluator):
	"""Knowledge Graph Alignment and Uniformity training loop for KG encoders."""

	def __init__(self, args, ngpus_per_node):
		super().__init__(args)
		self.ngpus_per_node = ngpus_per_node
		self.entity_dict = get_entity_dict()
		self.relation_to_idx = _load_relation_to_idx(args)
		self.model = _load_encoder(args)
		logger.info('=> creating model')
		logger.info(self.model)

		if torch.cuda.device_count() > 1:
			self.model = torch.nn.DataParallel(self.model).cuda()
		elif torch.cuda.is_available():
			self.model.cuda()
		self.device = next(self.model.parameters()).device

		report_num_trainable_parameters(get_model_obj(self.model))

		lam = getattr(args, 'lam', getattr(args, 'weight_decay', 0.0))
		n_batch = getattr(args, 'n_batch', getattr(args, 'batch_size', 1))
		self.weight_decay = lam / max(n_batch, 1)
		self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=self.weight_decay)

		from models.losses.au_loss import KGAULoss

		# Support multiple config names: `tuni` preferred, fall back to `temperature` or `t`.
		tuni_val = getattr(args, 'tuni', getattr(args, 'temperature', getattr(args, 't', 2.0)))

		self.criterion = KGAULoss(
			gamma_q=getattr(args, 'gamma_q', 1.0),
			gamma_t=getattr(args, 'gamma_t', 1.0),
			gamma_h=getattr(args, 'gamma_h', 0.5),
			gamma_ent=getattr(args, 'gamma_ent', 0.0),
			tuni=tuni_val,
		).to(self.device)

		self.train_examples = load_data(args.train_path, add_forward_triplet=False, add_backward_triplet=False)
		self.train_src, self.train_rel, self.train_dst = self._examples_to_tensors(self.train_examples)
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

	def _iter_batches(self, src, rel, dst, batch_size) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Iterate over batches of examples."""

		for start in range(0, len(src), batch_size):
			end = start + batch_size
			yield src[start:end], rel[start:end], dst[start:end]

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

	def _validation_eval_path(self) -> str:
		"""Determine the path to use for validation evaluation."""

		candidates = [getattr(self.args, 'valid_path', '')]
		if self.args.valid_label_path:
			label_dir = os.path.dirname(self.args.valid_label_path)
			candidates.extend([
				os.path.join(label_dir, 'valid.txt.json'),
				os.path.join(label_dir, 'valid.txt'),
			])
		for candidate in candidates:
			if candidate and os.path.exists(candidate):
				return candidate
		return ''

	def train_epoch(self, epoch) -> float:
		"""Train the model for one epoch and return the average training loss."""

		self.model.train()
		epoch_loss = 0.0
		batch_size = getattr(self.args, 'n_batch', getattr(self.args, 'batch_size', 1024))
		model = get_model_obj(self.model)

		for ss, rs, ts in self._iter_batches(self.train_src, self.train_rel, self.train_dst, batch_size):
			ss = ss.to(self.device)
			rs = rs.to(self.device)
			ts = ts.to(self.device)
			self.optimizer.zero_grad()
			q_raw, t_raw, h_raw = model.get_queries_targets(ss, rs, ts)
			ent_raw = model.entity_embeddings(device=self.device) if getattr(self.criterion, 'gamma_ent', 0.0) > 0 else None
			loss, _, _ = self.criterion(q_raw, t_raw, h_raw, ent_raw)
			loss.backward()
			self.optimizer.step()
			epoch_loss += loss.item() * ss.size(0)

		avg_loss = epoch_loss / max(len(self.train_src), 1)
		logger.info('[EPOCH %s] train loss: %.6f', epoch, avg_loss)
		return avg_loss

	@torch.no_grad()
	def eval_epoch(self, epoch, train_loss=None) -> dict:
		"""Evaluate the model on the validation set and return a dictionary of metrics."""

		metric_dict = {}
		valid_eval_path = self._validation_eval_path()
		if valid_eval_path:
			valid_entity_dict = get_entity_dict()
			valid_output_path = os.path.join(self.args.model_dir, 'valid_link_prediction.log')
			forward_metrics = self.evaluate_link_prediction_inplace(
				self.model, valid_eval_path, valid_entity_dict, valid_output_path, eval_forward=True)
			backward_metrics = self.evaluate_link_prediction_inplace(
				self.model, valid_eval_path, valid_entity_dict, valid_output_path, eval_forward=False)
			if forward_metrics and backward_metrics:
				metric_dict['mrr'] = round((forward_metrics.get('mrr', 0) + backward_metrics.get('mrr', 0)) / 2, 4)
				logger.info('[EPOCH %s] valid mrr(avg): %s', epoch, metric_dict['mrr'])
		else:
			if train_loss is not None:
				metric_dict['loss'] = round(train_loss, 4)
		return metric_dict

	def train_loop(self) -> dict:
		"""Execute the full training loop over multiple epochs, including checkpointing and timing."""

		if self.args.use_amp:
			logger.info('AMP is ignored for KGDirectAU and will not be used.')

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

			filename = checkpoint_path(self.args.model_dir, epoch)
			saved_checkpoint_path = save_checkpoint({
				'epoch': epoch,
				'best_epoch': epoch if is_best else None,
				'best_metric': self.best_metric,
				'args': self.args.__dict__,
				'state_dict': get_model_obj(self.model).state_dict(),
			}, is_best=is_best, filename=filename)
			if is_best:
				self.best_checkpoint_path = best_model_path(self.args.model_dir)
			elif self.best_checkpoint_path is None:
				self.best_checkpoint_path = saved_checkpoint_path
			delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir), keep=self.args.max_to_keep)

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
