"""Training strategy for listwise DistMult and ComplEx models."""

import os
import time
from typing import Callable, Optional

import torch
from torch.optim import Adam

from base.evaluator import Evaluator
from configs.config import args
from data.dataset import load_data
from data.dict_hub import get_entity_dict, get_relation_id_map
from models.losses.infonce_loss import compute_listwise_loss
from models.builder import import_module_from_path, load_attr_from_path
from models.samplers.bernoulli_sampler import BernoulliListwiseSampler
from utils.checkpoint import best_model_path, delete_old_ckt, last_model_path, save_checkpoint
from utils.device import get_model_obj
from utils.logger import logger


class SoftmaxStrategy(object):
	"""Listwise softmax training loop for KG encoders."""

	def __init__(self, encoder, args):
		self.encoder = encoder
		self.args = args
		lam = getattr(args, 'lam', getattr(args, 'weight_decay', 0.0))
		n_batch = getattr(args, 'n_batch', getattr(args, 'batch_size', 1))
		self.weight_decay = lam / max(n_batch, 1)
		self.optimizer = Adam(self.encoder.parameters(), weight_decay=self.weight_decay)
		self.device = next(self.encoder.parameters()).device

	def _iter_batches(self, src, rel, dst, batch_size) -> (torch.Tensor, torch.Tensor, torch.Tensor):
		"""Yield batches of triples from the provided tensors."""

		for start in range(0, len(src), batch_size):
			end = start + batch_size
			yield src[start:end], rel[start:end], dst[start:end]

	def pretrain(self, train_data, sampler, tester: Optional[Callable[[], float]] = None) -> float:
		"""Train the encoder with Bernoulli listwise negatives."""

		src, rel, dst = train_data
		n_train = len(src)
		n_epoch = getattr(self.args, 'n_epoch', getattr(self.args, 'epochs', 1))
		n_batch = getattr(self.args, 'n_batch', getattr(self.args, 'batch_size', 1))
		sample_freq = getattr(self.args, 'sample_freq', 1)
		best_perf = 0.0

		for epoch in range(n_epoch):
			if epoch % sample_freq == 0:
				src_corrupted, rel_corrupted, dst_corrupted = sampler.corrupt(src, rel, dst)
			else:
				src_corrupted, rel_corrupted, dst_corrupted = src, rel, dst

			epoch_loss = 0.0
			for ss, rs, ts in self._iter_batches(src_corrupted, rel_corrupted, dst_corrupted, n_batch):
				ss = ss.to(self.device)
				rs = rs.to(self.device)
				ts = ts.to(self.device)
				scores = self.encoder.forward(ss, rs, ts)
				truth = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
				loss = compute_listwise_loss(scores, truth)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				epoch_loss += loss.item() * ss.size(0)

			if tester is not None:
				perf = tester()
				if perf > best_perf:
					best_perf = perf

		return best_perf if tester is not None else epoch_loss / max(n_train, 1)


class DistMultTrainer:
	"""Concrete trainer for DistMult and ComplEx style encoders."""

	def __init__(self, train_args, ngpus_per_node):
		self.args = train_args
		self.ngpus_per_node = ngpus_per_node
		self.best_metric = None
		self.best_checkpoint_path = None
		self.train_time = 0.0
		self.valid_time = 0.0
		self.total_time = 0.0

		entity_dict = get_entity_dict()
		relation_id_map = get_relation_id_map()
		self.entity_ids = [entity_ex.entity_id for entity_ex in entity_dict.entity_exs]
		self.entity_to_idx = entity_dict.entity_to_idx
		self.relation_to_idx = relation_id_map

		encoder_path = getattr(train_args, 'model_encoder_path', '') or 'models/encoders/distmult_encoder.py'
		try:
			build_model = load_attr_from_path(encoder_path, 'build_model')
		except Exception:
			mod = import_module_from_path(encoder_path)
			build_model = getattr(mod, 'build_model')

		self.model = build_model(train_args)
		if torch.cuda.is_available():
			self.model.cuda()

		self.optimizer = Adam(self.model.parameters(), lr=train_args.lr, weight_decay=train_args.weight_decay)
		self.train_examples = load_data(train_args.train_path, add_forward_triplet=True, add_backward_triplet=False)
		if not self.train_examples:
			raise ValueError(f'No training examples loaded from {train_args.train_path}')
		self.train_tensors = self._examples_to_tensors(self.train_examples)
		self.sampler = BernoulliListwiseSampler(self.train_tensors, len(entity_dict), max(len(relation_id_map), 1), getattr(train_args, 'n_sample', getattr(train_args, 'batch_size', 1)))
		self.valid_path = getattr(train_args, 'valid_path', '')
		self.evaluator = Evaluator(train_args)

	def _examples_to_tensors(self, examples):
		src = torch.tensor([self.entity_to_idx(ex.head_id) for ex in examples], dtype=torch.long)
		rel = torch.tensor([self.relation_to_idx.get(ex.relation, self.relation_to_idx.get(ex.relation.replace('inverse ', ''), 0)) for ex in examples], dtype=torch.long)
		dst = torch.tensor([self.entity_to_idx(ex.tail_id) for ex in examples], dtype=torch.long)
		return src, rel, dst

	def _resolve_label_path(self, path: str) -> str:
		"""Resolve a labeled TSV file from a preprocessed JSON path when needed."""

		if not path:
			return path
		if path.endswith('.json'):
			parent_dir = os.path.dirname(os.path.dirname(path))
			candidate = os.path.join(parent_dir, os.path.basename(path)[:-5])
			if os.path.exists(candidate):
				return candidate
			candidate = os.path.join(os.path.dirname(path), os.path.basename(path)[:-5])
			if os.path.exists(candidate):
				return candidate
		return path

	def _resolve_link_prediction_path(self, path: str) -> str:
		"""Resolve a raw link-prediction split from a labeled validation/test path."""

		if not path:
			return path

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
		return path

	def _average_metric_dict(self, forward_metrics, backward_metrics):
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

	def _move_model(self):
		if torch.cuda.is_available():
			self.model.cuda()

	def _batch_iter(self, src, rel, dst, batch_size):
		for start in range(0, len(src), batch_size):
			end = start + batch_size
			yield src[start:end], rel[start:end], dst[start:end]

	def train_epoch(self, epoch):
		self.model.train()
		n_batch = getattr(self.args, 'batch_size', 1)
		src, rel, dst = self.train_tensors
		src_corrupted, rel_corrupted, dst_corrupted = self.sampler.corrupt(src, rel, dst)
		epoch_loss = 0.0
		for ss, rs, ts in self._batch_iter(src_corrupted, rel_corrupted, dst_corrupted, n_batch):
			if torch.cuda.is_available():
				ss = ss.cuda(non_blocking=True)
				rs = rs.cuda(non_blocking=True)
				ts = ts.cuda(non_blocking=True)
			scores = self.model.forward(ss, rs, ts)
			truth = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
			loss = compute_listwise_loss(scores, truth)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			epoch_loss += loss.item() * ss.size(0)

		log_str = f"[EPOCH {epoch}] Loss: {epoch_loss / max(len(src), 1):.4f}"
		print(log_str)
		logger.info(log_str)

	@torch.no_grad()
	def eval_epoch(self, epoch):
		metric_dict = {}
		valid_eval_path = self._resolve_link_prediction_path(self.valid_path)
		if not valid_eval_path or not os.path.exists(valid_eval_path):
			valid_eval_path = self._resolve_label_path(self.valid_path)

		if valid_eval_path and os.path.exists(valid_eval_path):
			valid_exs = load_data(valid_eval_path, add_forward_triplet=True, add_backward_triplet=False)
			if valid_exs:
				epoch_loss = 0.0
				batch_size = max(getattr(self.args, 'batch_size', 1), 1)
				for i in range(0, len(valid_exs), batch_size):
					batch = valid_exs[i:i + batch_size]
					scores = self.model.score_batch(
						[ex.head_id for ex in batch],
						[ex.relation for ex in batch],
						[ex.tail_id for ex in batch],
					)
					if not isinstance(scores, torch.Tensor):
						scores = torch.tensor(scores)
					truth = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
					loss = compute_listwise_loss(scores, truth)
					epoch_loss += loss.item() * len(batch)
				metric_dict['loss'] = epoch_loss / max(len(valid_exs), 1)

			valid_entity_dict = get_entity_dict()
			valid_output_path = os.path.join(self.args.model_dir, 'valid_link_prediction.log')
			forward_metrics = self.evaluator.evaluate_link_prediction_inplace(
				self.model, valid_eval_path, valid_entity_dict, valid_output_path, eval_forward=True)
			backward_metrics = self.evaluator.evaluate_link_prediction_inplace(
				self.model, valid_eval_path, valid_entity_dict, valid_output_path, eval_forward=False)
			metric_dict.update(self._average_metric_dict(forward_metrics, backward_metrics))

		metric_dict = {key: metric_dict[key] for key in ('loss', 'mrr') if key in metric_dict}

		if metric_dict:
			logger.info('Epoch {}, valid metric: {}'.format(epoch, metric_dict))
		return metric_dict

	def _extract_monitor_value(self, metric_dict):
		if not metric_dict:
			return None
		for key in ('mrr', 'loss'):
			if key in metric_dict:
				return metric_dict[key] if key != 'loss' else -metric_dict[key]
		for value in metric_dict.values():
			if isinstance(value, (int, float)):
				return value
		return None

	def train_loop(self):
		total_start = time.time()
		if torch.cuda.is_available():
			self._move_model()
		for epoch in range(self.args.epochs):
			epoch_start = time.time()
			self.train_epoch(epoch)
			self.train_time += time.time() - epoch_start
			eval_start = time.time()
			metric_dict = self.eval_epoch(epoch)
			self.valid_time += time.time() - eval_start
			monitor_value = self._extract_monitor_value(metric_dict)
			is_best = monitor_value is not None and (self.best_metric is None or monitor_value > self.best_metric.get('score', float('-inf')))
			if is_best:
				self.best_metric = {'score': monitor_value, 'metrics': metric_dict, 'epoch': epoch}
			saved_checkpoint_path = save_checkpoint({
				'epoch': epoch,
				'best_epoch': epoch if is_best else None,
				'best_metric': self.best_metric,
				'args': self.args.__dict__,
				'state_dict': self.model.state_dict(),
			}, is_best=is_best, filename=last_model_path(self.args.model_dir))
			if is_best:
				self.best_checkpoint_path = best_model_path(self.args.model_dir)
			elif self.best_checkpoint_path is None:
				self.best_checkpoint_path = saved_checkpoint_path
		self.total_time = time.time() - total_start
		return {
			'best_epoch': None if self.best_metric is None else self.best_metric.get('epoch'),
			'best_mrr': None if self.best_metric is None else self.best_metric.get('score'),
			'train_time': self.train_time,
			'valid_time': self.valid_time,
			'total_time': self.total_time,
			'best_checkpoint_path': self.best_checkpoint_path,
		}