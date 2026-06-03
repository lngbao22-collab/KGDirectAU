"""Training strategy for listwise DistMult and ComplEx models."""

import os
import time
import torch
from torch.optim import Adam

from base.evaluator import Evaluator
from data.dataset import Example, load_data, reverse_triplet
from data.dict_hub import get_entity_dict
from models.losses.infonce_loss import compute_listwise_loss
from utils.checkpoint import best_model_path, delete_old_ckt, last_model_path, save_checkpoint
from utils.device import get_model_obj
from utils.logger import logger


class SoftmaxStrategy(object):
	"""Listwise softmax training loop for KG encoders."""

	def __init__(self, encoder, sampler, args, num_train_triples, train_data=None):
		self.encoder = encoder
		self.model = encoder
		self.sampler = sampler
		self.args = args
		self.train_data = train_data
		self.entity_dict = get_entity_dict()
		self.entity_ids = [entity_ex.entity_id for entity_ex in self.entity_dict.entity_exs]
		self.valid_path = getattr(args, 'valid_path', '')
		self.evaluator = Evaluator(args)
		self.best_metric = None
		self.best_checkpoint_path = None
		self.train_time = 0.0
		self.valid_time = 0.0
		self.total_time = 0.0

		if torch.cuda.is_available():
			self.model.cuda()
		self.device = next(self.model.parameters()).device

		batch_size = max(getattr(args, 'batch_size', 1), 1)
		num_batches = max(num_train_triples // batch_size, 1)
		weight_decay = getattr(args, 'weight_decay', None)
		if weight_decay is None:
			weight_decay = 0.0
		self.weight_decay = float(weight_decay) / num_batches
		self.optimizer = Adam(self.model.parameters(), lr=getattr(args, 'lr', 2e-5), weight_decay=self.weight_decay)

	def _iter_batches(self, src, rel, dst, batch_size):
		"""Yield batches of triples from the provided tensors."""

		for start in range(0, len(src), batch_size):
			end = start + batch_size
			yield src[start:end], rel[start:end], dst[start:end]

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

	def _format_link_metrics(self, epoch: int, direction: str, metrics: dict) -> str:
		"""Format link prediction metrics for a single direction."""

		return (
			f"[EPOCH {epoch}] Valid ({direction}) | "
			f"MR: {metrics.get('mr', metrics.get('mean_rank', 0.0)):.4f} | "
			f"MRR: {metrics.get('mrr', 0.0):.4f} | "
			f"H@1: {metrics.get('hit@1', metrics.get('hits@1', 0.0)):.4f} | "
			f"H@3: {metrics.get('hit@3', metrics.get('hits@3', 0.0)):.4f} | "
			f"H@10: {metrics.get('hit@10', metrics.get('hits@10', 0.0)):.4f}"
		)

	def _format_avg_metrics(self, epoch: int, metrics: dict) -> str:
		"""Format averaged validation metrics for logging."""

		return (
			f"[EPOCH {epoch}] Valid (Avg) | Loss: {metrics.get('loss', 0.0):.4f} | "
			f"MR: {metrics.get('mr', metrics.get('mean_rank', 0.0)):.4f} | "
			f"MRR: {metrics.get('mrr', 0.0):.4f} | "
			f"H@1: {metrics.get('hit@1', metrics.get('hits@1', 0.0)):.4f} | "
			f"H@3: {metrics.get('hit@3', metrics.get('hits@3', 0.0)):.4f} | "
			f"H@10: {metrics.get('hit@10', metrics.get('hits@10', 0.0)):.4f}\n"
		)

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

	def train_epoch(self, epoch):
		self.model.train()
		batch_size = max(getattr(self.args, 'batch_size', 1), 1)
		if self.train_data is None:
			raise ValueError('SoftmaxStrategy requires train_data to train an epoch')
		src, rel, dst = self.train_data
		src_corrupted, rel_corrupted, dst_corrupted = self.sampler.corrupt(src, rel, dst)
		epoch_loss = 0.0
		for ss, rs, ts in self._iter_batches(src_corrupted, rel_corrupted, dst_corrupted, batch_size):
			ss = ss.to(self.device)
			rs = rs.to(self.device)
			ts = ts.to(self.device)
			scores = self.model.forward(ss, rs, ts)
			truth = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
			loss = compute_listwise_loss(scores, truth)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			epoch_loss += loss.item() * ss.size(0)

		display_epoch = epoch + 1
		logger.info(f"[EPOCH {display_epoch}] Train | Loss: {epoch_loss / max(len(src), 1):.4f}")
		return epoch_loss / max(len(src), 1)

	@torch.no_grad()
	def eval_epoch(self, epoch):
		metric_dict = {}
		valid_eval_path = self._resolve_link_prediction_path(self.valid_path)
		if not valid_eval_path or not os.path.exists(valid_eval_path):
			valid_eval_path = self._resolve_label_path(self.valid_path)

		if valid_eval_path and os.path.exists(valid_eval_path):
			valid_exs = load_data(valid_eval_path, add_forward_triplet=True, add_backward_triplet=False)
			valid_backward_exs = [
				Example(**reverse_triplet({
					'head_id': ex.head_id,
					'head': ex.head,
					'relation': ex.relation,
					'tail_id': ex.tail_id,
					'tail': ex.tail,
				}))
				for ex in valid_exs
			]
			if valid_exs:
				epoch_loss = 0.0
				batch_size = max(getattr(self.args, 'batch_size', 1), 1)
				for i in range(0, len(valid_exs), batch_size):
					batch = valid_exs[i:i + batch_size]
					scores = self.model.score_batch(
						[ex.head_id for ex in batch],
						[ex.relation for ex in batch],
						self.entity_ids,
					)
					if not isinstance(scores, torch.Tensor):
						scores = torch.tensor(scores, device=self.device)
					tail_indices = torch.tensor(
						[self.entity_dict.entity_to_idx(ex.tail_id) for ex in batch],
						dtype=torch.long,
						device=scores.device,
					)
					loss = compute_listwise_loss(scores, tail_indices)
					epoch_loss += loss.item() * len(batch)
				metric_dict['loss'] = epoch_loss / max(len(valid_exs), 1)

			valid_output_path = os.path.join(self.args.output_dir, 'valid_link_prediction.log')
			display_epoch = epoch + 1
			forward_metrics = self.evaluator.evaluate_link_prediction_inplace(
				self.model, valid_eval_path, self.entity_dict, valid_output_path, eval_forward=True, examples=valid_exs)
			backward_metrics = self.evaluator.evaluate_link_prediction_inplace(
				self.model, valid_eval_path, self.entity_dict, valid_output_path, eval_forward=False, examples=valid_backward_exs)
			logger.info(self._format_link_metrics(display_epoch, 'Fwd', forward_metrics))
			logger.info(self._format_link_metrics(display_epoch, 'Bwd', backward_metrics))
			metric_dict.update(self._average_metric_dict(forward_metrics, backward_metrics))
			avg_metrics = dict(metric_dict)
			avg_metrics['loss'] = metric_dict.get('loss', 0.0)
			logger.info(self._format_avg_metrics(display_epoch, avg_metrics))

		return {key: metric_dict[key] for key in ('loss', 'mrr') if key in metric_dict}

	def train_loop(self):
		total_start = time.time()
		for epoch in range(getattr(self.args, 'epochs', 1)):
			epoch_start = time.time()
			train_loss = self.train_epoch(epoch)
			self.train_time += time.time() - epoch_start
			eval_start = time.time()
			metric_dict = self.eval_epoch(epoch)
			self.valid_time += time.time() - eval_start
			if not metric_dict:
				metric_dict = {'loss': round(train_loss, 4)}
			monitor_value = self._extract_monitor_value(metric_dict)
			is_best = monitor_value is not None and (self.best_metric is None or monitor_value > self.best_metric.get('score', float('-inf')))
			if is_best:
				self.best_metric = {'score': monitor_value, 'metrics': metric_dict, 'epoch': epoch}
			saved_checkpoint_path = save_checkpoint({
				'epoch': epoch,
				'best_epoch': epoch if is_best else None,
				'best_metric': self.best_metric,
				'args': self.args.__dict__,
				'state_dict': get_model_obj(self.model).state_dict(),
			}, is_best=is_best, filename=last_model_path(self.args.output_dir))
			if is_best:
				self.best_checkpoint_path = best_model_path(self.args.output_dir)
			elif self.best_checkpoint_path is None:
				self.best_checkpoint_path = saved_checkpoint_path
			delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.output_dir), keep=getattr(self.args, 'max_to_keep', 5))

		self.total_time = time.time() - total_start
		return {
			'best_epoch': None if self.best_metric is None else self.best_metric.get('epoch', 0) + 1,
			'best_mrr': None if self.best_metric is None else self.best_metric.get('score'),
			'train_time': self.train_time,
			'valid_time': self.valid_time,
			'total_time': self.total_time,
			'best_checkpoint_path': self.best_checkpoint_path,
		}
