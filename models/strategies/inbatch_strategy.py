"""In-batch contrastive training paradigm (SimKGC-style)."""

import json
import os

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from base.evaluator import Evaluator
from data.dataset import Dataset
from data.dataloader import collate
from data.dict_hub import build_tokenizer, get_entity_dict
from metrics.ranking import topk_accuracy as accuracy
from models.builder import import_module_from_path, init_index_kge_trainer, load_attr_from_path, load_loss_fn, run_kge_train_loop
from utils.checkpoint import best_model_path, last_model_path, save_checkpoint
from utils.device import get_model_obj, move_to_cuda, report_num_trainable_parameters
from utils.training_cadence import init_step_cadence_state
from utils.memory import PhaseMemoryTracker, format_memory


class InBatchStrategy(Evaluator):
	"""Train with in-batch negatives: broadcast queries against batch targets only."""

	def __init__(self, model, sampler, loss_fn, args, ngpus_per_node=1, **_kwargs):
		del sampler
		Evaluator.__init__(self)
		self.args = args
		self.ngpus_per_node = ngpus_per_node
		build_tokenizer(args)

		if model is None:
			from models.builder import build_model

			model = build_model(args)
		logger.info(model)

		from data.dict_hub import warmup_data_structures

		warmup_data_structures()
		init_index_kge_trainer(self, model, args)
		self.criterion = loss_fn if loss_fn is not None else load_loss_fn(args)
		self._load_loss_and_sampler_helpers(args)

		self.optimizer = AdamW(
			[p for p in self.model.parameters() if p.requires_grad],
			lr=args.lr,
			weight_decay=float(args.weight_decay) if args.weight_decay is not None else 0.0,
		)
		report_num_trainable_parameters(self.model)

		train_dataset = Dataset(path=args.train_path, task=args.dataset)
		valid_dataset = Dataset(path=args.valid_path, task=args.dataset) if args.valid_path else None
		num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size, 1)
		args.warmup = min(args.warmup, num_training_steps // 10)
		logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
		self.scheduler = self._create_lr_scheduler(num_training_steps)

		self.train_loader = torch.utils.data.DataLoader(
			train_dataset,
			batch_size=args.batch_size,
			shuffle=True,
			collate_fn=collate,
			num_workers=args.workers,
			pin_memory=True,
			drop_last=True,
		)

		self.valid_loader = None
		if valid_dataset:
			self.valid_loader = torch.utils.data.DataLoader(
				valid_dataset,
				batch_size=args.batch_size * 2,
				shuffle=True,
				collate_fn=collate,
				num_workers=args.workers,
				pin_memory=True,
			)

		if args.use_amp:
			self.scaler = torch.cuda.amp.GradScaler()

		init_step_cadence_state(self)

	def _load_loss_and_sampler_helpers(self, args) -> None:
		loss_path = getattr(args, 'model_loss_path', '') or 'models/losses/infonce_loss.py'
		try:
			self.ModelOutput = load_attr_from_path(loss_path, 'ModelOutput')
			self.compute_infonce_logits = load_attr_from_path(loss_path, 'compute_infonce_logits')
		except Exception:
			loss_mod = import_module_from_path(loss_path)
			self.ModelOutput = getattr(loss_mod, 'ModelOutput')
			self.compute_infonce_logits = getattr(loss_mod, 'compute_infonce_logits')

		sampler_path = getattr(args, 'model_sampler_path', '') or 'models/samplers/masking_sampler.py'
		try:
			self.construct_mask = load_attr_from_path(sampler_path, 'construct_mask')
		except Exception:
			sampler_mod = import_module_from_path(sampler_path)
			self.construct_mask = getattr(sampler_mod, 'construct_mask')

	def _create_lr_scheduler(self, num_training_steps):
		if self.args.lr_scheduler == 'linear':
			return get_linear_schedule_with_warmup(
				optimizer=self.optimizer,
				num_warmup_steps=self.args.warmup,
				num_training_steps=num_training_steps,
			)
		if self.args.lr_scheduler == 'cosine':
			return get_cosine_schedule_with_warmup(
				optimizer=self.optimizer,
				num_warmup_steps=self.args.warmup,
				num_training_steps=num_training_steps,
			)
		raise ValueError(f'Unknown lr scheduler: {self.args.lr_scheduler}')

	def train_loop(self) -> dict:
		return run_kge_train_loop(self)

	def _compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
		model = get_model_obj(self.model)
		hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
		batch_size = hr_vector.size(0)
		labels = torch.arange(batch_size, device=hr_vector.device)

		logits = self.compute_infonce_logits(
			query_vec=hr_vector,
			candidate_vec=tail_vector,
			temp=model.log_inv_t,
			margin=model.add_margin if model.training else 0.0,
		)

		triplet_mask = batch_dict.get('triplet_mask', None)
		if triplet_mask is not None:
			logits.masked_fill_(~triplet_mask.to(hr_vector.device), -1e4)

		if model.pre_batch > 0 and model.training:
			pre_batch_logits = self._compute_pre_batch_logits(model, hr_vector, tail_vector, batch_dict)
			logits = torch.cat([logits, pre_batch_logits], dim=-1)

		if self.args.use_self_negative and model.training:
			head_vector = output_dict['head_vector']
			self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * model.log_inv_t.exp()
			self_negative_mask = batch_dict.get('self_negative_mask', None)
			if self_negative_mask is None:
				self_negative_mask = torch.ones(batch_size, dtype=torch.bool, device=hr_vector.device)
			else:
				self_negative_mask = self_negative_mask.to(hr_vector.device).bool()
			self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
			logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

		return {
			'logits': logits,
			'labels': labels,
			'inv_t': model.log_inv_t.detach().exp(),
			'hr_vector': hr_vector.detach(),
			'tail_vector': tail_vector.detach(),
		}

	def _compute_pre_batch_logits(self, model, hr_vector: torch.Tensor, tail_vector: torch.Tensor, batch_dict: dict) -> torch.Tensor:
		assert tail_vector.size(0) == model.batch_size
		batch_exs = batch_dict['batch_data']
		pre_batch_logits = self.compute_infonce_logits(hr_vector, model.pre_batch_vectors.clone(), model.log_inv_t)
		pre_batch_logits *= model.args.pre_batch_weight
		if model.pre_batch_exs[-1] is not None:
			pre_triplet_mask = self.construct_mask(batch_exs, model.pre_batch_exs).to(hr_vector.device)
			pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

		model.pre_batch_vectors[model.offset:(model.offset + model.batch_size)] = tail_vector.data.clone()
		model.pre_batch_exs[model.offset:(model.offset + model.batch_size)] = batch_exs
		model.offset = (model.offset + model.batch_size) % len(model.pre_batch_exs)

		return pre_batch_logits

	def _compute_infonce_loss(
		self,
		logits: torch.Tensor,
		labels: torch.Tensor,
		batch_size: int,
		*,
		symmetric: bool,
	) -> torch.Tensor:
		loss = self.criterion(logits, labels)
		if symmetric:
			loss = loss + self.criterion(logits[:, :batch_size].t(), labels)
		return loss

	def train_epoch(self, epoch) -> None:
		losses = AverageMeter('Loss', ':.4')
		top1 = AverageMeter('Acc@1', ':6.2f')
		top3 = AverageMeter('Acc@3', ':6.2f')
		inv_t = AverageMeter('InvT', ':6.2f')
		progress = ProgressMeter(
			len(self.train_loader),
			[losses, inv_t, top1, top3],
			prefix='Epoch: [{}]'.format(epoch),
		)

		from utils.training_cadence import (
			increment_trainer_global_step,
			maybe_decay_lr_at_step,
			resolve_max_steps,
			resolve_valid_steps,
			should_stop_at_step,
		)

		eval_interval = resolve_valid_steps(self.args)
		if eval_interval is None:
			eval_interval = getattr(self.args, 'eval_every_n_step', None)

		for i, batch_dict in enumerate(self.train_loader):
			self.model.train()

			if torch.cuda.is_available():
				batch_dict = move_to_cuda(batch_dict)
			batch_size = len(batch_dict['batch_data'])

			if self.args.use_amp:
				with torch.cuda.amp.autocast():
					outputs = self.model(**batch_dict)
			else:
				outputs = self.model(**batch_dict)

			outputs = self._compute_logits(output_dict=outputs, batch_dict=batch_dict)
			outputs = self.ModelOutput(**outputs)
			logits, labels = outputs.logits, outputs.labels
			assert logits.size(0) == batch_size

			loss = self._compute_infonce_loss(logits, labels, batch_size, symmetric=True)

			acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
			top1.update(acc1.item(), batch_size)
			top3.update(acc3.item(), batch_size)
			inv_t.update(outputs.inv_t, 1)
			losses.update(loss.item(), batch_size)

			self.optimizer.zero_grad()
			grad_clip = getattr(self.args, 'grad_clip', None)
			if self.args.use_amp:
				self.scaler.scale(loss).backward()
				self.scaler.unscale_(self.optimizer)
				if grad_clip is not None:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
				self.scaler.step(self.optimizer)
				self.scaler.update()
			else:
				loss.backward()
				if grad_clip is not None:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
				self.optimizer.step()
			self.scheduler.step()

			step = increment_trainer_global_step(self)
			maybe_decay_lr_at_step(self, step)

			if i % self.args.print_freq == 0:
				progress.display(i)
			if eval_interval and step % int(eval_interval) == 0:
				self._run_eval(epoch=epoch, step=step)
			if resolve_max_steps(self.args) is not None and should_stop_at_step(self, step):
				self._stop_training = True
				break

		logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))
		log_str = f"[EPOCH {epoch}] Loss: {losses.avg:.4f} | Acc@1: {top1.avg:.2f} | Acc@3: {top3.avg:.2f}"
		logger.info(log_str)

	@torch.no_grad()
	def eval_epoch(self, epoch) -> dict:
		metric_dict = {}
		if self.valid_loader:
			losses = AverageMeter('Loss', ':.4')
			top1 = AverageMeter('Acc@1', ':6.2f')
			top3 = AverageMeter('Acc@3', ':6.2f')

			for _, batch_dict in enumerate(self.valid_loader):
				self.model.eval()
				if torch.cuda.is_available():
					batch_dict = move_to_cuda(batch_dict)
				batch_size = len(batch_dict['batch_data'])

				outputs = self.model(**batch_dict)
				outputs = self._compute_logits(output_dict=outputs, batch_dict=batch_dict)
				outputs = self.ModelOutput(**outputs)
				logits, labels = outputs.logits, outputs.labels
				loss = self._compute_infonce_loss(logits, labels, batch_size, symmetric=False)
				losses.update(loss.item(), batch_size)

				acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
				top1.update(acc1.item(), batch_size)
				top3.update(acc3.item(), batch_size)

			metric_dict.update({
				'Acc@1': round(top1.avg, 3),
				'Acc@3': round(top3.avg, 3),
				'loss': round(losses.avg, 3),
			})

		valid_eval_path = None
		if self.args.valid_path:
			if self.args.valid_path.endswith('_w_label.txt'):
				cand_txt = self.args.valid_path.replace('valid_w_label.txt', 'valid.txt')
				cand_json = self.args.valid_path.replace('valid_w_label.txt', 'valid.txt.json')
				if os.path.exists(cand_json):
					valid_eval_path = cand_json
				elif os.path.exists(cand_txt):
					valid_eval_path = cand_txt
			elif self.args.valid_path.endswith('.txt.json') or self.args.valid_path.endswith('.txt'):
				valid_eval_path = self.args.valid_path

		if valid_eval_path and os.path.exists(valid_eval_path):
			valid_entity_dict = get_entity_dict()
			valid_output_path = os.path.join(self.args.output_dir, 'valid_link_prediction.log')
			forward_metrics = self.evaluate_link_prediction_inplace(
				self.model, valid_eval_path, valid_entity_dict, valid_output_path, eval_forward=True)
			backward_metrics = self.evaluate_link_prediction_inplace(
				self.model, valid_eval_path, valid_entity_dict, valid_output_path, eval_forward=False)
			if forward_metrics and backward_metrics:
				metric_dict['mrr'] = round((forward_metrics.get('mrr', 0) + backward_metrics.get('mrr', 0)) / 2, 4)
				logger.info(f"[EPOCH {epoch}] Validation link-pred MRR(avg): {metric_dict['mrr']}")

		if metric_dict:
			logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
		return metric_dict

	@torch.no_grad()
	def _run_eval(self, epoch, step=0) -> dict:
		eval_start = __import__('time').time()
		self.memory_tracker.begin_phase()
		metric_dict = self.eval_epoch(epoch)
		self.memory_tracker.end_phase('eval')
		self.valid_time += __import__('time').time() - eval_start
		monitor_value = self._extract_monitor_value(metric_dict)
		is_best = monitor_value is not None and (
			self.best_metric is None or monitor_value > self.best_metric.get('score', float('-inf'))
		)
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
		return metric_dict

	def _extract_monitor_value(self, metric_dict, valid_metric='mrr'):
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


SimKGCStrategy = InBatchStrategy
Strategy = InBatchStrategy
