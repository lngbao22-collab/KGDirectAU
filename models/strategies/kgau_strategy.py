"""Training strategy for KGAU."""

from __future__ import annotations

import inspect
import os
import math
import time
from typing import Iterator

import torch
from torch import optim
from torch.optim import Adam

from base.embeddings import use_reciprocal_relations
from base.evaluator import Evaluator, log_bidirectional_link_metrics
from data.dataloader import collate
from data.dataset import Dataset, load_data
from data.dict_hub import get_entity_dict, get_relation_id_map
from base.embeddings import compute_kge_regularization, embedding_l3_penalty
from models.builder import build_lr_scheduler, config_bool, load_attr_from_path, step_lr_scheduler
from utils.checkpoint import best_model_path, checkpoint_path, delete_old_ckt, last_model_path, save_checkpoint
from utils.device import get_model_obj, move_to_cuda, report_num_trainable_parameters
from utils.logger import AverageMeter, ProgressMeter, logger, time_per_train_epoch
from utils.memory import PhaseMemoryTracker, format_memory
from utils.training_cadence import (
	init_step_cadence_state,
	increment_trainer_global_step,
	maybe_decay_lr_at_step,
)
from models.losses.au_loss import KGAULoss, distinct_first_indices, select_distinct_rows, _GAMMA_NAMES


def _load_encoder(args) -> torch.nn.Module:
	from models.builder import build_model

	return build_model(args)


def _uses_text_inputs(args, model=None) -> bool:
	if model is not None:
		from utils.device import get_model_obj

		return getattr(get_model_obj(model), 'training_input_mode', 'indices') == 'tokens'
	embedder_path = str(getattr(args, 'model_embedder_path', '') or '').replace('\\', '/')
	if embedder_path.endswith('text_embedder.py'):
		return True
	scorer_path = str(getattr(args, 'model_scorer_path', '') or getattr(args, 'model_encoder_path', '') or '')
	return os.path.basename(scorer_path) in {'bert_encoder.py', 'simkgc_scorer.py'}


def _config_float(args, name: str, default: float) -> float:
	"""Read a float hyperparameter from args, treating JSON null as unset."""

	value = getattr(args, name, None)
	return default if value is None else float(value)


def _is_dabr_encoder(args) -> bool:
	"""Return True when the configured model is DaBR (or DaBR-AU)."""

	scorer_path = str(
		getattr(args, 'model_scorer_path', '') or getattr(args, 'model_encoder_path', '') or ''
	).lower()
	model_name = str(getattr(args, 'model', '') or '').lower()
	return 'dabr' in scorer_path or 'dabr' in model_name


def _build_relation_to_idx() -> dict[str, int]:
	"""Build the relation->index map with distinct IDs for inverse relations."""

	from base.embeddings import add_inverse_relations

	base = {str(key): int(value) for key, value in get_relation_id_map().items()}
	return add_inverse_relations(base)


def _build_optimizer(args, parameters, weight_decay: float):
	"""Build the training optimizer; respects ``optim`` in config (adam/adagrad/sgd)."""

	lr = float(getattr(args, 'lr', getattr(args, 'learning_rate', 2e-5)))
	optim_name = str(getattr(args, 'optim', 'adam')).lower()
	if optim_name == 'adagrad':
		return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
	if optim_name == 'sgd':
		return optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
	return Adam(parameters, lr=lr, weight_decay=weight_decay)


def _resolve_learnable_au_flags(args) -> tuple[bool, bool]:
	"""Return (learnable_au_alpha, learnable_au_gammas); ``learnable_au_scales`` enables both."""

	if config_bool(args, 'learnable_au_scales', False):
		return True, True
	return (
		config_bool(args, 'learnable_au_alpha', False),
		config_bool(args, 'learnable_au_gammas', False),
	)


def _build_kgau_optimizer(args, model, criterion: KGAULoss, weight_decay: float):
	"""Build optimizer over model + KGAU loss params (optional learnable ``tuni`` / gammas)."""

	lr = float(getattr(args, 'lr', getattr(args, 'learning_rate', 2e-5)))
	base_params = [p for p in model.parameters() if p.requires_grad]
	log_tuni_param = None
	log_alpha_param = None
	log_gamma_params = []
	aux_other_params = []
	for name, param in criterion.named_parameters():
		if not param.requires_grad:
			continue
		if name == 'log_tuni' or name.endswith('.log_tuni'):
			log_tuni_param = param
		elif name == 'log_alpha_adj' or name.endswith('.log_alpha_adj'):
			log_alpha_param = param
		elif name.startswith('log_gamma_adj_'):
			log_gamma_params.append(param)
		else:
			aux_other_params.append(param)

	param_groups = []
	base_group_params = base_params + aux_other_params
	if base_group_params:
		param_groups.append({'params': base_group_params, 'lr': lr, 'weight_decay': weight_decay})
	if log_tuni_param is not None:
		log_tuni_lr = _config_float(args, 'log_uniformity_lr', lr)
		param_groups.append({'params': [log_tuni_param], 'lr': log_tuni_lr, 'weight_decay': 0.0})
	if log_alpha_param is not None:
		log_alpha_lr = _config_float(args, 'log_au_alpha_lr', lr)
		param_groups.append({'params': [log_alpha_param], 'lr': log_alpha_lr, 'weight_decay': 0.0})
	if log_gamma_params:
		log_gamma_lr = _config_float(args, 'log_au_gamma_lr', lr)
		param_groups.append({
			'params': log_gamma_params,
			'lr': log_gamma_lr,
			'weight_decay': 0.0,
			'param_group': 'log_gamma',
		})

	if not param_groups:
		return _build_optimizer(args, model.parameters(), weight_decay)

	optim_name = str(getattr(args, 'optim', 'adam')).lower()
	if optim_name == 'adagrad':
		return optim.Adagrad(param_groups)
	if optim_name == 'sgd':
		return optim.SGD(param_groups)
	return Adam(param_groups)


def _tuni_scalar(criterion: KGAULoss) -> float:
	"""Return the current uniformity scale as a Python float for logging."""

	value = criterion.tuni
	if torch.is_tensor(value):
		return float(value.detach().cpu().item())
	return float(value)


def _scheduled_tuni_value(args, epoch: int) -> float:
	"""Linear tuni schedule: ``start`` at ``start_epoch``, ``end`` at the last scheduled epoch."""

	start_epoch = int(getattr(args, 'tuni_schedule_start_epoch', 0) or 0)
	tuni_default = _config_float(args, 'tuni', _config_float(args, 'temperature', _config_float(args, 't', 2.0)))
	start_raw = getattr(args, 'tuni_schedule_start', None)
	end_raw = getattr(args, 'tuni_schedule_end', None)
	start_scale = tuni_default if start_raw is None else float(start_raw)
	end_scale = tuni_default if end_raw is None else float(end_raw)

	schedule_epochs = int(getattr(args, 'tuni_schedule_epochs', 0) or 0)
	if schedule_epochs <= 0:
		schedule_span = max(1, int(getattr(args, 'epochs', 1)) - 1 - start_epoch)
	else:
		schedule_span = max(1, schedule_epochs - 1)

	if epoch < start_epoch:
		return start_scale
	progress = min(1.0, max(0.0, (epoch - start_epoch) / float(schedule_span)))
	return start_scale + (end_scale - start_scale) * progress


def _gamma_schedule_enabled(args) -> bool:
	if getattr(args, 'gamma_linear_schedule', None) is not None:
		return bool(args.gamma_linear_schedule)
	return (
		config_bool(args, 'learnable_au_gammas', False)
		or config_bool(args, 'learnable_au_scales', False)
	)


def _alpha_schedule_enabled(args) -> bool:
	if getattr(args, 'alpha_linear_schedule', None) is not None:
		return bool(args.alpha_linear_schedule)
	return (
		config_bool(args, 'learnable_au_alpha', False)
		or config_bool(args, 'learnable_au_scales', False)
	)


def _gamma_schedule_span(args) -> tuple[int, int]:
	"""Return ``(start_epoch, schedule_span)`` shared by gamma multiplier and LR schedules."""

	start_epoch = int(getattr(args, 'gamma_schedule_start_epoch', 0) or 0)
	schedule_epochs = int(getattr(args, 'gamma_schedule_epochs', 0) or 0)
	if schedule_epochs <= 0:
		schedule_span = max(1, int(getattr(args, 'epochs', 1)) - 1 - start_epoch)
	else:
		schedule_span = max(1, schedule_epochs - 1)
	return start_epoch, schedule_span


def _gamma_schedule_progress(args, epoch: int) -> float:
	start_epoch, schedule_span = _gamma_schedule_span(args)
	if epoch < start_epoch:
		return 0.0
	return min(1.0, max(0.0, (epoch - start_epoch) / float(schedule_span)))


def _scheduled_gamma_mult(args, epoch: int) -> float:
	"""Linear gamma multiplier: 1.0 at start, ``gamma_schedule_end`` at the last scheduled epoch."""

	end_mult = float(getattr(args, 'gamma_schedule_end', 0.1))
	progress = _gamma_schedule_progress(args, epoch)
	return 1.0 + (end_mult - 1.0) * progress


def _log_au_gamma_lr_schedule_enabled(args) -> bool:
	"""Whether to linearly ramp ``log_au_gamma_lr`` toward ``gamma_schedule_end``."""

	_, learnable_au_gammas = _resolve_learnable_au_flags(args)
	if not learnable_au_gammas:
		return False
	if getattr(args, 'log_au_gamma_lr_linear_schedule', None) is not None:
		return bool(args.log_au_gamma_lr_linear_schedule)
	return _gamma_schedule_enabled(args)


def _scheduled_log_au_gamma_lr(args, epoch: int) -> float:
	"""Linear ``log_au_gamma_lr``: initial config value at start, ``gamma_schedule_end`` at the last scheduled epoch."""

	lr = _config_float(args, 'lr', getattr(args, 'learning_rate', 2e-5))
	start_lr = _config_float(args, 'log_au_gamma_lr', lr)
	end_lr = float(getattr(args, 'gamma_schedule_end', start_lr))
	progress = _gamma_schedule_progress(args, epoch)
	return start_lr + (end_lr - start_lr) * progress


def _set_log_au_gamma_lr(optimizer, lr: float) -> bool:
	"""Update the learnable-gamma param group LR; return True when that group exists."""

	for group in optimizer.param_groups:
		if group.get('param_group') == 'log_gamma':
			group['lr'] = float(lr)
			return True
	return False


def _scheduled_alpha_mult(args, epoch: int) -> float:
	"""Linear alpha multiplier: 1.0 at start, ``alpha_schedule_end`` at the last scheduled epoch."""

	start_epoch = int(getattr(args, 'alpha_schedule_start_epoch', 0) or 0)
	end_mult = float(getattr(args, 'alpha_schedule_end', 10.0))
	schedule_epochs = int(getattr(args, 'alpha_schedule_epochs', 0) or 0)
	if schedule_epochs <= 0:
		schedule_span = max(1, int(getattr(args, 'epochs', 1)) - 1 - start_epoch)
	else:
		schedule_span = max(1, schedule_epochs - 1)

	if epoch < start_epoch:
		return 1.0
	progress = min(1.0, max(0.0, (epoch - start_epoch) / float(schedule_span)))
	return 1.0 + (end_mult - 1.0) * progress


def _gamma_log_suffix(criterion: KGAULoss) -> str:
	parts = []
	if (
		criterion.tuni_as_alpha
		or criterion.learnable_au_alpha
		or criterion.alpha_schedule_mult_value() != 1.0
	):
		parts.append(f'alpha={criterion.alpha_value():.4f}')
		if criterion.alpha_schedule_mult_value() != 1.0:
			parts.append(f'alpha_mult={criterion.alpha_schedule_mult_value():.4f}')
	if criterion.learnable_au_gammas or criterion.gamma_schedule_mult_value() != 1.0:
		parts.append(f'mult={criterion.gamma_schedule_mult_value():.4f}')
	for name in _GAMMA_NAMES:
		if criterion.gamma_active(name):
			parts.append(f'{name}={criterion.gamma_value(name):.4f}')
	if not parts:
		return ''
	return ' | gammas: ' + ', '.join(parts)


def _build_text_train_loader(args, train_examples) -> torch.utils.data.DataLoader:
	"""Build the tokenized training loader (SimKGC/BERT only)."""

	from data.dict_hub import build_tokenizer, init_dataloader_worker, warmup_data_structures

	build_tokenizer(args)
	warmup_data_structures()

	train_workers = int(getattr(args, 'workers', 0))
	train_loader_kwargs = {
		'dataset': Dataset(path='', examples=train_examples, task=args.dataset),
		'batch_size': max(getattr(args, 'batch_size', 1), 1),
		'shuffle': True,
		'collate_fn': collate,
		'num_workers': train_workers,
		'pin_memory': True,
		'drop_last': True,
	}
	if train_workers > 0:
		train_loader_kwargs['worker_init_fn'] = init_dataloader_worker
		train_loader_kwargs['persistent_workers'] = True
	return torch.utils.data.DataLoader(**train_loader_kwargs)


def _entity_embeddings_call_kwargs(model, device: torch.device, criterion: KGAULoss) -> dict:
	"""Build ``entity_embeddings`` kwargs supported by the configured encoder."""

	supported = inspect.signature(model.entity_embeddings).parameters
	kwargs: dict = {}
	if 'device' in supported:
		kwargs['device'] = device
	if 'max_samples' in supported:
		max_samples = int(getattr(criterion, 'max_uniformity_samples', 0) or 0)
		kwargs['max_samples'] = max_samples or None
	return kwargs


class KGAUStrategy(Evaluator):
	"""Knowledge Graph Alignment and Uniformity training loop for KG encoders."""

	def __init__(self, model, sampler, loss_fn, args, ngpus_per_node=1, **_kwargs):
		del sampler, loss_fn
		super().__init__(args)
		self.ngpus_per_node = ngpus_per_node
		self.uses_text_inputs = _uses_text_inputs(args, model)
		self.kgau_bidirectional = (
			config_bool(args, 'kgau_bidirectional', False) and not self.uses_text_inputs
		)
		# GB-Magic-style head/tail batches use forward relation at eval; reciprocal triplets
		# are only needed for sp_inverse training (legacy KGAU path).
		add_backward_triplet = self.uses_text_inputs or (
			use_reciprocal_relations(args) and not self.kgau_bidirectional
		)
		self.train_examples = load_data(
			args.train_path, add_forward_triplet=True, add_backward_triplet=add_backward_triplet)
		logger.info(
			'Training examples: %d (backward triplets=%s, kgau_bidirectional=%s)',
			len(self.train_examples), add_backward_triplet, self.kgau_bidirectional,
		)
		self.entity_dict = get_entity_dict()
		self.model = model if model is not None else _load_encoder(args)
		self.uses_text_inputs = _uses_text_inputs(args, self.model)
		logger.info(self.model)
		self.relation_to_idx = _build_relation_to_idx()
		if not self.uses_text_inputs:
			self.train_src, self.train_rel, self.train_dst = self._examples_to_tensors(self.train_examples)
		else:
			self.train_loader = _build_text_train_loader(args, self.train_examples)

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
		self.au_deduplicate = config_bool(args, 'au_deduplicate', True)
		self.weight_decay = float(weight_decay) / num_batches

		tuni_val = _config_float(args, 'tuni', _config_float(args, 'temperature', _config_float(args, 't', 2.0)))
		learnable_tuni = config_bool(args, 'learnable_uniformity_scale', False)
		tuni_as_alpha = config_bool(args, 'tuni_as_alpha', False)
		learnable_au_alpha, learnable_au_gammas = _resolve_learnable_au_flags(args)
		if tuni_as_alpha and learnable_au_alpha:
			logger.warning('tuni_as_alpha is enabled; ignoring learnable_au_alpha')
			learnable_au_alpha = False

		# Alignment mode is opt-in: cosine by default for all encoders; only pRotatE-AU sets
		# ``sin_phase`` (via config and/or encoder ``kgau_alignment_mode``).
		model_obj = get_model_obj(self.model)
		encoder_align = getattr(model_obj, 'kgau_alignment_mode', None)
		alignment_mode = getattr(args, 'alignment_mode', None) or encoder_align or 'cosine'
		normalize_uniformity = getattr(args, 'normalize_uniformity', None)
		if normalize_uniformity is None:
			normalize_uniformity = alignment_mode not in ('phase_residual', 'sin_phase')
		if alignment_mode != 'cosine':
			logger.info('KGAU alignment mode: %s (normalize_uniformity=%s)', alignment_mode, normalize_uniformity)
		normalize_au = getattr(model_obj, 'normalize_au_vectors', None)
		normalize_lp = getattr(model_obj, 'normalize_lp_scores', None)
		if normalize_au is not None or normalize_lp is not None:
			logger.info(
				'KGAU scoring modes: normalize_au_vectors=%s (training), normalize_lp_scores=%s (link prediction)',
				normalize_au,
				normalize_lp,
			)
		self.criterion = KGAULoss(
			gamma_q=_config_float(args, 'gamma_q', 1.0),
			gamma_t=_config_float(args, 'gamma_t', 1.0),
			gamma_h=_config_float(args, 'gamma_h', 0.0),
			gamma_ent=_config_float(args, 'gamma_ent', 0.0),
			gamma_cross=_config_float(args, 'gamma_cross', 0.0),
			alpha=_config_float(args, 'alpha', 1.0),
			tuni=tuni_val,
			learnable_tuni=learnable_tuni,
			learnable_au_alpha=learnable_au_alpha,
			learnable_au_gammas=learnable_au_gammas,
			tuni_as_alpha=tuni_as_alpha,
			max_uniformity_samples=int(_config_float(args, 'max_uniformity_samples', 1024)),
			additive_margin=_config_float(args, 'additive_margin', 0.0),
			alignment_mode=alignment_mode,
			normalize_uniformity=bool(normalize_uniformity),
			average_uniformity_terms=config_bool(args, 'average_uniformity_terms', False),
			uniformity_full_pdist=config_bool(args, 'uniformity_full_pdist', False),
		).to(self.device)
		if config_bool(args, 'average_uniformity_terms', False):
			logger.info('KGAU average_uniformity_terms: enabled (sum active terms / count)')
		if config_bool(args, 'uniformity_full_pdist', False):
			logger.info('KGAU uniformity_full_pdist: full-batch torch.pdist (GB-Magic au style)')
		if learnable_tuni:
			if tuni_as_alpha:
				logger.info(
					'Learnable tuni (alignment scale + uniformity temperature): initial=%.4f, '
					'log_uniformity_lr=%.2e',
					tuni_val,
					_config_float(args, 'log_uniformity_lr', _config_float(args, 'lr', 2e-5)),
				)
			else:
				logger.info(
					'Learnable uniformity scale (tuni): initial=%.4f, log_uniformity_lr=%.2e',
					tuni_val,
					_config_float(args, 'log_uniformity_lr', _config_float(args, 'lr', 2e-5)),
				)
		elif tuni_as_alpha:
			logger.info('tuni scales alignment and uniformity (fixed): %.4f', tuni_val)
		elif config_bool(args, 'tuni_linear_schedule', False):
			start_scale = _scheduled_tuni_value(args, 0)
			end_scale = _scheduled_tuni_value(args, max(int(getattr(args, 'epochs', 1)) - 1, 0))
			logger.info(
				'Linear tuni schedule: epoch 0=%.4f -> last epoch=%.4f (start_epoch=%d)',
				start_scale,
				end_scale,
				int(getattr(args, 'tuni_schedule_start_epoch', 0) or 0),
			)
		if learnable_tuni and config_bool(args, 'tuni_linear_schedule', False):
			logger.warning('tuni_linear_schedule is ignored when learnable_uniformity_scale is enabled')
		if learnable_au_alpha or config_bool(args, 'learnable_au_scales', False):
			logger.info(
				'Learnable AU alpha: init=%.4f, schedule mult 1.0 -> %.4f, log_au_alpha_lr=%.2e',
				self.criterion.alpha_value(),
				float(getattr(args, 'alpha_schedule_end', 10.0)),
				_config_float(args, 'log_au_alpha_lr', _config_float(args, 'lr', 2e-5)),
			)
		if learnable_au_gammas:
			logger.info(
				'Learnable AU gammas: init q/t/h/ent/cross=%.4f/%.4f/%.4f/%.4f/%.4f, '
				'schedule mult 1.0 -> %.4f, log_au_gamma_lr=%.2e',
				self.criterion._gamma_init_value('q'),
				self.criterion._gamma_init_value('t'),
				self.criterion._gamma_init_value('h'),
				self.criterion._gamma_init_value('ent'),
				self.criterion._gamma_init_value('cross'),
				float(getattr(args, 'gamma_schedule_end', 0.1)),
				_config_float(args, 'log_au_gamma_lr', _config_float(args, 'lr', 2e-5)),
			)
		if _gamma_schedule_enabled(args):
			end_mult = float(getattr(args, 'gamma_schedule_end', 0.1))
			logger.info(
				'Gamma linear schedule: multiplier epoch 0=1.0000 -> last=%.4f (start_epoch=%d)',
				end_mult,
				int(getattr(args, 'gamma_schedule_start_epoch', 0) or 0),
			)
		if _log_au_gamma_lr_schedule_enabled(args):
			start_lr = _config_float(args, 'log_au_gamma_lr', _config_float(args, 'lr', 2e-5))
			end_lr = float(getattr(args, 'gamma_schedule_end', start_lr))
			logger.info(
				'log_au_gamma_lr linear schedule: epoch 0=%.2e -> last=%.2e (start_epoch=%d)',
				start_lr,
				end_lr,
				int(getattr(args, 'gamma_schedule_start_epoch', 0) or 0),
			)
		if _alpha_schedule_enabled(args):
			end_mult = float(getattr(args, 'alpha_schedule_end', 10.0))
			logger.info(
				'Alpha linear schedule: multiplier epoch 0=1.0000 -> last=%.4f (start_epoch=%d)',
				end_mult,
				int(getattr(args, 'alpha_schedule_start_epoch', 0) or 0),
			)
		self.optimizer = _build_kgau_optimizer(args, self.model, self.criterion, self.weight_decay)
		self.lr_scheduler = build_lr_scheduler(args, self.optimizer)
		init_step_cadence_state(self)
		warm_up_epochs = getattr(args, 'warm_up_epochs', None)
		if warm_up_epochs is not None and getattr(args, 'warm_up_steps', None) is None:
			num_batches = max(math.ceil(len(self.train_examples) / batch_size), 1)
			if self.kgau_bidirectional:
				num_batches *= 2
			self.next_lr_decay_step = int(warm_up_epochs) * num_batches
			logger.info(
				'warm_up_epochs=%d -> warm_up_steps=%d (%d batches/epoch)',
				int(warm_up_epochs),
				self.next_lr_decay_step,
				num_batches,
			)
		logger.info('KGAU au_deduplicate: %s', self.au_deduplicate)
		if config_bool(args, 'entity_uniformity_batch', False):
			logger.info('KGAU entity_uniformity_batch: batch cat(head, tail) (GB-Magic au style)')
		if self.kgau_bidirectional:
			logger.info(
				'KGAU kgau_bidirectional: tail-batch + head-batch per epoch (GB-Magic); '
				'head eval uses po_forward',
			)
		if self.criterion.gamma_active('ent') and self.uses_text_inputs:
			ent_mode = 'deduplicated' if self.au_deduplicate else 'all batch'
			logger.info(
				'Entity uniformity (text encoder): gamma_ent uses %s head+tail vectors '
				'(max_uniformity_samples=%d)',
				ent_mode,
				self.criterion.max_uniformity_samples,
			)
		self.best_metric = None
		self.best_checkpoint_path = None
		self.train_time = 0.0
		self.valid_time = 0.0
		self.total_time = 0.0
		self.memory_tracker = PhaseMemoryTracker()

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
		*,
		predict_head: bool = False,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Build deduplication keys for query, align-target, and head uniformity terms."""

		device = head_indices.device
		if predict_head:
			q_keys = torch.stack(
				[
					relation_indices.to(device=device, dtype=torch.long),
					tail_indices.to(device=device, dtype=torch.long),
				],
				dim=1,
			)
			t_keys = head_indices.to(device=device, dtype=torch.long)
		else:
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
		"""Select uniformity rows: deduplicated by id keys, or full batch (DirectAU-style)."""

		n_unique_q = q_raw.size(0) if self.criterion.gamma_active('q') else 0
		n_unique_t = t_raw.size(0) if self.criterion.gamma_active('t') else 0
		if not self.au_deduplicate:
			return None, None, None, n_unique_q, n_unique_t

		q_uni = select_distinct_rows(q_raw, q_keys) if self.criterion.gamma_active('q') else None
		t_uni = select_distinct_rows(t_raw, t_keys) if self.criterion.gamma_active('t') else None
		h_uni = select_distinct_rows(h_raw, h_keys) if self.criterion.gamma_active('h') else None
		n_unique_q = q_uni.size(0) if q_uni is not None else 0
		n_unique_t = t_uni.size(0) if t_uni is not None else 0
		return q_uni, t_uni, h_uni, n_unique_q, n_unique_t

	@staticmethod
	def _merge_cross_uniformity_vectors(
		q_uni: torch.Tensor | None,
		t_uni: torch.Tensor | None,
	) -> torch.Tensor | None:
		"""Pool deduplicated query and tail rows for cross uniformity (shared LP space)."""

		parts = [x for x in (q_uni, t_uni) if x is not None and x.size(0) > 0]
		if not parts:
			return None
		cross = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
		return cross if cross.size(0) >= 2 else None

	def _cross_uniformity_vectors(
		self,
		q_raw: torch.Tensor,
		t_raw: torch.Tensor,
		q_keys: torch.Tensor,
		t_keys: torch.Tensor,
	) -> torch.Tensor | None:
		"""Build pooled query+tail vectors for ``gamma_cross`` uniformity."""

		if not self.criterion.gamma_active('cross'):
			return None
		if self.au_deduplicate:
			q_uni = select_distinct_rows(q_raw, q_keys)
			t_uni = select_distinct_rows(t_raw, t_keys)
		else:
			q_uni, t_uni = q_raw, t_raw
		return self._merge_cross_uniformity_vectors(q_uni, t_uni)

	def _count_unique_uniformity_keys(
		self,
		head_indices: torch.Tensor,
		relation_indices: torch.Tensor,
		tail_indices: torch.Tensor,
		*,
		predict_head: bool = False,
	) -> tuple[int, int]:
		"""Count uniformity rows used for logging (unique keys or batch size)."""

		if not self.au_deduplicate:
			n_unique_q = int(head_indices.numel()) if self.criterion.gamma_active('q') else 0
			n_unique_t = int(tail_indices.numel()) if self.criterion.gamma_active('t') else 0
			return n_unique_q, n_unique_t

		q_keys, t_keys, _ = self._uniformity_keys(
			head_indices, relation_indices, tail_indices, predict_head=predict_head)
		n_unique_q = int(distinct_first_indices(q_keys).numel()) if self.criterion.gamma_active('q') else 0
		n_unique_t = int(distinct_first_indices(t_keys).numel()) if self.criterion.gamma_active('t') else 0
		return n_unique_q, n_unique_t

	def _embedding_l3_regularization(self, model) -> torch.Tensor | None:
		"""Optional L3 embedding penalty (adversarial / legacy scalar ``regularization``)."""

		model_obj = get_model_obj(model)
		return embedding_l3_penalty(model_obj)

	def _apply_embedding_regularization(
		self,
		loss: torch.Tensor,
		*,
		batch_triples: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Add L3 embedding penalty (LibKGE weights or legacy ``regularization`` scalar)."""

		zero = loss.new_zeros(())
		ent_weight = float(getattr(self.args, 'entity_regularize_weight', 0.0) or 0.0)
		rel_weight = float(getattr(self.args, 'relation_regularize_weight', 0.0) or 0.0)
		if ent_weight > 0.0 or rel_weight > 0.0:
			reg_term = compute_kge_regularization(
				get_model_obj(self.model),
				self.args,
				batch_triples=batch_triples,
			)
			if reg_term is None:
				return loss, zero
			return loss + reg_term, reg_term

		reg_coef = _config_float(self.args, 'regularization', 0.0)
		if reg_coef <= 0.0:
			return loss, zero
		l3_term = self._embedding_l3_regularization(self.model)
		if l3_term is None:
			return loss, zero
		l_reg = reg_coef * l3_term
		return loss + l_reg, l_reg

	def _au_loss_with_distinct_keys(
		self,
		q_raw: torch.Tensor,
		t_raw: torch.Tensor,
		h_raw: torch.Tensor,
		ent_raw: torch.Tensor | None,
		q_keys: torch.Tensor,
		t_keys: torch.Tensor,
		h_keys: torch.Tensor,
		*,
		batch_triples: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, float]:
		"""KGAU loss with deduplicated uniformity inputs (by entity/relation id keys)."""

		q_uni, t_uni, h_uni, n_unique_q, n_unique_t = self._distinct_uniformity_inputs(
			q_raw, t_raw, h_raw, q_keys, t_keys, h_keys)
		cross_uni = self._cross_uniformity_vectors(q_raw, t_raw, q_keys, t_keys)
		au_loss, l_align, l_unif, margin_active_frac = self.criterion(
			q_raw, t_raw, h_raw, ent_raw, q_uni=q_uni, t_uni=t_uni, h_uni=h_uni,
			cross_uni=cross_uni, return_stats=True)
		loss, l_reg = self._apply_embedding_regularization(au_loss, batch_triples=batch_triples)
		return loss, l_align, l_unif, l_reg, n_unique_q, n_unique_t, margin_active_frac

	def _batch_entity_uniformity_vectors(
		self,
		h_raw: torch.Tensor,
		t_raw: torch.Tensor,
		h_keys: torch.Tensor,
		t_keys: torch.Tensor,
	) -> torch.Tensor | None:
		"""Build entity-uniformity inputs from batch heads and tails (SimKGC-style).

		When ``au_deduplicate`` is enabled, keeps one vector per unique entity id.
		Otherwise pools all batch head/tail rows (DirectAU-style batch uniformity).
		"""

		if not self.criterion.gamma_active('ent') or h_raw.size(0) == 0:
			return None

		if not self.au_deduplicate:
			ent = torch.cat([h_raw, t_raw], dim=0)
			return ent if ent.size(0) >= 2 else None

		seen: set[int] = set()
		rows: list[torch.Tensor] = []
		head_ids = h_keys.reshape(-1).tolist()
		tail_ids = t_keys.reshape(-1).tolist()
		for i, (head_id, tail_id) in enumerate(zip(head_ids, tail_ids)):
			if head_id not in seen:
				seen.add(head_id)
				rows.append(h_raw[i])
			if tail_id not in seen:
				seen.add(tail_id)
				rows.append(t_raw[i])
		if len(rows) < 2:
			return None
		return torch.stack(rows, dim=0)

	def _catalog_entity_uniformity_vectors(self, model) -> torch.Tensor | None:
		"""Full entity-table vectors for embedding encoders (ComplEx, DistMult, DaBR, etc.)."""

		if not self.criterion.gamma_active('ent'):
			return None
		kwargs = _entity_embeddings_call_kwargs(model, self.device, self.criterion)
		# Optional encoder hook (pRotatE-AU only); all other encoders use ``entity_embeddings``.
		if hasattr(model, 'au_entity_embeddings'):
			ent = model.au_entity_embeddings(**kwargs)
		elif hasattr(model, 'entity_embeddings'):
			ent = model.entity_embeddings(**kwargs)
		else:
			return None
		if ent is not None and hasattr(model, '_normalize_au_vector'):
			ent = model._normalize_au_vector(ent)
		return ent

	def _entity_uniformity_vectors_for_loss(
		self,
		model,
		h_raw: torch.Tensor,
		t_raw: torch.Tensor,
		h_keys: torch.Tensor,
		t_keys: torch.Tensor,
		*,
		head_indices: torch.Tensor | None = None,
		tail_indices: torch.Tensor | None = None,
	) -> torch.Tensor | None:
		"""Entity vectors for ``gamma_ent``: batch dedup (text) or full table (embedding encoders)."""

		if config_bool(self.args, 'entity_uniformity_batch', False) or self.uses_text_inputs:
			if (
				not self.uses_text_inputs
				and head_indices is not None
				and tail_indices is not None
				and config_bool(self.args, 'entity_uniformity_batch', False)
			):
				model_obj = get_model_obj(model)
				h_ent = model_obj.embed_s(head_indices)
				t_ent = model_obj.embed_o(tail_indices)
				if hasattr(model_obj, '_normalize_au_vector'):
					h_ent = model_obj._normalize_au_vector(h_ent)
					t_ent = model_obj._normalize_au_vector(t_ent)
				return self._batch_entity_uniformity_vectors(h_ent, t_ent, h_keys, t_keys)
			return self._batch_entity_uniformity_vectors(h_raw, t_raw, h_keys, t_keys)
		return self._catalog_entity_uniformity_vectors(model)

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

	def _optimizer_step(self, use_amp: bool) -> None:
		grad_clip = getattr(self.args, 'grad_clip', None)
		if use_amp:
			self.scaler.unscale_(self.optimizer)
			if grad_clip is not None:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
			self.scaler.step(self.optimizer)
			self.scaler.update()
		else:
			if grad_clip is not None:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
			self.optimizer.step()
		self.criterion.clamp_learnable_gamma_adj()
		self.criterion.clamp_learnable_alpha_adj()
		self.criterion.clamp_learnable_tuni()
		step = increment_trainer_global_step(self)
		maybe_decay_lr_at_step(self, step)

	def _train_au_tensor_batch(
		self,
		model,
		ss: torch.Tensor,
		rs: torch.Tensor,
		ts: torch.Tensor,
		use_amp: bool,
		*,
		predict_head: bool = False,
	) -> tuple[float, float, float, float, int, int, float, int]:
		"""Run one optimizer step on a head/relation/tail batch.

		Non-DaBR encoders use a single forward/backward over the full batch (unchanged).
		DaBR may split into smaller chunks when ``train_micro_batch_size`` or the default cap applies.
		"""

		total = ss.size(0)
		n_uq_log, n_ut_log = self._count_unique_uniformity_keys(
			ss, rs, ts, predict_head=predict_head)
		micro_batch = min(self._train_micro_batch_size(total), total)

		if micro_batch >= total:
			return self._train_au_tensor_batch_single(
				model, ss, rs, ts, use_amp, n_uq_log, n_ut_log, total,
				predict_head=predict_head,
			)
		return self._train_au_tensor_batch_micro(
			model, ss, rs, ts, use_amp, n_uq_log, n_ut_log, total, micro_batch,
			predict_head=predict_head,
		)

	def _au_representation_batch(
		self,
		model,
		ss: torch.Tensor,
		rs: torch.Tensor,
		ts: torch.Tensor,
		*,
		predict_head: bool = False,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Fetch AU vectors via ``KGEModel.get_queries_targets`` (scorer-owned math)."""

		return get_model_obj(model).get_queries_targets(ss, rs, ts, predict_head=predict_head)

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
		*,
		predict_head: bool = False,
	) -> tuple[float, float, float, float, int, int, float, int]:
		"""Full-batch training step (DistMult-AU, ComplEx-AU, RotatE-AU, etc.)."""

		self.optimizer.zero_grad()
		q_keys, t_keys, h_keys = self._uniformity_keys(ss, rs, ts, predict_head=predict_head)
		batch_triples = torch.stack([ss, rs, ts], dim=1)
		if use_amp:
			with torch.amp.autocast(device_type='cuda'):
				q_raw, t_raw, h_raw = self._au_representation_batch(
					model, ss, rs, ts, predict_head=predict_head)
				ent_raw = self._entity_uniformity_vectors_for_loss(
					model, h_raw, t_raw, h_keys, t_keys,
					head_indices=ss, tail_indices=ts)
				loss, l_align, l_unif, l_reg, _, _, margin_active = self._au_loss_with_distinct_keys(
					q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys, batch_triples=batch_triples)
			self.scaler.scale(loss).backward()
			self._optimizer_step(use_amp)
		else:
			q_raw, t_raw, h_raw = self._au_representation_batch(
				model, ss, rs, ts, predict_head=predict_head)
			ent_raw = self._entity_uniformity_vectors_for_loss(
				model, h_raw, t_raw, h_keys, t_keys,
				head_indices=ss, tail_indices=ts)
			loss, l_align, l_unif, l_reg, _, _, margin_active = self._au_loss_with_distinct_keys(
				q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys, batch_triples=batch_triples)
			loss.backward()
			self._optimizer_step(use_amp)
		return (
			loss.item(), l_align.item(), l_unif.item(), l_reg.item(),
			n_uq_log, n_ut_log, margin_active, total,
		)

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
		*,
		predict_head: bool = False,
	) -> tuple[float, float, float, float, int, int, float, int]:
		"""DaBR-only: gradient accumulation over micro-batches to avoid OOM."""

		loss_sum = 0.0
		align_sum = 0.0
		unif_sum = 0.0
		reg_sum = 0.0
		margin_acc = 0.0
		margin_batches = 0

		self.optimizer.zero_grad()
		for start in range(0, total, micro_batch):
			end = min(start + micro_batch, total)
			fraction = (end - start) / total
			q_keys, t_keys, h_keys = self._uniformity_keys(
				ss[start:end], rs[start:end], ts[start:end], predict_head=predict_head)
			batch_triples = torch.stack([ss[start:end], rs[start:end], ts[start:end]], dim=1)
			if use_amp:
				with torch.amp.autocast(device_type='cuda'):
					q_raw, t_raw, h_raw = self._au_representation_batch(
						model, ss[start:end], rs[start:end], ts[start:end],
						predict_head=predict_head)
					ent_raw = self._entity_uniformity_vectors_for_loss(
						model, h_raw, t_raw, h_keys, t_keys,
						head_indices=ss[start:end], tail_indices=ts[start:end])
					loss, l_align, l_unif, l_reg, _, _, margin_active = self._au_loss_with_distinct_keys(
						q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys, batch_triples=batch_triples)
				self._backward_au_loss(loss, fraction, use_amp=True)
			else:
				q_raw, t_raw, h_raw = self._au_representation_batch(
					model, ss[start:end], rs[start:end], ts[start:end],
					predict_head=predict_head)
				ent_raw = self._entity_uniformity_vectors_for_loss(
					model, h_raw, t_raw, h_keys, t_keys,
					head_indices=ss[start:end], tail_indices=ts[start:end])
				loss, l_align, l_unif, l_reg, _, _, margin_active = self._au_loss_with_distinct_keys(
					q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys, batch_triples=batch_triples)
				self._backward_au_loss(loss, fraction, use_amp=False)
			chunk = end - start
			loss_sum += loss.item() * chunk
			align_sum += l_align.item() * chunk
			unif_sum += l_unif.item() * chunk
			reg_sum += l_reg.item() * chunk
			margin_acc += margin_active
			margin_batches += 1

		self._optimizer_step(use_amp)

		avg_margin = (margin_acc / margin_batches) if margin_batches > 0 else 0.0
		return (
			loss_sum / total, align_sum / total, unif_sum / total, reg_sum / total,
			n_uq_log, n_ut_log, avg_margin, total,
		)

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

	def train_epoch(self, epoch) -> float:
		"""Train the model for one epoch and return the average training loss."""

		self.model.train()
		epoch_loss = 0.0
		epoch_align_loss = 0.0
		epoch_unif_loss = 0.0
		epoch_reg_loss = 0.0
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
						ent_raw = self._entity_uniformity_vectors_for_loss(
							model, h_raw, t_raw, h_keys, t_keys)
						loss, l_align, l_unif, l_reg, n_uq, n_ut, margin_active = self._au_loss_with_distinct_keys(
							q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys)
					self.scaler.scale(loss).backward()
					self._optimizer_step(use_amp)
				else:
					outputs = self.model(**batch_dict)
					q_raw = outputs['hr_vector']
					t_raw = outputs['tail_vector']
					h_raw = outputs['head_vector']
					ent_raw = self._entity_uniformity_vectors_for_loss(
						model, h_raw, t_raw, h_keys, t_keys)
					loss, l_align, l_unif, l_reg, n_uq, n_ut, margin_active = self._au_loss_with_distinct_keys(
						q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys)
					loss.backward()
					self._optimizer_step(use_amp)
				batch_examples = len(batch_dict['batch_data'])
				losses.update(loss.item(), batch_examples)
				epoch_align_loss += l_align.item() * batch_examples
				epoch_unif_loss += l_unif.item() * batch_examples
				epoch_reg_loss += l_reg.item() * batch_examples
				epoch_loss += loss.item() * batch_examples
				epoch_unique_q += n_uq
				epoch_unique_t += n_ut
				epoch_margin_active += margin_active
				epoch_batches += 1
				if i % self.args.print_freq == 0:
					progress.display(i)
		else:
			directions = (False, True) if self.kgau_bidirectional else (False,)
			for predict_head in directions:
				for batch_idx, (ss, rs, ts) in enumerate(
					self._iter_batches(self.train_src, self.train_rel, self.train_dst, batch_size, shuffle=True),
				):
					loss, l_align, l_unif, l_reg, n_uq, n_ut, margin_active, n_examples = self._train_au_tensor_batch(
						model, ss, rs, ts, use_amp, predict_head=predict_head,
					)
					epoch_align_loss += l_align * n_examples
					epoch_unif_loss += l_unif * n_examples
					epoch_reg_loss += l_reg * n_examples
					epoch_loss += loss * n_examples
					epoch_unique_q += n_uq
					epoch_unique_t += n_ut
					epoch_margin_active += margin_active
					epoch_batches += 1

		avg_count = max(len(self.train_examples) * (2 if self.kgau_bidirectional else 1), 1)
		avg_loss = epoch_loss / avg_count
		avg_align_loss = epoch_align_loss / avg_count
		avg_unif_loss = epoch_unif_loss / avg_count
		avg_reg_loss = epoch_reg_loss / avg_count
		avg_au_loss = avg_align_loss + avg_unif_loss
		display_epoch = epoch + 1
		if epoch_batches > 0:
			avg_margin_active = epoch_margin_active / epoch_batches
			avg_unique_q = epoch_unique_q / epoch_batches
			avg_unique_t = epoch_unique_t / epoch_batches
		else:
			avg_margin_active = 0.0
			avg_unique_q = avg_unique_t = 0.0
		tuni_suffix = ''
		if self._should_log_tuni():
			tuni_suffix = f' | tuni: {_tuni_scalar(self.criterion):.4f}'
		gamma_suffix = _gamma_log_suffix(self.criterion) if (
			self._should_log_gammas() or self._should_log_alpha()
		) else ''
		if float(self.criterion.additive_margin) > 0.0:
			logger.info(
				'[EPOCH %s] train loss: %.6f | au: %.6f | align: %.6f | uniformity: %.6f | reg: %.6f | '
				'unique q/t per batch: %.0f/%.0f (of %d) | margin-buffer pairs: %.2f%%%s',
				display_epoch, avg_loss, avg_au_loss, avg_align_loss, avg_unif_loss, avg_reg_loss,
				avg_unique_q, avg_unique_t, batch_size,
				100.0 * avg_margin_active,
				tuni_suffix + gamma_suffix,
			)
		else:
			logger.info(
				'[EPOCH %s] train loss: %.6f | au: %.6f | align: %.6f | uniformity: %.6f | reg: %.6f | '
				'unique q/t per batch: %.0f/%.0f (of %d)%s',
				display_epoch, avg_loss, avg_au_loss, avg_align_loss, avg_unif_loss, avg_reg_loss,
				avg_unique_q, avg_unique_t, batch_size,
				tuni_suffix + gamma_suffix,
			)
		self.train_component_losses = {
			'loss': avg_loss,
			'au': avg_au_loss,
			'align': avg_align_loss,
			'uniformity': avg_unif_loss,
			'reg': avg_reg_loss,
			'num_examples': avg_count,
			'avg_unique_q': avg_unique_q,
			'avg_unique_t': avg_unique_t,
		}
		if float(self.criterion.additive_margin) > 0.0:
			self.train_component_losses['margin_buffer_pair_frac'] = avg_margin_active
		if hasattr(self.criterion, 'log_tuni') or config_bool(self.args, 'tuni_linear_schedule', False):
			if not config_bool(self.args, 'tuni_as_alpha', False):
				self.train_component_losses['tuni'] = _tuni_scalar(self.criterion)
		if self._should_log_gammas():
			for name in _GAMMA_NAMES:
				if self.criterion.gamma_active(name):
					self.train_component_losses[f'gamma_{name}'] = self.criterion.gamma_value(name)
		if self._should_log_alpha():
			self.train_component_losses['alpha'] = self.criterion.alpha_value()
		return avg_loss

	def _maybe_update_alpha_schedule(self, epoch: int) -> None:
		"""Apply linear alpha multiplier growth before each training epoch."""

		if not _alpha_schedule_enabled(self.args):
			return

		mult = _scheduled_alpha_mult(self.args, epoch)
		if mult <= 0.0:
			logger.warning('Scheduled alpha multiplier <= 0 (%.6f) at epoch %d; skip update', mult, epoch)
			return

		self.criterion.set_alpha_schedule_mult(mult)

	def _maybe_update_gamma_schedule(self, epoch: int) -> None:
		"""Apply linear gamma multiplier decay before each training epoch."""

		if not _gamma_schedule_enabled(self.args):
			return

		mult = _scheduled_gamma_mult(self.args, epoch)
		if mult <= 0.0:
			logger.warning('Scheduled gamma multiplier <= 0 (%.6f) at epoch %d; skip update', mult, epoch)
			return

		self.criterion.set_gamma_schedule_mult(mult)

	def _maybe_update_log_au_gamma_lr_schedule(self, epoch: int) -> None:
		"""Linearly ramp ``log_au_gamma_lr`` toward ``gamma_schedule_end`` before each training epoch."""

		if not _log_au_gamma_lr_schedule_enabled(self.args):
			return

		lr = _scheduled_log_au_gamma_lr(self.args, epoch)
		if lr <= 0.0:
			logger.warning('Scheduled log_au_gamma_lr <= 0 (%.6e) at epoch %d; skip update', lr, epoch)
			return

		if not _set_log_au_gamma_lr(self.optimizer, lr):
			return

		self.args.log_au_gamma_lr = float(lr)

	def _should_log_gammas(self) -> bool:
		return self.criterion.learnable_au_gammas or _gamma_schedule_enabled(self.args)

	def _should_log_alpha(self) -> bool:
		if config_bool(self.args, 'tuni_as_alpha', False):
			return hasattr(self.criterion, 'log_tuni') or config_bool(self.args, 'tuni_linear_schedule', False)
		return self.criterion.learnable_au_alpha or _alpha_schedule_enabled(self.args)

	def _maybe_update_tuni_schedule(self, epoch: int) -> None:
		"""Apply linear tuni schedule before each training epoch (fixed scale only)."""

		if not config_bool(self.args, 'tuni_linear_schedule', False):
			return
		if hasattr(self.criterion, 'log_tuni'):
			return

		scale_value = _scheduled_tuni_value(self.args, epoch)
		if scale_value <= 0.0:
			logger.warning('Scheduled tuni <= 0 (%.6f) at epoch %d; skip update', scale_value, epoch)
			return

		self.criterion.tuni = float(scale_value)
		self.args.tuni = float(scale_value)
		logger.info('Linear tuni schedule at epoch %d: %.6f', epoch + 1, scale_value)

	def _should_log_tuni(self) -> bool:
		if config_bool(self.args, 'tuni_as_alpha', False):
			return False
		return hasattr(self.criterion, 'log_tuni') or config_bool(self.args, 'tuni_linear_schedule', False)

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
			metric_dict.update(
				log_bidirectional_link_metrics(
					f'[EPOCH {display_epoch}] Valid', forward_metrics, backward_metrics
				)
			)
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

		patience = getattr(self.args, 'early_stopping_patience', None)
		patience = int(patience) if patience else None
		min_epochs = int(getattr(self.args, 'early_stopping_min_epochs', 0) or 0)
		min_metric = getattr(self.args, 'early_stopping_min_metric', None)
		bad_counts = 0
		if patience is not None and patience > 0:
			logger.info(
				'KGAU early stopping: stop after %d validation(s) without MRR improvement '
				'(min_epochs=%d).',
				patience,
				min_epochs,
			)
		else:
			patience = None

		total_start_time = time.time()
		num_train_epochs = 0
		for epoch in range(self.args.epochs):
			self._maybe_update_alpha_schedule(epoch)
			self._maybe_update_gamma_schedule(epoch)
			self._maybe_update_log_au_gamma_lr_schedule(epoch)
			self._maybe_update_tuni_schedule(epoch)
			epoch_train_start = time.time()
			self.memory_tracker.begin_phase()
			train_loss = self.train_epoch(epoch)
			self.memory_tracker.end_phase('train')
			self.train_time += time.time() - epoch_train_start
			num_train_epochs = epoch + 1

			from utils.au_reporter import report_au_after_epoch
			report_au_after_epoch(self, epoch)

			validated = self._should_validate(epoch)
			metric_dict: dict = {}
			if validated:
				eval_start = time.time()
				self.memory_tracker.begin_phase()
				metric_dict = self.eval_epoch(epoch, train_loss=train_loss)
				self.memory_tracker.end_phase('eval')
				self.valid_time += time.time() - eval_start

			from utils.au_reporter import report_au_validation
			report_au_validation(self, epoch, metric_dict if validated else None)

			is_best = False
			if validated and metric_dict and 'mrr' in metric_dict:
				monitor_value = metric_dict['mrr']
				is_best = self.best_metric is None or monitor_value > self.best_metric.get('score', float('-inf'))
				if is_best:
					self.best_metric = {'score': monitor_value, 'metrics': metric_dict, 'epoch': epoch}
					bad_counts = 0
				else:
					best_mrr = None if self.best_metric is None else self.best_metric.get('score')
					if min_metric is None or (best_mrr is not None and best_mrr >= float(min_metric)):
						bad_counts += 1

			max_to_keep = getattr(self.args, 'max_to_keep', 5)
			if max_to_keep is None:
				max_to_keep = 5
			filename = (
				last_model_path(self.args.output_dir)
				if max_to_keep == 0
				else checkpoint_path(self.args.output_dir, epoch)
			)
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
			delete_old_ckt(
				path_pattern='{}/checkpoint_*.mdl'.format(self.args.output_dir),
				keep=max_to_keep,
			)

			step_lr_scheduler(self.lr_scheduler, metric_dict)

			if patience is not None and bad_counts >= patience and (epoch + 1) >= min_epochs:
				logger.info(
					'[EARLY STOP] No validation MRR improvement for %d evaluations (epoch %s).',
					patience, epoch + 1,
				)
				break

		self.total_time = time.time() - total_start_time
		epoch_time = time_per_train_epoch(self.train_time, num_train_epochs)
		logger.info('[Timing] Training time (s): %.2f', round(self.train_time, 2))
		logger.info('[Timing] Valid time (s): %.2f', round(self.valid_time, 2))
		logger.info('[Timing] Total run time (s): %.2f', round(self.total_time, 2))
		if epoch_time is not None:
			logger.info('[Timing] Time per training epoch (s): %.2f', round(epoch_time, 2))
		logger.info('[Memory] Training peak: %s', format_memory(self.memory_tracker.train_peak_mb))
		logger.info('[Memory] Eval peak: %s', format_memory(self.memory_tracker.eval_peak_mb))
		logger.info('[Memory] Peak memory: %s', format_memory(self.memory_tracker.peak_memory_mb))

		from utils.au_reporter import finalize_au_report
		finalize_au_report(self)

		return {
			'best_epoch': None if self.best_metric is None else self.best_metric.get('epoch'),
			'best_mrr': None if self.best_metric is None else self.best_metric.get('score'),
			'best_checkpoint_path': self.best_checkpoint_path,
			'train_time': self.train_time,
			'valid_time': self.valid_time,
			'total_time': self.total_time,
			'num_train_epochs': num_train_epochs,
			'time_per_train_epoch': epoch_time,
			**self.memory_tracker.to_dict(),
		}

Strategy = KGAUStrategy
