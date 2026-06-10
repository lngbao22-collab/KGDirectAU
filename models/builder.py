"""Five-pillar component factory: embedder, scorer, loss, sampler, strategy."""

from __future__ import annotations

from importlib import import_module
import importlib.util
import inspect
import os
import sys
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn

from base.model import DaBRModel, KGEModel
from data.dataset import PointwiseDataset, load_data
from data.dict_hub import get_entity_dict, get_relation_id_map


def _normalize_path(path: str) -> str:
	if path.endswith('.py'):
		path = path[:-3]
		path = path.replace('/', '.').replace('\\', '.')
	if path.startswith('./') or path.startswith('.\\'):
		path = path[2:]
	return path


def import_module_from_path(path: str) -> ModuleType:
	if not path:
		raise ValueError('Empty module path')
	if os.path.exists(path) and path.endswith('.py'):
		spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
		module = importlib.util.module_from_spec(spec)
		sys.modules[spec.name] = module
		spec.loader.exec_module(module)
		return module
	return import_module(_normalize_path(path))


def load_attr_from_path(path: str, attr: str) -> Any:
	module = import_module_from_path(path)
	if not hasattr(module, attr):
		raise AttributeError(f'Module {path} has no attribute {attr}')
	return getattr(module, attr)


def _config_path(args, name: str, legacy_name: str = '') -> str:
	value = getattr(args, name, '') or ''
	if value:
		return value
	if legacy_name:
		return getattr(args, legacy_name, '') or ''
	return ''


def _is_text_model(args) -> bool:
	return 'simkgc' in str(getattr(args, 'model', '') or '').lower()


def _strategy_paradigm(strategy_path: str) -> str:
	path = strategy_path.replace('\\', '/').lower()
	if 'negsamp' in path or 'adversarial' in path or 'pointwise' in path:
		return 'negsamp'
	if '1vsall' in path or 'softmax' in path:
		return '1vsall'
	if 'kgau' in path:
		return 'kgau'
	if 'inbatch' in path or 'contrastive' in path or 'simkgc' in path:
		return 'inbatch'
	return 'generic'


def build_entity_embedder(args) -> nn.Module:
	embedder_path = _config_path(args, 'model_embedder_path')
	if not embedder_path:
		raise ValueError('model_embedder_path is required')
	return load_attr_from_path(embedder_path, 'build_entity_embedder')(args)


def build_relation_embedder(args) -> nn.Module:
	embedder_path = _config_path(args, 'model_embedder_path')
	if not embedder_path:
		raise ValueError('model_embedder_path is required')
	return load_attr_from_path(embedder_path, 'build_relation_embedder')(args)


def build_scorer_module(args) -> nn.Module:
	scorer_path = _config_path(args, 'model_scorer_path', 'model_encoder_path')
	if not scorer_path:
		raise ValueError('model_scorer_path is required')
	return load_attr_from_path(scorer_path, 'build_scorer')(args)


def bind_model(args, ent_embedder: nn.Module, rel_embedder: nn.Module, scorer: nn.Module) -> nn.Module:
	"""Bind embedders and scorer into ``KGEModel`` (or ``DaBRModel`` when needed)."""

	model_name = str(getattr(args, 'model', '') or '').lower()
	if 'dabr' in model_name:
		embedder_path = _config_path(args, 'model_embedder_path')
		dr_embedder = load_attr_from_path(embedder_path, 'build_dr_embedder')(args)
		return DaBRModel(
			args,
			ent_embedder=ent_embedder,
			rel_embedder=rel_embedder,
			scorer=scorer,
			dr_embedder=dr_embedder,
		)
	return KGEModel(ent_embedder, rel_embedder, scorer, args)


def build_model(args) -> nn.Module:
	"""Assemble the model pillar (embedder + scorer bound by ``KGEModel``)."""

	if _is_text_model(args):
		scorer_path = _config_path(args, 'model_scorer_path', 'model_encoder_path')
		return load_attr_from_path(scorer_path, 'build_model')(args)

	ent_embedder = build_entity_embedder(args)
	rel_embedder = build_relation_embedder(args)
	scorer = build_scorer_module(args)
	return bind_model(args, ent_embedder, rel_embedder, scorer)


def load_loss_fn(args):
	"""Load the loss pillar from ``model_loss_path`` via ``compute_loss``."""

	loss_path = getattr(args, 'model_loss_path', '') or ''
	if not loss_path:
		return None

	compute_loss = load_attr_from_path(loss_path, 'compute_loss')
	if not callable(compute_loss):
		raise TypeError(f'{loss_path} compute_loss must be callable')

	if len(inspect.signature(compute_loss).parameters) == 1:
		return compute_loss(args)
	return compute_loss


def load_sampler(args, model: nn.Module | None = None, train_triples: torch.Tensor | None = None):
	"""Load the sampler pillar when configured (1vsAll/KGAU/in-batch skip it)."""

	sampler_path = getattr(args, 'model_sampler_path', '') or ''
	if not sampler_path:
		return None

	module = import_module_from_path(sampler_path)
	build_sampler = getattr(module, 'build_sampler', None)
	if callable(build_sampler):
		try:
			return build_sampler(args, train_triples, model)
		except TypeError:
			return build_sampler(args)

	sampler_cls = getattr(module, 'FilteredSubsampler', None)
	if sampler_cls is not None and train_triples is not None:
		nentity = _resolve_nentity(args, model)
		num_neg = int(getattr(args, 'n_sample', 1))
		return sampler_cls(train_triples, nentity, num_neg)

	raise AttributeError(f'Module {sampler_path} must define build_sampler or FilteredSubsampler')


def config_float(args, name: str, default: float) -> float:
	value = getattr(args, name, None)
	return default if value is None else float(value)


def config_int(args, name: str, default: int | None = None) -> int | None:
	value = getattr(args, name, None)
	if value is None:
		return default
	return int(value)


def config_bool(args, name: str, default: bool = False) -> bool:
	value = getattr(args, name, None)
	if value is None:
		return default
	return bool(value)


def build_optimizer(args, parameters, weight_decay: float):
	from torch import optim
	from torch.optim import Adam

	lr = float(getattr(args, 'lr', getattr(args, 'learning_rate', 1e-3)))
	optim_name = str(getattr(args, 'optim', 'adam')).lower()
	if optim_name == 'adagrad':
		return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
	if optim_name == 'sgd':
		return optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
	return Adam(parameters, lr=lr, weight_decay=weight_decay)


def load_loss_fn_for_paradigm(args, paradigm: str):
	"""Load a loss factory from ``model_loss_path`` for negsamp/1vsall fallbacks."""

	loss_path = getattr(args, 'model_loss_path', '') or ''
	if not loss_path:
		raise ValueError(f'model_loss_path is required for {paradigm} training')

	module = import_module_from_path(loss_path)
	for builder_name in (f'build_{paradigm}_loss_fn', 'build_loss_fn'):
		builder = getattr(module, builder_name, None)
		if callable(builder):
			return builder(args)

	compute_fn = getattr(module, 'compute_loss', None)
	if callable(compute_fn):
		if len(inspect.signature(compute_fn).parameters) == 1:
			built = compute_fn(args)
			if callable(built):
				return built
		return compute_fn

	raise AttributeError(
		f'Module {loss_path} must define build_{paradigm}_loss_fn, build_loss_fn, or compute_loss'
	)


def _resolve_nentity(args, model: nn.Module) -> int:
	from utils.device import get_model_obj

	for attr in ('nentity', 'ent_total'):
		value = getattr(args, attr, None)
		if value is not None:
			return int(value)
	model_obj = get_model_obj(model)
	if hasattr(model_obj, 'entity_embedding'):
		return int(model_obj.entity_embedding.size(0))
	if hasattr(model_obj, 'ent_embedder') and hasattr(model_obj.ent_embedder, 'embedding'):
		return int(model_obj.ent_embedder.embedding.num_embeddings)
	if hasattr(model_obj, 'ent_embeddings') and hasattr(model_obj.ent_embeddings, 'embedding'):
		return int(model_obj.ent_embeddings.embedding.num_embeddings)
	raise ValueError('Could not resolve entity count for sampler construction')


def init_index_kge_trainer(trainer, model: nn.Module, args) -> None:
	"""Attach shared evaluation/checkpoint state to index-based KGE strategies."""

	from base.evaluator import Evaluator
	from utils.memory import PhaseMemoryTracker

	trainer.model = model
	trainer.encoder = model
	trainer.args = args
	trainer.entity_dict = get_entity_dict()
	trainer.entity_ids = [ex.entity_id for ex in trainer.entity_dict.entity_exs]
	trainer.evaluator = Evaluator(args)
	trainer.best_metric = None
	trainer.best_checkpoint_path = None
	trainer.train_time = 0.0
	trainer.valid_time = 0.0
	trainer.total_time = 0.0
	trainer.memory_tracker = PhaseMemoryTracker()
	trainer._cached_valid_exs = None
	trainer._cached_valid_backward_exs = None

	if torch.cuda.is_available():
		trainer.model.cuda()
	trainer.device = next(trainer.model.parameters()).device


def _kge_validation_interval(args) -> int:
	raw = getattr(args, 'epoch_per_eval', None)
	interval = int(raw) if raw is not None else 0
	if interval <= 0 or interval > int(getattr(args, 'epochs', 1)):
		return 1
	return interval


def _kge_should_validate(args, epoch: int) -> bool:
	interval = _kge_validation_interval(args)
	epoch_number = epoch + 1
	max_epochs = max(int(getattr(args, 'epochs', 1)), 1)
	return epoch_number % interval == 0 or epoch_number >= max_epochs


def _kge_get_valid_examples(trainer):
	from data.dataset import Example, load_data, reverse_triplet
	from utils.device import get_model_obj

	if trainer._cached_valid_exs is not None:
		return trainer._cached_valid_exs, trainer._cached_valid_backward_exs

	valid_path = getattr(trainer.args, 'valid_path', '')
	valid_exs = load_data(valid_path, add_forward_triplet=True, add_backward_triplet=False)
	model_obj = get_model_obj(trainer.model)
	if getattr(model_obj, 'bidirectional_score_batch', False):
		valid_backward_exs = valid_exs
	else:
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
	trainer._cached_valid_exs = valid_exs
	trainer._cached_valid_backward_exs = valid_backward_exs
	return valid_exs, valid_backward_exs


def _kge_resolve_link_prediction_path(path: str) -> str:
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


def _kge_average_metric_dict(forward_metrics: dict, backward_metrics: dict) -> dict:
	if not forward_metrics or not backward_metrics:
		return forward_metrics or backward_metrics or {}
	averaged = {}
	for key in forward_metrics.keys() & backward_metrics.keys():
		fwd = forward_metrics[key]
		bwd = backward_metrics[key]
		if isinstance(fwd, (int, float)) and isinstance(bwd, (int, float)):
			averaged[key] = (fwd + bwd) / 2
	return averaged


def _kge_extract_monitor_value(metric_dict: dict, train_loss: float | None = None):
	if metric_dict and 'mrr' in metric_dict:
		return metric_dict['mrr']
	if metric_dict and 'loss' in metric_dict:
		return -metric_dict['loss']
	if train_loss is not None:
		return -float(train_loss)
	for value in metric_dict.values():
		if isinstance(value, (int, float)):
			return value
	return None


def eval_index_kge_epoch(trainer, epoch: int) -> dict:
	"""Run validation link prediction for index-based KGE strategies."""

	from utils.logger import logger

	trainer.model.eval()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	metric_dict = {}
	valid_path = getattr(trainer.args, 'valid_path', '')
	valid_eval_path = _kge_resolve_link_prediction_path(valid_path)
	if not valid_eval_path or not os.path.exists(valid_eval_path):
		return metric_dict

	valid_exs, valid_backward_exs = _kge_get_valid_examples(trainer)
	valid_output_path = os.path.join(trainer.args.output_dir, 'valid_link_prediction.log')
	forward_metrics = trainer.evaluator.evaluate_link_prediction_inplace(
		trainer.model, valid_eval_path, trainer.entity_dict, valid_output_path,
		eval_forward=True, examples=valid_exs,
	)
	backward_metrics = trainer.evaluator.evaluate_link_prediction_inplace(
		trainer.model, valid_eval_path, trainer.entity_dict, valid_output_path,
		eval_forward=False, examples=valid_backward_exs,
	)
	if forward_metrics and backward_metrics:
		metric_dict.update(_kge_average_metric_dict(forward_metrics, backward_metrics))
		logger.info(
			'[EPOCH %s] Valid | MRR: %.4f | H@10: %.4f',
			epoch + 1,
			metric_dict.get('mrr', 0.0),
			metric_dict.get('hit@10', metric_dict.get('hits@10', 0.0)),
		)
	return metric_dict


def _save_index_kge_checkpoint(trainer, epoch: int, is_best: bool) -> str:
	from utils.checkpoint import best_model_path, delete_old_ckt, last_model_path, save_checkpoint
	from utils.device import get_model_obj

	saved_checkpoint_path = save_checkpoint(
		{
			'epoch': epoch,
			'best_epoch': epoch if is_best else None,
			'best_metric': trainer.best_metric,
			'args': trainer.args.__dict__,
			'state_dict': get_model_obj(trainer.model).state_dict(),
		},
		is_best=is_best,
		filename=last_model_path(trainer.args.output_dir),
	)
	if is_best:
		trainer.best_checkpoint_path = best_model_path(trainer.args.output_dir)
	elif trainer.best_checkpoint_path is None:
		trainer.best_checkpoint_path = saved_checkpoint_path
	delete_old_ckt(
		path_pattern='{}/checkpoint_*.mdl'.format(trainer.args.output_dir),
		keep=getattr(trainer.args, 'max_to_keep', 5),
	)
	return saved_checkpoint_path


def run_index_kge_train_loop(trainer, dataloader=None) -> dict:
	"""Shared epoch loop for index-based KGE strategies."""

	import time

	from utils.logger import logger
	from utils.memory import format_memory

	total_start = time.time()
	max_epochs = max(int(getattr(trainer.args, 'epochs', 1)), 1)
	patience = config_int(trainer.args, 'early_stopping_patience', None)
	bad_counts = 0
	if patience is not None and patience <= 0:
		patience = None

	for epoch in range(max_epochs):
		train_start = time.time()
		trainer.memory_tracker.begin_phase()
		train_loss = trainer.train_epoch(dataloader, epoch)
		trainer.memory_tracker.end_phase('train')
		trainer.train_time += time.time() - train_start

		metric_dict = {}
		if _kge_should_validate(trainer.args, epoch):
			eval_start = time.time()
			trainer.memory_tracker.begin_phase()
			metric_dict = eval_index_kge_epoch(trainer, epoch)
			trainer.memory_tracker.end_phase('eval')
			trainer.valid_time += time.time() - eval_start

		monitor_value = _kge_extract_monitor_value(metric_dict, train_loss)
		is_best = trainer.best_metric is None or monitor_value > trainer.best_metric.get('score', float('-inf'))
		if is_best:
			trainer.best_metric = {'score': monitor_value, 'metrics': metric_dict, 'epoch': epoch}
			if metric_dict and 'mrr' in metric_dict:
				bad_counts = 0
		elif metric_dict and 'mrr' in metric_dict:
			bad_counts += 1

		_save_index_kge_checkpoint(trainer, epoch, is_best)

		if patience is not None and bad_counts >= patience:
			logger.info('[EARLY STOP] No validation MRR improvement for %d evaluations.', patience)
			break

	trainer.total_time = time.time() - total_start
	logger.info('[Memory] Training peak: %s', format_memory(trainer.memory_tracker.train_peak_mb))
	logger.info('[Memory] Eval peak: %s', format_memory(trainer.memory_tracker.eval_peak_mb))
	logger.info('[Memory] Peak memory: %s', format_memory(trainer.memory_tracker.peak_memory_mb))
	return {
		'best_epoch': None if trainer.best_metric is None else trainer.best_metric.get('epoch', 0) + 1,
		'best_mrr': None if trainer.best_metric is None else trainer.best_metric.get('score'),
		'train_time': trainer.train_time,
		'valid_time': trainer.valid_time,
		'total_time': trainer.total_time,
		'best_checkpoint_path': trainer.best_checkpoint_path,
		**trainer.memory_tracker.to_dict(),
	}


def load_strategy_class(args):
	strategy_path = getattr(args, 'model_strategy_path', '') or ''
	for attr in ('Strategy', 'NegSampStrategy', 'OneVsAllStrategy', 'KGAUStrategy', 'InBatchStrategy', 'SimKGCStrategy'):
		try:
			cls = load_attr_from_path(strategy_path, attr)
		except AttributeError:
			continue
		if isinstance(cls, type):
			return cls
	raise AttributeError(f'Could not find Strategy class in {strategy_path}')


def _resolve_relation_index(relation: str, relation_to_idx: dict) -> int:
	if relation in relation_to_idx:
		return relation_to_idx[relation]
	normalized = ' '.join(relation.split())
	if normalized in relation_to_idx:
		return relation_to_idx[normalized]
	if relation.startswith('inverse '):
		base_relation = relation[len('inverse '):]
		if base_relation in relation_to_idx and f'inverse {base_relation}' not in relation_to_idx:
			return relation_to_idx[base_relation]
	raise KeyError(relation)


def _examples_to_tensors(examples, entity_dict, relation_to_idx):
	head_indices = torch.tensor([entity_dict.entity_to_idx(example.head_id) for example in examples], dtype=torch.long)
	relation_indices = torch.tensor(
		[_resolve_relation_index(example.relation, relation_to_idx) for example in examples],
		dtype=torch.long,
	)
	tail_indices = torch.tensor([entity_dict.entity_to_idx(example.tail_id) for example in examples], dtype=torch.long)
	return head_indices, relation_indices, tail_indices


def _prepare_train_triples(args, model: nn.Module) -> torch.Tensor:
	entity_dict = get_entity_dict()
	relation_to_idx = getattr(model, 'rel_to_idx', None) or get_relation_id_map()
	train_examples = load_data(args.train_path, add_forward_triplet=True, add_backward_triplet=False)
	if not train_examples:
		raise ValueError(f'No training examples loaded from {args.train_path}')
	src, rel, dst = _examples_to_tensors(train_examples, entity_dict, relation_to_idx)
	return torch.stack([src, rel, dst], dim=-1)


def _prepare_pointwise_dataloader(args):
	from data.dataloader import collate_pointwise

	train_examples = load_data(args.train_path, add_forward_triplet=True, add_backward_triplet=False)
	train_dataset = PointwiseDataset(train_examples)
	n_batches = getattr(args, 'n_batches', None)
	if n_batches:
		args.batch_size = max(len(train_examples) // int(n_batches), 1)
	batch_size = max(getattr(args, 'batch_size', 1), 1)
	return torch.utils.data.DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		collate_fn=collate_pointwise,
		num_workers=getattr(args, 'workers', 0),
		pin_memory=torch.cuda.is_available(),
		drop_last=True,
	)


def _strategy_init_kwargs(args, strategy_path: str, model: nn.Module, train_triples: torch.Tensor | None):
	paradigm = _strategy_paradigm(strategy_path)
	kwargs: dict[str, Any] = {}

	if paradigm == 'negsamp':
		sampler_path = (getattr(args, 'model_sampler_path', '') or '').replace('\\', '/')
		if sampler_path.endswith('uniform_pointwise_sampler.py'):
			kwargs['train_dataloader'] = _prepare_pointwise_dataloader(args)
		else:
			kwargs['train_triples'] = train_triples if train_triples is not None else _prepare_train_triples(args, model)
	elif paradigm == '1vsall':
		add_backward = getattr(args, 'add_reciprocal_relations', False)
		entity_dict = get_entity_dict()
		train_examples = load_data(args.train_path, add_forward_triplet=True, add_backward_triplet=add_backward)
		rel_map = getattr(model, 'rel_to_idx', None) or get_relation_id_map()
		kwargs['train_data'] = _examples_to_tensors(train_examples, entity_dict, rel_map)

	return kwargs


def build_pipeline(args, ngpus_per_node: int = 1):
	"""Wire all five pillars: embedder, scorer, loss, sampler, strategy."""

	strategy_path = getattr(args, 'model_strategy_path', '') or ''
	paradigm = _strategy_paradigm(strategy_path)
	StrategyClass = load_strategy_class(args)

	if paradigm == 'inbatch':
		loss_fn = load_loss_fn(args)
		sampler = load_sampler(args)
		return StrategyClass(None, sampler, loss_fn, args, ngpus_per_node=ngpus_per_node)

	ent_embedder = build_entity_embedder(args)
	rel_embedder = build_relation_embedder(args)
	scorer = build_scorer_module(args)
	model = bind_model(args, ent_embedder, rel_embedder, scorer)
	if torch.cuda.is_available():
		model.cuda()

	loss_fn = load_loss_fn(args)
	train_triples = _prepare_train_triples(args, model) if paradigm == 'negsamp' else None
	sampler = load_sampler(args, model, train_triples)
	strategy_kwargs = _strategy_init_kwargs(args, strategy_path, model, train_triples)

	if paradigm == 'kgau':
		return StrategyClass(model, sampler, loss_fn, args, ngpus_per_node=ngpus_per_node, **strategy_kwargs)

	return StrategyClass(model, sampler, loss_fn, args, **strategy_kwargs)
