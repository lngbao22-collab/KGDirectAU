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
from base.evaluator import Evaluator
from data.dataloader import collate
from data.dataset import Dataset, load_data
from data.dict_hub import get_entity_dict, get_relation_id_map
from models.builder import apply_kge_regularization, build_lr_scheduler, config_bool, load_attr_from_path
from utils.checkpoint import best_model_path, checkpoint_path, delete_old_ckt, save_checkpoint
from utils.device import get_model_obj, move_to_cuda, report_num_trainable_parameters
from utils.logger import AverageMeter, ProgressMeter, logger
from utils.memory import PhaseMemoryTracker, format_memory
from models.losses.au_loss import KGAULoss, distinct_first_indices, select_distinct_rows


def _load_encoder(args) -> torch.nn.Module:
	from models.builder import build_model

	return build_model(args)


def _uses_text_inputs(args) -> bool:
	if 'simkgc' in str(getattr(args, 'model', '') or '').lower():
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


def _build_kgau_optimizer(args, model, criterion: KGAULoss, weight_decay: float):
	"""Build optimizer over model + KGAU loss params (optional learnable ``tuni`` group)."""

	lr = float(getattr(args, 'lr', getattr(args, 'learning_rate', 2e-5)))
	base_params = [p for p in model.parameters() if p.requires_grad]
	log_tuni_param = None
	aux_other_params = []
	for name, param in criterion.named_parameters():
		if not param.requires_grad:
			continue
		if name == 'log_tuni' or name.endswith('.log_tuni'):
			log_tuni_param = param
		else:
			aux_other_params.append(param)

	param_groups = []
	base_group_params = base_params + aux_other_params
	if base_group_params:
		param_groups.append({'params': base_group_params, 'lr': lr, 'weight_decay': weight_decay})
	if log_tuni_param is not None:
		log_tuni_lr = float(getattr(args, 'log_uniformity_lr', lr))
		param_groups.append({'params': [log_tuni_param], 'lr': log_tuni_lr, 'weight_decay': 0.0})

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
		self.uses_text_inputs = _uses_text_inputs(args)
		# Text encoders (SimKGC) already encode both directions in one forward triplet.
		# Index KGE with reciprocal relations needs explicit inverse triplets so
		# inverse-relation embeddings are trained (backward LP eval uses them).
		add_backward_triplet = self.uses_text_inputs or use_reciprocal_relations(args)
		self.train_examples = load_data(
			args.train_path, add_forward_triplet=True, add_backward_triplet=add_backward_triplet)
		logger.info(
			'Training examples: %d (backward triplets=%s)',
			len(self.train_examples), add_backward_triplet,
		)
		self.entity_dict = get_entity_dict()
		self.model = model if model is not None else _load_encoder(args)
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
		self.au_per_epoch = config_bool(args, 'au_per_epoch', False)
		if self.au_per_epoch and self.uses_text_inputs:
			logger.warning('au_per_epoch is not supported for text encoders; using per-batch AU.')
			self.au_per_epoch = False
		if self.au_per_epoch:
			self.weight_decay = float(weight_decay)
			self._build_epoch_uniformity_representatives()
			logger.info(
				'KGAU au_per_epoch: one optimizer step per epoch; '
				'alignment/uniformity use the full training set (batch_size=%d is forward chunk size only).',
				batch_size,
			)
		else:
			self.weight_decay = float(weight_decay) / num_batches

		tuni_val = _config_float(args, 'tuni', _config_float(args, 'temperature', _config_float(args, 't', 2.0)))
		learnable_tuni = config_bool(args, 'learnable_uniformity_scale', False)

		# Alignment mode is opt-in: cosine by default for all encoders; only pRotatE-AU sets
		# ``sin_phase`` (via config and/or encoder ``kga_u_alignment_mode``).
		model_obj = get_model_obj(self.model)
		encoder_align = getattr(model_obj, 'kga_u_alignment_mode', None)
		alignment_mode = getattr(args, 'alignment_mode', None) or encoder_align or 'cosine'
		normalize_uniformity = getattr(args, 'normalize_uniformity', None)
		if normalize_uniformity is None:
			normalize_uniformity = alignment_mode not in ('phase_residual', 'sin_phase')
		if alignment_mode != 'cosine':
			logger.info('KGAU alignment mode: %s (normalize_uniformity=%s)', alignment_mode, normalize_uniformity)
		self.criterion = KGAULoss(
			gamma_q=_config_float(args, 'gamma_q', 1.0),
			gamma_t=_config_float(args, 'gamma_t', 1.0),
			gamma_h=_config_float(args, 'gamma_h', 0.0),
			gamma_ent=_config_float(args, 'gamma_ent', 0.0),
			gamma_cross=_config_float(args, 'gamma_cross', 0.0),
			tuni=tuni_val,
			learnable_tuni=learnable_tuni,
			max_uniformity_samples=int(_config_float(args, 'max_uniformity_samples', 1024)),
			additive_margin=_config_float(args, 'additive_margin', 0.0),
			alignment_mode=alignment_mode,
			normalize_uniformity=bool(normalize_uniformity),
		).to(self.device)
		if learnable_tuni:
			logger.info(
				'Learnable uniformity scale (tuni): initial=%.4f, log_uniformity_lr=%.2e',
				tuni_val,
				float(getattr(args, 'log_uniformity_lr', getattr(args, 'lr', 2e-5))),
			)
		self.optimizer = _build_kgau_optimizer(args, self.model, self.criterion, self.weight_decay)
		self.lr_scheduler = build_lr_scheduler(args, self.optimizer)
		if self.criterion.gamma_ent > 0 and self.uses_text_inputs:
			logger.info(
				'Entity uniformity (text encoder): gamma_ent uses deduplicated batch head+tail vectors '
				'(max_uniformity_samples=%d)',
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

	def _build_epoch_uniformity_representatives(self) -> None:
		"""Precompute one training-row index per unique query/tail/head key for epoch AU."""

		q_keys, t_keys, h_keys = self._uniformity_keys(
			self.train_src, self.train_rel, self.train_dst,
		)
		self.epoch_q_rep_idx = distinct_first_indices(q_keys)
		self.epoch_t_rep_idx = distinct_first_indices(t_keys)
		self.epoch_h_rep_idx = distinct_first_indices(h_keys)

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

		if self.criterion.gamma_cross <= 0:
			return None
		q_uni = select_distinct_rows(q_raw, q_keys)
		t_uni = select_distinct_rows(t_raw, t_keys)
		return self._merge_cross_uniformity_vectors(q_uni, t_uni)

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

	def _embedding_l3_regularization(self, model) -> torch.Tensor | None:
		"""Optional L3 embedding penalty (same form as adversarial DistMult/ComplEx training)."""

		model_obj = get_model_obj(model)
		if hasattr(model_obj, 'adversarial_l3_regularization'):
			return model_obj.adversarial_l3_regularization()
		if hasattr(model_obj, 'entity_embedding') and hasattr(model_obj, 'relation_embedding'):
			return model_obj.entity_embedding.norm(p=3) ** 3 + model_obj.relation_embedding.norm(p=3) ** 3
		return None

	def _apply_embedding_regularization(
		self,
		loss: torch.Tensor,
		*,
		batch_triples: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Add L3 embedding penalty (LibKGE weights or legacy ``regularization`` scalar)."""

		ent_weight = float(getattr(self.args, 'entity_regularize_weight', 0.0) or 0.0)
		rel_weight = float(getattr(self.args, 'relation_regularize_weight', 0.0) or 0.0)
		if ent_weight > 0.0 or rel_weight > 0.0:
			return apply_kge_regularization(
				loss,
				self.model,
				self.args,
				batch_triples=batch_triples,
			)

		reg_coef = _config_float(self.args, 'regularization', 0.0)
		if reg_coef <= 0.0:
			return loss
		l3_term = self._embedding_l3_regularization(self.model)
		if l3_term is None:
			return loss
		return loss + reg_coef * l3_term

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
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, float]:
		"""KGAU loss with deduplicated uniformity inputs (by entity/relation id keys)."""

		q_uni, t_uni, h_uni, n_unique_q, n_unique_t = self._distinct_uniformity_inputs(
			q_raw, t_raw, h_raw, q_keys, t_keys, h_keys)
		cross_uni = self._cross_uniformity_vectors(q_raw, t_raw, q_keys, t_keys)
		loss, l_align, l_unif, margin_active_frac = self.criterion(
			q_raw, t_raw, h_raw, ent_raw, q_uni=q_uni, t_uni=t_uni, h_uni=h_uni,
			cross_uni=cross_uni, return_stats=True)
		loss = self._apply_embedding_regularization(loss, batch_triples=batch_triples)
		return loss, l_align, l_unif, n_unique_q, n_unique_t, margin_active_frac

	def _batch_entity_uniformity_vectors(
		self,
		h_raw: torch.Tensor,
		t_raw: torch.Tensor,
		h_keys: torch.Tensor,
		t_keys: torch.Tensor,
	) -> torch.Tensor | None:
		"""Build entity-uniformity inputs from unique batch heads and tails (SimKGC-style).

		Matches ``--uniformity-on-entity`` in the standalone SimKGC repo: one pooled
		uniformity term over deduplicated head/tail entity vectors from the current
		batch, with gradients flowing through the main forward pass.
		"""

		if self.criterion.gamma_ent <= 0 or h_raw.size(0) == 0:
			return None

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

		if self.criterion.gamma_ent <= 0:
			return None
		kwargs = _entity_embeddings_call_kwargs(model, self.device, self.criterion)
		# Optional encoder hook (pRotatE-AU only); all other encoders use ``entity_embeddings``.
		if hasattr(model, 'au_entity_embeddings'):
			ent = model.au_entity_embeddings(**kwargs)
		elif hasattr(model, 'entity_embeddings'):
			ent = model.entity_embeddings(**kwargs)
		else:
			return None
		if ent is not None and hasattr(model, '_normalize_lp_vector'):
			ent = model._normalize_lp_vector(ent)
		return ent

	def _entity_uniformity_vectors_for_loss(
		self,
		model,
		h_raw: torch.Tensor,
		t_raw: torch.Tensor,
		h_keys: torch.Tensor,
		t_keys: torch.Tensor,
	) -> torch.Tensor | None:
		"""Entity vectors for ``gamma_ent``: batch dedup (text) or full table (embedding encoders)."""

		if self.uses_text_inputs:
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

	def _au_vectors_at_indices(
		self,
		model,
		indices: torch.Tensor,
		chunk_size: int,
		vector: str,
	) -> torch.Tensor:
		"""Fetch query, tail, or head AU vectors for training rows (chunked for memory)."""

		parts: list[torch.Tensor] = []
		for start in range(0, indices.numel(), chunk_size):
			chunk_idx = indices[start:start + chunk_size]
			ss = self.train_src.index_select(0, chunk_idx)
			rs = self.train_rel.index_select(0, chunk_idx)
			ts = self.train_dst.index_select(0, chunk_idx)
			q_raw, t_raw, h_raw = self._au_representation_batch(model, ss, rs, ts)
			if vector == 'q':
				parts.append(q_raw)
			elif vector == 't':
				parts.append(t_raw)
			else:
				parts.append(h_raw)
		return torch.cat(parts, dim=0)

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

	def _train_au_epoch(
		self,
		model,
		epoch: int,
		chunk_size: int,
		use_amp: bool,
	) -> tuple[float, float, float, int, int, float]:
		"""One optimizer step per epoch: global alignment mean + full-set uniformity."""

		del epoch
		num_examples = self.train_src.size(0)
		perm = torch.randperm(num_examples, device=self.train_src.device)
		src = self.train_src.index_select(0, perm)
		rel = self.train_rel.index_select(0, perm)
		dst = self.train_dst.index_select(0, perm)

		self.optimizer.zero_grad()
		align_loss_sum = 0.0

		for start in range(0, num_examples, chunk_size):
			end = min(start + chunk_size, num_examples)
			ss, rs, ts = src[start:end], rel[start:end], dst[start:end]
			chunk = end - start
			fraction = chunk / num_examples
			if use_amp:
				with torch.amp.autocast(device_type='cuda'):
					q_raw, t_raw, _ = self._au_representation_batch(model, ss, rs, ts)
					l_align = self.criterion.forward_alignment(q_raw, t_raw)
				self._backward_au_loss(l_align, fraction, use_amp=True)
			else:
				q_raw, t_raw, _ = self._au_representation_batch(model, ss, rs, ts)
				l_align = self.criterion.forward_alignment(q_raw, t_raw)
				self._backward_au_loss(l_align, fraction, use_amp=False)
			align_loss_sum += l_align.item() * chunk

		q_uni = self._au_vectors_at_indices(model, self.epoch_q_rep_idx, chunk_size, 'q') if (
			self.criterion.gamma_q > 0 or self.criterion.gamma_cross > 0
		) else None
		t_uni = self._au_vectors_at_indices(model, self.epoch_t_rep_idx, chunk_size, 't') if (
			self.criterion.gamma_t > 0 or self.criterion.gamma_cross > 0
		) else None
		h_uni = self._au_vectors_at_indices(model, self.epoch_h_rep_idx, chunk_size, 'h') if self.criterion.gamma_h > 0 else None
		ent_raw = self._catalog_entity_uniformity_vectors(model)
		cross_uni = self._merge_cross_uniformity_vectors(q_uni, t_uni)

		dummy_q = q_uni if q_uni is not None else t_uni if t_uni is not None else h_uni
		if dummy_q is None:
			dummy_q = ent_raw if ent_raw is not None else next(model.parameters())
		dummy_t = t_uni if t_uni is not None else dummy_q

		if use_amp:
			with torch.amp.autocast(device_type='cuda'):
				l_unif, margin_active = self.criterion.forward_uniformity(
					dummy_q, dummy_t, q_uni=q_uni, t_uni=t_uni, h=h_uni, h_uni=h_uni, ent=ent_raw,
					cross_uni=cross_uni,
				)
				loss = self._apply_embedding_regularization(l_unif)
			self.scaler.scale(loss).backward()
		else:
			l_unif, margin_active = self.criterion.forward_uniformity(
				dummy_q, dummy_t, q_uni=q_uni, t_uni=t_uni, h=h_uni, h_uni=h_uni, ent=ent_raw,
				cross_uni=cross_uni,
			)
			loss = self._apply_embedding_regularization(l_unif)
			loss.backward()

		self._optimizer_step(use_amp)

		n_unique_q = int(self.epoch_q_rep_idx.numel()) if self.criterion.gamma_q > 0 else 0
		n_unique_t = int(self.epoch_t_rep_idx.numel()) if self.criterion.gamma_t > 0 else 0
		avg_align = align_loss_sum / max(num_examples, 1)
		avg_unif = l_unif.item()
		avg_loss = avg_align + avg_unif
		return avg_loss, avg_align, avg_unif, n_unique_q, n_unique_t, margin_active

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

	def _au_representation_batch(
		self,
		model,
		ss: torch.Tensor,
		rs: torch.Tensor,
		ts: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Fetch AU vectors via decoupled embedders and optional scorer query builder."""

		if hasattr(model, 'get_queries_targets'):
			return model.get_queries_targets(ss, rs, ts)

		h_emb = model.embed_s(ss)
		r_emb = model.embed_p(rs)
		t_emb = model.embed_o(ts)
		scorer = model.get_scorer()
		if hasattr(scorer, 'build_query'):
			try:
				q_emb = scorer.build_query(h_emb, r_emb)
				return q_emb, t_emb, h_emb
			except NotImplementedError:
				pass
		return h_emb * r_emb, t_emb, h_emb

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
		batch_triples = torch.stack([ss, rs, ts], dim=1)
		if use_amp:
			with torch.amp.autocast(device_type='cuda'):
				q_raw, t_raw, h_raw = self._au_representation_batch(model, ss, rs, ts)
				ent_raw = self._entity_uniformity_vectors_for_loss(
					model, h_raw, t_raw, h_keys, t_keys)
				loss, l_align, l_unif, _, _, margin_active = self._au_loss_with_distinct_keys(
					q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys, batch_triples=batch_triples)
			self.scaler.scale(loss).backward()
			self.scaler.unscale_(self.optimizer)
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
			self.scaler.step(self.optimizer)
			self.scaler.update()
		else:
			q_raw, t_raw, h_raw = self._au_representation_batch(model, ss, rs, ts)
			ent_raw = self._entity_uniformity_vectors_for_loss(
				model, h_raw, t_raw, h_keys, t_keys)
			loss, l_align, l_unif, _, _, margin_active = self._au_loss_with_distinct_keys(
				q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys, batch_triples=batch_triples)
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
			batch_triples = torch.stack([ss[start:end], rs[start:end], ts[start:end]], dim=1)
			if use_amp:
				with torch.amp.autocast(device_type='cuda'):
					q_raw, t_raw, h_raw = self._au_representation_batch(
						model, ss[start:end], rs[start:end], ts[start:end])
					ent_raw = self._entity_uniformity_vectors_for_loss(
						model, h_raw, t_raw, h_keys, t_keys)
					loss, l_align, l_unif, _, _, margin_active = self._au_loss_with_distinct_keys(
						q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys, batch_triples=batch_triples)
				self._backward_au_loss(loss, fraction, use_amp=True)
			else:
				q_raw, t_raw, h_raw = self._au_representation_batch(
					model, ss[start:end], rs[start:end], ts[start:end])
				ent_raw = self._entity_uniformity_vectors_for_loss(
					model, h_raw, t_raw, h_keys, t_keys)
				loss, l_align, l_unif, _, _, margin_active = self._au_loss_with_distinct_keys(
					q_raw, t_raw, h_raw, ent_raw, q_keys, t_keys, h_keys, batch_triples=batch_triples)
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
						ent_raw = self._entity_uniformity_vectors_for_loss(
							model, h_raw, t_raw, h_keys, t_keys)
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
					ent_raw = self._entity_uniformity_vectors_for_loss(
						model, h_raw, t_raw, h_keys, t_keys)
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
		elif self.au_per_epoch:
			loss, l_align, l_unif, n_uq, n_ut, margin_active = self._train_au_epoch(
				model, epoch, batch_size, use_amp,
			)
			n_train = len(self.train_examples)
			epoch_loss = loss * n_train
			epoch_align_loss = l_align * n_train
			epoch_unif_loss = l_unif * n_train
			epoch_unique_q = float(n_uq)
			epoch_unique_t = float(n_ut)
			epoch_margin_active = margin_active
			epoch_batches = 1
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
		unique_scope = 'epoch' if self.au_per_epoch else 'batch'
		tuni_suffix = ''
		if hasattr(self.criterion, 'log_tuni'):
			tuni_suffix = f' | tuni: {_tuni_scalar(self.criterion):.4f}'
		if float(self.criterion.additive_margin) > 0.0:
			logger.info(
				'[EPOCH %s] train loss: %.6f | align: %.6f | uniformity: %.6f | '
				'unique q/t per %s: %.0f/%.0f%s | margin-buffer pairs: %.2f%%%s',
				display_epoch, avg_loss, avg_align_loss, avg_unif_loss,
				unique_scope, avg_unique_q, avg_unique_t,
				'' if self.au_per_epoch else f' (of {batch_size})',
				100.0 * avg_margin_active,
				tuni_suffix,
			)
		else:
			logger.info(
				'[EPOCH %s] train loss: %.6f | align: %.6f | uniformity: %.6f | '
				'unique q/t per %s: %.0f/%.0f%s%s',
				display_epoch, avg_loss, avg_align_loss, avg_unif_loss,
				unique_scope, avg_unique_q, avg_unique_t,
				'' if self.au_per_epoch else f' (of {batch_size})',
				tuni_suffix,
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
		if hasattr(self.criterion, 'log_tuni'):
			self.train_component_losses['tuni'] = _tuni_scalar(self.criterion)
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
		for epoch in range(self.args.epochs):
			epoch_train_start = time.time()
			self.memory_tracker.begin_phase()
			train_loss = self.train_epoch(epoch)
			self.memory_tracker.end_phase('train')
			self.train_time += time.time() - epoch_train_start

			validated = self._should_validate(epoch)
			metric_dict: dict = {}
			if validated:
				eval_start = time.time()
				self.memory_tracker.begin_phase()
				metric_dict = self.eval_epoch(epoch, train_loss=train_loss)
				self.memory_tracker.end_phase('eval')
				self.valid_time += time.time() - eval_start

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

			if self.lr_scheduler is not None and metric_dict and 'mrr' in metric_dict:
				from torch.optim.lr_scheduler import ReduceLROnPlateau
				if isinstance(self.lr_scheduler, ReduceLROnPlateau):
					self.lr_scheduler.step(metric_dict['mrr'])

			if patience is not None and bad_counts >= patience and (epoch + 1) >= min_epochs:
				logger.info(
					'[EARLY STOP] No validation MRR improvement for %d evaluations (epoch %s).',
					patience, epoch + 1,
				)
				break

		self.total_time = time.time() - total_start_time
		logger.info('[Timing] Training time (s): %.2f', round(self.train_time, 2))
		logger.info('[Timing] Valid time (s): %.2f', round(self.valid_time, 2))
		logger.info('[Timing] Total run time (s): %.2f', round(self.total_time, 2))
		logger.info('[Memory] Training peak: %s', format_memory(self.memory_tracker.train_peak_mb))
		logger.info('[Memory] Eval peak: %s', format_memory(self.memory_tracker.eval_peak_mb))
		logger.info('[Memory] Peak memory: %s', format_memory(self.memory_tracker.peak_memory_mb))

		return {
			'best_epoch': None if self.best_metric is None else self.best_metric.get('epoch'),
			'best_mrr': None if self.best_metric is None else self.best_metric.get('score'),
			'train_time': self.train_time,
			'valid_time': self.valid_time,
			'total_time': self.total_time,
			**self.memory_tracker.to_dict(),
		}

Strategy = KGAUStrategy
