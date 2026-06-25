"""Report alignment-uniformity (AU) loss during training without affecting gradients."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

import torch

from models.builder import config_bool
from models.losses.au_loss import KGAULoss, select_distinct_rows
from models.strategies.kgau_strategy import (
	_alpha_schedule_enabled,
	_config_float,
	_gamma_schedule_enabled,
	_scheduled_alpha_mult,
	_scheduled_gamma_mult,
	_scheduled_tuni_value,
)
from utils.device import get_model_obj, move_to_cuda
from utils.logger import logger


def report_au_enabled(args) -> bool:
	return config_bool(args, 'report_au', False)


def _build_au_criterion(args, device: torch.device, *, alignment_mode: str | None = None) -> KGAULoss:
	tuni_val = _config_float(args, 'tuni', _config_float(args, 'temperature', _config_float(args, 't', 2.0)))
	alignment_mode = alignment_mode or getattr(args, 'alignment_mode', None) or 'cosine'
	normalize_uniformity = getattr(args, 'normalize_uniformity', None)
	if normalize_uniformity is None:
		normalize_uniformity = alignment_mode not in ('phase_residual', 'sin_phase')
	return KGAULoss(
		gamma_q=_config_float(args, 'gamma_q', 1.0),
		gamma_t=_config_float(args, 'gamma_t', 1.0),
		gamma_h=_config_float(args, 'gamma_h', 0.0),
		gamma_ent=_config_float(args, 'gamma_ent', 0.0),
		gamma_cross=_config_float(args, 'gamma_cross', 0.0),
		alpha=_config_float(args, 'alpha', 1.0),
		tuni=tuni_val,
		learnable_tuni=False,
		learnable_au_alpha=False,
		learnable_au_gammas=False,
		tuni_as_alpha=config_bool(args, 'tuni_as_alpha', False),
		max_uniformity_samples=int(_config_float(args, 'max_uniformity_samples', 1024)),
		additive_margin=_config_float(args, 'additive_margin', 0.0),
		alignment_mode=alignment_mode,
		normalize_uniformity=bool(normalize_uniformity),
		average_uniformity_terms=config_bool(args, 'average_uniformity_terms', False),
		uniformity_full_pdist=config_bool(args, 'uniformity_full_pdist', False),
	).to(device)


def _uniformity_keys(
	head_indices: torch.Tensor,
	relation_indices: torch.Tensor,
	tail_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _merge_cross_uniformity_vectors(
	q_uni: torch.Tensor | None,
	t_uni: torch.Tensor | None,
) -> torch.Tensor | None:
	parts = [x for x in (q_uni, t_uni) if x is not None and x.size(0) > 0]
	if not parts:
		return None
	cross = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
	return cross if cross.size(0) >= 2 else None


class AUReporter:
	"""Compute and log AU metrics each epoch using ``tuni`` / ``gamma_*`` config only."""

	def __init__(self, args, trainer) -> None:
		self.args = args
		self.trainer = trainer
		self.device = trainer.device
		self.model = trainer.model
		model_obj = get_model_obj(self.model)
		encoder_align = getattr(model_obj, 'kgau_alignment_mode', None)
		alignment_mode = getattr(args, 'alignment_mode', None) or encoder_align or 'cosine'
		self.criterion = _build_au_criterion(args, self.device, alignment_mode=alignment_mode)
		self.au_deduplicate = config_bool(args, 'au_deduplicate', True)
		self.uses_text_inputs = self._resolve_uses_text_inputs()
		self.max_batches = getattr(args, 'report_au_max_batches', None)
		if self.max_batches is not None:
			self.max_batches = max(int(self.max_batches), 1)
		self.output_dir = Path(args.output_dir)
		self.json_path = self.output_dir / 'au_loss_report.json'
		self.csv_path = self.output_dir / 'au_loss_report.csv'
		self.mrr_loss_chart_path = self.output_dir / 'au_mrr_loss_by_epoch.png'
		self.align_uniformity_chart_path = self.output_dir / 'au_align_uniformity_by_epoch.png'
		self.history: list[dict[str, Any]] = []
		self._cached_ent_raw: torch.Tensor | None = None
		self._cached_ent_epoch: int | None = None
		logger.info(
			'AU report mode enabled: tuni=%.4f, gammas q/t/h/ent/cross=%.4g/%.4g/%.4g/%.4g/%.4g',
			float(self.criterion.tuni if not torch.is_tensor(self.criterion.tuni) else self.criterion.tuni.item()),
			self.criterion._gamma_init_value('q'),
			self.criterion._gamma_init_value('t'),
			self.criterion._gamma_init_value('h'),
			self.criterion._gamma_init_value('ent'),
			self.criterion._gamma_init_value('cross'),
		)

	def _resolve_uses_text_inputs(self) -> bool:
		if hasattr(self.trainer, 'uses_text_inputs'):
			return bool(self.trainer.uses_text_inputs)
		model_obj = get_model_obj(self.model)
		return getattr(model_obj, 'training_input_mode', 'indices') == 'tokens'

	def _apply_epoch_schedules(self, epoch: int) -> None:
		if _alpha_schedule_enabled(self.args):
			self.criterion.set_alpha_schedule_mult(_scheduled_alpha_mult(self.args, epoch))
		if _gamma_schedule_enabled(self.args):
			self.criterion.set_gamma_schedule_mult(_scheduled_gamma_mult(self.args, epoch))
		if config_bool(self.args, 'tuni_linear_schedule', False):
			self.criterion.tuni = float(_scheduled_tuni_value(self.args, epoch))

	def _reporting_criterion(self) -> KGAULoss:
		"""Prefer the trainer's live criterion (already scheduled for this epoch)."""

		return getattr(self.trainer, 'criterion', self.criterion)

	def _epoch_config_snapshot(self) -> dict[str, float]:
		criterion = self._reporting_criterion()
		snapshot = {'tuni': float(criterion.tuni if not torch.is_tensor(criterion.tuni) else criterion.tuni.item())}
		if not config_bool(self.args, 'tuni_as_alpha', False):
			snapshot['alpha'] = criterion.alpha_value()
		for name in ('q', 't', 'h', 'ent', 'cross'):
			if criterion.gamma_active(name):
				snapshot[f'gamma_{name}'] = criterion.gamma_value(name)
		return snapshot

	def _training_example_count(self) -> int:
		if hasattr(self.trainer, 'train_examples'):
			count = len(self.trainer.train_examples)
			if getattr(self.trainer, 'kgau_bidirectional', False):
				count *= 2
			return count
		if hasattr(self.trainer, 'train_src'):
			return int(self.trainer.train_src.size(0))
		return 0

	def _reuse_train_component_losses(self) -> dict[str, float] | None:
		"""Return epoch AU metrics already accumulated during training, if available."""

		if self.max_batches is not None:
			return None
		components = getattr(self.trainer, 'train_component_losses', None)
		if not isinstance(components, dict):
			return None
		if 'align' not in components or 'uniformity' not in components:
			return None
		return components

	def _record_epoch_metrics(
		self,
		epoch: int,
		*,
		au_loss: float,
		align_loss: float,
		uniformity_loss: float,
		num_examples: int,
	) -> None:
		record = {
			'epoch': epoch + 1,
			'au_loss': au_loss,
			'align_loss': align_loss,
			'uniformity_loss': uniformity_loss,
			'mrr': None,
			'num_examples': num_examples,
			**self._epoch_config_snapshot(),
		}
		self.history.append(record)
		self._write_files()
		logger.info(
			'[AU REPORT] Epoch %d | loss: %.6f | align: %.6f | uniformity: %.6f',
			record['epoch'], record['au_loss'], record['align_loss'], record['uniformity_loss'],
		)

	def _distinct_uniformity_inputs(
		self,
		q_raw: torch.Tensor,
		t_raw: torch.Tensor,
		h_raw: torch.Tensor,
		q_keys: torch.Tensor,
		t_keys: torch.Tensor,
		h_keys: torch.Tensor,
	) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
		if not self.au_deduplicate:
			return None, None, None
		q_uni = select_distinct_rows(q_raw, q_keys) if self.criterion.gamma_active('q') else None
		t_uni = select_distinct_rows(t_raw, t_keys) if self.criterion.gamma_active('t') else None
		h_uni = select_distinct_rows(h_raw, h_keys) if self.criterion.gamma_active('h') else None
		return q_uni, t_uni, h_uni

	def _cross_uniformity_vectors(
		self,
		q_raw: torch.Tensor,
		t_raw: torch.Tensor,
		q_keys: torch.Tensor,
		t_keys: torch.Tensor,
	) -> torch.Tensor | None:
		if not self.criterion.gamma_active('cross'):
			return None
		if self.au_deduplicate:
			q_uni = select_distinct_rows(q_raw, q_keys)
			t_uni = select_distinct_rows(t_raw, t_keys)
		else:
			q_uni, t_uni = q_raw, t_raw
		return _merge_cross_uniformity_vectors(q_uni, t_uni)

	def _entity_uniformity_vectors(
		self,
		model,
		h_raw: torch.Tensor,
		t_raw: torch.Tensor,
		h_keys: torch.Tensor,
		t_keys: torch.Tensor,
		*,
		epoch: int,
		head_indices: torch.Tensor | None = None,
		tail_indices: torch.Tensor | None = None,
	) -> torch.Tensor | None:
		if not self.criterion.gamma_active('ent'):
			return None
		if self.uses_text_inputs or config_bool(self.args, 'entity_uniformity_batch', False):
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
		if self._cached_ent_epoch == epoch and self._cached_ent_raw is not None:
			return self._cached_ent_raw
		model_obj = get_model_obj(model)
		if hasattr(model_obj, 'au_entity_embeddings'):
			ent = model_obj.au_entity_embeddings(device=self.device)
		elif hasattr(model_obj, 'entity_embeddings'):
			max_samples = int(getattr(self.criterion, 'max_uniformity_samples', 0) or 0)
			ent = model_obj.entity_embeddings(device=self.device, max_samples=max_samples or None)
		else:
			return None
		if ent is not None and hasattr(model_obj, '_normalize_au_vector'):
			ent = model_obj._normalize_au_vector(ent)
		self._cached_ent_raw = ent
		self._cached_ent_epoch = epoch
		return ent

	def _batch_entity_uniformity_vectors(
		self,
		h_raw: torch.Tensor,
		t_raw: torch.Tensor,
		h_keys: torch.Tensor,
		t_keys: torch.Tensor,
	) -> torch.Tensor | None:
		if h_raw.size(0) == 0:
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

	def _compute_batch_au(
		self,
		model,
		q_raw: torch.Tensor,
		t_raw: torch.Tensor,
		h_raw: torch.Tensor,
		q_keys: torch.Tensor,
		t_keys: torch.Tensor,
		h_keys: torch.Tensor,
		*,
		epoch: int,
		head_indices: torch.Tensor | None = None,
		tail_indices: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		q_uni, t_uni, h_uni = self._distinct_uniformity_inputs(q_raw, t_raw, h_raw, q_keys, t_keys, h_keys)
		cross_uni = self._cross_uniformity_vectors(q_raw, t_raw, q_keys, t_keys)
		ent_raw = self._entity_uniformity_vectors(
			model, h_raw, t_raw, h_keys, t_keys, epoch=epoch,
			head_indices=head_indices, tail_indices=tail_indices,
		)
		total, l_align, l_unif = self.criterion(
			q_raw, t_raw, h_raw, ent_raw, q_uni=q_uni, t_uni=t_uni, h_uni=h_uni, cross_uni=cross_uni,
		)
		return total, l_align, l_unif

	def _examples_to_tensors(self, examples) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		entity_dict = self.trainer.entity_dict
		relation_to_idx = getattr(self.trainer, 'relation_to_idx', None)
		if relation_to_idx is None:
			from models.builder import _relation_index_map

			relation_to_idx = _relation_index_map(get_model_obj(self.model))
		head_indices = torch.tensor(
			[entity_dict.entity_to_idx(example.head_id) for example in examples], dtype=torch.long,
		)
		relation_indices = torch.tensor(
			[relation_to_idx[example.relation] for example in examples], dtype=torch.long,
		)
		tail_indices = torch.tensor(
			[entity_dict.entity_to_idx(example.tail_id) for example in examples], dtype=torch.long,
		)
		return head_indices, relation_indices, tail_indices

	def _uniformity_keys_from_examples(self, examples) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		head_indices, relation_indices, tail_indices = self._examples_to_tensors(examples)
		return _uniformity_keys(
			head_indices.to(self.device),
			relation_indices.to(self.device),
			tail_indices.to(self.device),
		)

	def _resolve_index_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
		if hasattr(self.trainer, 'train_src'):
			return self.trainer.train_src, self.trainer.train_rel, self.trainer.train_dst
		if getattr(self.trainer, 'train_triples', None) is not None:
			triples = self.trainer.train_triples
			return triples[:, 0], triples[:, 1], triples[:, 2]
		return None

	def _iter_index_batches(self, epoch: int) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
		tensors = self._resolve_index_tensors()
		if tensors is None:
			return
		src, rel, dst = tensors
		batch_size = max(getattr(self.args, 'batch_size', 1), 1)
		num_examples = src.size(0)
		order = torch.randperm(num_examples, device=src.device)
		src = src.index_select(0, order)
		rel = rel.index_select(0, order)
		dst = dst.index_select(0, order)
		batch_count = 0
		for start in range(0, num_examples, batch_size):
			if self.max_batches is not None and batch_count >= self.max_batches:
				break
			end = start + batch_size
			yield src[start:end], rel[start:end], dst[start:end]
			batch_count += 1

	def _iter_text_batches(self) -> Iterator[dict]:
		loader = getattr(self.trainer, 'train_loader', None)
		if loader is None:
			return
		batch_count = 0
		for batch_dict in loader:
			if self.max_batches is not None and batch_count >= self.max_batches:
				break
			yield batch_dict
			batch_count += 1

	@torch.no_grad()
	def on_epoch_end(self, epoch: int) -> None:
		components = self._reuse_train_component_losses()
		if components is not None:
			align_loss = float(components['align'])
			uniformity_loss = float(components['uniformity'])
			au_loss = float(components.get('au', align_loss + uniformity_loss))
			num_examples = int(components.get('num_examples') or self._training_example_count())
			self._record_epoch_metrics(
				epoch,
				au_loss=au_loss,
				align_loss=align_loss,
				uniformity_loss=uniformity_loss,
				num_examples=num_examples,
			)
			return

		self._apply_epoch_schedules(epoch)
		model = get_model_obj(self.model)
		was_training = self.model.training
		self.model.eval()

		loss_sum = 0.0
		align_sum = 0.0
		unif_sum = 0.0
		num_examples = 0

		if self.uses_text_inputs:
			for batch_dict in self._iter_text_batches():
				if torch.cuda.is_available():
					batch_dict = move_to_cuda(batch_dict)
				examples = batch_dict['batch_data']
				q_keys, t_keys, h_keys = self._uniformity_keys_from_examples(examples)
				outputs = self.model(**batch_dict)
				q_raw = outputs['hr_vector']
				t_raw = outputs['tail_vector']
				h_raw = outputs['head_vector']
				batch_n = len(examples)
				total, l_align, l_unif = self._compute_batch_au(
					model, q_raw, t_raw, h_raw, q_keys, t_keys, h_keys, epoch=epoch,
				)
				loss_sum += float(total.item()) * batch_n
				align_sum += float(l_align.item()) * batch_n
				unif_sum += float(l_unif.item()) * batch_n
				num_examples += batch_n
		else:
			for ss, rs, ts in self._iter_index_batches(epoch):
				ss = ss.to(self.device)
				rs = rs.to(self.device)
				ts = ts.to(self.device)
				q_keys, t_keys, h_keys = _uniformity_keys(ss, rs, ts)
				q_raw, t_raw, h_raw = model.get_queries_targets(ss, rs, ts)
				batch_n = ss.size(0)
				total, l_align, l_unif = self._compute_batch_au(
					model, q_raw, t_raw, h_raw, q_keys, t_keys, h_keys, epoch=epoch,
					head_indices=ss, tail_indices=ts,
				)
				loss_sum += float(total.item()) * batch_n
				align_sum += float(l_align.item()) * batch_n
				unif_sum += float(l_unif.item()) * batch_n
				num_examples += batch_n

		if was_training:
			self.model.train()

		if num_examples == 0:
			logger.warning('[AU REPORT] Epoch %d: no training batches available; skipping', epoch + 1)
			return

		self._record_epoch_metrics(
			epoch,
			au_loss=loss_sum / num_examples,
			align_loss=align_sum / num_examples,
			uniformity_loss=unif_sum / num_examples,
			num_examples=num_examples,
		)

	def record_validation(self, epoch: int, metric_dict: dict | None) -> None:
		target_epoch = epoch + 1
		mrr = None
		if metric_dict and 'mrr' in metric_dict:
			mrr = float(metric_dict['mrr'])
		for record in reversed(self.history):
			if record['epoch'] == target_epoch:
				record['mrr'] = mrr
				self._write_files()
				if mrr is not None:
					logger.info('[AU REPORT] Epoch %d | validation MRR: %.6f', target_epoch, mrr)
				return

	def _write_files(self) -> None:
		self.output_dir.mkdir(parents=True, exist_ok=True)
		payload = {
			'config': {
				'tuni': _config_float(self.args, 'tuni', 2.0),
				'gamma_q': _config_float(self.args, 'gamma_q', 1.0),
				'gamma_t': _config_float(self.args, 'gamma_t', 1.0),
				'gamma_h': _config_float(self.args, 'gamma_h', 0.0),
				'gamma_ent': _config_float(self.args, 'gamma_ent', 0.0),
				'gamma_cross': _config_float(self.args, 'gamma_cross', 0.0),
				'alpha': _config_float(self.args, 'alpha', 1.0),
				'tuni_as_alpha': config_bool(self.args, 'tuni_as_alpha', False),
				'au_deduplicate': self.au_deduplicate,
			},
			'history': self.history,
		}
		self.json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

		fieldnames = ['epoch', 'au_loss', 'align_loss', 'uniformity_loss', 'mrr', 'num_examples', 'tuni']
		if not config_bool(self.args, 'tuni_as_alpha', False):
			fieldnames.append('alpha')
		for name in ('q', 't', 'h', 'ent', 'cross'):
			if self._reporting_criterion()._gamma_init_value(name) > 0.0:
				fieldnames.append(f'gamma_{name}')
		with self.csv_path.open('w', newline='', encoding='utf-8') as handle:
			writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
			writer.writeheader()
			for row in self.history:
				writer.writerow(row)

	def _save_charts(self) -> None:
		try:
			import matplotlib.pyplot as plt
		except ImportError:
			logger.warning('[AU REPORT] matplotlib not installed; saved %s and %s only', self.json_path, self.csv_path)
			return

		epochs = [row['epoch'] for row in self.history]
		au_losses = [row['au_loss'] for row in self.history]
		align_losses = [row['align_loss'] for row in self.history]
		unif_losses = [row['uniformity_loss'] for row in self.history]
		mrr_values = [row.get('mrr') for row in self.history]

		fig, ax_loss = plt.subplots(figsize=(8, 5))
		ax_loss.plot(epochs, au_losses, marker='o', color='tab:blue', label='AU loss', linewidth=2)
		ax_loss.set_xlabel('Epoch')
		ax_loss.set_ylabel('AU loss', color='tab:blue')
		ax_loss.tick_params(axis='y', labelcolor='tab:blue')
		ax_loss.grid(True, alpha=0.3)

		mrr_epochs = [epoch for epoch, mrr in zip(epochs, mrr_values) if mrr is not None]
		mrr_series = [mrr for mrr in mrr_values if mrr is not None]
		if mrr_series:
			ax_mrr = ax_loss.twinx()
			ax_mrr.plot(mrr_epochs, mrr_series, marker='s', color='tab:orange', label='MRR', linewidth=2)
			ax_mrr.set_ylabel('MRR', color='tab:orange')
			ax_mrr.tick_params(axis='y', labelcolor='tab:orange')
			lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
			lines_mrr, labels_mrr = ax_mrr.get_legend_handles_labels()
			ax_loss.legend(lines_loss + lines_mrr, labels_loss + labels_mrr, loc='best')
		else:
			ax_loss.legend(loc='best')

		ax_loss.set_title('AU loss and validation MRR by epoch')
		fig.tight_layout()
		fig.savefig(self.mrr_loss_chart_path, dpi=150)
		plt.close(fig)

		fig, ax = plt.subplots(figsize=(8, 5))
		ax.plot(epochs, align_losses, marker='s', label='Alignment', linewidth=2)
		ax.plot(epochs, unif_losses, marker='^', label='Uniformity', linewidth=2)
		ax.set_xlabel('Epoch')
		ax.set_ylabel('Loss')
		ax.set_title('Alignment and uniformity loss by epoch')
		ax.grid(True, alpha=0.3)
		ax.legend()
		fig.tight_layout()
		fig.savefig(self.align_uniformity_chart_path, dpi=150)
		plt.close(fig)
		logger.info(
			'[AU REPORT] Saved charts to %s and %s',
			self.mrr_loss_chart_path,
			self.align_uniformity_chart_path,
		)

	def finalize(self) -> None:
		if not self.history:
			logger.warning('[AU REPORT] No AU history recorded; skipping charts')
			return
		self._write_files()
		self._save_charts()


def attach_au_reporter(trainer, args) -> None:
	if not report_au_enabled(args):
		return
	trainer.au_reporter = AUReporter(args, trainer)


def report_au_after_epoch(trainer, epoch: int) -> None:
	reporter = getattr(trainer, 'au_reporter', None)
	if reporter is not None:
		reporter.on_epoch_end(epoch)


def report_au_validation(trainer, epoch: int, metric_dict: dict | None) -> None:
	reporter = getattr(trainer, 'au_reporter', None)
	if reporter is not None:
		reporter.record_validation(epoch, metric_dict)


def finalize_au_report(trainer) -> None:
	reporter = getattr(trainer, 'au_reporter', None)
	if reporter is not None:
		reporter.finalize()
