"""Universal step-based vs epoch-based training cadence helpers."""

from __future__ import annotations

from typing import Any


def config_int(args: Any, name: str, default: int | None = None) -> int | None:
	value = getattr(args, name, None)
	if value is None:
		return default
	return int(value)


def config_float(args: Any, name: str, default: float) -> float:
	value = getattr(args, name, None)
	return default if value is None else float(value)


def uses_step_cadence(args: Any) -> bool:
	"""Return True when training should follow optimizer steps instead of epoch boundaries."""

	cadence = getattr(args, 'training_cadence', None)
	if cadence is not None:
		return str(cadence).lower() == 'step'
	return config_int(args, 'max_steps', None) is not None


def resolve_max_steps(args: Any) -> int | None:
	return config_int(args, 'max_steps', None)


def resolve_valid_steps(args: Any) -> int | None:
	for key in ('valid_steps', 'eval_every_n_step'):
		value = config_int(args, key, None)
		if value is not None and value > 0:
			return value
	return None


def resolve_save_checkpoint_steps(args: Any) -> int | None:
	value = config_int(args, 'save_checkpoint_steps', None)
	if value is not None and value > 0:
		return value
	return None


def resolve_warm_up_steps(args: Any) -> int | None:
	explicit = config_int(args, 'warm_up_steps', None)
	if explicit is not None:
		return explicit
	max_steps = resolve_max_steps(args)
	if max_steps is not None:
		return max_steps // 2
	return None


def _resolve_warm_up_ratio_values(args: Any) -> list[float]:
	"""Return sorted warm-up ratios in (0, 1), or an empty list when unset/invalid."""

	raw = getattr(args, 'warm_up_ratio', None)
	if raw is None:
		return []
	if isinstance(raw, (float, int, str)):
		raw = [raw]

	ratios: list[float] = []
	for value in raw:
		try:
			ratio = float(value)
		except (TypeError, ValueError):
			continue
		if 0.0 < ratio < 1.0:
			ratios.append(ratio)

	if not ratios:
		return []

	# Keep deterministic order and remove duplicates after float parsing.
	return sorted(set(ratios))


def resolve_warm_up_ratio_steps(args: Any) -> list[int]:
	"""Resolve ratio-based LR decay steps from ``warm_up_ratio`` and ``max_steps``.

	Example: ``warm_up_ratio=[0.2, 0.5, 0.8]`` with ``max_steps=100000``
	produces ``[20000, 50000, 80000]``.
	"""

	max_steps = resolve_max_steps(args)
	if max_steps is None or max_steps <= 0:
		return []

	ratios = _resolve_warm_up_ratio_values(args)
	if not ratios:
		return []

	steps: list[int] = []
	for ratio in ratios:
		step = max(1, int(max_steps * ratio))
		if step >= max_steps:
			continue
		if not steps or step > steps[-1]:
			steps.append(step)

	return steps


def resolve_lr_decay_factor(args: Any) -> float:
	return config_float(args, 'lr_decay_factor', 0.1)


def should_validate_at_step(step: int, args: Any) -> bool:
	interval = resolve_valid_steps(args)
	return interval is not None and step > 0 and step % interval == 0


def should_save_checkpoint_at_step(step: int, args: Any) -> bool:
	interval = resolve_save_checkpoint_steps(args)
	return interval is not None and step > 0 and step % interval == 0


def trainer_supports_step_batches(trainer: Any) -> bool:
	return (
		callable(getattr(trainer, 'iter_training_batches', None))
		and callable(getattr(trainer, 'train_batch', None))
	)


def get_trainer_global_step(trainer: Any) -> int:
	return int(getattr(trainer, 'global_step', 0) or 0)


def init_step_cadence_state(trainer: Any) -> None:
	"""Attach step-cadence fields to any strategy trainer before the step loop runs."""

	if not hasattr(trainer, 'global_step'):
		trainer.global_step = 0
	trainer.lr_decay_steps = resolve_warm_up_ratio_steps(trainer.args)
	trainer.lr_decay_step_index = 0
	if trainer.lr_decay_steps:
		trainer.next_lr_decay_step = trainer.lr_decay_steps[0]
	else:
		trainer.next_lr_decay_step = resolve_warm_up_steps(trainer.args)
	trainer.lr_decay_factor = resolve_lr_decay_factor(trainer.args)
	trainer._stop_training = False


def increment_trainer_global_step(trainer: Any) -> int:
	trainer.global_step = get_trainer_global_step(trainer) + 1
	return trainer.global_step


def _rebuild_trainer_optimizer(trainer: Any, new_lr: float) -> None:
	"""Recreate the optimizer at a new LR (legacy KnowledgeGraphEmbedding warm-up decay)."""

	model = getattr(trainer, 'model', None)
	args = getattr(trainer, 'args', None)
	if model is None or args is None:
		return

	from models.builder import build_optimizer

	weight_decay = config_float(args, 'weight_decay', 0.0)
	params = filter(lambda p: p.requires_grad, model.parameters())
	trainer.optimizer = build_optimizer(args, params, weight_decay)
	for param_group in trainer.optimizer.param_groups:
		param_group['lr'] = new_lr


def _scale_optimizer_lrs(optimizer, scale: float) -> None:
	"""Multiply every param-group LR by ``scale`` (GB-Magic / RotatE-style decay)."""

	for group in optimizer.param_groups:
		group['lr'] = float(group['lr']) * scale


def maybe_decay_lr_at_step(trainer: Any, step: int) -> None:
	"""Apply LR decays at configured milestone steps.

	Priority:
	1) ratio milestones from ``warm_up_ratio`` + ``max_steps``
	2) legacy single ``warm_up_steps`` then geometric ``*3`` schedule
	"""

	next_step = getattr(trainer, 'next_lr_decay_step', None)
	optimizer = getattr(trainer, 'optimizer', None)
	if next_step is None or optimizer is None or step < int(next_step):
		return

	decay_factor = max(float(getattr(trainer, 'lr_decay_factor', 0.1)), 0.0)
	args = getattr(trainer, 'args', None)
	preserve = bool(getattr(args, 'lr_decay_preserve_optimizer', False)) if args is not None else False
	if preserve:
		_scale_optimizer_lrs(optimizer, decay_factor)
		new_lr = float(optimizer.param_groups[0]['lr'])
	else:
		new_lr = float(optimizer.param_groups[0]['lr']) * decay_factor
		_rebuild_trainer_optimizer(trainer, new_lr)

	from utils.logger import logger

	logger.info('Change learning rate to %.8f at step %d', new_lr, step)

	decay_steps = getattr(trainer, 'lr_decay_steps', None) or []
	if decay_steps:
		decay_idx = int(getattr(trainer, 'lr_decay_step_index', 0) or 0) + 1
		trainer.lr_decay_step_index = decay_idx
		trainer.next_lr_decay_step = decay_steps[decay_idx] if decay_idx < len(decay_steps) else None
		return

	trainer.next_lr_decay_step = int(next_step * 3)


def should_stop_at_step(trainer: Any, step: int) -> bool:
	max_steps = resolve_max_steps(trainer.args)
	if max_steps is not None and step >= max_steps:
		return True
	return bool(getattr(trainer, '_stop_training', False))
