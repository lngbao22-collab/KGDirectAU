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
	if max_steps is not None and uses_step_cadence(args):
		return max_steps // 2
	return None


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
	trainer.next_lr_decay_step = resolve_warm_up_steps(trainer.args)
	trainer.lr_decay_factor = resolve_lr_decay_factor(trainer.args)
	trainer._stop_training = False


def increment_trainer_global_step(trainer: Any) -> int:
	trainer.global_step = get_trainer_global_step(trainer) + 1
	return trainer.global_step


def maybe_decay_lr_at_step(trainer: Any, step: int) -> None:
	"""RotatE-style LR decay: multiply LR by ``lr_decay_factor`` once ``step >= warm_up_steps``."""

	next_step = getattr(trainer, 'next_lr_decay_step', None)
	optimizer = getattr(trainer, 'optimizer', None)
	if next_step is None or optimizer is None or step < int(next_step):
		return

	decay_factor = max(float(getattr(trainer, 'lr_decay_factor', 0.1)), 0.0)
	new_lr = optimizer.param_groups[0]['lr'] * decay_factor
	for param_group in optimizer.param_groups:
		param_group['lr'] = new_lr

	from utils.logger import logger

	logger.info('Change learning rate to %.8f at step %d', new_lr, step)
	trainer.next_lr_decay_step = int(next_step * 3)


def should_stop_at_step(trainer: Any, step: int) -> bool:
	max_steps = resolve_max_steps(trainer.args)
	if max_steps is not None and step >= max_steps:
		return True
	return bool(getattr(trainer, '_stop_training', False))
