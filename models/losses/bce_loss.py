"""Helpers for treating ComplEx adversarial BCE scores as classification logits."""

import math
import os


_BCE_OFFSET_LOSS_BASENAMES = frozenset({'bce_loss.py', 'bce_loss'})
_LOGIT_CLASSIFICATION_LOSS_BASENAMES = frozenset({
	'bce_loss.py',
	'bce_loss',
	'adversarial_bce_loss.py',
	'adversarial_bce_loss',
})


def _loss_basename(args) -> str:
	loss_path = str(getattr(args, 'model_loss_path', '') or '').lower()
	return os.path.basename(loss_path)


def uses_bce_logit_offset(args) -> bool:
	"""Return True when inference should add the BCE logit offset to raw scores."""

	return _loss_basename(args) in _BCE_OFFSET_LOSS_BASENAMES


def uses_logit_classification_scores(args) -> bool:
	"""Return True when triple classification should sigmoid scores into probabilities.

	AU scores are not calibrated logits; sigmoid saturates large values and
	collapses the TC threshold. Those losses keep raw scores.
	"""

	return _loss_basename(args) in _LOGIT_CLASSIFICATION_LOSS_BASENAMES


def bce_logit_offset(args) -> float:
	"""Return the BCE logit offset (``train.loss_arg``) for inference."""

	if not uses_bce_logit_offset(args):
		return 0.0
	offset = getattr(args, 'bce_offset', None)
	if offset is not None:
		return float(offset)
	raw = getattr(args, 'loss_arg', None)
	if raw is None or (isinstance(raw, float) and math.isnan(raw)):
		return 0.0
	return float(raw)
