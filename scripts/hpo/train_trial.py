"""Run one HPO trial: training only (no test evaluation)."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _repo_root() -> str:
	return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_trial_args(config_path: str):
	"""Load args for a trial config in an isolated interpreter."""

	repo_root = _repo_root()
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)
	sys.argv = ['train_trial.py', '--config-path', config_path]
	if 'configs.config' in sys.modules:
		del sys.modules['configs.config']
	import configs.config as cfg  # noqa: WPS433 — must import after argv
	return cfg.args


def main() -> int:
	parser = argparse.ArgumentParser(description='Run a single HPO training trial')
	parser.add_argument('--trial-config', required=True, help='Path to trial JSON config')
	cli = parser.parse_args()

	repo_root = _repo_root()
	os.chdir(repo_root)
	config_path = os.path.abspath(cli.trial_config)
	if not os.path.exists(config_path):
		print(json.dumps({'ok': False, 'error': f'missing config: {config_path}'}))
		return 1

	args = _load_trial_args(config_path)

	from utils.device import init_hardware
	from models.builder import build_pipeline

	ngpus = init_hardware(args)
	trainer = build_pipeline(args, ngpus_per_node=ngpus)
	train_dataloader = getattr(trainer, 'train_dataloader', None)
	if train_dataloader is not None:
		summary = trainer.train_loop(train_dataloader)
	else:
		summary = trainer.train_loop()

	best_mrr = summary.get('best_mrr')
	if best_mrr is None and getattr(trainer, 'best_metric', None):
		best_mrr = trainer.best_metric.get('score')

	result = {
		'ok': True,
		'best_mrr': float(best_mrr) if best_mrr is not None else 0.0,
		'best_epoch': summary.get('best_epoch'),
		'output_dir': getattr(args, 'output_dir', ''),
		'train_time': summary.get('train_time'),
		'valid_time': summary.get('valid_time'),
	}
	print(json.dumps(result))
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
