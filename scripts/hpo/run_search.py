"""Optuna hyperparameter search for ComplEx-AU (LibKGE-style space)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any

import optuna
from optuna.samplers import TPESampler


def _repo_root() -> str:
	return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_json(path: str) -> dict:
	with open(path, 'r', encoding='utf-8') as handle:
		return json.load(handle)


def _save_json(path: str, payload: dict) -> None:
	os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
	with open(path, 'w', encoding='utf-8') as handle:
		json.dump(payload, handle, indent=2, sort_keys=True)


def _suggest_from_spec(trial: optuna.Trial, name: str, spec: dict, sampled: dict) -> Any:
	param_type = spec['type']
	when = spec.get('when')
	if when:
		for key, expected in when.items():
			if sampled.get(key) != expected:
				return None

	if param_type == 'loguniform':
		return trial.suggest_float(name, float(spec['low']), float(spec['high']), log=True)
	if param_type == 'uniform':
		return trial.suggest_float(name, float(spec['low']), float(spec['high']))
	if param_type == 'int':
		return trial.suggest_int(name, int(spec['low']), int(spec['high']))
	if param_type == 'categorical':
		return trial.suggest_categorical(name, spec['choices'])
	raise ValueError(f'Unsupported parameter type for {name}: {param_type}')


def _uniformity_active(params: dict) -> bool:
	return any(float(params.get(key, 0.0) or 0.0) > 0.0 for key in ('gamma_q', 'gamma_t', 'gamma_h', 'gamma_ent', 'gamma_cross'))


def _build_trial_config(
	base_cfg: dict,
	search_space: dict,
	params: dict,
	*,
	output_dir: str,
	screening: bool,
	screening_epochs: int | None = None,
) -> dict:
	cfg = deepcopy(base_cfg)
	cfg.update(search_space.get('fixed_overrides', {}))
	if screening:
		overrides = dict(search_space.get('screening_overrides', {}))
		if screening_epochs is not None:
			overrides['epochs'] = int(screening_epochs)
		cfg.update(overrides)
	cfg.update({key: value for key, value in params.items() if value is not None})
	cfg['output_dir'] = output_dir
	cfg['output_dir_prefix'] = ''
	return cfg


def _run_trial_subprocess(repo_root: str, trial_config_path: str) -> dict:
	env = os.environ.copy()
	env['PYTHONPATH'] = repo_root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
	cmd = [sys.executable, os.path.join(repo_root, 'scripts', 'hpo', 'train_trial.py'), '--trial-config', trial_config_path]
	proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, env=env)
	if proc.returncode != 0:
		stderr = (proc.stderr or '').strip()
		stdout = (proc.stdout or '').strip()
		raise RuntimeError(f'trial failed (code={proc.returncode})\nstdout={stdout}\nstderr={stderr}')
	lines = [line.strip() for line in (proc.stdout or '').splitlines() if line.strip()]
	for line in reversed(lines):
		try:
			payload = json.loads(line)
			if isinstance(payload, dict) and 'ok' in payload:
				return payload
		except json.JSONDecodeError:
			continue
	raise RuntimeError(f'could not parse trial output:\n{proc.stdout}')


def main() -> int:
	parser = argparse.ArgumentParser(description='Run ComplEx-AU HPO with Optuna')
	parser.add_argument(
		'--search-space',
		default='scripts/hpo/search_space_complex_au_wn18rr.json',
		help='Search space JSON (relative to repo root)',
	)
	parser.add_argument('--n-trials', type=int, default=30, help='Number of Optuna trials')
	parser.add_argument('--n-startup-trials', type=int, default=10, help='Random trials before TPE')
	parser.add_argument('--study-name', default='complex_au_wn18rr', help='Optuna study name')
	parser.add_argument(
		'--storage',
		default='',
		help='Optuna storage URL (default: sqlite under logs/hpo/)',
	)
	parser.add_argument('--screening-epochs', type=int, default=30, help='Epoch cap when --screening is enabled')
	parser.add_argument('--no-screening', dest='screening', action='store_false', help='Full-epoch search')
	parser.set_defaults(screening=True)
	parser.add_argument('--smoke-test', action='store_true', help='Run a single quick trial to verify pipeline')
	args = parser.parse_args()

	repo_root = _repo_root()
	os.chdir(repo_root)
	search_space_path = args.search_space
	if not os.path.isabs(search_space_path):
		search_space_path = os.path.join(repo_root, search_space_path)
	search_space = _load_json(search_space_path)

	base_config_path = search_space['base_config']
	if not os.path.isabs(base_config_path):
		base_config_path = os.path.join(repo_root, base_config_path)
	base_cfg = _load_json(base_config_path)

	study_root = os.path.join(repo_root, 'logs', 'hpo', 'ComplEx-AU_WN18RR')
	os.makedirs(study_root, exist_ok=True)
	storage = args.storage or f'sqlite:///{os.path.join(study_root, "optuna.db")}'
	study = optuna.create_study(
		study_name=args.study_name,
		storage=storage,
		direction='maximize',
		load_if_exists=True,
		sampler=TPESampler(n_startup_trials=args.n_startup_trials, seed=42),
	)

	n_trials = 1 if args.smoke_test else args.n_trials
	param_specs = search_space['parameters']

	def objective(trial: optuna.Trial) -> float:
		sampled: dict[str, Any] = {}
		for name, spec in param_specs.items():
			value = _suggest_from_spec(trial, name, spec, sampled)
			if value is not None:
				sampled[name] = value

		if not _uniformity_active(sampled):
			raise optuna.TrialPruned('at least one gamma_* must be > 0')

		trial_dir = os.path.join(study_root, f'trial_{trial.number:04d}')
		os.makedirs(trial_dir, exist_ok=True)
		trial_config_path = os.path.join(trial_dir, 'config.json')
		trial_cfg = _build_trial_config(
			base_cfg,
			search_space,
			sampled,
			output_dir=trial_dir,
			screening=args.screening,
			screening_epochs=args.screening_epochs if args.screening else None,
		)
		_save_json(trial_config_path, trial_cfg)

		started = datetime.now().isoformat(timespec='seconds')
		try:
			result = _run_trial_subprocess(repo_root, trial_config_path)
		except Exception as exc:
			_save_json(os.path.join(trial_dir, 'error.json'), {'error': str(exc), 'started': started})
			raise

		best_mrr = float(result.get('best_mrr', 0.0) or 0.0)
		_save_json(
			os.path.join(trial_dir, 'result.json'),
			{'best_mrr': best_mrr, 'started': started, 'finished': datetime.now().isoformat(timespec='seconds'), **result},
		)
		return best_mrr

	study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

	best = study.best_trial
	best_params = dict(best.params)
	best_config = _build_trial_config(
		base_cfg,
		search_space,
		best_params,
		output_dir=os.path.join(study_root, 'best_config'),
		screening=False,
	)
	best_config_path = os.path.join(study_root, 'best_config.json')
	_save_json(best_config_path, best_config)

	summary = {
		'study_name': args.study_name,
		'n_trials': len(study.trials),
		'best_trial': best.number,
		'best_mrr': best.value,
		'best_params': best_params,
		'best_config_path': best_config_path,
		'screening': args.screening,
		'screening_epochs': args.screening_epochs if args.screening else None,
	}
	_save_json(os.path.join(study_root, 'study_summary.json'), summary)
	print(json.dumps(summary, indent=2))
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
