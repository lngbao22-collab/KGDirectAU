"""Optuna hyperparameter search for KGAU configs."""

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any

import optuna
from optuna.samplers import GridSampler, TPESampler


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


def _uniformity_active(params: dict, base_cfg: dict) -> bool:
	merged = dict(base_cfg)
	merged.update(params)
	return any(float(merged.get(key, 0.0) or 0.0) > 0.0 for key in ('gamma_q', 'gamma_t', 'gamma_h', 'gamma_ent', 'gamma_cross'))


def _grid_search_space(param_specs: dict) -> dict[str, list[Any]]:
	"""Build an Optuna GridSampler search space from categorical parameter specs."""

	grid: dict[str, list[Any]] = {}
	for name, spec in param_specs.items():
		if spec.get('when'):
			raise ValueError(f'grid sampler does not support conditional param {name!r}')
		if spec['type'] != 'categorical':
			raise ValueError(
				f'grid sampler requires categorical parameters; got {name!r} type={spec["type"]!r}'
			)
		grid[name] = list(spec['choices'])
	if not grid:
		raise ValueError('grid sampler requires at least one categorical parameter')
	return grid


def _grid_size(grid: dict[str, list[Any]]) -> int:
	size = 1
	for values in grid.values():
		size *= max(len(values), 1)
	return size


def _resolve_sampler(search_space: dict, *, n_startup_trials: int, seed: int = 42):
	"""Return ``(sampler, grid_or_none)`` from search-space ``sampler`` field."""

	sampler_name = str(search_space.get('sampler', 'tpe') or 'tpe').strip().lower()
	param_specs = search_space['parameters']
	if sampler_name in {'grid', 'gridsampler'}:
		grid = _grid_search_space(param_specs)
		return GridSampler(grid, seed=seed), grid
	if sampler_name in {'tpe', 'tpesampler', ''}:
		return TPESampler(n_startup_trials=n_startup_trials, seed=seed), None
	raise ValueError(f'Unsupported sampler {sampler_name!r}; use "tpe" or "grid"')


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


def _run_trial_subprocess(repo_root: str, trial_config_path: str, *, log_path: str | None = None) -> dict:
	env = os.environ.copy()
	env['PYTHONPATH'] = repo_root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
	cmd = [sys.executable, os.path.join(repo_root, 'scripts', 'hpo', 'train_trial.py'), '--trial-config', trial_config_path]
	proc = subprocess.Popen(
		cmd,
		cwd=repo_root,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		env=env,
		bufsize=1,
	)
	stdout_lines: list[str] = []
	log_handle = open(log_path, 'w', encoding='utf-8') if log_path else None
	try:
		assert proc.stdout is not None
		for line in proc.stdout:
			stdout_lines.append(line)
			sys.stdout.write(line)
			sys.stdout.flush()
			if log_handle is not None:
				log_handle.write(line)
				log_handle.flush()
	finally:
		if log_handle is not None:
			log_handle.close()
	return_code = proc.wait()
	stdout = ''.join(stdout_lines)
	if return_code != 0:
		raise RuntimeError(f'trial failed (code={return_code})\nstdout={stdout.strip()}')
	lines = [line.strip() for line in stdout.splitlines() if line.strip()]
	for line in reversed(lines):
		try:
			payload = json.loads(line)
			if isinstance(payload, dict) and 'ok' in payload:
				return payload
		except json.JSONDecodeError:
			continue
	raise RuntimeError(f'could not parse trial output:\n{stdout}')


def main() -> int:
	parser = argparse.ArgumentParser(description='Run KGAU HPO with Optuna')
	parser.add_argument(
		'--search-space',
		default='scripts/hpo/search_space_complex_au_wn18rr.json',
		help='Search space JSON (relative to repo root)',
	)
	parser.add_argument(
		'--n-trials',
		type=int,
		default=None,
		help='Number of Optuna trials (default: 30 for TPE, full grid size for grid sampler)',
	)
	parser.add_argument('--n-startup-trials', type=int, default=10, help='Random trials before TPE')
	parser.add_argument(
		'--study-name',
		default='',
		help='Optuna study name (default: search-space study_name, else complex_au_wn18rr)',
	)
	parser.add_argument(
		'--storage',
		default='',
		help='Optuna storage URL (default: sqlite under logs/hpo/)',
	)
	parser.add_argument(
		'--screening-epochs',
		type=int,
		default=None,
		help='Epoch cap when screening is enabled (default: screening_overrides.epochs, else 30)',
	)
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

	study_dir = search_space.get('study_dir', 'ComplEx-AU_WN18RR')
	study_root = os.path.join(repo_root, 'logs', 'hpo', study_dir)
	os.makedirs(study_root, exist_ok=True)
	storage = args.storage or f'sqlite:///{os.path.join(study_root, "optuna.db")}'
	study_name = args.study_name or search_space.get('study_name') or 'complex_au_wn18rr'
	sampler, grid = _resolve_sampler(search_space, n_startup_trials=args.n_startup_trials)
	study = optuna.create_study(
		study_name=study_name,
		storage=storage,
		direction='maximize',
		load_if_exists=True,
		sampler=sampler,
	)

	param_specs = search_space['parameters']
	if args.smoke_test:
		n_trials = 1
	elif args.n_trials is not None:
		n_trials = args.n_trials
	elif grid is not None:
		n_trials = _grid_size(grid)
	else:
		n_trials = 30

	if grid is not None and not args.smoke_test:
		remaining = max(_grid_size(grid) - len(study.trials), 0)
		if remaining == 0:
			print(f'Grid already exhausted ({len(study.trials)} trials); summarizing existing study.', flush=True)
			n_trials = 0
		else:
			n_trials = min(n_trials, remaining)

	screening_epochs = args.screening_epochs
	if args.screening and screening_epochs is None:
		screening_epochs = search_space.get('screening_overrides', {}).get('epochs', 30)

	print(
		f'Study={study_name} sampler={type(sampler).__name__} n_trials={n_trials} '
		f'screening={args.screening} screening_epochs={screening_epochs if args.screening else None}',
		flush=True,
	)

	def objective(trial: optuna.Trial) -> float:
		sampled: dict[str, Any] = {}
		for name, spec in param_specs.items():
			value = _suggest_from_spec(trial, name, spec, sampled)
			if value is not None:
				sampled[name] = value

		if not _uniformity_active(sampled, base_cfg):
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
			screening_epochs=int(screening_epochs) if args.screening else None,
		)
		_save_json(trial_config_path, trial_cfg)

		print(f'\n{"=" * 72}', flush=True)
		print(f'Trial {trial.number} started at {datetime.now().isoformat(timespec="seconds")}', flush=True)
		print(f'Config: {trial_config_path}', flush=True)
		print(f'Params: {json.dumps(sampled, sort_keys=True)}', flush=True)
		print(f'{"=" * 72}\n', flush=True)

		started = datetime.now().isoformat(timespec='seconds')
		trial_log_path = os.path.join(trial_dir, 'trial_console.log')
		try:
			result = _run_trial_subprocess(repo_root, trial_config_path, log_path=trial_log_path)
		except Exception as exc:
			_save_json(os.path.join(trial_dir, 'error.json'), {'error': str(exc), 'started': started})
			raise

		best_mrr = float(result.get('best_mrr', 0.0) or 0.0)
		print(
			f'\nTrial {trial.number} finished: best_mrr={best_mrr:.6f} '
			f'(epoch={result.get("best_epoch")}, train_time={result.get("train_time")}s)\n',
			flush=True,
		)
		_save_json(
			os.path.join(trial_dir, 'result.json'),
			{'best_mrr': best_mrr, 'started': started, 'finished': datetime.now().isoformat(timespec='seconds'), **result},
		)
		return best_mrr

	if n_trials > 0:
		study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

	completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
	if not completed:
		print('No completed trials; nothing to summarize.', flush=True)
		return 1

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
		'study_name': study_name,
		'sampler': type(sampler).__name__,
		'n_trials': len(study.trials),
		'best_trial': best.number,
		'best_mrr': best.value,
		'best_params': best_params,
		'best_config_path': best_config_path,
		'screening': args.screening,
		'screening_epochs': int(screening_epochs) if args.screening else None,
	}
	if grid is not None:
		summary['grid_size'] = _grid_size(grid)
	_save_json(os.path.join(study_root, 'study_summary.json'), summary)
	print(json.dumps(summary, indent=2))
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
