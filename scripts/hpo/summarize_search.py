"""Summarize an Optuna HPO study and export the best trial config."""

from __future__ import annotations

import argparse
import json
import os
import sys

import optuna
import pandas as pd


def _repo_root() -> str:
	return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
	parser = argparse.ArgumentParser(description='Summarize ComplEx-AU HPO study')
	parser.add_argument('--study-name', default='complex_au_wn18rr')
	parser.add_argument(
		'--storage',
		default='',
		help='Optuna storage URL (default: logs/hpo/ComplEx-AU_WN18RR/optuna.db)',
	)
	parser.add_argument('--top-k', type=int, default=10, help='Number of top trials to list')
	args = parser.parse_args()

	repo_root = _repo_root()
	study_root = os.path.join(repo_root, 'logs', 'hpo', 'ComplEx-AU_WN18RR')
	storage = args.storage or f'sqlite:///{os.path.join(study_root, "optuna.db")}'
	study = optuna.load_study(study_name=args.study_name, storage=storage)

	rows = []
	for trial in study.trials:
		if trial.state != optuna.trial.TrialState.COMPLETE:
			continue
		row = {'trial': trial.number, 'mrr': trial.value}
		row.update(trial.params)
		rows.append(row)

	if not rows:
		print('No completed trials found.')
		return 1

	df = pd.DataFrame(rows).sort_values('mrr', ascending=False)
	csv_path = os.path.join(study_root, 'trials_ranked.csv')
	df.to_csv(csv_path, index=False)

	print(f'Top {args.top_k} trials (valid MRR):')
	print(df.head(args.top_k).to_string(index=False))
	print(f'\nSaved: {csv_path}')
	print(f'Best config: {os.path.join(study_root, "best_config.json")}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
