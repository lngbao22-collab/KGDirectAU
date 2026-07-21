"""Reorder config JSON keys and drop default-off hyperparameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'configs'

KEY_ORDER = [
    'model', 'dataset', 'task',
    'model_loss_path', 'model_strategy_path', 'model_sampler_path',
    'kbc_reciprocal_relations', 'add_reciprocal_relations', 'bidirectional_1vsall',
    'head_eval_mode', 'kvsall_query_types',
    'dim', 'batch_size', 'epochs', 'optim', 'lr', 'weight_decay',
    'adversarial_training', 'adversarial_temperature', 'margin',
    'n_sample', 'n_sample_t', 'n_sample_h',
    'neg_score_chunk_size', 'neg_weight_chunk_size', 'sample_freq', 'loss_arg',
    'lam', 'lmbda', 'lmbda_two',
    'regularizer', 'regularize_weight', 'regularization',
    'label_smoothing',
    'entity_dropout', 'relation_dropout',
    'entity_regularize_weight', 'relation_regularize_weight',
    'entity_regularize_weighted', 'relation_regularize_weighted', 'regularize_p',
    'init_method', 'init_scale', 'init_uniform_a', 'init_xavier_gain', 'sparse_embeddings',
    'training_cadence', 'max_steps', 'valid_steps', 'eval_every_n_step',
    'save_checkpoint_steps', 'epoch_per_eval', 'warm_up_steps', 'warm_up_ratio', 'lr_decay_factor', 'shuffle_train',
    'lr_scheduler', 'lr_scheduler_mode', 'lr_scheduler_factor',
    'lr_scheduler_patience', 'lr_scheduler_threshold', 'lr_scheduler_step_size', 'warmup',
    'alpha', 'gamma_q', 'gamma_t', 'gamma_h', 'gamma_ent', 'gamma_cross', 'tuni',
    'au_deduplicate', 'normalize_au_vectors', 'normalize_lp_scores',
    'learnable_au_scales', 'learnable_au_alpha', 'learnable_au_gammas', 'learnable_uniformity_scale',
    'log_au_alpha_lr', 'log_au_gamma_lr', 'log_uniformity_lr',
    'alpha_linear_schedule', 'alpha_schedule_end', 'alpha_schedule_start_epoch', 'alpha_schedule_epochs',
    'gamma_linear_schedule', 'gamma_schedule_end', 'gamma_schedule_start_epoch', 'gamma_schedule_epochs',
    'tuni_linear_schedule', 'tuni_schedule_start', 'tuni_schedule_end',
    'tuni_schedule_start_epoch', 'tuni_schedule_epochs',
    'eval_batch_size', 'test_batch_size',
    'early_stopping_patience', 'early_stopping_min_epochs', 'early_stopping_min_metric',
    'tie_handling', 'tie_rtol', 'tie_atol',
    'l_norm', 'normalize_phases',
    'score_query_chunk_size', 'eval_candidate_chunk_size', 'eval_entity_chunk_size',
    'bert_encoder', 'pooling', 'dropout', 'infonce_t', 'additive_margin', 'finetune_t',
    'pre_batch', 'pre_batch_weight', 'grad_clip', 'max_num_tokens',
    'use_link_graph', 'use_self_negative', 'neighbor_weight', 'rerank_n_hop',
    'workers', 'print_freq', 'max_to_keep',
    'use_amp', 'eval_use_amp', 'seed',
    'output_dir_prefix',
]

DROP_KEYS = frozenset({
    'model_embedder_path', 'model_scorer_path', 'model_encoder_path',
    'output_dir', 'unparsed_args', 'is_test',
    'train_path', 'valid_path', 'test_path', 'valid_w_label_path', 'test_w_label_path',
})

DEFAULT_OMIT: dict[str, Any] = {
    'task': 'both',
    'weight_decay': 0.0,
    'entity_dropout': 0.0,
    'relation_dropout': 0.0,
    'gamma_h': 0.0,
    'gamma_ent': 0.0,
    'gamma_cross': 0.0,
    'au_deduplicate': True,
    'normalize_lp_scores': False,
    'bidirectional_1vsall': False,
    'tie_handling': 'rounded_mean_rank',
    'tie_rtol': 0.0001,
    'tie_atol': 1e-05,
    'neg_score_chunk_size': 0,
    'neg_weight_chunk_size': 0,
    'shuffle_train': False,
    'entity_regularize_weight': 0.0,
    'relation_regularize_weight': 0.0,
    'entity_regularize_weighted': False,
    'relation_regularize_weighted': False,
    'adversarial_training': False,
    'add_reciprocal_relations': False,
    'kbc_reciprocal_relations': False,
    'sparse_embeddings': False,
    'use_self_negative': False,
    'neighbor_weight': 0.0,
    'pre_batch': 0,
    'seed': None,
    'model_sampler_path': '',
    'lr_scheduler': 'none',
    'normalize_phases': False,
    'tuni_linear_schedule': False,
    'alpha_linear_schedule': False,
    'gamma_linear_schedule': False,
    'learnable_au_gammas': False,
    'learnable_au_alpha': False,
    'learnable_au_scales': False,
    'learnable_uniformity_scale': False,
}

LR_SCHEDULER_KEYS = frozenset({
    'lr_scheduler_mode', 'lr_scheduler_factor', 'lr_scheduler_patience',
    'lr_scheduler_threshold', 'lr_scheduler_step_size',
})


def _should_omit(key: str, value: Any, cfg: dict[str, Any]) -> bool:
    if key in DROP_KEYS:
        return True
    if key in DEFAULT_OMIT and value == DEFAULT_OMIT[key]:
        return True
    if key in LR_SCHEDULER_KEYS and cfg.get('lr_scheduler', 'none') in (None, 'none'):
        return True
    if key == 'output_dir_prefix' and value in (None, ''):
        return True
    if key == 'weight_decay' and value in (0, 0.0):
        return True
    return False


def _clean(cfg: dict[str, Any]) -> dict[str, Any]:
    cleaned = {k: v for k, v in cfg.items() if k not in DROP_KEYS}
    return {k: v for k, v in cleaned.items() if not _should_omit(k, v, cleaned)}


def _order(cfg: dict[str, Any]) -> dict[str, Any]:
    ordered = {k: cfg[k] for k in KEY_ORDER if k in cfg}
    for k in sorted(cfg):
        if k not in ordered:
            ordered[k] = cfg[k]
    return ordered


def _fix_special(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    if path.name == 'ComplEx-AU_WN18RR_libkge_best_config.json':
        cfg.pop('output_dir', None)
        if not cfg.get('output_dir_prefix'):
            cfg['output_dir_prefix'] = 'logs/ComplEx-AU_WN18RR_libkge_best'
    return cfg


def refactor(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as fh:
        raw = json.load(fh)
    return _order(_clean(_fix_special(path, raw)))


def main() -> None:
    paths = sorted(
        p for p in CONFIG_DIR.glob('*.json')
        if 'TEMPLATE' not in p.name.upper()
    )
    for path in paths:
        out = refactor(path)
        path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
        print(f'{path.name}: {len(out)} keys')


if __name__ == '__main__':
    main()
