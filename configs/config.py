"""Config parsing and global args."""

import argparse
import json
import os
import random
import sys
import warnings
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.backends.cudnn as cudnn
from typing import Dict, Any


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the KG training and evaluation script."""

    parser = argparse.ArgumentParser(description='Generic KG arguments')

    parser.add_argument('--config-path', default='', type=str,
                        help='path to a JSON config file in configs/ or an absolute config path')

    parser.add_argument('--model', default='simkgc', type=str,
                        help='model family name, e.g. simkgc, transe, transd, rotate')
    parser.add_argument('--model-encoder-path', default='', type=str,
                        help='path to encoder module, e.g. models/encoders/bert_encoder.py')
    parser.add_argument('--model-loss-path', default='', type=str,
                        help='path to loss module, e.g. models/losses/infonce_loss.py')
    parser.add_argument('--model-sampler-path', default='', type=str,
                        help='path to sampler module, e.g. models/samplers/masking_sampler.py')
    parser.add_argument('--model-strategy-path', default='', type=str,
                        help='path to strategy module, e.g. models/strategies/simkgc_strategy.py')
    parser.add_argument('--task', default='both', type=str,
                        help='link prediction/triple classification/both')
    parser.add_argument('--bert-encoder', '--encoder', default='distilbert-base-uncased', type=str, dest='bert_encoder',
                        help='pretrained text encoder name or path')
    parser.add_argument('--dataset', default='wn18rr', type=str,
                        help='dataset or benchmark name')

    # Core data/model paths.
    parser.add_argument('--train-path', default='', type=str,
                        help='path to training data')
    parser.add_argument('--valid-path', default='', type=str,
                        help='path to validation data')
    parser.add_argument('--test-path', default='', type=str,
                        help='path to test data')
    parser.add_argument('--valid-w-label-path', default='', type=str,
                        help='path to validation data for triple classification')
    parser.add_argument('--test-w-label-path', default='', type=str,
                        help='path to test data for triple classification')
    # in default, paths for .txt.json (preprocess) or .txt (unprocessed) are taken by dataset in 'data/<dataset>/preprocessed' folder e.g. data/WN18RR/preprocessed/train.txt.json, data/WN18RR/preprocessed/valid.txt.json, data/WN18RR/preprocessed/test.txt.json

    parser.add_argument('--eval-model-path', default='', type=str,
                        help='path to model to evaluate')
    # in default, eval_model_path is taken from best_model.mdl in output-dir if exists; otherwise, it needs to be specified.

    parser.add_argument('--output-dir-prefix', default='', type=str,
                        help='prefix for the directory used to save checkpoints, predictions, and logs; a timestamp will be appended when used')
    # in default, output is saved in 'logs/<model>_<dataset>' folder e.g. logs/SimKGC_WN18RR.
    # This folder will contain: train.log (Text training output), results.txt (Final result metrics + best valid + time), best_model.mdl  (Best model weights)

    # Hyperparameters and settings.
    parser.add_argument('--additive-margin', default=0.0, type=float,
                        help='additive margin for contrastive loss')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='mini-batch size')
    parser.add_argument('--dim', default=768, type=int,
                        help='embedding dimension for non-text KG models')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout rate')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of epochs to run')
    parser.add_argument('--eval-every-n-step', default=10000, type=int,
                        help='evaluate every n steps')
    parser.add_argument('--finetune-t', action='store_true',
                        help='make InfoNCE temperature trainable')
    parser.add_argument('--grad-clip', default=10.0, type=float,
                        help='gradient clipping')
    parser.add_argument('--is-test', action='store_true',
                        help='run test-mode evaluation')
    parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float, dest='lr',
                        help='initial learning rate')
    parser.add_argument('--lr-scheduler', default='linear', type=str,
                        help='learning-rate scheduler')
    parser.add_argument('--max-num-tokens', default=50, type=int,
                        help='maximum number of tokens for text-based models')
    parser.add_argument('--max-to-keep', default=5, type=int,
                        help='maximum number of checkpoints to keep')
    parser.add_argument('--neighbor-weight', default=0.0, type=float,
                        help='reranking weight')
    parser.add_argument('--pooling', default='cls', type=str,
                        help='pooling strategy for text encoders')
    parser.add_argument('--pre-batch', default=0, type=int,
                        help='number of pre-batch negatives')
    parser.add_argument('--pre-batch-weight', default=0.5, type=float,
                        help='weight for pre-batch negatives')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        help='logging frequency')
    parser.add_argument('--rerank-n-hop', default=2, type=int,
                        help='neighbor hops for reranking during evaluation')
    parser.add_argument('--seed', default=None, type=int,
                        help='random seed')
    parser.add_argument('--infonce-t', '--t', default=0.05, type=float, dest='infonce_t',
                        help='InfoNCE temperature parameter')
    parser.add_argument('--use-amp', action='store_true',
                        help='use AMP if available')
    parser.add_argument('--use-link-graph', action='store_true',
                        help='use neighbors from link graph as context')
    parser.add_argument('--use-self-negative', action='store_true',
                        help='use head entity as negative')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        dest='weight_decay', help='weight decay')
    parser.add_argument('-j', '--workers', default=2, type=int,
                        help='number of data loading workers')
    parser.add_argument('--warmup', default=400, type=int,
                        help='warmup steps')

    return parser


def _resolve_output_dir() -> str:
    """Determine the output directory for checkpoints and logs, creating it if necessary."""

    def _default_run_dir() -> str:
        """Construct a default run directory based on the model and dataset names, with a timestamp for uniqueness."""

        base_dir = os.path.join(os.getcwd(), 'logs')
        run_name = f'{_format_model_name(args.model)}_{_format_dataset_name(args.dataset)}'
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return os.path.join(base_dir, f'{run_name}_{timestamp}')

    def _is_default_placeholder(path: str) -> bool:
        """Check if the given path is empty or matches the default placeholder pattern for this model and dataset."""
        
        if not path:
            return True
        placeholder = os.path.join('logs', f'{_format_model_name(args.model)}_{_format_dataset_name(args.dataset)}')
        normalized_path = os.path.normpath(path)
        normalized_placeholder = os.path.normpath(placeholder)
        absolute_placeholder = os.path.normpath(os.path.join(os.getcwd(), placeholder))
        return normalized_path in {normalized_placeholder, absolute_placeholder}
    # starting candidate list: explicit output_dir, prefix, or fallback defaults
    candidates = [getattr(args, 'output_dir', ''), getattr(args, 'output_dir_prefix', '')]

    if args.eval_model_path:
        candidates.append(os.path.dirname(args.eval_model_path))
    candidates.append(_default_run_dir())
    candidates.append(os.getcwd())

    for candidate in candidates:
        if _is_default_placeholder(candidate):
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
        except OSError:
            continue
        if os.access(candidate, os.W_OK):
            return candidate

    return os.getcwd()


def _format_model_name(model: str) -> str:
    """Format the model name for consistent config lookup and output naming."""

    mapping = {
        'simkgc': 'SimKGC',
        'transe': 'TransE',
        'transd': 'TransD',
        'rotate': 'RotatE',
    }
    return mapping.get(model.lower(), model)


def _format_dataset_name(dataset: str) -> str:
    """Format the dataset name for consistent config lookup and output naming."""

    mapping = {
        'wn18rr': 'WN18RR',
        'fb15k237': 'FB15k237',
        'wiki5m_ind': 'Wiki5M_Ind',
    }
    return mapping.get(dataset.lower(), dataset)


def _resolve_config_path() -> str:
    """Resolve the config JSON path, preferring an explicit path and then configs/ fallbacks."""

    explicit_path = getattr(args, 'config_path', '')
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        candidate = os.path.join('configs', explicit_path)
        if os.path.exists(candidate):
            return candidate

    candidates = [
        os.path.join('configs', f'{_format_model_name(args.model)}_{_format_dataset_name(args.dataset)}.json'),
        os.path.join('configs', f'{args.model.lower()}_{args.dataset.lower()}.json'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _load_json_defaults(path: str) -> Dict[str, Any]:
    """Load configuration defaults from a JSON object file if it exists."""

    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Config file must contain a JSON object: {path}')
    return cfg


def _resolve_case_insensitive_path(path: str) -> str:
    """Resolve an existing path on case-sensitive filesystems when only letter case differs."""

    if not path:
        return path
    norm_path = os.path.normpath(path)
    if os.path.exists(norm_path):
        return norm_path

    drive, tail = os.path.splitdrive(norm_path)
    current = drive + os.sep if drive else os.sep if norm_path.startswith(os.sep) else ''
    parts = [p for p in tail.split(os.sep) if p]

    if not parts:
        return norm_path

    for part in parts:
        if not current or not os.path.isdir(current):
            return norm_path
        try:
            entries = os.listdir(current)
        except OSError:
            return norm_path

        matched = None
        lower_part = part.lower()
        for entry in entries:
            if entry.lower() == lower_part:
                matched = entry
                break

        if matched is None:
            return norm_path
        current = os.path.join(current, matched)

    return current if os.path.exists(current) else norm_path


def _resolve_data_path(path: str) -> str:
    """Resolve data paths, falling back from preprocessed *.txt.json to raw *.txt when needed."""

    if not path:
        return path

    candidate = _resolve_case_insensitive_path(path)
    if os.path.exists(candidate):
        return candidate

    if candidate.endswith('.txt.json'):
        raw_candidate = candidate[:-5]
        raw_candidate = _resolve_case_insensitive_path(raw_candidate)
        if os.path.exists(raw_candidate):
            return raw_candidate

    return candidate


def _replace_split_suffix(path: str, source_suffix: str, target_suffix: str) -> str:
    """Replace a dataset split suffix inside a file name while preserving the directory."""

    if not path:
        return path

    directory, basename = os.path.split(path)
    if source_suffix not in basename:
        return path
    return os.path.join(directory, basename.replace(source_suffix, target_suffix))


def _derive_split_variant(path: str, *, split_name: str, labeled: bool) -> str:
    """Map between the raw and labeled split variants for a given split name."""

    if not path:
        return path

    if labeled:
        source_suffix = f'{split_name}.txt'
        target_suffix = f'{split_name}_w_label.txt'
    else:
        source_suffix = f'{split_name}_w_label.txt'
        target_suffix = f'{split_name}.txt'

    return _replace_split_suffix(path, source_suffix, target_suffix)


def _cuda_unavailable_reason() -> str:
    """Return a human-readable reason when CUDA is unavailable in the current Python env."""

    torch_cuda = getattr(torch.version, 'cuda', None)
    torch_version = getattr(torch, '_version_', 'unknown')
    executable = sys.executable

    if not torch_cuda:
        return (
            'CPU-only PyTorch build detected '
            f'(python={executable}, torch={torch_version}). '
            'Install a CUDA wheel in this same environment.'
        )

    return (
        'CUDA runtime is bundled with PyTorch but no GPU is usable in this environment '
        f'(python={executable}, torch={torch_version}, torch_cuda={torch_cuda}). '
        'This is commonly caused by a CUDA-runtime/driver mismatch or running with a different Python env than expected.'
    )


def _resolve_data_path(path: str) -> str:
    """Resolve a dataset path against the repo root and common layout variants."""

    if not path:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join(repo_root, path),
    ]

    if '/preprocessed/' in path:
        candidates.append(path.replace('/preprocessed/', '/'))
        candidates.append(os.path.join(repo_root, path.replace('/preprocessed/', '/')))

    if path.endswith('.json'):
        candidates.append(path[:-5])
        candidates.append(os.path.join(repo_root, path[:-5]))
    elif path.endswith('.txt'):
        candidates.append(path + '.json')
        candidates.append(os.path.join(repo_root, path + '.json'))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return path


parser = build_parser()
args, unknown_args = parser.parse_known_args()

config_path = _resolve_config_path()
config_defaults = _load_json_defaults(config_path)
if config_defaults:
    parser.set_defaults(**config_defaults)
    args, unknown_args = parser.parse_known_args()

args.unparsed_args = unknown_args

args.train_path = _resolve_data_path(getattr(args, 'train_path', ''))
args.valid_path = _resolve_data_path(_derive_split_variant(getattr(args, 'valid_path', ''), split_name='valid', labeled=False))
args.test_path = _resolve_data_path(_derive_split_variant(getattr(args, 'test_path', ''), split_name='test', labeled=False))
args.valid_w_label_path = _resolve_data_path(
    getattr(args, 'valid_w_label_path', '')
    or _derive_split_variant(args.valid_path, split_name='valid', labeled=True)
)
args.test_w_label_path = _resolve_data_path(
    getattr(args, 'test_w_label_path', '')
    or _derive_split_variant(args.test_path, split_name='test', labeled=True)
)
assert not args.train_path or os.path.exists(args.train_path)
assert args.pooling in ['cls', 'mean', 'max']
assert args.lr_scheduler in ['linear', 'cosine']

args.config_path = config_path

_model_name = (args.model or '').lower()
_is_text_model = _model_name not in {'distmult', 'complex'}

if _is_text_model:
    args.encoder = args.bert_encoder
    args.pretrained_model = args.bert_encoder
else:
    args.bert_encoder = ''
    args.encoder = ''
    args.pretrained_model = ''

if not args.model_strategy_path:
    if _model_name in {'distmult', 'complex'}:
        args.model_strategy_path = 'models/strategies/softmax_strategy.py'
    else:
        args.model_strategy_path = 'models/strategies/simkgc_strategy.py'

if not args.model_encoder_path:
    if _model_name == 'distmult':
        args.model_encoder_path = 'models/encoders/distmult_encoder.py'
    elif _model_name == 'complex':
        args.model_encoder_path = 'models/encoders/complex_encoder.py'
    else:
        args.model_encoder_path = 'models/encoders/bert_encoder.py'

if not args.model_sampler_path:
    if _model_name in {'distmult', 'complex'}:
        args.model_sampler_path = 'models/samplers/bernoulli_sampler.py'
    else:
        args.model_sampler_path = 'models/samplers/masking_sampler.py'

if not args.model_loss_path and _model_name in {'distmult', 'complex'}:
    args.model_loss_path = 'models/losses/infonce_loss.py'

# --task is a separate flag controlling which evaluations to run
# (link prediction / triple classification / both). Do NOT overwrite it
# with args.dataset here so users can specify evaluation task independently.

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        torch.cuda.manual_seed_all(args.seed)
    except Exception:
        # cuda may not be available in all environments
        pass
    cudnn.deterministic = True

try:
    if args.use_amp:
        import torch.cuda.amp
except Exception:
    args.use_amp = False
    warnings.warn('AMP training is not available, set use_amp=False')

if not torch.cuda.is_available():
    args.use_amp = False
    args.print_freq = 1
    warnings.warn(
        'GPU is not available, set use_amp=False and print_freq=1. '
        + _cuda_unavailable_reason()
    )

# Ensure args exposes output_dir (parser flags were removed).
if not hasattr(args, 'output_dir'):
    args.output_dir = ''

# If a user provided an output_dir_prefix (e.g., "logs/Model_Dataset"),
# convert it into a timestamped run directory and prefer it when writable.
if getattr(args, 'output_dir_prefix', ''):
    prefix = args.output_dir_prefix.rstrip('/\\')
    import re, datetime
    ts_pattern = re.compile(r'.*\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
    if ts_pattern.match(prefix):
        chosen = prefix
    else:
        chosen = prefix + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    try:
        os.makedirs(chosen, exist_ok=True)
        if os.access(chosen, os.W_OK):
            args.output_dir = chosen
    except Exception:
        # ignore and fall back to resolver
        pass

# If no explicit output_dir was chosen above, resolve a sensible default.
if not args.output_dir:
    args.output_dir = _resolve_output_dir()
    
def apply_train_args(train_args: SimpleNamespace) -> SimpleNamespace:
    """Merge training-time args from a checkpoint with current global args.

    Ensures any missing flags are filled from current parser defaults and
    updates global args for evaluation flags like use_link_graph and is_test.
    """

    train_args_dict = vars(train_args)
    for k, v in vars(args).items():
        if k not in train_args_dict:
            train_args_dict[k] = v

    # Export training flags to global args used at runtime
    args.use_link_graph = getattr(train_args, 'use_link_graph', args.use_link_graph)
    # When applying training args for evaluation, prefer explicit test flag if present,
    # otherwise set evaluation mode to True to indicate we're loading a checkpoint for eval.
    args.is_test = getattr(train_args, 'is_test', True)
    return train_args


def _merge_with_defaults(cfg: Dict[str, Any]) -> SimpleNamespace:
    """Return a SimpleNamespace merged with current parser defaults.

    This fills in any missing keys from the current args defaults so
    downstream code can rely on a complete args namespace (useful when
    loading hyperparameters from JSON files).
    """

    merged = dict(vars(args))
    merged.update(cfg)
    return SimpleNamespace(**merged)


def load_args_from_json(path: str) -> SimpleNamespace:
    """Load args from a JSON file and merge with parser defaults.

    Returns a SimpleNamespace suitable to pass to apply_train_args.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return _merge_with_defaults(cfg)


def save_args_to_json(namespace: SimpleNamespace, path: str) -> None:
    """Save an args namespace to a JSON file (converting to plain dict)."""
    
    d = dict(vars(namespace))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, sort_keys=True)
