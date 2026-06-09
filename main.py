import json
import inspect
import os
import time

import torch

from configs.config import args
from base.evaluator import Evaluator
from data.dataset import Dataset, PointwiseDataset, load_data
from data.dataloader import collate, collate_pointwise
from data.dict_hub import get_entity_dict, get_relation_id_map
from models.builder import import_module_from_path, load_attr_from_path
from models.samplers.bernoulli_sampler import BernoulliListwiseSampler
from utils.device import init_hardware
from utils.checkpoint import best_model_path, last_model_path
from utils.logger import setup_logger, write_results_report, _format_metric_key
from utils.memory import PhaseMemoryTracker


logger = setup_logger(log_file=os.path.join(args.output_dir, 'run.log'))


def _resolve_test_lp_path(current_args) -> str:
    """Resolve the test path for link prediction evaluation, trying multiple candidates in order of preference."""

    candidates = []
    for source_path in [current_args.test_path, current_args.valid_path, current_args.train_path]:
        if not source_path:
            continue
        source_dir = os.path.dirname(source_path)
        candidates.append(os.path.join(source_dir, 'test.txt.json'))
        candidates.append(os.path.join(source_dir, 'test.txt'))

    candidates.append(os.path.join('data', getattr(current_args, 'dataset', ''), 'preprocessed', 'test.txt.json'))
    candidates.append(os.path.join('data', getattr(current_args, 'dataset', ''), 'preprocessed', 'test.txt'))
    candidates.append(os.path.join('data', getattr(current_args, 'dataset', ''), 'test.txt.json'))
    candidates.append(os.path.join('data', getattr(current_args, 'dataset', ''), 'test.txt'))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ''


def _write_results(current_args, train_summary, evaluator, link_metrics, triple_metrics, test_time, configs_snapshot, memory_tracker=None) -> None:
    """Write the evaluation results and training summary to a report file."""

    if link_metrics:
        logger.info('Link prediction metrics on test set:\n{}'.format(json.dumps(link_metrics, indent=4)))
    if triple_metrics:
        logger.info('Triple classification metrics on test set:\n{}'.format(json.dumps(triple_metrics, indent=4)))

    checkpoint = getattr(evaluator, 'checkpoint', {}) or {}
    best_metric = checkpoint.get('best_metric') or {}
    best_epoch = train_summary.get('best_epoch') if train_summary else None
    best_mrr = train_summary.get('best_mrr') if train_summary else None
    best_monitor_metric = train_summary.get('best_monitor_metric') if train_summary else None
    best_monitor_score = train_summary.get('best_monitor_score') if train_summary else None

    if best_epoch is None:
        best_epoch = best_metric.get('epoch')
    if best_mrr is None:
        checkpoint_metrics = best_metric.get('metrics') or {}
        best_mrr = checkpoint_metrics.get('mrr')
    if best_monitor_metric is None:
        best_monitor_metric = best_metric.get('metric')
    if best_monitor_score is None:
        best_monitor_score = best_metric.get('score')

    train_time = train_summary.get('train_time') if train_summary else None
    valid_time = train_summary.get('valid_time') if train_summary else None
    total_time = None
    if train_summary and train_summary.get('total_time') is not None:
        total_time = train_summary['total_time'] + test_time

    memory_summary = memory_tracker.to_dict() if memory_tracker is not None else {}
    train_peak_mb = memory_summary.get('train_peak_mb')
    eval_peak_mb = memory_summary.get('eval_peak_mb')
    peak_memory_mb = memory_summary.get('peak_memory_mb')

    best_valid_extra = {}
    if best_monitor_metric and best_monitor_score is not None:
        best_valid_extra[f'Best {_format_metric_key(best_monitor_metric)}'] = best_monitor_score

    write_results_report(
        os.path.join(current_args.output_dir, 'results.txt'),
        link_metrics=link_metrics,
        triple_metrics=triple_metrics,
        best_epoch=best_epoch,
        best_mrr=best_mrr,
        train_time=train_time,
        valid_time=valid_time,
        test_time=test_time,
        total_time=total_time,
        train_peak_mb=train_peak_mb,
        eval_peak_mb=eval_peak_mb,
        peak_memory_mb=peak_memory_mb,
        configs=configs_snapshot,
        extra_sections={'Best Valid Monitor': best_valid_extra} if best_valid_extra else None,
    )


def _release_gpu_memory() -> None:
    """Return cached GPU memory to the allocator between heavy eval passes."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _average_link_metrics(forward_metrics, backward_metrics) -> dict:
    """Average the link prediction metrics from forward and backward evaluations."""

    if not forward_metrics or not backward_metrics:
        return forward_metrics or backward_metrics

    averaged_metrics = {}
    for key in forward_metrics.keys() & backward_metrics.keys():
        forward_value = forward_metrics[key]
        backward_value = backward_metrics[key]
        if isinstance(forward_value, (int, float)) and isinstance(backward_value, (int, float)):
            averaged_metrics[key] = (forward_value + backward_value) / 2
    return averaged_metrics


def _resolve_relation_index(relation: str, relation_to_idx: dict) -> int:
    """Map a relation string to its embedding index.

    Inverse relations must resolve to their own indices when reciprocal
    relations are enabled; do not silently collapse them onto the forward ID.
    """

    if relation in relation_to_idx:
        return relation_to_idx[relation]
    normalized = ' '.join(relation.split())
    if normalized in relation_to_idx:
        return relation_to_idx[normalized]
    if relation.startswith('inverse '):
        base_relation = relation[len('inverse '):]
        if base_relation in relation_to_idx and f'inverse {base_relation}' not in relation_to_idx:
            return relation_to_idx[base_relation]
    raise KeyError(relation)


def _examples_to_tensors(examples, entity_dict, relation_to_idx):
    """Convert examples into head, relation, and tail index tensors."""

    head_indices = torch.tensor([entity_dict.entity_to_idx(example.head_id) for example in examples], dtype=torch.long)
    relation_indices = torch.tensor(
        [_resolve_relation_index(example.relation, relation_to_idx) for example in examples],
        dtype=torch.long,
    )
    tail_indices = torch.tensor([entity_dict.entity_to_idx(example.tail_id) for example in examples], dtype=torch.long)
    return head_indices, relation_indices, tail_indices


def _build_softmax_trainer(current_args):
    """Build the encoder, sampler, and strategy used by the softmax configs."""

    strategy_mod = import_module_from_path(current_args.model_strategy_path)
    strategy_cls = getattr(strategy_mod, 'SoftmaxStrategy', None)
    if strategy_cls is None:
        raise ImportError(f'Could not find SoftmaxStrategy in {current_args.model_strategy_path}')

    encoder_path = getattr(current_args, 'model_encoder_path', '') or 'models/encoders/distmult_encoder.py'
    try:
        build_model = load_attr_from_path(encoder_path, 'build_model')
    except Exception:
        encoder_mod = import_module_from_path(encoder_path)
        build_model = getattr(encoder_mod, 'build_model')

    model = build_model(current_args)
    if torch.cuda.is_available():
        model.cuda()

    entity_dict = get_entity_dict()
    # When add_reciprocal_relations is set, load inverse triples too.
    # The encoder's rel_to_idx already includes inverse-relation indices.
    add_backward = getattr(current_args, 'add_reciprocal_relations', False)
    train_examples = load_data(current_args.train_path, add_forward_triplet=True, add_backward_triplet=add_backward)
    if not train_examples:
        raise ValueError(f'No training examples loaded from {current_args.train_path}')

    # Use the encoder's relation map so inverse-relation indices line up
    # with what the embedding table was sized for.
    rel_map = getattr(model, 'rel_to_idx', None) or get_relation_id_map()
    inverse_relations = sum(1 for relation in rel_map if str(relation).startswith('inverse '))
    logger.info(
        'Training examples: %d (reciprocal=%s, relations=%d, inverse=%d)',
        len(train_examples), add_backward, len(rel_map), inverse_relations,
    )
    train_tensors = _examples_to_tensors(train_examples, entity_dict, rel_map)
    sampler = BernoulliListwiseSampler(
        train_tensors,
        len(entity_dict),
        max(len(rel_map), 1),
        getattr(current_args, 'n_sample', getattr(current_args, 'batch_size', 1)),
    )
    return strategy_cls(model, sampler, current_args, len(train_examples), train_data=train_tensors)


def _build_adversarial_trainer(current_args):
    """Build the encoder and indexed training triples for adversarial RotatE strategy."""

    strategy_mod = import_module_from_path(current_args.model_strategy_path)
    strategy_cls = getattr(strategy_mod, 'AdversarialStrategy', None)
    if strategy_cls is None:
        strategy_cls = getattr(strategy_mod, 'Strategy', None)
    if strategy_cls is None:
        raise ImportError(f'Could not find AdversarialStrategy in {current_args.model_strategy_path}')

    encoder_path = getattr(current_args, 'model_encoder_path', '') or 'models/encoders/rotate_encoder.py'
    try:
        build_model = load_attr_from_path(encoder_path, 'build_model')
    except Exception:
        encoder_mod = import_module_from_path(encoder_path)
        build_model = getattr(encoder_mod, 'build_model')

    model = build_model(current_args)
    if torch.cuda.is_available():
        model.cuda()

    entity_dict = get_entity_dict()
    relation_to_idx = getattr(model, 'rel_to_idx', None) or get_relation_id_map()
    train_examples = load_data(current_args.train_path, add_forward_triplet=True, add_backward_triplet=False)
    if not train_examples:
        raise ValueError(f'No training examples loaded from {current_args.train_path}')

    src, rel, dst = _examples_to_tensors(train_examples, entity_dict, relation_to_idx)
    train_triples = torch.stack([src, rel, dst], dim=-1)
    return strategy_cls(model, current_args, train_triples)


def _build_pointwise_trainer(current_args):
    """Build the encoder and dataloader for DaBR pointwise training."""

    strategy_mod = import_module_from_path(current_args.model_strategy_path)
    strategy_cls = getattr(strategy_mod, 'PointwiseStrategy', None)
    if strategy_cls is None:
        strategy_cls = getattr(strategy_mod, 'Strategy', None)
    if strategy_cls is None:
        raise ImportError(f'Could not find PointwiseStrategy in {current_args.model_strategy_path}')

    encoder_path = getattr(current_args, 'model_encoder_path', '') or 'models/encoders/dabr_encoder.py'
    try:
        build_model = load_attr_from_path(encoder_path, 'build_model')
    except Exception:
        encoder_mod = import_module_from_path(encoder_path)
        build_model = getattr(encoder_mod, 'build_model')

    model = build_model(current_args)
    if torch.cuda.is_available():
        model.cuda()

    train_examples = load_data(current_args.train_path, add_forward_triplet=True, add_backward_triplet=False)
    train_dataset = PointwiseDataset(train_examples)

    # DaBR/OpenKE convention: a fixed number of batches per epoch, so the batch
    # size is derived from the training set size (batch_size = train_total // n_batches).
    # This reproduces the paper's "100 batches" setting exactly for any dataset.
    n_batches = getattr(current_args, 'n_batches', None)
    if n_batches:
        derived_batch_size = max(len(train_examples) // int(n_batches), 1)
        current_args.batch_size = derived_batch_size
        logger.info(
            'DaBR pointwise batching: n_batches=%d -> batch_size=%d (train_total=%d)',
            int(n_batches), derived_batch_size, len(train_examples),
        )
    batch_size = max(getattr(current_args, 'batch_size', 1), 1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pointwise,
        num_workers=getattr(current_args, 'workers', 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return strategy_cls(model, current_args), train_dataloader


def main():
    ngpus_per_node = init_hardware(args)

    logger.info('Use {} gpus for this run'.format(ngpus_per_node))
    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    config_snapshot = dict(args.__dict__)

    # Determine which evaluation tasks to run: link prediction, triple classification, or both
    task_flag = (args.task or 'both').lower()
    run_lp = False
    run_tc = False
    if 'both' in task_flag or task_flag == 'both':
        run_lp = True
        run_tc = True
    else:
        if 'link' in task_flag or 'pred' in task_flag or 'lp' in task_flag:
            run_lp = True
        if 'triple' in task_flag or 'class' in task_flag or 'tc' in task_flag:
            run_tc = True

    if args.is_test:
        evaluator = Evaluator(args)
        eval_model_path = args.eval_model_path or best_model_path(args.output_dir)
        evaluator.load(eval_model_path)
        memory_tracker = PhaseMemoryTracker()
        test_start = time.time()
        memory_tracker.begin_phase()
        link_metrics = None
        triple_metrics = None
        test_lp_path = _resolve_test_lp_path(args)
        if run_lp and test_lp_path:
            entity_dict = get_entity_dict()
            test_lp_log_path = os.path.join(args.output_dir, 'test_link_prediction.log')
            forward_metrics = evaluator.evaluate_link_prediction_inplace(
                evaluator.model, test_lp_path, entity_dict, test_lp_log_path, eval_forward=True)
            _release_gpu_memory()
            backward_metrics = evaluator.evaluate_link_prediction_inplace(
                evaluator.model, test_lp_path, entity_dict, test_lp_log_path, eval_forward=False)
            link_metrics = _average_link_metrics(forward_metrics, backward_metrics)
        if run_tc:
            triple_metrics = evaluator.evaluate_test_triple_classification()
        memory_tracker.end_phase('eval')
        test_time = time.time() - test_start
        _write_results(args, None, evaluator, link_metrics, triple_metrics, test_time, config_snapshot, memory_tracker)
        return

    strategy_path = args.model_strategy_path
    if strategy_path.replace('\\', '/').endswith('softmax_strategy.py'):
        trainer = _build_softmax_trainer(args)
    elif strategy_path.replace('\\', '/').endswith('adversarial_strategy.py'):
        trainer = _build_adversarial_trainer(args)
    elif strategy_path.replace('\\', '/').endswith('pointwise_strategy.py'):
        trainer, train_dataloader = _build_pointwise_trainer(args)
        train_summary = trainer.train_loop(train_dataloader)
        eval_model_path = train_summary.get('best_checkpoint_path') or best_model_path(args.output_dir)
        if not os.path.exists(eval_model_path):
            eval_model_path = last_model_path(args.output_dir)
        del trainer
        _release_gpu_memory()
        evaluator = Evaluator(args)
        evaluator.load(eval_model_path)
        memory_tracker = PhaseMemoryTracker()
        memory_tracker.update_from_summary(train_summary)
        test_start = time.time()
        memory_tracker.begin_phase()
        link_metrics = None
        triple_metrics = None
        test_lp_path = _resolve_test_lp_path(args)
        if run_lp and test_lp_path:
            entity_dict = get_entity_dict()
            test_lp_log_path = os.path.join(args.output_dir, 'test_link_prediction.log')
            forward_metrics = evaluator.evaluate_link_prediction_inplace(
                evaluator.model, test_lp_path, entity_dict, test_lp_log_path, eval_forward=True)
            _release_gpu_memory()
            backward_metrics = evaluator.evaluate_link_prediction_inplace(
                evaluator.model, test_lp_path, entity_dict, test_lp_log_path, eval_forward=False)
            link_metrics = _average_link_metrics(forward_metrics, backward_metrics)
        if run_tc:
            triple_metrics = evaluator.evaluate_test_triple_classification()
        memory_tracker.end_phase('eval')
        test_time = time.time() - test_start
        _write_results(args, train_summary, evaluator, link_metrics, triple_metrics, test_time, config_snapshot, memory_tracker)
        return
    else:
        strategy_mod = import_module_from_path(strategy_path)
        trainer_cls = None
        preferred_names = (
            'SoftmaxTrainer',
            'SimKGCStrategy',
            'ContrastiveTrainer',
            'SimKGCTrainer',
            'Strategy',
        )
        for cand in preferred_names:
            cls = getattr(strategy_mod, cand, None)
            if inspect.isclass(cls) and cls.__module__ == strategy_mod.__name__ and not inspect.isabstract(cls):
                trainer_cls = cls
                break

        if trainer_cls is None:
            for name, obj in vars(strategy_mod).items():
                if (
                    isinstance(obj, type)
                    and obj.__module__ == strategy_mod.__name__
                    and name != 'Trainer'
                ):
                    trainer_cls = obj
                    break

        if trainer_cls is None:
            raise ImportError(f'Could not find a Trainer class in {strategy_path}')

        trainer = trainer_cls(args, ngpus_per_node=ngpus_per_node)
    train_summary = trainer.train_loop()
    eval_model_path = train_summary.get('best_checkpoint_path') or best_model_path(args.output_dir)
    del trainer
    _release_gpu_memory()
    evaluator = Evaluator(args)
    evaluator.load(eval_model_path)
    memory_tracker = PhaseMemoryTracker()
    memory_tracker.update_from_summary(train_summary)
    test_start = time.time()
    memory_tracker.begin_phase()
    link_metrics = None
    triple_metrics = None
    test_lp_path = _resolve_test_lp_path(args)
    if run_lp and test_lp_path:
        entity_dict = get_entity_dict()
        test_lp_log_path = os.path.join(args.output_dir, 'test_link_prediction.log')
        forward_metrics = evaluator.evaluate_link_prediction_inplace(
            evaluator.model, test_lp_path, entity_dict, test_lp_log_path, eval_forward=True)
        _release_gpu_memory()
        backward_metrics = evaluator.evaluate_link_prediction_inplace(
            evaluator.model, test_lp_path, entity_dict, test_lp_log_path, eval_forward=False)
        link_metrics = _average_link_metrics(forward_metrics, backward_metrics)
    if run_tc:
        triple_metrics = evaluator.evaluate_test_triple_classification()
    memory_tracker.end_phase('eval')
    test_time = time.time() - test_start
    _write_results(args, train_summary, evaluator, link_metrics, triple_metrics, test_time, config_snapshot, memory_tracker)


if __name__ == '__main__':
    main()