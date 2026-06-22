import json
import os
import time

import torch

from configs.config import args
from base.evaluator import Evaluator
from data.dict_hub import get_entity_dict
from models.builder import build_pipeline
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
        if _is_dual_link_metrics(link_metrics):
            for scorer_label, metrics in link_metrics.items():
                logger.info(
                    'Link prediction metrics on test set (%s scorer):\n%s',
                    scorer_label,
                    json.dumps(metrics, indent=4),
                )
        else:
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


def _is_dual_link_metrics(link_metrics: dict | None) -> bool:
    """Return True when link metrics contain separate cosine and original scorer results."""

    if not link_metrics:
        return False
    return (
        'cosine' in link_metrics
        and 'original' in link_metrics
        and isinstance(link_metrics['cosine'], dict)
        and isinstance(link_metrics['original'], dict)
    )


def _run_test_link_prediction(evaluator, test_lp_path: str, entity_dict, output_dir: str) -> dict:
    """Evaluate test link prediction with cosine and native KGE scorers."""

    return evaluator.evaluate_dual_test_link_prediction(test_lp_path, entity_dict, output_dir)


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
            link_metrics = _run_test_link_prediction(evaluator, test_lp_path, entity_dict, args.output_dir)
            _release_gpu_memory()
        if run_tc:
            triple_metrics = evaluator.evaluate_test_triple_classification()
        memory_tracker.end_phase('eval')
        test_time = time.time() - test_start
        _write_results(args, None, evaluator, link_metrics, triple_metrics, test_time, config_snapshot, memory_tracker)
        return

    trainer = build_pipeline(args, ngpus_per_node=ngpus_per_node)
    train_dataloader = getattr(trainer, 'train_dataloader', None)
    if train_dataloader is not None:
        train_summary = trainer.train_loop(train_dataloader)
    else:
        train_summary = trainer.train_loop()
    eval_model_path = train_summary.get('best_checkpoint_path')
    if not eval_model_path or not os.path.exists(eval_model_path):
        for candidate in (best_model_path(args.output_dir), last_model_path(args.output_dir)):
            if os.path.exists(candidate):
                eval_model_path = candidate
                break
    if not eval_model_path or not os.path.exists(eval_model_path):
        raise FileNotFoundError(
            f'No checkpoint found under {args.output_dir}. '
            'Training must save at least last_model.mdl before test evaluation.'
        )
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
        link_metrics = _run_test_link_prediction(evaluator, test_lp_path, entity_dict, args.output_dir)
        _release_gpu_memory()
    if run_tc:
        triple_metrics = evaluator.evaluate_test_triple_classification()
    memory_tracker.end_phase('eval')
    test_time = time.time() - test_start
    _write_results(args, train_summary, evaluator, link_metrics, triple_metrics, test_time, config_snapshot, memory_tracker)


if __name__ == '__main__':
    main()