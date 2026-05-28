import json
import os
import time

from configs.config import args
from base.evaluator import Evaluator
from models.builder import import_module_from_path
from utils.device import init_hardware
from utils.checkpoint import best_model_path
from utils.logger import setup_logger, write_results_report
from data.dict_hub import get_entity_dict


logger = setup_logger(log_file=os.path.join(args.model_dir, 'run.log'))


def _resolve_test_lp_path(current_args):
    candidates = [current_args.test_path]

    if current_args.valid_path:
        valid_dir = os.path.dirname(current_args.valid_path)
        valid_name = os.path.basename(current_args.valid_path)
        candidates.append(os.path.join(valid_dir, valid_name.replace('valid', 'test')))
        candidates.append(os.path.join(valid_dir, 'test.txt'))

    if current_args.valid_label_path:
        label_dir = os.path.dirname(current_args.valid_label_path)
        candidates.append(os.path.join(label_dir, 'test.txt.json'))
        candidates.append(os.path.join(label_dir, 'test.txt'))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ''


def _write_results(current_args, train_summary, evaluator, link_metrics, triple_metrics, test_time, configs_snapshot):
    checkpoint = getattr(evaluator, 'checkpoint', {}) or {}
    best_metric = checkpoint.get('best_metric') or {}
    best_epoch = train_summary.get('best_epoch') if train_summary else None
    best_mrr = train_summary.get('best_mrr') if train_summary else None

    if best_epoch is None:
        best_epoch = best_metric.get('epoch')
    if best_mrr is None:
        best_mrr = best_metric.get('score')

    train_time = train_summary.get('train_time') if train_summary else None
    valid_time = train_summary.get('valid_time') if train_summary else None
    total_time = None
    if train_summary and train_summary.get('total_time') is not None:
        total_time = train_summary['total_time'] + test_time

    write_results_report(
        os.path.join(current_args.model_dir, 'results.txt'),
        link_metrics=link_metrics,
        triple_metrics=triple_metrics,
        best_epoch=best_epoch,
        best_mrr=best_mrr,
        train_time=train_time,
        valid_time=valid_time,
        test_time=test_time,
        total_time=total_time,
        configs=configs_snapshot,
    )


def _average_link_metrics(forward_metrics, backward_metrics):
    if not forward_metrics or not backward_metrics:
        return forward_metrics or backward_metrics

    averaged_metrics = {}
    for key in forward_metrics.keys() & backward_metrics.keys():
        forward_value = forward_metrics[key]
        backward_value = backward_metrics[key]
        if isinstance(forward_value, (int, float)) and isinstance(backward_value, (int, float)):
            averaged_metrics[key] = (forward_value + backward_value) / 2
    return averaged_metrics


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
        eval_model_path = args.eval_model_path or best_model_path(args.model_dir)
        evaluator.load(eval_model_path)
        test_start = time.time()
        link_metrics = None
        triple_metrics = None
        test_lp_path = _resolve_test_lp_path(args)
        if run_lp and test_lp_path:
            entity_dict = get_entity_dict()
            test_lp_log_path = os.path.join(args.model_dir, 'test_link_prediction.log')
            forward_metrics = evaluator.evaluate_link_prediction_inplace(
                evaluator.model, test_lp_path, entity_dict, test_lp_log_path, eval_forward=True)
            backward_metrics = evaluator.evaluate_link_prediction_inplace(
                evaluator.model, test_lp_path, entity_dict, test_lp_log_path, eval_forward=False)
            link_metrics = _average_link_metrics(forward_metrics, backward_metrics)
        if run_tc:
            triple_metrics = evaluator.evaluate_test_triple_classification()
        test_time = time.time() - test_start
        _write_results(args, None, evaluator, link_metrics, triple_metrics, test_time, config_snapshot)
        return

    # Dynamically load the strategy/trainer class from config
    strategy_path = args.model_strategy_path or args.model_strategy_path or 'models/strategies/simkgc_strategy.py'
    strategy_mod = import_module_from_path(strategy_path)
    # prefer common trainer names
    trainer_cls = None
    for cand in ('ContrastiveTrainer', 'Trainer', 'SimKGCStrategy', 'SimKGCTrainer', 'Strategy'):
        if hasattr(strategy_mod, cand):
            trainer_cls = getattr(strategy_mod, cand)
            break
    if trainer_cls is None:
        # fallback: find first class defined in module
        for v in vars(strategy_mod).values():
            try:
                if isinstance(v, type):
                    trainer_cls = v
                    break
            except Exception:
                continue
    if trainer_cls is None:
        raise ImportError(f'Could not find a Trainer class in {strategy_path}')

    trainer = trainer_cls(args, ngpus_per_node=ngpus_per_node)
    train_summary = trainer.train_loop()

    evaluator = Evaluator(args)
    eval_model_path = train_summary.get('best_checkpoint_path') or best_model_path(args.model_dir)
    evaluator.load(eval_model_path)
    test_start = time.time()
    link_metrics = None
    triple_metrics = None
    test_lp_path = _resolve_test_lp_path(args)
    if run_lp and test_lp_path:
        entity_dict = get_entity_dict()
        test_lp_log_path = os.path.join(args.model_dir, 'test_link_prediction.log')
        forward_metrics = evaluator.evaluate_link_prediction_inplace(
            evaluator.model, test_lp_path, entity_dict, test_lp_log_path, eval_forward=True)
        backward_metrics = evaluator.evaluate_link_prediction_inplace(
            evaluator.model, test_lp_path, entity_dict, test_lp_log_path, eval_forward=False)
        link_metrics = _average_link_metrics(forward_metrics, backward_metrics)
    if run_tc:
        triple_metrics = evaluator.evaluate_test_triple_classification()
    test_time = time.time() - test_start
    _write_results(args, train_summary, evaluator, link_metrics, triple_metrics, test_time, config_snapshot)


if __name__ == '__main__':
    main()
