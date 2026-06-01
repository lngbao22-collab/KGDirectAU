import json
import inspect
import os
import time

import torch

from configs.config import args
from base.evaluator import Evaluator
from data.dataset import Dataset, load_data
from data.dataloader import collate
from data.dict_hub import get_entity_dict, get_relation_id_map
from models.builder import import_module_from_path, load_attr_from_path
from models.samplers.bernoulli_sampler import BernoulliListwiseSampler
from utils.device import init_hardware
from utils.checkpoint import best_model_path
from utils.logger import setup_logger, write_results_report


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


def _write_results(current_args, train_summary, evaluator, link_metrics, triple_metrics, test_time, configs_snapshot) -> None:
    """Write the evaluation results and training summary to a report file."""

    if link_metrics:
        logger.info('Link prediction metrics on test set:\n{}'.format(json.dumps(link_metrics, indent=4)))

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
        os.path.join(current_args.output_dir, 'results.txt'),
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


def _examples_to_tensors(examples, entity_dict, relation_to_idx):
    """Convert examples into head, relation, and tail index tensors."""

    head_indices = torch.tensor([entity_dict.entity_to_idx(example.head_id) for example in examples], dtype=torch.long)
    relation_indices = torch.tensor([relation_to_idx.get(example.relation, relation_to_idx.get(example.relation.replace('inverse ', ''), 0)) for example in examples], dtype=torch.long)
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
    relation_to_idx = get_relation_id_map()
    train_examples = load_data(current_args.train_path, add_forward_triplet=True, add_backward_triplet=False)
    if not train_examples:
        raise ValueError(f'No training examples loaded from {current_args.train_path}')

    train_tensors = _examples_to_tensors(train_examples, entity_dict, relation_to_idx)
    sampler = BernoulliListwiseSampler(
        train_tensors,
        len(entity_dict),
        max(len(relation_to_idx), 1),
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
    relation_to_idx = get_relation_id_map()
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
    train_dataset = Dataset(path=current_args.train_path, task=current_args.dataset, examples=train_examples)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=max(getattr(current_args, 'batch_size', 1), 1),
        shuffle=True,
        collate_fn=collate,
        num_workers=getattr(current_args, 'workers', 2),
        pin_memory=True,
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
        test_start = time.time()
        link_metrics = None
        triple_metrics = None
        test_lp_path = _resolve_test_lp_path(args)
        if run_lp and test_lp_path:
            entity_dict = get_entity_dict()
            test_lp_log_path = os.path.join(args.output_dir, 'test_link_prediction.log')
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

    strategy_path = args.model_strategy_path
    if strategy_path.replace('\\', '/').endswith('softmax_strategy.py'):
        trainer = _build_softmax_trainer(args)
    elif strategy_path.replace('\\', '/').endswith('adversarial_strategy.py'):
        trainer = _build_adversarial_trainer(args)
    elif strategy_path.replace('\\', '/').endswith('pointwise_strategy.py'):
        trainer, train_dataloader = _build_pointwise_trainer(args)
        train_summary = trainer.train_loop(train_dataloader)
        evaluator = Evaluator(args)
        eval_model_path = train_summary.get('best_checkpoint_path') or best_model_path(args.output_dir)
        evaluator.load(eval_model_path)
        test_start = time.time()
        link_metrics = None
        triple_metrics = None
        test_lp_path = _resolve_test_lp_path(args)
        if run_lp and test_lp_path:
            entity_dict = get_entity_dict()
            test_lp_log_path = os.path.join(args.output_dir, 'test_link_prediction.log')
            forward_metrics = evaluator.evaluate_link_prediction_inplace(
                evaluator.model, test_lp_path, entity_dict, test_lp_log_path, eval_forward=True)
            backward_metrics = evaluator.evaluate_link_prediction_inplace(
                evaluator.model, test_lp_path, entity_dict, test_lp_log_path, eval_forward=False)
            link_metrics = _average_link_metrics(forward_metrics, backward_metrics)
        if run_tc:
            triple_metrics = evaluator.evaluate_test_triple_classification()
        test_time = time.time() - test_start
        _write_results(args, train_summary, evaluator, link_metrics, triple_metrics, test_time, config_snapshot)
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

    evaluator = Evaluator(args)
    eval_model_path = train_summary.get('best_checkpoint_path') or best_model_path(args.output_dir)
    evaluator.load(eval_model_path)
    test_start = time.time()
    link_metrics = None
    triple_metrics = None
    test_lp_path = _resolve_test_lp_path(args)
    if run_lp and test_lp_path:
        entity_dict = get_entity_dict()
        test_lp_log_path = os.path.join(args.output_dir, 'test_link_prediction.log')
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