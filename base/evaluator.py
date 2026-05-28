"""Abstract evaluation loop shared by KG evaluators."""

from typing import List, Optional, Sequence, Tuple
import os
from types import SimpleNamespace

import torch
import tqdm

from utils.logger import logger
from utils.device import get_model_obj, move_to_cuda

from data.dict_hub import get_all_triplet_dict, get_entity_dict
from data.dataset import Example, load_data
from data.dataloader import collate
from metrics.ranking import ranking_metrics_from_ranks
from metrics.classification import classification_metrics, find_global_threshold

from configs.config import args as global_args
from data.dict_hub import build_tokenizer
from models.encoders.bert import build_model
from utils.checkpoint import load_state_dict_clean, load_checkpoint, best_model_path, checkpoint_path
from configs.config import apply_train_args
import numpy as np
import json


class ModelInterfaceError(RuntimeError):
    """Custom error for when a model does not conform to the expected evaluation interface."""
    pass


def _filter_known(batch_score: torch.Tensor, examples: List[Example], all_triplet_dict, entity_dict) -> None:
    """Mask known neighbors for filtered link-prediction evaluation."""
    for idx, ex in enumerate(examples):
        gold_neighbor_ids = all_triplet_dict.get_neighbors(ex.head_id, ex.relation)
        if not gold_neighbor_ids:
            continue

        mask_indices = [
            entity_dict.entity_to_idx(entity_id)
            for entity_id in gold_neighbor_ids
            if entity_id != ex.tail_id
        ]
        if not mask_indices:
            continue

        mask_tensor = torch.LongTensor(mask_indices).to(batch_score.device)
        batch_score[idx].index_fill_(0, mask_tensor, float('-inf'))


def _infer_target_indices(examples: Sequence[Example], entity_dict) -> torch.Tensor:
    """Infer target entity indices for a batch of examples."""
    target_indices = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    return torch.LongTensor(target_indices)


def _score_by_embedding_adapter(model, examples: List[Example], entity_tensor: torch.Tensor) -> torch.Tensor:
    """Score examples using the model's embedding adapters."""
    hr_tensor = model.hr_embeddings(examples, entity_tensor.device)
    if hr_tensor.size(1) != entity_tensor.size(1):
        raise ValueError('hr_embeddings and entity_embeddings must have the same hidden size')
    return hr_tensor


def evaluate_model(
    model,
    eval_path: str,
    entity_dict=None,
    all_triplet_dict=None,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
    chunk_size: Optional[int] = None,
    topk: int = 10,
    filter_known: bool = True,
) -> Tuple[List[List[float]], List[List[int]], dict]:
    """Evaluate a KG model on link prediction.

    Returns:
        topk_scores, topk_indices, metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if entity_dict is None:
        entity_dict = get_entity_dict()
    if all_triplet_dict is None:
        all_triplet_dict = get_all_triplet_dict()

    examples = load_data(eval_path, add_forward_triplet=True, add_backward_triplet=False)
    total = len(examples)

    if total == 0:
        raise ValueError(f'No examples found in {eval_path}')

    if chunk_size is None:
        chunk_size = getattr(model, 'chunk_size', 8192)

    use_embedding_path = hasattr(model, 'entity_embeddings') and hasattr(model, 'hr_embeddings')

    topk_scores_all: List[List[float]] = []
    topk_indices_all: List[List[int]] = []
    ranks_all: List[int] = []

    if use_embedding_path:
        entity_tensor = model.entity_embeddings(device).to(device)
        hr_tensor = _score_by_embedding_adapter(model, examples, entity_tensor).to(device)

        for start in tqdm.tqdm(range(0, total, batch_size)):
            end = min(start + batch_size, total)
            batch_hr = hr_tensor[start:end, :]
            batch_examples = examples[start:end]

            batch_score = torch.zeros(
                batch_hr.size(0),
                entity_tensor.size(0),
                device=device,
                dtype=batch_hr.dtype,
            )
            for entity_start in range(0, entity_tensor.size(0), chunk_size):
                entity_end = min(entity_start + chunk_size, entity_tensor.size(0))
                batch_score[:, entity_start:entity_end] = torch.mm(
                    batch_hr,
                    entity_tensor[entity_start:entity_end, :].t(),
                )

            if filter_known:
                _filter_known(batch_score, batch_examples, all_triplet_dict, entity_dict)

            batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
            target_indices = _infer_target_indices(batch_examples, entity_dict).to(device)
            target_rank = torch.nonzero(batch_sorted_indices.eq(target_indices.unsqueeze(-1)).long(), as_tuple=False)
            if target_rank.size(0) != batch_score.size(0):
                raise RuntimeError('Unable to compute one rank per example')

            for idx in range(target_rank.size(0)):
                row = target_rank[idx].tolist()
                if row[0] != idx:
                    raise RuntimeError('Target rank rows are misaligned')
                ranks_all.append(row[1] + 1)

            topk_scores_all.extend(batch_sorted_score[:, :topk].tolist())
            topk_indices_all.extend(batch_sorted_indices[:, :topk].tolist())

    else:
        if not hasattr(model, 'score_batch'):
            raise ModelInterfaceError('Model must expose either embedding-style adapters or `score_batch`.')

        all_entity_ids = [entity_ex.entity_id for entity_ex in entity_dict.entity_exs]

        for start in tqdm.tqdm(range(0, total, batch_size)):
            end = min(start + batch_size, total)
            batch = examples[start:end]

            batch_score = torch.zeros(len(batch), len(all_entity_ids), device=device)
            for entity_start in range(0, len(all_entity_ids), chunk_size):
                entity_end = min(entity_start + chunk_size, len(all_entity_ids))
                entity_chunk = all_entity_ids[entity_start:entity_end]
                scores_chunk = model.score_batch(
                    [ex.head_id for ex in batch],
                    [ex.relation for ex in batch],
                    entity_chunk,
                )
                if not isinstance(scores_chunk, torch.Tensor):
                    scores_chunk = torch.tensor(scores_chunk, device=device)
                batch_score[:, entity_start:entity_end] = scores_chunk.to(device)

            if filter_known:
                _filter_known(batch_score, batch, all_triplet_dict, entity_dict)

            batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
            target_indices = _infer_target_indices(batch, entity_dict).to(device)
            target_rank = torch.nonzero(batch_sorted_indices.eq(target_indices.unsqueeze(-1)).long(), as_tuple=False)
            if target_rank.size(0) != batch_score.size(0):
                raise RuntimeError('Unable to compute one rank per example')

            for idx in range(target_rank.size(0)):
                row = target_rank[idx].tolist()
                if row[0] != idx:
                    raise RuntimeError('Target rank rows are misaligned')
                ranks_all.append(row[1] + 1)

            topk_scores_all.extend(batch_sorted_score[:, :topk].tolist())
            topk_indices_all.extend(batch_sorted_indices[:, :topk].tolist())

    metrics = ranking_metrics_from_ranks(ranks_all)
    return topk_scores_all, topk_indices_all, metrics

class Evaluator:
    """Helper to load encoder checkpoints and run model-based evaluations."""

    def __init__(self, args=None):
        self.args = args if args is not None else global_args
        self.model = None
        self.train_args: SimpleNamespace | None = None
        self.use_cuda = False

    def load(self, ckt_path: str, use_data_parallel: bool = False):
        """Load checkpoint, apply training args, build tokenizer and model, and load weights."""

        checkpoint = load_checkpoint(ckt_path, map_location='cpu')
        self.checkpoint = checkpoint
        self.train_args = SimpleNamespace(**checkpoint['args'])

        apply_train_args(self.train_args)
        build_tokenizer(self.train_args)

        self.model = build_model(self.train_args)
        load_state_dict_clean(self.model, ckt_path)
        self.model.eval()

        if use_data_parallel and torch.cuda.device_count() > 1:
            logger.info('Use data parallel evaluator model')
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.use_cuda = True
        elif torch.cuda.is_available():
            self.model.cuda()
            self.use_cuda = True

        logger.info('Load model from %s successfully', ckt_path)

    @torch.no_grad()
    def evaluate_triple_classification_inplace(self, model, label_file, output_log_path, batch_size=128):
        """Evaluate triple classification using the model's forward pass."""

        model = get_model_obj(model)
        model.eval()
        if not os.path.exists(label_file):
            print(f"[EVAL] {label_file} not found, skip evaluation.")
            return
        eval_set = 'TEST' if 'test' in label_file else 'VALID'
        print(f"\n[{eval_set}] Evaluating triple classification inplace on {label_file} ...")
        eval_exs = load_data(label_file, add_forward_triplet=False, add_backward_triplet=False)
        y_true = [ex.label for ex in eval_exs]
        y_prob = []
        with torch.no_grad():
            for i in range(0, len(eval_exs), batch_size):
                batch = eval_exs[i:i + batch_size]
                batch_vec = [ex.vectorize() for ex in batch]
                batch_dict = collate(batch_vec)
                if torch.cuda.is_available():
                    batch_dict = move_to_cuda(batch_dict)
                    model.cuda()
                output_dict = model(**batch_dict)
                logits = model.compute_logits(output_dict=output_dict, batch_dict=batch_dict)['logits']
                prob = torch.sigmoid(logits.diag()).detach().cpu().numpy().reshape(-1)
                y_prob.extend(prob.tolist())

        threshold = find_global_threshold(y_true, y_prob)
        y_pred = (np.array(y_prob) > threshold).astype(int).tolist()
        metrics_cls = classification_metrics(y_true, y_pred, y_prob)
        log_thresh = f"[{eval_set}] Best threshold: {threshold:.6f}"
        log_cls = f"[{eval_set}] Triple Classification: {json.dumps(metrics_cls)}"
        print(log_thresh)
        print(log_cls)
        logger.info(log_thresh)
        logger.info(log_cls)
        with open(output_log_path, 'a', encoding='utf-8') as f:
            f.write(log_thresh + '\n')
            f.write(log_cls + '\n')
        return metrics_cls

    @torch.no_grad()
    def evaluate_link_prediction_inplace(self, model, eval_path, entity_dict, output_log_path, batch_size=128, eval_forward=True):
        """Evaluate link prediction using the model's forward pass."""

        model = get_model_obj(model)
        model.eval()
        if not os.path.exists(eval_path):
            print(f"[EVAL] {eval_path} not found, skip link prediction evaluation.")
            return
        eval_set = 'TEST' if 'test' in eval_path else 'VALID'
        print(f"\n[{eval_set}] Evaluating link prediction inplace on {eval_path} ...")
        examples = load_data(eval_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)

        hr_tensor, _ = model.predict_by_examples(examples, batch_size=batch_size)
        entity_examples = [Example(head_id='', relation='', tail_id=entity_ex.entity_id) for entity_ex in entity_dict.entity_exs]
        entities_tensor = model.predict_by_entities(entity_examples, batch_size=max(batch_size, 512))

        if torch.cuda.is_available():
            hr_tensor = hr_tensor.cuda()
            entities_tensor = entities_tensor.cuda()

        score = torch.mm(hr_tensor, entities_tensor.t())
        all_triplet_dict = get_all_triplet_dict()
        _filter_known(score, examples, all_triplet_dict, entity_dict)
        target = torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in examples])
        sorted_indices = torch.sort(score, dim=-1, descending=True).indices
        target_rank = torch.nonzero(sorted_indices.eq(target.unsqueeze(-1)).long(), as_tuple=False)
        if target_rank.size(0) != score.size(0):
            raise RuntimeError('Unable to compute one rank per example')

        ranks = []
        for idx in range(target_rank.size(0)):
            row = target_rank[idx].tolist()
            if row[0] != idx:
                raise RuntimeError('Target rank rows are misaligned')
            ranks.append(row[1] + 1)

        metrics = ranking_metrics_from_ranks(ranks)
        log_str = f"[{eval_set}] Link Prediction Metrics: {json.dumps(metrics)}"
        print(log_str)
        with open(output_log_path, 'a', encoding='utf-8') as f:
            f.write(log_str + '\n')
        return metrics

    def evaluate_test_triple_classification(self, epoch=None):
        """Evaluate triple classification on the test split using the loaded checkpoint."""

        args = self.args if self.args is not None else global_args
        test_label_path = ''
        valid_label_path = getattr(args, 'valid_label_path', '')
        if valid_label_path:
            test_label_path = valid_label_path.replace('valid_w_label.txt', 'test_w_label.txt')
        if not test_label_path:
            candidate_dirs = []
            for source_path in [getattr(args, 'test_path', ''), getattr(args, 'valid_path', ''), getattr(args, 'train_path', '')]:
                if source_path:
                    candidate_dirs.append(os.path.dirname(source_path))
            candidate_dirs.append(os.path.join('data', getattr(args, 'dataset', '')))
            for candidate_dir in candidate_dirs:
                for candidate_name in ['test_w_label.txt', 'test_label.txt']:
                    candidate_path = os.path.join(candidate_dir, candidate_name)
                    if os.path.exists(candidate_path):
                        test_label_path = candidate_path
                        break
                if test_label_path:
                    break

        if not os.path.exists(test_label_path):
            print('[TEST] test_w_label.txt not found, skip test evaluation.')
            return

        print('\n[TEST] Evaluating triple classification on test set...')
        test_exs = load_data(test_label_path, add_forward_triplet=False, add_backward_triplet=False)
        y_true = [ex.label for ex in test_exs]
        y_prob = []
        batch_size = 128

        if epoch is None:
            ckt_path = getattr(args, 'eval_model_path', '') or best_model_path(args.model_dir)
        else:
            ckt_path = checkpoint_path(args.model_dir, epoch)
            if not os.path.exists(ckt_path):
                ckt_path = checkpoint_path(args.model_dir, epoch, 0)
            if not os.path.exists(ckt_path):
                ckt_path = getattr(args, 'eval_model_path', '') or best_model_path(args.model_dir)

        if self.model is None:
            self.load(ckt_path)
        self.model.eval()

        for i in range(0, len(test_exs), batch_size):
            batch = test_exs[i:i + batch_size]
            batch_vec = [ex.vectorize() for ex in batch]
            batch_dict = collate(batch_vec)
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
                self.model.cuda()
            output_dict = self.model(**batch_dict)
            logits = self.model.compute_logits(output_dict=output_dict, batch_dict=batch_dict)['logits']
            prob = torch.sigmoid(logits.diag()).detach().cpu().numpy().reshape(-1)
            y_prob.extend(prob.tolist())

        threshold = find_global_threshold(y_true, y_prob)
        y_pred = (np.array(y_prob) > threshold).astype(int).tolist()
        metrics_cls = classification_metrics(y_true, y_pred, y_prob)
        log_thresh = f'[TEST] Best threshold on test: {threshold:.6f}'
        log_cls = f'[TEST] Triple Classification: {json.dumps(metrics_cls)}'
        print(log_thresh)
        print(log_cls)
        logger.info(log_thresh)
        logger.info(log_cls)
        with open(os.path.join(args.model_dir, 'test_metrics.log'), 'a', encoding='utf-8') as f:
            f.write(log_thresh + '\n')
            f.write(log_cls + '\n')

        return metrics_cls

