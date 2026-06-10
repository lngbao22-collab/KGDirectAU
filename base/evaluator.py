"""Abstract evaluation loop shared by KG evaluators."""

from typing import List, Optional, Sequence, Tuple
import inspect
import os
from types import SimpleNamespace

import torch
import tqdm

from base.model import KGEModel
from utils.logger import logger
from utils.device import get_model_obj, move_to_cuda

from data.dict_hub import get_all_triplet_dict, get_entity_dict
from data.dataset import Example, load_data
from data.dataloader import collate
from metrics.ranking import ranking_metrics_from_ranks
from metrics.classification import classification_metrics, find_global_threshold

from configs.config import args as global_args
from data.dict_hub import build_tokenizer
from models.builder import import_module_from_path, load_attr_from_path
from utils.checkpoint import load_state_dict_clean, load_checkpoint, best_model_path, checkpoint_path
from configs.config import apply_train_args
import numpy as np
import json


FILTER_MASK_VALUE = -1e30


class ModelInterfaceError(RuntimeError):
	"""Custom error for when a model does not conform to the expected evaluation interface."""
	pass


def _supports_kge_1vsall_eval(model) -> bool:
	"""Return True when the model exposes LibKGE-style 1-vs-all tail/head scoring."""

	return hasattr(model, 'predict_tail_sp_') or isinstance(model, KGEModel)


def _resolve_relation_index(relation: str, relation_to_idx: dict) -> int:
	"""Map a relation string to its embedding index."""

	if relation in relation_to_idx:
		return relation_to_idx[relation]
	normalized = ' '.join(relation.split())
	if normalized in relation_to_idx:
		return relation_to_idx[normalized]
	if relation.startswith('inverse '):
		base_relation = relation[len('inverse '):]
		if f'inverse {base_relation}' in relation_to_idx:
			return relation_to_idx[f'inverse {base_relation}']
		if base_relation in relation_to_idx:
			return relation_to_idx[base_relation]
	raise KeyError(relation)


def _relation_lookup(model):
	"""Return a callable that maps relation strings to embedding indices."""

	if hasattr(model, '_relation_to_idx') and callable(model._relation_to_idx):
		return model._relation_to_idx
	rel_to_idx = getattr(model, 'rel_to_idx', None)
	if rel_to_idx is None:
		raise RuntimeError('Model is missing a relation index lookup for fast evaluation')
	return lambda relation: _resolve_relation_index(relation, rel_to_idx)


def _build_filter_index_maps(all_triplet_dict, entity_dict, relation_lookup) -> tuple[dict, dict]:
	"""Build filtered-evaluation maps over integer (h, r) and (r, t) keys."""

	entity_to_idx = entity_dict.entity_to_idx
	sp_to_tails: dict[tuple[int, int], list[int]] = {}
	for (head_id, relation), tail_ids in all_triplet_dict.hr2tails.items():
		try:
			h_idx = entity_to_idx[head_id]
			r_idx = relation_lookup(relation)
		except KeyError:
			continue
		tails = [entity_to_idx[tail_id] for tail_id in tail_ids if tail_id in entity_to_idx]
		if tails:
			sp_to_tails[(h_idx, r_idx)] = tails

	po_to_heads: dict[tuple[int, int], list[int]] = {}
	for (relation, tail_id), head_ids in all_triplet_dict.rt2heads.items():
		try:
			r_idx = relation_lookup(relation)
			t_idx = entity_to_idx[tail_id]
		except KeyError:
			continue
		heads = [entity_to_idx[head_id] for head_id in head_ids if head_id in entity_to_idx]
		if heads:
			po_to_heads[(r_idx, t_idx)] = heads
	return sp_to_tails, po_to_heads


def _apply_filter_mask(
	scores: torch.Tensor,
	h_idx: torch.Tensor,
	r_idx: torch.Tensor,
	t_idx: torch.Tensor,
	filter_map: dict[tuple[int, int], list[int]],
	*,
	predict_head: bool,
) -> torch.Tensor:
	"""Mask known alternative true entities to ``FILTER_MASK_VALUE``, keeping the target."""

	rows: list[int] = []
	cols: list[int] = []
	for i in range(h_idx.size(0)):
		if predict_head:
			key = (int(r_idx[i].item()), int(t_idx[i].item()))
			target = int(h_idx[i].item())
		else:
			key = (int(h_idx[i].item()), int(r_idx[i].item()))
			target = int(t_idx[i].item())
		for candidate in filter_map.get(key, ()):
			if candidate != target:
				rows.append(i)
				cols.append(candidate)
	if rows:
		row_tensor = torch.tensor(rows, device=scores.device, dtype=torch.long)
		col_tensor = torch.tensor(cols, device=scores.device, dtype=torch.long)
		scores[row_tensor, col_tensor] = FILTER_MASK_VALUE
	return scores


def _evaluate_kge_1vsall_batch(
	model,
	h_idx: torch.Tensor,
	r_idx: torch.Tensor,
	t_idx: torch.Tensor,
	sp_filter: dict[tuple[int, int], list[int]],
	po_filter: dict[tuple[int, int], list[int]],
	*,
	predict_head: bool,
	filter_known: bool,
) -> list[int]:
	"""Score and rank one batch with full-matrix ``sp_`` or ``_po`` broadcasting."""

	device = next(model.parameters()).device
	h_idx = h_idx.to(device)
	r_idx = r_idx.to(device)
	t_idx = t_idx.to(device)

	if predict_head:
		scores = model.predict_head_po_(r_idx, t_idx)
		if filter_known:
			scores = _apply_filter_mask(scores, h_idx, r_idx, t_idx, po_filter, predict_head=True)
		target_indices = h_idx
	else:
		scores = model.predict_tail_sp_(h_idx, r_idx)
		if filter_known:
			scores = _apply_filter_mask(scores, h_idx, r_idx, t_idx, sp_filter, predict_head=False)
		target_indices = t_idx

	return _ranks_from_score_matrix(scores, target_indices)


def _evaluate_kge_link_prediction(
	model,
	examples: Sequence[Example],
	entity_dict,
	batch_size: int,
	*,
	eval_forward: bool,
	filter_known: bool,
) -> list[int]:
	"""Fast filtered link prediction for ``KGEModel`` instances."""

	relation_lookup = _relation_lookup(model)
	sp_filter, po_filter = _build_filter_index_maps(get_all_triplet_dict(), entity_dict, relation_lookup)
	predict_head = (not eval_forward) and bool(getattr(model, 'bidirectional_score_batch', False))
	scoring_examples = _coerce_forward_examples(examples) if predict_head else list(examples)
	h_all, r_all, t_all = _examples_to_query_index_tensors(scoring_examples, entity_dict, model)

	ranks: list[int] = []
	iterator = range(0, len(scoring_examples), batch_size)
	for start in tqdm.tqdm(iterator, disable=len(scoring_examples) <= batch_size):
		end = min(start + batch_size, len(scoring_examples))
		batch_ranks = _evaluate_kge_1vsall_batch(
			model,
			h_all[start:end],
			r_all[start:end],
			t_all[start:end],
			sp_filter,
			po_filter,
			predict_head=predict_head,
			filter_known=filter_known,
		)
		ranks.extend(batch_ranks)
	return ranks


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


def _filter_known_heads(batch_score: torch.Tensor, examples: List[Example], all_triplet_dict, entity_dict) -> None:
    """Mask other known heads for filtered head-prediction evaluation."""

    for idx, ex in enumerate(examples):
        gold_head_ids = all_triplet_dict.get_heads(ex.relation, ex.tail_id)
        if not gold_head_ids:
            continue

        mask_indices = [
            entity_dict.entity_to_idx(entity_id)
            for entity_id in gold_head_ids
            if entity_id != ex.head_id
        ]
        if not mask_indices:
            continue

        mask_tensor = torch.LongTensor(mask_indices).to(batch_score.device)
        batch_score[idx].index_fill_(0, mask_tensor, float('-inf'))


def _coerce_forward_examples(examples: Sequence[Example]) -> List[Example]:
    """Normalize backward/reversed examples to forward (head, relation, tail) form."""

    normalized: List[Example] = []
    for ex in examples:
        relation = ex.relation
        head_id = ex.head_id
        tail_id = ex.tail_id
        head = getattr(ex, 'head', head_id)
        tail = getattr(ex, 'tail', tail_id)

        if str(relation).startswith('inverse '):
            relation = relation[len('inverse '):]
            head_id, tail_id = ex.tail_id, ex.head_id
            head = getattr(ex, 'tail', tail_id)
            tail = getattr(ex, 'head', head_id)

        normalized.append(Example(
            head_id=head_id,
            head=head,
            relation=relation,
            tail_id=tail_id,
            tail=tail,
            label=getattr(ex, 'label', None),
        ))
    return normalized


def _uses_head_batch_scoring(model) -> bool:
    """Return True when the model exposes native head-batch link-prediction scoring."""

    return bool(getattr(model, 'bidirectional_score_batch', False))


def _score_batch_supports_mode(model) -> bool:
    """Return True when score_batch accepts an explicit batch mode argument."""

    if not hasattr(model, 'score_batch'):
        return False
    return 'mode' in inspect.signature(model.score_batch).parameters


def _examples_to_query_index_tensors(examples: Sequence[Example], entity_dict, model) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert examples to head/relation/tail index tensors once per evaluation pass."""

    head_indices = [entity_dict.entity_to_idx(ex.head_id) for ex in examples]
    tail_indices = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    relation_lookup = getattr(model, '_relation_to_idx', getattr(model, 'rel_to_idx', None))
    if relation_lookup is None:
        raise RuntimeError('Model is missing a relation index lookup for fast evaluation')
    relation_indices = [relation_lookup(ex.relation) for ex in examples]
    return (
        torch.tensor(head_indices, dtype=torch.long),
        torch.tensor(relation_indices, dtype=torch.long),
        torch.tensor(tail_indices, dtype=torch.long),
    )


def _entity_indices(entity_dict, entity_ids: Sequence[str]) -> torch.Tensor:
    """Convert entity id strings to a single index tensor."""

    return torch.tensor([entity_dict.entity_to_idx(entity_id) for entity_id in entity_ids], dtype=torch.long)


def _infer_target_indices(examples: Sequence[Example], entity_dict, predict_head: bool = False) -> torch.Tensor:
    """Infer target entity indices for a batch of examples."""

    if predict_head:
        target_indices = [entity_dict.entity_to_idx(ex.head_id) for ex in examples]
    else:
        target_indices = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    return torch.LongTensor(target_indices)


def _ranks_from_score_matrix(score: torch.Tensor, target_indices: torch.Tensor) -> list[int]:
    """Compute 1-based filtered ranks without sorting the full score matrix."""

    target_scores = score.gather(1, target_indices.unsqueeze(1))
    return (score > target_scores).sum(dim=1).add(1).tolist()


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

    model = get_model_obj(model)
    if _supports_kge_1vsall_eval(model):
        ranks_all = _evaluate_kge_link_prediction(
            model,
            examples,
            entity_dict,
            batch_size,
            eval_forward=True,
            filter_known=filter_known,
        )
        metrics = ranking_metrics_from_ranks(ranks_all)
        return [], [], metrics

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

    def load(self, ckt_path: str, use_data_parallel: bool = False) -> None:
        """Load checkpoint, apply training args, build tokenizer and model, and load weights."""

        checkpoint = load_checkpoint(ckt_path, map_location='cpu')
        self.checkpoint = checkpoint
        self.train_args = SimpleNamespace(**checkpoint['args'])

        apply_train_args(self.train_args)

        from models.builder import build_model

        model_name = str(getattr(self.train_args, 'model', '') or '').lower()
        scorer_path = (
            getattr(self.train_args, 'model_scorer_path', '')
            or getattr(self.train_args, 'model_encoder_path', '')
            or ''
        )
        if 'simkgc' in model_name or scorer_path.endswith('simkgc_scorer.py'):
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
    def evaluate_triple_classification_inplace(self, model, label_file, output_log_path, batch_size=128) -> dict:
        """Evaluate triple classification using the model's forward pass."""

        model = get_model_obj(model)
        model.eval()
        if not os.path.exists(label_file):
            logger.info(f"[EVAL] {label_file} not found, skip evaluation.")
            return
        eval_set = 'TEST' if 'test' in label_file else 'VALID'
        eval_exs = [
            ex for ex in load_data(
                label_file,
                add_forward_triplet=label_file.endswith('.json'),
                add_backward_triplet=False,
            )
            if ex.label is not None
        ]
        y_true = [int(ex.label) for ex in eval_exs]
        y_prob = []
        if hasattr(model, 'score_batch'):
            if not eval_exs:
                logger.info(f"[EVAL] {label_file} has no labeled examples, skip evaluation.")
                return
            for i in range(0, len(eval_exs), batch_size):
                batch = eval_exs[i:i + batch_size]
                scores = model.score_batch(
                    [ex.head_id for ex in batch],
                    [ex.relation for ex in batch],
                    [ex.tail_id for ex in batch],
                )
                if not isinstance(scores, torch.Tensor):
                    scores = torch.tensor(scores)
                if scores.dim() == 2 and scores.size(0) == scores.size(1):
                    scores = scores.diag()
                y_prob.extend(torch.sigmoid(scores).detach().cpu().numpy().reshape(-1).tolist())
        else:
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
        logger.info(log_thresh)
        logger.info(log_cls)
        with open(output_log_path, 'a', encoding='utf-8') as f:
            f.write(log_thresh + '\n')
            f.write(log_cls + '\n')
        return metrics_cls

    @torch.inference_mode()
    def evaluate_link_prediction_inplace(self, model, eval_path, entity_dict, output_log_path, batch_size=128, eval_forward=True, examples=None) -> dict:
        """Evaluate link prediction using the model's forward pass."""
        model = get_model_obj(model)
        model.eval()
        if not os.path.exists(eval_path):
            logger.info(f"[EVAL] {eval_path} not found, skip link prediction evaluation.")
            return {}
        if examples is None:
            examples = load_data(eval_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)

        if _supports_kge_1vsall_eval(model):
            ranks = _evaluate_kge_link_prediction(
                model,
                examples,
                entity_dict,
                batch_size,
                eval_forward=eval_forward,
                filter_known=True,
            )
            return ranking_metrics_from_ranks(ranks)

        predict_head = (not eval_forward) and _uses_head_batch_scoring(model)
        scoring_examples = _coerce_forward_examples(examples) if predict_head else list(examples)

        if hasattr(model, 'score_batch'):
            all_entity_ids = [entity_ex.entity_id for entity_ex in entity_dict.entity_exs]
            score_device = next(model.parameters()).device
            head_ids = [ex.head_id for ex in scoring_examples]
            relations = [ex.relation for ex in scoring_examples]
            score_batch_mode = 'head-batch' if predict_head and _score_batch_supports_mode(model) else 'tail-batch'
            use_fast_indices = hasattr(model, 'score_batch_from_indices')
            query_head_idx = query_rel_idx = query_tail_idx = None
            all_entity_idx = None
            if use_fast_indices:
                query_head_idx, query_rel_idx, query_tail_idx = _examples_to_query_index_tensors(
                    scoring_examples, entity_dict, model
                )
                all_entity_idx = _entity_indices(entity_dict, all_entity_ids)

            if (
                score_batch_mode == 'head-batch'
                and hasattr(model, 'prepare_head_prediction_queries')
                and hasattr(model, 'score_head_prediction_full')
            ):
                tail_ids = [ex.tail_id for ex in scoring_examples]
                query_cache = model.prepare_head_prediction_queries(tail_ids, relations)
                score = model.score_head_prediction_full(query_cache)
                if score.size(0) != len(scoring_examples) or score.size(1) != len(all_entity_ids):
                    raise RuntimeError('DaBR fast head-prediction score matrix has unexpected shape')
            elif (
                score_batch_mode == 'tail-batch'
                and hasattr(model, 'prepare_link_prediction_queries')
                and hasattr(model, 'score_link_prediction_full')
            ):
                query_cache = model.prepare_link_prediction_queries(head_ids, relations)
                score = model.score_link_prediction_full(query_cache)
                if score.size(0) != len(scoring_examples) or score.size(1) != len(all_entity_ids):
                    raise RuntimeError('DaBR fast link-prediction score matrix has unexpected shape')
            else:
                score = torch.zeros(len(scoring_examples), len(all_entity_ids), device=score_device)
                entity_chunk_size = getattr(model, 'eval_entity_chunk_size', None)
                if entity_chunk_size is None:
                    entity_chunk_size = getattr(model.config, 'eval_entity_chunk_size', None) if hasattr(model, 'config') else None
                if entity_chunk_size is None:
                    entity_chunk_size = getattr(global_args, 'eval_entity_chunk_size', None)
                if entity_chunk_size is None:
                    entity_chunk_size = max(batch_size, 4096)
                entity_chunk_size = max(int(entity_chunk_size), 1)

                if score_batch_mode == 'head-batch' and hasattr(model, 'prepare_head_prediction_queries'):
                    tail_ids = [ex.tail_id for ex in scoring_examples]
                    query_cache = model.prepare_head_prediction_queries(tail_ids, relations)
                    score_candidates = getattr(model, 'score_head_prediction_candidates', None)
                elif hasattr(model, 'prepare_link_prediction_queries'):
                    query_cache = model.prepare_link_prediction_queries(head_ids, relations)
                    score_candidates = model.score_link_prediction_candidates
                else:
                    query_cache = None
                    score_candidates = None

                use_eval_amp = (
                    score_device.type == 'cuda'
                    and bool(getattr(global_args, 'eval_use_amp', getattr(global_args, 'use_amp', False)))
                )
                for start in range(0, len(all_entity_ids), entity_chunk_size):
                    end = min(start + entity_chunk_size, len(all_entity_ids))
                    if score_candidates is not None:
                        chunk_score = score_candidates(query_cache, (start, end))
                    elif use_fast_indices:
                        candidate_idx = all_entity_idx[start:end].to(score_device)
                        with torch.autocast(device_type='cuda', enabled=use_eval_amp):
                            if score_batch_mode == 'head-batch':
                                chunk_score = model.score_batch_from_indices(
                                    query_rel_idx,
                                    candidate_idx,
                                    mode='head-batch',
                                    query_tail_indices=query_tail_idx,
                                )
                            else:
                                chunk_score = model.score_batch_from_indices(
                                    query_rel_idx,
                                    candidate_idx,
                                    mode='tail-batch',
                                    query_head_indices=query_head_idx,
                                )
                    else:
                        entity_chunk = all_entity_ids[start:end]
                        score_batch_kwargs = {}
                        if score_batch_mode == 'head-batch':
                            score_batch_kwargs['mode'] = 'head-batch'
                            score_batch_kwargs['query_tail_ids'] = [ex.tail_id for ex in scoring_examples]
                        with torch.autocast(device_type='cuda', enabled=use_eval_amp):
                            if score_batch_mode == 'head-batch':
                                chunk_score = model.score_batch(
                                    head_ids,
                                    relations,
                                    entity_chunk,
                                    **score_batch_kwargs,
                                )
                            else:
                                chunk_score = model.score_batch(head_ids, relations, entity_chunk)
                    if not isinstance(chunk_score, torch.Tensor):
                        chunk_score = torch.tensor(chunk_score, device=score_device)
                    score[:, start:end] = chunk_score
        else:
            hr_tensor, _ = model.predict_by_examples(scoring_examples, batch_size=batch_size)
            entity_examples = [Example(head_id='', relation='', tail_id=entity_ex.entity_id) for entity_ex in entity_dict.entity_exs]
            entities_tensor = model.predict_by_entities(entity_examples, batch_size=max(batch_size, 512))

            if torch.cuda.is_available():
                hr_tensor = hr_tensor.cuda()
                entities_tensor = entities_tensor.cuda()
            score = torch.mm(hr_tensor, entities_tensor.t())
        all_triplet_dict = get_all_triplet_dict()
        if predict_head:
            _filter_known_heads(score, scoring_examples, all_triplet_dict, entity_dict)
        else:
            _filter_known(score, scoring_examples, all_triplet_dict, entity_dict)
        target_indices = _infer_target_indices(scoring_examples, entity_dict, predict_head=predict_head).to(score.device)
        ranks = _ranks_from_score_matrix(score, target_indices)
        metrics = ranking_metrics_from_ranks(ranks)
        return metrics

    def evaluate_test_triple_classification(self, epoch=None) -> dict:
        """Evaluate triple classification on the test split using the loaded checkpoint."""

        args = self.args if self.args is not None else global_args
        test_label_path = getattr(args, 'test_w_label_path', '')
        if not test_label_path or not os.path.exists(test_label_path):
            candidate_dirs = []
            for source_path in [getattr(args, 'valid_w_label_path', ''), getattr(args, 'valid_path', ''), getattr(args, 'train_path', ''), getattr(args, 'test_path', '')]:
                if source_path:
                    candidate_dirs.append(os.path.dirname(source_path))
            candidate_dirs.append(os.path.join('data', getattr(args, 'dataset', '')))
            candidate_dirs.append(os.path.join('data', getattr(args, 'dataset', ''), 'preprocessed'))
            for candidate_dir in candidate_dirs:
                for candidate_name in ['test_w_label.txt.json', 'test_w_label.txt', 'test_label.txt.json', 'test_label.txt']:
                    candidate_path = os.path.join(candidate_dir, candidate_name)
                    if os.path.exists(candidate_path):
                        test_label_path = candidate_path
                        break
                if test_label_path:
                    break

        if not os.path.exists(test_label_path):
            logger.info('[TEST] test_w_label.txt not found, skip test evaluation.')
            return {}

        logger.info('[TEST] Evaluating triple classification on test set...')
        test_exs = [
            ex
            for ex in load_data(
                test_label_path,
                add_forward_triplet=True,
                add_backward_triplet=False,
            )
            if ex.label is not None
        ]
        if not test_exs:
            logger.info(f"[TEST] {test_label_path} has no labeled examples, skip test evaluation.")
            return {}
        y_true = [int(ex.label) for ex in test_exs]
        y_prob = []
        batch_size = 128

        if epoch is None:
            ckt_path = getattr(args, 'eval_model_path', '') or best_model_path(args.output_dir)
        else:
            ckt_path = checkpoint_path(args.output_dir, epoch)
            if not os.path.exists(ckt_path):
                ckt_path = checkpoint_path(args.output_dir, epoch, 0)
            if not os.path.exists(ckt_path):
                ckt_path = getattr(args, 'eval_model_path', '') or best_model_path(args.output_dir)

        if self.model is None:
            self.load(ckt_path)
        self.model.eval()

        if hasattr(self.model, 'score_batch'):
            for i in range(0, len(test_exs), batch_size):
                batch = test_exs[i:i + batch_size]
                scores = self.model.score_batch(
                    [ex.head_id for ex in batch],
                    [ex.relation for ex in batch],
                    [ex.tail_id for ex in batch],
                )
                if not isinstance(scores, torch.Tensor):
                    scores = torch.tensor(scores)
                if scores.dim() == 2 and scores.size(0) == scores.size(1):
                    scores = scores.diag()
                prob = torch.sigmoid(scores).detach().cpu().numpy().reshape(-1)
                y_prob.extend(prob.tolist())
        else:
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
        logger.info(log_thresh)
        logger.info(log_cls)
        return metrics_cls