import os
import json
import tqdm
import torch

from time import time
from typing import List, Tuple
from dataclasses import dataclass, asdict

from config import args
from doc import load_data, Example, collate
from predict import BertPredictor
from dict_hub import get_entity_dict, get_all_triplet_dict
from triplet import EntityDict
from rerank import rerank_by_graph
from metric_classification import classification_metrics, find_global_threshold
from utils import get_model_obj, call_model_forward
from logger_config import logger


def _setup_entity_dict() -> EntityDict:
    if args.task == 'wiki5m_ind':
        return EntityDict(entity_dict_dir=os.path.dirname(args.valid_path),
                          inductive_test_path=args.valid_path)
    return get_entity_dict()


def _resolve_label_path() -> str:
    candidates = []
    if args.valid_path.endswith('.json'):
        candidates.append(args.valid_path[:-5])
    candidates.append(args.valid_path)
    if args.valid_path.endswith('.txt'):
        candidates.append(args.valid_path + '.json')

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ''


def _load_classification_examples(label_path: str) -> List[Example]:
    examples = []
    if label_path.endswith('.json'):
        data = json.load(open(label_path, 'r', encoding='utf-8'))
        for obj in data:
            if 'label' in obj:
                examples.append(Example(**obj))
        return examples

    if label_path.endswith('.txt'):
        with open(label_path, 'r', encoding='utf-8') as reader:
            for line in reader:
                fs = line.strip().split('\t')
                if len(fs) != 4:
                    continue
                head_id, relation, tail_id, label = fs
                examples.append(Example(head_id=head_id, relation=relation, tail_id=tail_id, label=label))
        return examples

    raise ValueError('Unsupported label format: {}'.format(label_path))


entity_dict = _setup_entity_dict()
all_triplet_dict = get_all_triplet_dict()


@dataclass
class PredInfo:
    head: str
    relation: str
    tail: str
    pred_tail: str
    pred_score: float
    topk_score_info: str
    rank: int
    correct: bool


@torch.no_grad()
def compute_metrics(hr_tensor: torch.tensor,
                    entities_tensor: torch.tensor,
                    target: List[int],
                    examples: List[Example],
                    k=3, batch_size=256) -> Tuple:
    assert hr_tensor.size(1) == entities_tensor.size(1)
    total = hr_tensor.size(0)
    entity_cnt = len(entity_dict)
    assert entity_cnt == entities_tensor.size(0)
    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)
    topk_scores, topk_indices = [], []
    ranks = []

    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0

    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size
        # batch_size * entity_cnt
        batch_score = torch.mm(hr_tensor[start:end, :], entities_tensor.t())
        assert entity_cnt == batch_score.size(1)
        batch_target = target[start:end]

        # re-ranking based on topological structure
        # rerank_by_graph(batch_score, examples[start:end], entity_dict=entity_dict)

        # filter known triplets
        for idx in range(batch_score.size(0)):
            mask_indices = []
            cur_ex = examples[start + idx]
            gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
            if len(gold_neighbor_ids) > 10000:
                logger.debug('{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
            for e_id in gold_neighbor_ids:
                if e_id == cur_ex.tail_id:
                    continue
                mask_indices.append(entity_dict.entity_to_idx(e_id))
            mask_indices = torch.LongTensor(mask_indices).to(batch_score.device)
            batch_score[idx].index_fill_(0, mask_indices, -1)

        batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
        target_rank = torch.nonzero(batch_sorted_indices.eq(batch_target).long(), as_tuple=False)
        assert target_rank.size(0) == batch_score.size(0)
        for idx in range(batch_score.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx
            cur_rank = idx_rank[1]

            # 0-based -> 1-based
            cur_rank += 1
            mean_rank += cur_rank
            mrr += 1.0 / cur_rank
            hit1 += 1 if cur_rank <= 1 else 0
            hit3 += 1 if cur_rank <= 3 else 0
            hit10 += 1 if cur_rank <= 10 else 0
            ranks.append(cur_rank)

        topk_scores.extend(batch_sorted_score[:, :k].tolist())
        topk_indices.extend(batch_sorted_indices[:, :k].tolist())

    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    metrics = {k: round(v / total, 4) for k, v in metrics.items()}
    assert len(topk_scores) == total
    return topk_scores, topk_indices, metrics, ranks


def predict_by_split():
    assert os.path.exists(args.valid_path)
    assert os.path.exists(args.train_path)

    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path)
    entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)

    # For link prediction in test mode, use unlabeled test.txt instead of labeled test_w_label.txt
    linkpred_eval_path = None
    if args.is_test:
        data_dir = os.path.dirname(args.valid_path)
        test_unlabeled_path = os.path.join(data_dir, 'test.txt')
        if os.path.exists(test_unlabeled_path):
            linkpred_eval_path = test_unlabeled_path
            logger.info(f'Using unlabeled test file for link prediction: {test_unlabeled_path}')

    forward_metrics = eval_single_direction(predictor,
                                            entity_tensor=entity_tensor,
                                            eval_forward=True,
                                            eval_path=linkpred_eval_path)
    backward_metrics = eval_single_direction(predictor,
                                             entity_tensor=entity_tensor,
                                             eval_forward=False,
                                             eval_path=linkpred_eval_path)
    metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
    logger.info('Averaged metrics: {}'.format(metrics))

    output_dir = args.output_dir or args.model_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(args.eval_model_path) if args.eval_model_path else 'model'
    split = os.path.basename(args.valid_path)
    with open(os.path.join(output_dir, f"metrics_{split}_{basename}.json"), 'w', encoding='utf-8') as writer:
        writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
        writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
        writer.write('average metrics: {}\n'.format(json.dumps(metrics)))

    label_path = _resolve_label_path()
    if label_path:
        classification_metrics_dict = evaluate_triple_classification(predictor, label_path, output_dir)
        logger.info('Triple classification metrics: {}'.format(json.dumps(classification_metrics_dict)))


def eval_single_direction(predictor: BertPredictor,
                          entity_tensor: torch.tensor,
                          eval_forward=True,
                          batch_size=256,
                          eval_path=None) -> dict:
    start_time = time()
    # Use provided eval_path or fall back to args.valid_path
    data_path = eval_path if eval_path else args.valid_path
    print(f"Evaluating {'forward' if eval_forward else 'backward'} direction using data from: {data_path}")
    examples = load_data(data_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)

    hr_tensor, _ = predictor.predict_by_examples(examples)
    hr_tensor = hr_tensor.to(entity_tensor.device)
    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    logger.info('predict tensor done, compute metrics...')

    #chunk_size = getattr(args, 'chunk_size', 8192)
    topk_scores, topk_indices, metrics, ranks = compute_metrics(hr_tensor=hr_tensor, entities_tensor=entity_tensor,
                                                                target=target, examples=examples,
                                                                batch_size=batch_size)
    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))

    pred_infos = []
    for idx, ex in enumerate(examples):
        cur_topk_scores = topk_scores[idx]
        cur_topk_indices = topk_indices[idx]
        pred_idx = cur_topk_indices[0]
        cur_score_info = {entity_dict.get_entity_by_idx(topk_idx).entity: round(topk_score, 3)
                          for topk_score, topk_idx in zip(cur_topk_scores, cur_topk_indices)}

        pred_info = PredInfo(head=ex.head, relation=ex.relation,
                             tail=ex.tail, pred_tail=entity_dict.get_entity_by_idx(pred_idx).entity,
                             pred_score=round(cur_topk_scores[0], 4),
                             topk_score_info=json.dumps(cur_score_info),
                             rank=ranks[idx],
                             correct=pred_idx == target[idx])
        pred_infos.append(pred_info)

    output_dir = args.output_dir or args.model_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(args.eval_model_path) if args.eval_model_path else 'model'
    split = os.path.basename(args.valid_path)
    out_path = os.path.join(output_dir, f"eval_{split}_{eval_dir}_{basename}.json")
    with open(out_path, 'w', encoding='utf-8') as writer:
        writer.write(json.dumps([asdict(info) for info in pred_infos], ensure_ascii=False, indent=4))

    logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
    return metrics

@torch.no_grad()
def evaluate_triple_classification(predictor: BertPredictor, label_path: str, output_dir: str,
                                   batch_size=128) -> dict:
    examples = _load_classification_examples(label_path)
    if not examples or all(ex.label is None for ex in examples):
        logger.info('No labeled examples found in {}; skip triple classification evaluation.'.format(label_path))
        return {}

    y_true = [ex.label for ex in examples]
    y_prob = []
    predictor.model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        batch_vec = [ex.vectorize() for ex in batch]
        batch_dict = collate(batch_vec)
        if torch.cuda.is_available():
            for key in batch_dict:
                if isinstance(batch_dict[key], torch.Tensor):
                    batch_dict[key] = batch_dict[key].cuda()
            predictor.model.cuda()
        output_dict = predictor.model(**batch_dict)
        logits = predictor.model.compute_logits(output_dict=output_dict, batch_dict=batch_dict)['logits']
        prob = torch.sigmoid(logits.diag()).detach().cpu().numpy().reshape(-1)
        y_prob.extend(prob.tolist())

    threshold = find_global_threshold(y_true, y_prob)
    y_pred = (torch.tensor(y_prob) > threshold).long().tolist()
    metrics_cls = classification_metrics(y_true, y_pred, y_prob)

    log_thresh = '[TEST] Best threshold on test: {:.6f}'.format(threshold)
    log_cls = '[TEST] Triple Classification: {}'.format(json.dumps(metrics_cls))
    print(log_thresh)
    print(log_cls)
    logger.info(log_thresh)
    logger.info(log_cls)

    with open(os.path.join(output_dir, 'test_metrics.log'), 'a', encoding='utf-8') as writer:
        writer.write(log_thresh + '\n')
        writer.write(log_cls + '\n')

    return metrics_cls


if __name__ == '__main__':
    predict_by_split()
