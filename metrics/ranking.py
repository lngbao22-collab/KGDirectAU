"""Ranking metrics for evaluating KG models."""

from typing import Sequence, List, Tuple
import torch

from configs.config import args
from data.dataset import EntityDict, Example
from data.dict_hub import get_link_graph


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """Compute top-k classification accuracy (percentage) for each k in `topk`.

    Returns a list of tensors containing the percentage accuracy for each requested k.
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results: List[torch.Tensor] = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list:
    """Backward-compatible alias for top-k accuracy."""

    return topk_accuracy(output, target, topk=topk)


def ranking_metrics_from_ranks(ranks: Sequence[int]) -> dict:
    """Compute link-prediction metrics from 1-based ranks.

    Returns a dictionary containing 'mr'/'mean_rank', 'mrr', and hit@k metrics.
    """

    ranks_list = list(ranks)
    if not ranks_list:
        raise ValueError('ranks must not be empty')

    total = float(len(ranks_list))
    mr = sum(ranks_list) / total
    mrr = sum(1.0 / rank for rank in ranks_list) / total
    hit_at_1 = sum(1 for rank in ranks_list if rank <= 1) / total
    hit_at_3 = sum(1 for rank in ranks_list if rank <= 3) / total
    hit_at_10 = sum(1 for rank in ranks_list if rank <= 10) / total
    return {
        'mr': round(mr, 4),
        'mean_rank': round(mr, 4),
        'mrr': round(mrr, 4),
        'hit@1': round(hit_at_1, 4),
        'hit@3': round(hit_at_3, 4),
        'hit@10': round(hit_at_10, 4),
    }


def ranking_metrics_from_scores(scores: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 3, 10)) -> Tuple[List[List[float]], List[List[int]], dict, List[int]]:
    """Compute link-prediction metrics from a score matrix and target indices.

    Args:
        scores: Tensor of shape (batch_size, num_entities), higher is better.
        targets: Tensor of shape (batch_size,) or (batch_size, 1) with target entity indices.
        topk: Retained for symmetry with accuracy-style helpers.

    Returns:
        A tuple of (topk_scores, topk_indices, metrics, ranks).
    """

    with torch.no_grad():
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.view(-1)
        elif targets.dim() != 1:
            raise ValueError('targets must have shape (batch_size,) or (batch_size, 1)')

        maxk = max(topk)
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        target_rank = torch.nonzero(sorted_indices.eq(targets.unsqueeze(-1)).long(), as_tuple=False)
        if target_rank.size(0) != scores.size(0):
            raise RuntimeError('Unable to locate one target rank per example')

        ranks: List[int] = []
        for idx in range(target_rank.size(0)):
            row = target_rank[idx].tolist()
            if row[0] != idx:
                raise RuntimeError('Target rank rows are misaligned')
            ranks.append(row[1] + 1)

        metrics = ranking_metrics_from_ranks(ranks)
        topk_scores = sorted_scores[:, :maxk].tolist()
        topk_indices = sorted_indices[:, :maxk].tolist()
        return topk_scores, topk_indices, metrics, ranks


def link_prediction_metrics(ranks: Sequence[int]) -> dict:
    """Alias for ranking_metrics_from_ranks for link prediction tasks."""

    return ranking_metrics_from_ranks(ranks)


def rerank_by_graph(batch_score: torch.Tensor, examples: Sequence[Example], entity_dict: EntityDict) -> None:
    """Re-rank entity scores using the local link graph.

    Modifies `batch_score` in-place by adding a small delta to entities
    that are within `args.rerank_n_hop` hops in the training graph.
    """
    
    if args.dataset == 'wiki5m_ind':
        assert args.neighbor_weight < 1e-6, 'Inductive setting can not use re-rank strategy'

    if args.neighbor_weight < 1e-6:
        return

    for idx in range(batch_score.size(0)):
        cur_ex = examples[idx]
        n_hop_indices = get_link_graph().get_n_hop_entity_indices(
            cur_ex.head_id,
            entity_dict=entity_dict,
            n_hop=args.rerank_n_hop,
        )
        delta = torch.tensor([args.neighbor_weight for _ in n_hop_indices]).to(batch_score.device)
        n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)

        batch_score[idx].index_add_(0, n_hop_indices, delta)

        # The test set of FB15k237 removes triples that are connected in train set,
        # so any two entities that are connected in train set will not appear in test,
        # however, this is not a trick that could generalize.
        # by default, we do not use this piece of code.

        # if args.dataset == 'FB15k237':
        #     n_hop_indices = get_link_graph().get_n_hop_entity_indices(cur_ex.head_id,
        #                                                               entity_dict=entity_dict,
        #                                                               n_hop=1)
        #     n_hop_indices.remove(entity_dict.entity_to_idx(cur_ex.head_id))
        #     delta = torch.tensor([-0.5 for _ in n_hop_indices]).to(batch_score.device)
        #     n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)
        #
        #     batch_score[idx].index_add_(0, n_hop_indices, delta)
