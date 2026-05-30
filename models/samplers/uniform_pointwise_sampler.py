"""Pointwise negative sampling for DaBR training."""

import torch


def get_pointwise_negatives(batch: dict, num_neg: int, num_entities: int) -> dict:
    """Create pointwise negative samples by uniformly corrupting head or tail.

    Returns concatenated positives followed by negatives and corresponding labels (+1, -1).
    """

    heads = batch['head_id']
    rels = batch['relation']
    tails = batch['tail_id']
    device = heads.device if isinstance(heads, torch.Tensor) else torch.device('cpu')

    n = heads.size(0)
    pos_labels = torch.ones(n, dtype=torch.float, device=device)

    neg_heads = []
    neg_rels = []
    neg_tails = []

    # For each positive triple, generate `num_neg` corruptions
    for _ in range(num_neg):
        corrupt_head = torch.rand(n, device=device) < 0.5
        # sample random entity ids
        random_entities = torch.randint(0, num_entities, (n,), device=device)
        nh = torch.where(corrupt_head, random_entities, heads)
        nt = torch.where(~corrupt_head, random_entities, tails)
        neg_heads.append(nh)
        neg_rels.append(rels)
        neg_tails.append(nt)

    neg_heads = torch.stack(neg_heads, dim=1).reshape(-1)
    neg_rels = torch.stack(neg_rels, dim=1).reshape(-1)
    neg_tails = torch.stack(neg_tails, dim=1).reshape(-1)

    neg_labels = -torch.ones(neg_heads.size(0), dtype=torch.float, device=device)

    out_heads = torch.cat([heads, neg_heads], dim=0)
    out_rels = torch.cat([rels, neg_rels], dim=0)
    out_tails = torch.cat([tails, neg_tails], dim=0)
    out_labels = torch.cat([pos_labels, neg_labels], dim=0)

    return {
        'head_id': out_heads,
        'relation': out_rels,
        'tail_id': out_tails,
        'labels': out_labels,
    }
