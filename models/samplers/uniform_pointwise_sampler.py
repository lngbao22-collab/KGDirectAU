"""Pointwise negative sampling for DaBR training."""

import torch

from data.dict_hub import get_entity_dict, get_relation_id_map


def get_pointwise_negatives(batch: dict, num_neg: int, num_entities: int) -> dict:
    """Create pointwise negative samples by uniformly corrupting head or tail.

    Returns concatenated positives followed by negatives and corresponding labels (+1, -1).
    """

    entity_dict = get_entity_dict()
    relation_to_idx = get_relation_id_map() or {}

    heads = _get_batch_field(batch, 'head_id', 'head_ids', entity_key='head_id')
    rels = _get_batch_field(batch, 'relation', 'relations', relation_key='relation')
    tails = _get_batch_field(batch, 'tail_id', 'tail_ids', entity_key='tail_id')

    device = heads.device if isinstance(heads, torch.Tensor) else torch.device('cpu')
    heads = _to_index_tensor(heads, entity_dict.entity_to_idx, device)
    rels = _to_index_tensor(rels, lambda relation: _relation_to_idx(relation, relation_to_idx), device)
    tails = _to_index_tensor(tails, entity_dict.entity_to_idx, device)

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


def _get_batch_field(batch: dict, *candidate_keys, entity_key: str | None = None, relation_key: str | None = None):
    """Resolve a batch field from tensors or from the collated Example objects."""

    for key in candidate_keys:
        if key in batch:
            return batch[key]

    batch_data = batch.get('batch_data')
    if batch_data:
        if entity_key == 'head_id':
            return [ex.head_id for ex in batch_data]
        if entity_key == 'tail_id':
            return [ex.tail_id for ex in batch_data]
        if relation_key == 'relation':
            return [ex.relation for ex in batch_data]

    raise KeyError(candidate_keys[0])


def _to_index_tensor(values, lookup, device: torch.device) -> torch.Tensor:
    """Convert entity or relation IDs into index tensors."""

    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.long)
    return torch.tensor([lookup(value) for value in values], dtype=torch.long, device=device)


def _relation_to_idx(relation: str, relation_to_idx: dict[str, int]) -> int:
    """Resolve relation variants used by inverse triplet generation."""

    if relation in relation_to_idx:
        return relation_to_idx[relation]
    if relation.startswith('inverse '):
        base_relation = relation[len('inverse '):]
        if base_relation in relation_to_idx:
            return relation_to_idx[base_relation]
    if relation.startswith('inverse_'):
        base_relation = relation[len('inverse_'):]
        candidate = '_' + base_relation if not base_relation.startswith('_') else base_relation
        if candidate in relation_to_idx:
            return relation_to_idx[candidate]
    normalized = ' '.join(relation.split())
    if normalized in relation_to_idx:
        return relation_to_idx[normalized]
    raise KeyError(relation)
