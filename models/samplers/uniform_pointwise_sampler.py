"""Pointwise negative sampling for DaBR training (Bernoulli corruption)."""

from collections import defaultdict

import torch

from configs.config import args
from data.dataset import load_data
from data.dict_hub import get_entity_dict, get_relation_id_map


_BERN_PROB_CACHE: torch.Tensor | None = None


def _get_bern_prob(num_relations: int) -> torch.Tensor:
    """Relation-specific Bernoulli head-corruption probabilities (OpenKE ``bern=1``).

    For each relation ``r`` we compute ``tph`` (average tails per head) and ``htp``
    (average heads per tail) over the training triples, then set the probability of
    corrupting the **head** to ``tph / (tph + htp)``. This matches the official DaBR
    training (``con.set_bern(1)``) and the existing ``BernoulliListwiseSampler``.
    The table is computed once and cached for the lifetime of the process.
    """

    global _BERN_PROB_CACHE
    if _BERN_PROB_CACHE is not None and _BERN_PROB_CACHE.size(0) >= max(num_relations, 1):
        return _BERN_PROB_CACHE

    entity_dict = get_entity_dict()
    relation_to_idx = get_relation_id_map() or {}

    edges: dict[int, dict[int, set]] = defaultdict(lambda: defaultdict(set))
    rev_edges: dict[int, dict[int, set]] = defaultdict(lambda: defaultdict(set))

    train_path = getattr(args, 'train_path', '')
    examples = (
        load_data(train_path, add_forward_triplet=True, add_backward_triplet=False)
        if train_path
        else []
    )
    for example in examples:
        try:
            head = entity_dict.entity_to_idx(example.head_id)
            tail = entity_dict.entity_to_idx(example.tail_id)
            relation = _relation_to_idx(example.relation, relation_to_idx)
        except KeyError:
            continue
        edges[relation][head].add(tail)
        rev_edges[relation][tail].add(head)

    size = max(num_relations, len(relation_to_idx), 1)
    bern_prob = torch.full((size,), 0.5, dtype=torch.float)
    for relation in edges:
        tph = sum(len(tails) for tails in edges[relation].values()) / max(len(edges[relation]), 1)
        htp = sum(len(heads) for heads in rev_edges[relation].values()) / max(len(rev_edges[relation]), 1)
        if relation < bern_prob.size(0):
            bern_prob[relation] = tph / (tph + htp) if (tph + htp) > 0 else 0.5

    _BERN_PROB_CACHE = bern_prob
    return bern_prob


def get_pointwise_negatives(batch: dict, num_neg: int, num_entities: int) -> dict:
    """Create pointwise negative samples via relation-specific Bernoulli corruption.

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

    bern_prob = _get_bern_prob(len(relation_to_idx)).to(device)
    head_corrupt_prob = bern_prob[rels.clamp(max=bern_prob.size(0) - 1)]

    neg_heads = []
    neg_rels = []
    neg_tails = []

    # For each positive triple, generate `num_neg` Bernoulli corruptions:
    # corrupt the head with probability tph/(tph+htp), otherwise corrupt the tail.
    for _ in range(num_neg):
        corrupt_head = torch.rand(n, device=device) < head_corrupt_prob
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


class PointwiseNegSampler:
    """Wrap pointwise Bernoulli corruption for the negative-sampling strategy."""

    def __init__(self, args, num_entities: int):
        self.args = args
        self.num_entities = int(num_entities)

    def sample(self, batch, mode: str | None = None):
        batch_data = batch.get('batch_data')
        pos_size = len(batch_data) if batch_data else batch['head_id'].size(0)
        sampled = get_pointwise_negatives(
            batch,
            getattr(self.args, 'n_sample', 1),
            self.num_entities,
        )
        pos_triples = torch.stack([
            sampled['head_id'][:pos_size],
            sampled['relation'][:pos_size],
            sampled['tail_id'][:pos_size],
        ], dim=-1)
        neg_triples = torch.stack([
            sampled['head_id'][pos_size:],
            sampled['relation'][pos_size:],
            sampled['tail_id'][pos_size:],
        ], dim=-1)
        return pos_triples, neg_triples, sampled['labels'], mode


def build_sampler(args, train_triples, model):
    """Construct a pointwise Bernoulli sampler for DaBR-style training."""

    del train_triples
    num_entities = model.ent_embeddings.num_embeddings
    return PointwiseNegSampler(args, num_entities)


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
