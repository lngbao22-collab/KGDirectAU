"""RotatE encoder adapted from the original KGEModel implementation."""

from __future__ import annotations

import json
import os
from typing import Sequence

import torch
import torch.nn as nn

from base.model import BaseModel
from data.dataset import Example, load_data
from data.dict_hub import get_entity_dict


def build_model(args) -> nn.Module:
    """Factory method to build a RotatEEncoder instance based on provided arguments."""

    entity_dict = get_entity_dict()
    relation_to_idx = _load_relation_to_idx(args)
    model = RotatEEncoder(len(entity_dict), len(relation_to_idx), args)
    model.rel_to_idx = relation_to_idx
    return model


class RotatEEncoder(BaseModel):
    """RotatE encoder with complex-valued entity embeddings and phase relations."""

    def __init__(self, n_ent: int, n_rel: int, args):
        super().__init__()
        self.args = args
        self.nentity = n_ent
        self.nrelation = n_rel
        self.hidden_dim = int(getattr(args, "dim", 500))
        self.epsilon = 2.0
        margin = float(getattr(args, "margin", 6.0))

        self.margin = nn.Parameter(torch.tensor([margin]), requires_grad=False)
        self.embedding_range = nn.Parameter(
            torch.tensor([(self.margin.item() + self.epsilon) / self.hidden_dim]),
            requires_grad=False,
        )

        # RotatE uses doubled entity embeddings and single-phase relation embeddings.
        self.entity_dim = self.hidden_dim * 2
        self.relation_dim = self.hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(n_ent, self.entity_dim))
        nn.init.uniform_(self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        self.relation_embedding = nn.Parameter(torch.zeros(n_rel, self.relation_dim))
        nn.init.uniform_(self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        self.entity_dict = get_entity_dict()
        self.rel_to_idx = _load_relation_to_idx(args)

    def _rotate_score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor, mode: str) -> torch.Tensor:
        """Compute the RotatE score for the provided head, relation, and tail embeddings."""

        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = relation / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == "head-batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        score = self.margin.item() - score.sum(dim=2)
        return score

    def _score(self, positive_sample: torch.Tensor, negative_sample: torch.Tensor | None = None, mode: str = "single") -> torch.Tensor:
        """Compute scores for positive and optional negative samples based on the specified mode."""

        if mode == "single":
            head = torch.index_select(self.entity_embedding, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=positive_sample[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=positive_sample[:, 2]).unsqueeze(1)
        elif mode == "head-batch":
            if negative_sample is None:
                raise ValueError("negative_sample is required for head-batch")
            batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
            head = torch.index_select(self.entity_embedding, dim=0, index=negative_sample.reshape(-1)).reshape(batch_size, negative_sample_size, -1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=positive_sample[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=positive_sample[:, 2]).unsqueeze(1)
        elif mode == "tail-batch":
            if negative_sample is None:
                raise ValueError("negative_sample is required for tail-batch")
            batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
            head = torch.index_select(self.entity_embedding, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=positive_sample[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=negative_sample.reshape(-1)).reshape(batch_size, negative_sample_size, -1)
        else:
            raise ValueError(f"mode {mode} not supported")

        return self._rotate_score(head, relation, tail, mode)

    def forward(self, positive_sample: torch.Tensor, negative_sample: torch.Tensor | None = None, mode: str = "single") -> dict:
        """Return positive and optional negative scores for adversarial training."""

        pos_scores = self._score(positive_sample, mode="single")
        neg_scores = None
        if negative_sample is not None:
            neg_scores = self._score(positive_sample, negative_sample=negative_sample, mode=mode)

        return {
            "positive_scores": pos_scores,
            "negative_scores": neg_scores,
        }

    def get_queries_targets(self, src: torch.Tensor, rel: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return AU-compatible query, target, and head representations.

        KGAU expects encoders to expose query/target tensors for alignment-uniformity
        training. For RotatE, the query is the rotated head embedding, the target is
        the tail entity embedding, and the head is the raw head embedding.
        """

        head = torch.index_select(self.entity_embedding, dim=0, index=src).unsqueeze(1)
        relation = torch.index_select(self.relation_embedding, dim=0, index=rel).unsqueeze(1)
        tail = torch.index_select(self.entity_embedding, dim=0, index=dst).unsqueeze(1)

        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        phase_relation = relation / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        re_query = re_head * re_relation - im_head * im_relation
        im_query = re_head * im_relation + im_head * re_relation
        query = torch.cat([re_query.squeeze(1), im_query.squeeze(1)], dim=-1)

        return query, tail.squeeze(1), head.squeeze(1)

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        """Compatibility adapter used by generic trainer paths."""

        pos_scores = output_dict["positive_scores"]
        labels = torch.ones(pos_scores.size(0), dtype=torch.long, device=pos_scores.device)
        return {
            "logits": pos_scores,
            "labels": labels,
        }

    def entity_embeddings(self, device: torch.device | None = None) -> torch.Tensor:
        """Return raw entity embeddings for retrieval."""

        entity_vectors = self.entity_embedding
        if device is not None:
            entity_vectors = entity_vectors.to(device)
        return entity_vectors

    def hr_embeddings(self, examples: Sequence[Example], device: torch.device | None = None) -> torch.Tensor:
        """Build query vectors as rotated heads."""

        if device is None:
            device = self.entity_embedding.device

        head_indices = _as_index_tensor([example.head_id for example in examples], self.entity_dict.entity_to_idx, device)
        relation_indices = _as_index_tensor([example.relation for example in examples], self._relation_to_idx, device)

        head = torch.index_select(self.entity_embedding, dim=0, index=head_indices).unsqueeze(1)
        relation = torch.index_select(self.relation_embedding, dim=0, index=relation_indices).unsqueeze(1)

        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        phase_relation = relation / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        re_query = re_head * re_relation - im_head * im_relation
        im_query = re_head * im_relation + im_head * re_relation
        query = torch.cat([re_query.squeeze(1), im_query.squeeze(1)], dim=-1)
        return query

    def predict_by_examples(self, examples: Sequence[Example], batch_size: int | None = None, num_workers: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Return query and target embeddings for link prediction evaluation."""

        device = self.entity_embedding.device
        query = self.hr_embeddings(examples, device=device)
        tail_indices = _as_index_tensor([example.tail_id for example in examples], self.entity_dict.entity_to_idx, device)
        tails = self.entity_embeddings(device=device)[tail_indices]
        return query, tails

    def predict_by_entities(self, entity_exs, batch_size: int | None = None, num_workers: int = 2) -> torch.Tensor:
        """Return entity embeddings for a list of entity examples."""

        device = self.entity_embedding.device
        entity_ids = [getattr(entity_ex, "entity_id", getattr(entity_ex, "tail_id", "")) for entity_ex in entity_exs]
        entity_indices = _as_index_tensor(entity_ids, self.entity_dict.entity_to_idx, device)
        return self.entity_embeddings(device=device)[entity_indices]

    def score_batch(self, head_ids, relations, tail_entity_ids) -> torch.Tensor:
        """Score a batch using RotatE tail-batch distance scores."""

        device = self.entity_embedding.device

        head_indices = _as_index_tensor(head_ids, self.entity_dict.entity_to_idx, device)
        relation_indices = _as_index_tensor(relations, self._relation_to_idx, device)
        candidate_indices = _as_index_tensor(tail_entity_ids, self.entity_dict.entity_to_idx, device)

        positive_sample = torch.stack([head_indices, relation_indices, torch.zeros_like(head_indices)], dim=-1)
        negative_sample = candidate_indices.unsqueeze(0).expand(len(head_ids), len(candidate_indices))
        return self._score(positive_sample, negative_sample, mode="tail-batch")

    def _relation_to_idx(self, relation: str) -> int:
        """Resolve relation variants used by preprocessing and inverse triplet generation."""

        if relation in self.rel_to_idx:
            return self.rel_to_idx[relation]
        if relation.startswith('inverse '):
            base_relation = relation[len('inverse '):]
            if base_relation in self.rel_to_idx:
                return self.rel_to_idx[base_relation]
        if relation.startswith('inverse_'):
            base_relation = relation[len('inverse_'):]
            candidate = '_' + base_relation if not base_relation.startswith('_') else base_relation
            if candidate in self.rel_to_idx:
                return self.rel_to_idx[candidate]
        normalized = ' '.join(relation.split())
        if normalized in self.rel_to_idx:
            return self.rel_to_idx[normalized]
        raise KeyError(relation)


def _relation_path_candidates(args) -> list[str]:
    """Return a list of candidate paths for loading the relation-to-index mapping."""

    paths = []
    for source_path in [getattr(args, "train_path", ""), getattr(args, "valid_path", ""), getattr(args, "test_path", "")]:
        if not source_path:
            continue
        paths.append(os.path.join(os.path.dirname(source_path), "relation2id.json"))
        paths.append(os.path.join(os.path.dirname(source_path), "relations.json"))
        paths.append(os.path.join(os.path.dirname(source_path), "relation2idx.json"))
    paths.append(os.path.join("data", getattr(args, "dataset", ""), "relation2id.json"))
    paths.append(os.path.join("data", getattr(args, "dataset", ""), "preprocessed", "relation2id.json"))
    return paths


def _load_relation_to_idx(args) -> dict[str, int]:
    """Load the relation-to-index mapping from candidate paths or construct it from training data."""

    for path in _relation_path_candidates(args):
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            mapping = json.load(handle)
        if isinstance(mapping, dict):
            return {str(key): int(value) for key, value in mapping.items()}

    relations = []
    seen = set()
    for example in load_data(getattr(args, "train_path", ""), add_forward_triplet=False, add_backward_triplet=False):
        if example.relation not in seen:
            seen.add(example.relation)
            relations.append(example.relation)
    return {relation: idx for idx, relation in enumerate(relations)}


def _as_index_tensor(values, lookup, device: torch.device) -> torch.Tensor:
    """Convert a list of values into a tensor of corresponding indices using the provided lookup."""

    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.long)
    return torch.tensor([lookup(value) for value in values], dtype=torch.long, device=device)
