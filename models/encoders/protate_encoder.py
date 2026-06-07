"""pRotatE encoder adapted from the original KGEModel implementation."""

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
    """Factory method to build a pRotatEEncoder instance based on provided arguments."""

    entity_dict = get_entity_dict()
    relation_to_idx = _load_relation_to_idx(args)
    model = pRotatEEncoder(len(entity_dict), len(relation_to_idx), args)
    model.rel_to_idx = relation_to_idx
    return model


class pRotatEEncoder(BaseModel):
    """pRotatE encoder with 1D phase entity embeddings and phase relations."""

    bidirectional_score_batch = True
    # Opt-in for KGAU: only this encoder uses sin-phase AU (see ``KGAULoss.alignment_mode``).
    kga_u_alignment_mode = 'sin_phase'

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

        # pRotatE uses 1D phase embeddings for both entities and relations
        self.entity_dim = self.hidden_dim
        self.relation_dim = self.hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(n_ent, self.entity_dim))
        nn.init.uniform_(self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        self.relation_embedding = nn.Parameter(torch.zeros(n_rel, self.relation_dim))
        nn.init.uniform_(self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        # pRotatE specific modulus parameter
        self.modulus = nn.Parameter(torch.tensor([[0.5 * self.embedding_range.item()]]))

        self.entity_dict = get_entity_dict()
        self.rel_to_idx = _load_relation_to_idx(args)

    @staticmethod
    def _pi() -> float:
        return 3.14159265358979323846

    def _phase_scale(self) -> float:
        """Scaling factor that maps raw embeddings into pRotatE phase coordinates."""

        return self.embedding_range.item() / self._pi()

    def _to_phase(self, vectors: torch.Tensor) -> torch.Tensor:
        """Convert raw entity/relation embeddings to the phase space used by pRotatE scoring."""

        return vectors / self._phase_scale()

    def _rotate_score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor, mode: str) -> torch.Tensor:
        """Compute the pRotatE score for the provided head, relation, and tail embeddings."""

        # Make phases of entities and relations uniformly distributed in [-pi, pi]
        phase_head = self._to_phase(head)
        phase_relation = self._to_phase(relation)
        phase_tail = self._to_phase(tail)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.margin.item() - score.sum(dim=2) * self.modulus
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
        """Return phase-space AU query, target, and head for sin-phase alignment.

        With ``alignment_mode=sin_phase`` in KGAULoss, alignment minimizes
        mean_i |sin(phase(h)_i + phase(r)_i - phase(t)_i)| — the same per-dimension
        quantity summed inside ``_rotate_score`` at link-prediction time.
        """

        head = torch.index_select(self.entity_embedding, dim=0, index=src)
        relation = torch.index_select(self.relation_embedding, dim=0, index=rel)
        tail = torch.index_select(self.entity_embedding, dim=0, index=dst)

        phase_head = self._to_phase(head)
        phase_relation = self._to_phase(relation)
        phase_tail = self._to_phase(tail)
        query = phase_head + phase_relation
        return query, phase_tail, phase_head

    def sin_phase_score_penalty(
        self,
        phase_query: torch.Tensor,
        phase_target: torch.Tensor,
    ) -> torch.Tensor:
        """Per-example pRotatE score penalty: sum_i |sin(phase_query_i - phase_target_i)|."""

        return torch.abs(torch.sin(phase_query - phase_target)).sum(dim=-1)

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

    def au_entity_embeddings(self, device: torch.device | None = None) -> torch.Tensor:
        """Entity phases for KGAU uniformity (same coordinates as sin-phase alignment)."""

        entity_vectors = self._to_phase(self.entity_embedding)
        if device is not None:
            entity_vectors = entity_vectors.to(device)
        return entity_vectors

    def hr_embeddings(self, examples: Sequence[Example], device: torch.device | None = None) -> torch.Tensor:
        """Build query vectors as phase(head) + phase(relation)."""

        if device is None:
            device = self.entity_embedding.device

        head_indices = _as_index_tensor([example.head_id for example in examples], self.entity_dict.entity_to_idx, device)
        relation_indices = _as_index_tensor([example.relation for example in examples], self._relation_to_idx, device)

        head = torch.index_select(self.entity_embedding, dim=0, index=head_indices)
        relation = torch.index_select(self.relation_embedding, dim=0, index=relation_indices)
        return self._to_phase(head) + self._to_phase(relation)

    def predict_by_examples(self, examples: Sequence[Example], batch_size: int | None = None, num_workers: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Return query and target embeddings for link prediction evaluation."""

        device = self.entity_embedding.device
        query = self.hr_embeddings(examples, device=device)
        tail_indices = _as_index_tensor([example.tail_id for example in examples], self.entity_dict.entity_to_idx, device)
        tails = self.au_entity_embeddings(device=device)[tail_indices]
        return query, tails

    def predict_by_entities(self, entity_exs, batch_size: int | None = None, num_workers: int = 2) -> torch.Tensor:
        """Return entity embeddings for a list of entity examples."""

        device = self.entity_embedding.device
        entity_ids = [getattr(entity_ex, "entity_id", getattr(entity_ex, "tail_id", "")) for entity_ex in entity_exs]
        entity_indices = _as_index_tensor(entity_ids, self.entity_dict.entity_to_idx, device)
        return self.entity_embeddings(device=device)[entity_indices]

    def _resolve_link_prediction_chunk_sizes(
        self,
        query_chunk_size: int | None,
        candidate_chunk_size: int | None,
    ) -> tuple[int, int]:
        """Resolve query/candidate chunk sizes for link-prediction scoring."""

        if query_chunk_size is None:
            query_chunk_size = _config_int(self.args, "score_query_chunk_size", 256)
        else:
            query_chunk_size = int(query_chunk_size)
        if candidate_chunk_size is None:
            candidate_chunk_size = _config_int(self.args, "eval_candidate_chunk_size", 2048)
        else:
            candidate_chunk_size = int(candidate_chunk_size)
        query_chunk_size = max(query_chunk_size, 1)
        candidate_chunk_size = max(candidate_chunk_size, 1)
        return _cap_link_prediction_chunks_for_memory(
            query_chunk_size,
            candidate_chunk_size,
            self.hidden_dim,
            self.entity_embedding.device,
        )

    @torch.inference_mode()
    def _score_link_prediction_matrix(
        self,
        relation_indices: torch.Tensor,
        candidate_indices: torch.Tensor,
        batch_mode: str,
        anchor_indices: torch.Tensor,
        query_chunk_size: int,
        candidate_chunk_size: int,
    ) -> torch.Tensor:
        """Score many queries against candidate entity indices using chunked tail/head batching."""

        device = self.entity_embedding.device
        num_queries = relation_indices.size(0)
        num_candidates = candidate_indices.size(0)
        if num_queries == 0 or num_candidates == 0:
            return torch.empty(num_queries, num_candidates, device=device)

        scores = torch.empty(num_queries, num_candidates, device=device, dtype=torch.float32)
        for cand_start in range(0, num_candidates, candidate_chunk_size):
            cand_end = min(cand_start + candidate_chunk_size, num_candidates)
            cand_chunk = candidate_indices[cand_start:cand_end]

            for q_start in range(0, num_queries, query_chunk_size):
                q_end = min(q_start + query_chunk_size, num_queries)
                query_slice = slice(q_start, q_end)
                zero_heads = torch.zeros(q_end - q_start, dtype=torch.long, device=device)
                if batch_mode == "tail-batch":
                    positive_sample = torch.stack(
                        [
                            anchor_indices[query_slice],
                            relation_indices[query_slice],
                            zero_heads,
                        ],
                        dim=-1,
                    )
                else:
                    positive_sample = torch.stack(
                        [
                            zero_heads,
                            relation_indices[query_slice],
                            anchor_indices[query_slice],
                        ],
                        dim=-1,
                    )
                negative_sample = cand_chunk.unsqueeze(0).expand(q_end - q_start, cand_chunk.size(0))
                try:
                    block_score = self._score(positive_sample, negative_sample, mode=batch_mode)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    block_score = self._score_link_prediction_block_safe(
                        relation_indices,
                        cand_chunk,
                        batch_mode,
                        anchor_indices,
                        q_start,
                        q_end,
                        max(query_chunk_size // 2, 64),
                        max(candidate_chunk_size // 2, 128),
                    )
                scores[q_start:q_end, cand_start:cand_end] = block_score.float()

        return scores

    def _score_link_prediction_block_safe(
        self,
        relation_indices: torch.Tensor,
        candidate_indices: torch.Tensor,
        batch_mode: str,
        anchor_indices: torch.Tensor,
        q_start: int,
        q_end: int,
        query_chunk_size: int,
        candidate_chunk_size: int,
    ) -> torch.Tensor:
        """Score one query/candidate tile, recursively splitting on CUDA OOM."""

        device = self.entity_embedding.device
        num_queries = q_end - q_start
        num_candidates = candidate_indices.size(0)
        scores = torch.empty(num_queries, num_candidates, device=device, dtype=torch.float32)

        for cand_start in range(0, num_candidates, candidate_chunk_size):
            cand_end = min(cand_start + candidate_chunk_size, num_candidates)
            cand_chunk = candidate_indices[cand_start:cand_end]
            for local_q_start in range(0, num_queries, query_chunk_size):
                local_q_end = min(local_q_start + query_chunk_size, num_queries)
                global_q_start = q_start + local_q_start
                global_q_end = q_start + local_q_end
                query_slice = slice(global_q_start, global_q_end)
                zero_heads = torch.zeros(global_q_end - global_q_start, dtype=torch.long, device=device)
                if batch_mode == "tail-batch":
                    positive_sample = torch.stack(
                        [
                            anchor_indices[query_slice],
                            relation_indices[query_slice],
                            zero_heads,
                        ],
                        dim=-1,
                    )
                else:
                    positive_sample = torch.stack(
                        [
                            zero_heads,
                            relation_indices[query_slice],
                            anchor_indices[query_slice],
                        ],
                        dim=-1,
                    )
                negative_sample = cand_chunk.unsqueeze(0).expand(global_q_end - global_q_start, cand_chunk.size(0))
                try:
                    block_score = self._score(positive_sample, negative_sample, mode=batch_mode)
                except torch.cuda.OutOfMemoryError:
                    if query_chunk_size <= 64 and candidate_chunk_size <= 128:
                        raise
                    torch.cuda.empty_cache()
                    block_score = self._score_link_prediction_block_safe(
                        relation_indices,
                        cand_chunk,
                        batch_mode,
                        anchor_indices,
                        global_q_start,
                        global_q_end,
                        max(query_chunk_size // 2, 64),
                        max(candidate_chunk_size // 2, 128),
                    )
                scores[local_q_start:local_q_end, cand_start:cand_end] = block_score.float()

        return scores

    @torch.inference_mode()
    def score_batch_from_indices(
        self,
        relation_indices: torch.Tensor,
        candidate_indices: torch.Tensor,
        mode: str = "tail-batch",
        query_head_indices: torch.Tensor | None = None,
        query_tail_indices: torch.Tensor | None = None,
        query_chunk_size: int | None = None,
        candidate_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Fast link-prediction scoring using precomputed index tensors."""

        device = self.entity_embedding.device
        batch_mode = str(mode or "tail-batch")
        if batch_mode not in {"head-batch", "tail-batch"}:
            raise ValueError(f"mode {batch_mode} not supported")

        relation_indices = relation_indices.to(device=device, dtype=torch.long)
        candidate_indices = candidate_indices.to(device=device, dtype=torch.long)
        query_q, query_c = self._resolve_link_prediction_chunk_sizes(query_chunk_size, candidate_chunk_size)

        if batch_mode == "tail-batch":
            if query_head_indices is None:
                raise ValueError("query_head_indices is required for tail-batch scoring")
            anchor_indices = query_head_indices.to(device=device, dtype=torch.long)
        else:
            if query_tail_indices is None:
                raise ValueError("query_tail_indices is required for head-batch scoring")
            anchor_indices = query_tail_indices.to(device=device, dtype=torch.long)

        return self._score_link_prediction_matrix(
            relation_indices,
            candidate_indices,
            batch_mode,
            anchor_indices,
            query_q,
            query_c,
        )

    @torch.inference_mode()
    def score_batch(
        self,
        head_ids,
        relations,
        tail_entity_ids,
        mode: str = "tail-batch",
        query_tail_ids=None,
        query_chunk_size: int | None = None,
        candidate_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Score queries against candidate entities.

        tail-batch (default): fix (head, relation), score candidate tails.
        head-batch: fix (relation, tail), score candidate heads via query_tail_ids.
        """

        device = self.entity_embedding.device
        batch_mode = str(mode or "tail-batch")

        relation_indices = _as_index_tensor(relations, self._relation_to_idx, device)
        candidate_indices = _as_index_tensor(tail_entity_ids, self.entity_dict.entity_to_idx, device)

        num_queries = relation_indices.size(0)
        num_candidates = candidate_indices.size(0)
        if num_queries == 0 or num_candidates == 0:
            return torch.empty(num_queries, num_candidates, device=device)

        if head_ids is not None and num_queries == num_candidates == len(relations):
            head_indices = _as_index_tensor(head_ids, self.entity_dict.entity_to_idx, device)
            positive_sample = torch.stack([head_indices, relation_indices, candidate_indices], dim=-1)
            return self._score(positive_sample, mode="single").squeeze(-1)

        if batch_mode not in {"head-batch", "tail-batch"}:
            raise ValueError(f"mode {batch_mode} not supported")

        query_q, query_c = self._resolve_link_prediction_chunk_sizes(query_chunk_size, candidate_chunk_size)

        if batch_mode == "tail-batch":
            anchor_indices = _as_index_tensor(head_ids, self.entity_dict.entity_to_idx, device)
        else:
            if query_tail_ids is None:
                raise ValueError("query_tail_ids is required for head-batch scoring")
            anchor_indices = _as_index_tensor(query_tail_ids, self.entity_dict.entity_to_idx, device)

        return self._score_link_prediction_matrix(
            relation_indices,
            candidate_indices,
            batch_mode,
            anchor_indices,
            query_q,
            query_c,
        )

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


def _config_int(args, key: str, default: int) -> int:
    """Read an integer config value from args, falling back to default when unset."""

    value = getattr(args, key, None)
    if value is None:
        return default
    return int(value)


def _cap_link_prediction_chunks_for_memory(
    query_chunk_size: int,
    candidate_chunk_size: int,
    hidden_dim: int,
    device: torch.device,
    min_query: int = 64,
    min_candidate: int = 128,
) -> tuple[int, int]:
    """Shrink link-prediction chunks so a Q x C scoring block fits in free GPU memory."""

    query_chunk_size = max(int(query_chunk_size), min_query)
    candidate_chunk_size = max(int(candidate_chunk_size), min_candidate)
    if device.type != "cuda":
        return query_chunk_size, candidate_chunk_size

    free_bytes, _ = torch.cuda.mem_get_info(device)
    # Rough budget for intermediate [Q, C, D] tensors created during pRotatE scoring.
    budget_bytes = int(free_bytes * 0.30)
    bytes_per_pair = max(hidden_dim * 6 * 4, 1)
    max_pairs = max(budget_bytes // bytes_per_pair, min_query * min_candidate)

    while query_chunk_size * candidate_chunk_size > max_pairs:
        if candidate_chunk_size > min_candidate and candidate_chunk_size >= query_chunk_size:
            candidate_chunk_size = max(candidate_chunk_size // 2, min_candidate)
        elif query_chunk_size > min_query:
            query_chunk_size = max(query_chunk_size // 2, min_query)
        elif candidate_chunk_size > min_candidate:
            candidate_chunk_size = max(candidate_chunk_size // 2, min_candidate)
        else:
            break

    return query_chunk_size, candidate_chunk_size


def _as_index_tensor(values, lookup, device: torch.device) -> torch.Tensor:
    """Convert a list of values into a tensor of corresponding indices using the provided lookup."""

    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.long)
    return torch.tensor([lookup(value) for value in values], dtype=torch.long, device=device)