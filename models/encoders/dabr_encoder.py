"""DaBR encoder adapted from classic DaBR implementation."""

from contextlib import nullcontext
import json
import os

import torch
import torch.nn as nn

from base.model import BaseModel
from data.dataset import load_data
from data.dict_hub import get_entity_dict


def _eval_use_amp(config) -> bool:
    """Whether link-prediction scoring should run under CUDA autocast."""

    if not torch.cuda.is_available():
        return False
    if bool(getattr(config, 'eval_use_amp', False)):
        return True
    return bool(getattr(config, 'use_amp', False))


def _autocast_context(config):
    """Return a CUDA autocast context when eval AMP is enabled."""

    if _eval_use_amp(config):
        return torch.amp.autocast(device_type='cuda')
    return nullcontext()


def build_model(args) -> nn.Module:
    entity_dict = get_entity_dict()
    relation_to_idx = _load_relation_to_idx(args)
    model = DaBREncoder(args, len(entity_dict), len(relation_to_idx))
    model.rel_to_idx = relation_to_idx
    model.entity_dict = entity_dict
    return model


class DaBREncoder(BaseModel):
    def __init__(self, args, n_ent=None, n_rel=None):
        super().__init__()
        self.config = args
        dim = getattr(args, 'dim', getattr(args, 'hidden_size', 100))
        emb_dim = 4 * int(dim)
        n_ent = getattr(args, 'ent_total', n_ent)
        n_rel = getattr(args, 'rel_total', n_rel)

        self.ent_embeddings = nn.Embedding(n_ent or 1, emb_dim)
        self.rel_embeddings = nn.Embedding(n_rel or 1, emb_dim)
        self.Dr = nn.Embedding(n_rel or 1, emb_dim)
        self.para = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.init_parameters()

    def init_parameters(self) -> None:
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.Dr.weight.data)

    @staticmethod
    def normalization(quaternion, split_dim=1):
        size = quaternion.size(split_dim) // 4
        quaternion = quaternion.reshape(-1, 4, size)
        quaternion = quaternion / torch.sqrt(torch.sum(quaternion ** 2, 1, True).clamp_min(1e-12))
        return quaternion.reshape(-1, 4 * size)

    @staticmethod
    def make_wise_quaternion(quaternion):
        if len(quaternion.size()) == 1:
            quaternion = quaternion.unsqueeze(0)
        size = quaternion.size(1) // 4
        r, i, j, k = torch.split(quaternion, size, dim=1)
        r2 = torch.cat([r, -i, -j, -k], dim=1)
        i2 = torch.cat([i, r, -k, j], dim=1)
        j2 = torch.cat([j, k, r, -i], dim=1)
        k2 = torch.cat([k, -j, i, r], dim=1)
        return r2, i2, j2, k2

    @staticmethod
    def get_quaternion_wise_mul(quaternion):
        size = quaternion.size(1) // 4
        quaternion = quaternion.view(-1, 4, size)
        return torch.sum(quaternion, 1)

    @staticmethod
    def vec_vec_wise_multiplication(q, p):
        normalized_p = DaBREncoder.normalization(p)
        return DaBREncoder.vec_vec_wise_multiplication_q(q, normalized_p)

    @staticmethod
    def vec_vec_wise_multiplication_q(q, p_normalized):
        """Quaternion multiply ``q * p`` when ``p`` is already normalized."""

        q_r, q_i, q_j, q_k = DaBREncoder.make_wise_quaternion(q)
        qp_r = DaBREncoder.get_quaternion_wise_mul(q_r * p_normalized)
        qp_i = DaBREncoder.get_quaternion_wise_mul(q_i * p_normalized)
        qp_j = DaBREncoder.get_quaternion_wise_mul(q_j * p_normalized)
        qp_k = DaBREncoder.get_quaternion_wise_mul(q_k * p_normalized)
        return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=1)

    @staticmethod
    def get_inv(quaternion):
        q_r, q_i, q_j, q_k = torch.chunk(quaternion, 4, dim=1)
        quaternion_norm = (q_r ** 2 + q_i ** 2 + q_j ** 2 + q_k ** 2).clamp_min(1e-12)
        return torch.cat([q_r / quaternion_norm, -q_i / quaternion_norm, -q_j / quaternion_norm, -q_k / quaternion_norm], dim=1)

    @staticmethod
    def _calc_from_hr(h, t, dr, hr, r_inv, para) -> torch.Tensor:
        """Score rows when ``hr = h⊗r`` and ``r_inv`` are already computed."""

        r_inv_norm = DaBREncoder.normalization(r_inv)
        return DaBREncoder._calc_from_hr_norm(h, t, dr, hr, r_inv_norm, para)

    @staticmethod
    def _calc_from_hr_norm(h, t, dr, hr, r_inv_norm, para) -> torch.Tensor:
        """Score rows using pre-normalized ``r_inv`` (link-prediction fast path)."""

        tr = DaBREncoder.vec_vec_wise_multiplication_q(t, r_inv_norm)
        score_s = hr * tr
        hrt = h + dr - t
        s_d, x_d, y_d, z_d = torch.chunk(hrt, 4, dim=1)
        score_d = s_d + x_d + y_d + z_d
        return -torch.sum(score_s, -1) - para * torch.norm(score_d, p=1, dim=-1)

    @staticmethod
    def _score_lp_block(h_rep, t_rep, dr_rep, hr_rep, r_inv_norm_rep, para) -> torch.Tensor:
        """Batched link-prediction scores for expanded query/candidate rows."""

        return DaBREncoder._calc_from_hr_norm(h_rep, t_rep, dr_rep, hr_rep, r_inv_norm_rep, para)

    _compiled_score_lp_block = None

    @classmethod
    def _score_lp_block_fn(cls, config):
        """Return (optionally compiled) link-prediction block scorer."""

        if getattr(config, 'compile_eval', True) and torch.cuda.is_available():
            if cls._compiled_score_lp_block is None:
                try:
                    cls._compiled_score_lp_block = torch.compile(
                        cls._score_lp_block,
                        mode='reduce-overhead',
                    )
                except Exception:
                    cls._compiled_score_lp_block = cls._score_lp_block
            return cls._compiled_score_lp_block
        return cls._score_lp_block

    @staticmethod
    def _calc(h, r, t, dr, para):
        hr = DaBREncoder.vec_vec_wise_multiplication(h, r)
        r_inv = DaBREncoder.get_inv(r)
        return DaBREncoder._calc_from_hr(h, t, dr, hr, r_inv, para)

    @staticmethod
    def regularization(quaternion):
        size = quaternion.size(1) // 4
        r, i, j, k = torch.split(quaternion, size, dim=1)
        return torch.mean(r ** 2) + torch.mean(i ** 2) + torch.mean(j ** 2) + torch.mean(k ** 2)

    def _head_relation_tail(self, src, rel, dst):
        """Look up head, relation, and tail embeddings for index tensors."""

        h = self.ent_embeddings(src)
        r = self.rel_embeddings(rel)
        t = self.ent_embeddings(dst)
        return h, r, t

    def get_queries_targets(self, src, rel, dst) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return AU query, target, and head vectors aligned with DaBR scoring.

        Query uses h composed with r (hr). Target uses t composed with r^{-1} (tr),
        matching the multiplicative term in ``_calc``. Head returns raw entity embeddings.
        """

        h, r, t = self._head_relation_tail(src, rel, dst)
        q = DaBREncoder.vec_vec_wise_multiplication(h, r)
        t_target = DaBREncoder.vec_vec_wise_multiplication(t, DaBREncoder.get_inv(r))
        return q, t_target, h

    def entity_embeddings(self, device: torch.device | None = None) -> torch.Tensor:
        """Return all entity embeddings (for optional entity-level uniformity)."""

        entity_vectors = self.ent_embeddings.weight
        if device is not None:
            entity_vectors = entity_vectors.to(device)
        return entity_vectors

    def forward(self, batch_dict: dict) -> dict:
        h = self.ent_embeddings(batch_dict['head_id'])
        r = self.rel_embeddings(batch_dict['relation'])
        t = self.ent_embeddings(batch_dict['tail_id'])
        dr = self.Dr(batch_dict['relation'])
        score = DaBREncoder._calc(h, r, t, dr, self.para)
        return {'scores': score, 'ent_emb': (h, t), 'rel_emb': (r, dr)}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        """Adapt DaBR forward outputs to the generic logits interface."""

        if torch.is_tensor(output_dict):
            return {'logits': output_dict}
        if isinstance(output_dict, dict):
            if 'logits' in output_dict:
                return output_dict
            if 'scores' in output_dict:
                return {'logits': output_dict['scores']}
        raise TypeError('Unsupported model output type for logits computation')

    @torch.inference_mode()
    def prepare_link_prediction_queries(self, head_ids, relations) -> dict:
        """Precompute query-side embeddings once per link-prediction eval direction."""

        device = self.ent_embeddings.weight.device
        head_indices = _as_index_tensor(head_ids, self.entity_dict.entity_to_idx, device)
        relation_indices = _as_index_tensor(relations, self._relation_to_idx, device)

        h = self.ent_embeddings(head_indices)
        r = self.rel_embeddings(relation_indices)
        dr = self.Dr(relation_indices)
        hr = DaBREncoder.vec_vec_wise_multiplication(h, r)
        r_inv = DaBREncoder.get_inv(r)
        r_inv_norm = DaBREncoder.normalization(r_inv)
        return {'h': h, 'dr': dr, 'hr': hr, 'r_inv': r_inv, 'r_inv_norm': r_inv_norm, 'para': self.para}

    def _score_query_candidate_block(
        self,
        h_chunk: torch.Tensor,
        dr_chunk: torch.Tensor,
        hr_chunk: torch.Tensor,
        r_inv_norm_chunk: torch.Tensor,
        t_candidates: torch.Tensor,
        para: torch.Tensor,
    ) -> torch.Tensor:
        """Score a query block against a candidate block without OOM from full Q×C expansion."""

        q_size, num_candidates = h_chunk.size(0), t_candidates.size(0)
        if q_size == 0 or num_candidates == 0:
            return torch.empty(q_size, num_candidates, device=h_chunk.device)

        t_rep = t_candidates.unsqueeze(0).expand(q_size, num_candidates, -1).reshape(q_size * num_candidates, -1)
        h_rep = h_chunk.unsqueeze(1).expand(q_size, num_candidates, -1).reshape(q_size * num_candidates, -1)
        dr_rep = dr_chunk.unsqueeze(1).expand(q_size, num_candidates, -1).reshape(q_size * num_candidates, -1)
        hr_rep = hr_chunk.unsqueeze(1).expand(q_size, num_candidates, -1).reshape(q_size * num_candidates, -1)
        r_inv_norm_rep = r_inv_norm_chunk.unsqueeze(1).expand(q_size, num_candidates, -1).reshape(q_size * num_candidates, -1)

        score_fn = DaBREncoder._score_lp_block_fn(self.config)
        with _autocast_context(self.config):
            block_scores = score_fn(h_rep, t_rep, dr_rep, hr_rep, r_inv_norm_rep, para)
        return block_scores.reshape(q_size, num_candidates).float()

    @torch.inference_mode()
    def score_link_prediction_candidates(
        self,
        query_cache: dict,
        tail_entity_ids,
        query_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Score cached queries against a candidate tail set (entity indices or ids)."""

        device = self.ent_embeddings.weight.device
        h = query_cache['h']
        dr = query_cache['dr']
        hr = query_cache['hr']
        r_inv_norm = query_cache.get('r_inv_norm')
        if r_inv_norm is None:
            r_inv_norm = DaBREncoder.normalization(query_cache['r_inv'])
        para = query_cache['para']

        if isinstance(tail_entity_ids, tuple) and len(tail_entity_ids) == 2:
            start_idx, end_idx = int(tail_entity_ids[0]), int(tail_entity_ids[1])
            t = self.ent_embeddings.weight[start_idx:end_idx]
        elif torch.is_tensor(tail_entity_ids):
            candidate_indices = tail_entity_ids.to(device=device, dtype=torch.long)
            t = self.ent_embeddings.weight.index_select(0, candidate_indices)
        else:
            candidate_indices = _as_index_tensor(tail_entity_ids, self.entity_dict.entity_to_idx, device)
            t = self.ent_embeddings.weight.index_select(0, candidate_indices)
        num_queries, num_candidates = h.size(0), t.size(0)
        if num_queries == 0 or num_candidates == 0:
            return torch.empty(num_queries, num_candidates, device=device)

        if query_chunk_size is None:
            query_chunk_size = getattr(self.config, 'score_query_chunk_size', 256)
        query_chunk_size = max(int(query_chunk_size), 1)
        candidate_chunk_size = int(getattr(self.config, 'eval_candidate_chunk_size', 1024))
        candidate_chunk_size = max(candidate_chunk_size, 1)

        scores = torch.empty(num_queries, num_candidates, device=device, dtype=torch.float32)
        for cand_start in range(0, num_candidates, candidate_chunk_size):
            cand_end = min(cand_start + candidate_chunk_size, num_candidates)
            t_block = t[cand_start:cand_end]
            for q_start in range(0, num_queries, query_chunk_size):
                q_end = min(q_start + query_chunk_size, num_queries)
                try:
                    block = self._score_query_candidate_block(
                        h[q_start:q_end],
                        dr[q_start:q_end],
                        hr[q_start:q_end],
                        r_inv_norm[q_start:q_end],
                        t_block,
                        para,
                    )
                except torch.OutOfMemoryError:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    half_q = max(query_chunk_size // 2, 1)
                    for sub_start in range(q_start, q_end, half_q):
                        sub_end = min(sub_start + half_q, q_end)
                        block_part = self._score_query_candidate_block(
                            h[sub_start:sub_end],
                            dr[sub_start:sub_end],
                            hr[sub_start:sub_end],
                            r_inv_norm[sub_start:sub_end],
                            t_block,
                            para,
                        )
                        scores[sub_start:sub_end, cand_start:cand_end] = block_part
                    continue
                scores[q_start:q_end, cand_start:cand_end] = block

        return scores

    def score_batch(self, head_ids, relations, tail_entity_ids, query_chunk_size: int | None = None) -> torch.Tensor:
        """Score queries against candidate tails (builds a fresh query cache)."""

        query_cache = self.prepare_link_prediction_queries(head_ids, relations)
        return self.score_link_prediction_candidates(query_cache, tail_entity_ids, query_chunk_size)

    def _relation_to_idx(self, relation: str) -> int:
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


def _relation_path_candidates(args):
    paths = []
    for source_path in [getattr(args, 'train_path', ''), getattr(args, 'valid_path', ''), getattr(args, 'test_path', '')]:
        if not source_path:
            continue
        paths.append(os.path.join(os.path.dirname(source_path), 'relation2id.json'))
    paths.append(os.path.join('data', getattr(args, 'dataset', ''), 'relation2id.json'))
    return paths


def _load_relation_to_idx(args):
    for path in _relation_path_candidates(args):
        if not path or not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as handle:
            mapping = json.load(handle)
        if isinstance(mapping, dict):
            return {str(key): int(value) for key, value in mapping.items()}

    relations, seen = [], set()
    for example in load_data(getattr(args, 'train_path', ''), add_forward_triplet=False, add_backward_triplet=False):
        if example.relation not in seen:
            seen.add(example.relation)
            relations.append(example.relation)
    return {relation: idx for idx, relation in enumerate(relations)}


def _as_index_tensor(values, lookup, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(values):
        return values.to(device=device, dtype=torch.long)
    return torch.tensor([lookup(value) for value in values], dtype=torch.long, device=device)
