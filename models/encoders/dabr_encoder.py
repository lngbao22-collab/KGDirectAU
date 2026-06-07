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


def _config_int(config, name: str, default: int) -> int:
    """Read an integer config value, treating JSON/null as unset."""

    value = getattr(config, name, default)
    if value is None:
        return default
    return int(value)


def _is_entity_index_slice(tail_entity_ids) -> bool:
    """Return True when ``tail_entity_ids`` is an entity row slice ``(start, end)``."""

    return (
        isinstance(tail_entity_ids, tuple)
        and len(tail_entity_ids) == 2
        and not isinstance(tail_entity_ids[0], (list, str))
    )


def build_model(args) -> nn.Module:
    entity_dict = get_entity_dict()
    relation_to_idx = _load_relation_to_idx(args)
    model = DaBREncoder(args, len(entity_dict), len(relation_to_idx))
    model.rel_to_idx = relation_to_idx
    model.entity_dict = entity_dict
    return model


class DaBREncoder(BaseModel):
    bidirectional_score_batch = True

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
        """Return AU query/target vectors whose alignment maximises the DaBR LP score.

        DaBR score = −dot(h⊗r, t⊗r⁻¹) − para·‖h + Dᵣ − t‖₁

        Both terms are ≤ 0; a correct triple scores close to 0 (highest) when:
          (a) dot(h⊗r, t⊗r⁻¹) ≤ 0  (anti-parallel or orthogonal) → −dot ≥ 0
          (b) h + Dr ≈ t             → L1 term ≈ 0

        AU alignment minimises ‖normalize(q) − normalize(t_target)‖², which is
        zero when q and t_target are **parallel**.  To satisfy (a) via alignment
        we must negate the multiplicative target so that "q ∥ t_target" means
        "h⊗r ∥ −(t⊗r⁻¹)", i.e. h⊗r and t⊗r⁻¹ are anti-parallel.

        Query  = concat[ h⊗r,     h + Dr ]
        Target = concat[ −(t⊗r⁻¹), t    ]   ← multiplicative branch negated
        """

        h, r, t = self._head_relation_tail(src, rel, dst)
        dr = self.Dr(rel)
        q_mult = DaBREncoder.vec_vec_wise_multiplication(h, r)
        t_mult = -DaBREncoder.vec_vec_wise_multiplication(t, DaBREncoder.get_inv(r))  # negated
        q_add = h + dr
        t_add = t
        q = torch.cat([q_mult, q_add], dim=-1)
        t_target = torch.cat([t_mult, t_add], dim=-1)
        return q, t_target, h

    def entity_embeddings(
        self,
        device: torch.device | None = None,
        max_samples: int | None = None,
    ) -> torch.Tensor:
        """Return entity embeddings for optional entity-level uniformity (subsampled when requested)."""

        entity_vectors = self.ent_embeddings.weight
        if max_samples is not None and int(max_samples) > 0 and entity_vectors.size(0) > int(max_samples):
            indices = torch.randperm(entity_vectors.size(0), device=entity_vectors.device)[: int(max_samples)]
            entity_vectors = entity_vectors.index_select(0, indices)
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

    def _resolve_link_prediction_chunk_sizes(
        self,
        query_chunk_size: int | None,
        candidate_chunk_size: int | None,
    ) -> tuple[int, int]:
        """Resolve query/candidate tile sizes for link-prediction scoring."""

        if query_chunk_size is None:
            query_chunk_size = _config_int(self.config, 'score_query_chunk_size', 256)
        else:
            query_chunk_size = int(query_chunk_size)
        query_chunk_size = max(query_chunk_size, 1)
        if candidate_chunk_size is None:
            candidate_chunk_size = _config_int(self.config, 'eval_candidate_chunk_size', 1024)
        else:
            candidate_chunk_size = int(candidate_chunk_size)
        candidate_chunk_size = max(candidate_chunk_size, 1)
        return query_chunk_size, candidate_chunk_size

    @torch.inference_mode()
    def prepare_link_prediction_queries(self, head_ids, relations) -> dict:
        """Precompute tail-batch query embeddings: fix (head, relation), score candidate tails."""

        device = self.ent_embeddings.weight.device
        head_indices = _as_index_tensor(head_ids, self.entity_dict.entity_to_idx, device)
        relation_indices = _as_index_tensor(relations, self._relation_to_idx, device)

        h = self.ent_embeddings(head_indices)
        r = self.rel_embeddings(relation_indices)
        dr = self.Dr(relation_indices)
        hr = DaBREncoder.vec_vec_wise_multiplication(h, r)
        r_inv = DaBREncoder.get_inv(r)
        r_inv_norm = DaBREncoder.normalization(r_inv)
        return {
            'h': h,
            'dr': dr,
            'hr': hr,
            'r_inv': r_inv,
            'r_inv_norm': r_inv_norm,
            'relation_indices': relation_indices,
            'para': self.para,
        }

    @torch.inference_mode()
    def prepare_head_prediction_queries(self, tail_ids, relations) -> dict:
        """Precompute head-batch query embeddings: fix (relation, tail), score candidate heads."""

        device = self.ent_embeddings.weight.device
        tail_indices = _as_index_tensor(tail_ids, self.entity_dict.entity_to_idx, device)
        relation_indices = _as_index_tensor(relations, self._relation_to_idx, device)

        t = self.ent_embeddings(tail_indices)
        r = self.rel_embeddings(relation_indices)
        dr = self.Dr(relation_indices)
        r_inv = DaBREncoder.get_inv(r)
        r_inv_norm = DaBREncoder.normalization(r_inv)
        return {
            't': t,
            'dr': dr,
            'r': r,
            'r_inv': r_inv,
            'r_inv_norm': r_inv_norm,
            'relation_indices': relation_indices,
            'para': self.para,
        }

    def _precompute_tail_relation_matrix(self, relation_index: int, entity_weight: torch.Tensor) -> torch.Tensor:
        """Precompute ``tr`` for every entity under one relation (``t ⊗ r^{-1}``)."""

        device = entity_weight.device
        rel_tensor = torch.tensor([relation_index], device=device, dtype=torch.long)
        r = self.rel_embeddings(rel_tensor)
        r_inv_norm = DaBREncoder.normalization(DaBREncoder.get_inv(r))
        entity_chunk = _config_int(self.config, 'eval_entity_chunk_size', 4096)
        entity_chunk = max(entity_chunk, 1)

        tail_rows = []
        with _autocast_context(self.config):
            for start in range(0, entity_weight.size(0), entity_chunk):
                end = min(start + entity_chunk, entity_weight.size(0))
                t_chunk = entity_weight[start:end]
                tr_chunk = DaBREncoder.vec_vec_wise_multiplication_q(
                    t_chunk,
                    r_inv_norm.expand(t_chunk.size(0), -1),
                )
                tail_rows.append(tr_chunk.float())
        return torch.cat(tail_rows, dim=0)

    def _precompute_head_relation_matrix(self, relation_index: int, entity_weight: torch.Tensor) -> torch.Tensor:
        """Precompute ``hr`` for every entity under one relation (``h ⊗ r``)."""

        device = entity_weight.device
        rel_tensor = torch.tensor([relation_index], device=device, dtype=torch.long)
        r = self.rel_embeddings(rel_tensor)
        entity_chunk = _config_int(self.config, 'eval_entity_chunk_size', 4096)
        entity_chunk = max(entity_chunk, 1)

        head_rows = []
        with _autocast_context(self.config):
            for start in range(0, entity_weight.size(0), entity_chunk):
                end = min(start + entity_chunk, entity_weight.size(0))
                h_chunk = entity_weight[start:end]
                hr_chunk = DaBREncoder.vec_vec_wise_multiplication(h_chunk, r.expand(h_chunk.size(0), -1))
                head_rows.append(hr_chunk.float())
        return torch.cat(head_rows, dim=0)

    def _distance_scores_one(
        self,
        h: torch.Tensor,
        dr: torch.Tensor,
        entity_weight: torch.Tensor,
        para: torch.Tensor,
        dist_chunk: int,
    ) -> torch.Tensor:
        """Distance term of DaBR for one query against all entities."""

        h_dr = h + dr
        parts = []
        for start in range(0, entity_weight.size(0), dist_chunk):
            end = min(start + dist_chunk, entity_weight.size(0))
            hrt = h_dr.unsqueeze(0) - entity_weight[start:end]
            s_d, x_d, y_d, z_d = torch.chunk(hrt, 4, dim=1)
            parts.append((s_d + x_d + y_d + z_d).abs().sum(dim=1))
        return -para * torch.cat(parts)

    def _distance_scores_head_one(
        self,
        t: torch.Tensor,
        dr: torch.Tensor,
        entity_weight: torch.Tensor,
        para: torch.Tensor,
        dist_chunk: int,
    ) -> torch.Tensor:
        """Distance term of DaBR for head prediction: fix (relation, tail), vary candidate heads."""

        t_exp = t.unsqueeze(0)
        dr_exp = dr.unsqueeze(0)
        parts = []
        for start in range(0, entity_weight.size(0), dist_chunk):
            end = min(start + dist_chunk, entity_weight.size(0))
            hrt = entity_weight[start:end] + dr_exp - t_exp
            s_d, x_d, y_d, z_d = torch.chunk(hrt, 4, dim=1)
            parts.append((s_d + x_d + y_d + z_d).abs().sum(dim=1))
        return -para * torch.cat(parts)

    @torch.inference_mode()
    def score_link_prediction_full(self, query_cache: dict) -> torch.Tensor:
        """Score all queries against all entities (fast path grouped by relation)."""

        if not bool(getattr(self.config, 'use_fast_link_prediction', True)):
            return self._score_link_prediction_full_slow(query_cache)

        device = self.ent_embeddings.weight.device
        entity_weight = self.ent_embeddings.weight
        n_ent = entity_weight.size(0)
        n_queries = query_cache['h'].size(0)
        scores = torch.zeros(n_queries, n_ent, device=device, dtype=torch.float32)
        if n_queries == 0 or n_ent == 0:
            return scores

        relation_indices = query_cache['relation_indices']
        hr = query_cache['hr']
        h = query_cache['h']
        dr = query_cache['dr']
        para = query_cache['para']

        tail_cache: dict[int, torch.Tensor] = {}
        dist_chunk = _config_int(self.config, 'eval_distance_chunk_size', 8192)
        dist_chunk = max(dist_chunk, 1)
        para_scalar = para.squeeze() if para.dim() else para

        for rel_idx in torch.unique(relation_indices).tolist():
            rel_idx = int(rel_idx)
            query_rows = (relation_indices == rel_idx).nonzero(as_tuple=True)[0]
            if query_rows.numel() == 0:
                continue

            if rel_idx not in tail_cache:
                tail_cache[rel_idx] = self._precompute_tail_relation_matrix(rel_idx, entity_weight)
            tail_matrix = tail_cache[rel_idx]

            hr_group = hr[query_rows]
            with _autocast_context(self.config):
                score_s = -torch.matmul(hr_group, tail_matrix.t()).float()

            for local_i in range(query_rows.numel()):
                global_i = int(query_rows[local_i])
                scores[global_i] = score_s[local_i]
                scores[global_i] += self._distance_scores_one(
                    h[global_i],
                    dr[global_i],
                    entity_weight,
                    para_scalar,
                    dist_chunk,
                )

        return scores

    def _score_link_prediction_full_slow(self, query_cache: dict) -> torch.Tensor:
        """Fallback full-matrix scorer that reuses candidate chunking."""

        n_queries = query_cache['h'].size(0)
        n_ent = self.ent_embeddings.weight.size(0)
        return self.score_link_prediction_candidates(query_cache, (0, n_ent))[:n_queries, :n_ent]

    @torch.inference_mode()
    def score_head_prediction_full(self, query_cache: dict) -> torch.Tensor:
        """Score all head-batch queries against all entities (fast path grouped by relation)."""

        if not bool(getattr(self.config, 'use_fast_link_prediction', True)):
            return self._score_head_prediction_full_slow(query_cache)

        device = self.ent_embeddings.weight.device
        entity_weight = self.ent_embeddings.weight
        n_ent = entity_weight.size(0)
        n_queries = query_cache['t'].size(0)
        scores = torch.zeros(n_queries, n_ent, device=device, dtype=torch.float32)
        if n_queries == 0 or n_ent == 0:
            return scores

        relation_indices = query_cache['relation_indices']
        t = query_cache['t']
        dr = query_cache['dr']
        r_inv_norm = query_cache['r_inv_norm']
        para = query_cache['para']

        head_cache: dict[int, torch.Tensor] = {}
        dist_chunk = _config_int(self.config, 'eval_distance_chunk_size', 8192)
        dist_chunk = max(dist_chunk, 1)
        para_scalar = para.squeeze() if para.dim() else para

        for rel_idx in torch.unique(relation_indices).tolist():
            rel_idx = int(rel_idx)
            query_rows = (relation_indices == rel_idx).nonzero(as_tuple=True)[0]
            if query_rows.numel() == 0:
                continue

            if rel_idx not in head_cache:
                head_cache[rel_idx] = self._precompute_head_relation_matrix(rel_idx, entity_weight)
            head_matrix = head_cache[rel_idx]

            t_group = t[query_rows]
            dr_group = dr[query_rows]
            r_inv_norm_group = r_inv_norm[query_rows]
            with _autocast_context(self.config):
                tr_group = DaBREncoder.vec_vec_wise_multiplication_q(t_group, r_inv_norm_group).float()
                score_s = -torch.matmul(tr_group, head_matrix.t())

            for local_i in range(query_rows.numel()):
                global_i = int(query_rows[local_i])
                scores[global_i] = score_s[local_i]
                scores[global_i] += self._distance_scores_head_one(
                    t[global_i],
                    dr[global_i],
                    entity_weight,
                    para_scalar,
                    dist_chunk,
                )

        return scores

    def _score_head_prediction_full_slow(self, query_cache: dict) -> torch.Tensor:
        """Fallback full-matrix head-batch scorer that reuses candidate chunking."""

        n_queries = query_cache['t'].size(0)
        n_ent = self.ent_embeddings.weight.size(0)
        return self.score_head_prediction_candidates(query_cache, (0, n_ent))[:n_queries, :n_ent]

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

    def _score_head_query_candidate_block(
        self,
        h_candidates: torch.Tensor,
        t_chunk: torch.Tensor,
        dr_chunk: torch.Tensor,
        r_chunk: torch.Tensor,
        r_inv_norm_chunk: torch.Tensor,
        para: torch.Tensor,
    ) -> torch.Tensor:
        """Score a head-batch query block against a candidate-head block."""

        q_size, num_candidates = t_chunk.size(0), h_candidates.size(0)
        if q_size == 0 or num_candidates == 0:
            return torch.empty(q_size, num_candidates, device=t_chunk.device)

        h_rep = h_candidates.unsqueeze(0).expand(q_size, num_candidates, -1).reshape(q_size * num_candidates, -1)
        t_rep = t_chunk.unsqueeze(1).expand(q_size, num_candidates, -1).reshape(q_size * num_candidates, -1)
        dr_rep = dr_chunk.unsqueeze(1).expand(q_size, num_candidates, -1).reshape(q_size * num_candidates, -1)
        r_rep = r_chunk.unsqueeze(1).expand(q_size, num_candidates, -1).reshape(q_size * num_candidates, -1)
        r_inv_norm_rep = r_inv_norm_chunk.unsqueeze(1).expand(q_size, num_candidates, -1).reshape(q_size * num_candidates, -1)
        hr_rep = DaBREncoder.vec_vec_wise_multiplication(h_rep, r_rep)

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
        candidate_chunk_size: int | None = None,
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

        query_chunk_size, candidate_chunk_size = self._resolve_link_prediction_chunk_sizes(
            query_chunk_size, candidate_chunk_size,
        )

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

    @torch.inference_mode()
    def score_head_prediction_candidates(
        self,
        query_cache: dict,
        head_entity_ids,
        query_chunk_size: int | None = None,
        candidate_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Score cached head-batch queries against a candidate head set."""

        device = self.ent_embeddings.weight.device
        t = query_cache['t']
        dr = query_cache['dr']
        r = query_cache['r']
        r_inv_norm = query_cache.get('r_inv_norm')
        if r_inv_norm is None:
            r_inv_norm = DaBREncoder.normalization(query_cache['r_inv'])
        para = query_cache['para']

        if isinstance(head_entity_ids, tuple) and len(head_entity_ids) == 2:
            start_idx, end_idx = int(head_entity_ids[0]), int(head_entity_ids[1])
            h = self.ent_embeddings.weight[start_idx:end_idx]
        elif torch.is_tensor(head_entity_ids):
            candidate_indices = head_entity_ids.to(device=device, dtype=torch.long)
            h = self.ent_embeddings.weight.index_select(0, candidate_indices)
        else:
            candidate_indices = _as_index_tensor(head_entity_ids, self.entity_dict.entity_to_idx, device)
            h = self.ent_embeddings.weight.index_select(0, candidate_indices)
        num_queries, num_candidates = t.size(0), h.size(0)
        if num_queries == 0 or num_candidates == 0:
            return torch.empty(num_queries, num_candidates, device=device)

        query_chunk_size, candidate_chunk_size = self._resolve_link_prediction_chunk_sizes(
            query_chunk_size, candidate_chunk_size,
        )

        scores = torch.empty(num_queries, num_candidates, device=device, dtype=torch.float32)
        for cand_start in range(0, num_candidates, candidate_chunk_size):
            cand_end = min(cand_start + candidate_chunk_size, num_candidates)
            h_block = h[cand_start:cand_end]
            for q_start in range(0, num_queries, query_chunk_size):
                q_end = min(q_start + query_chunk_size, num_queries)
                try:
                    block = self._score_head_query_candidate_block(
                        h_block,
                        t[q_start:q_end],
                        dr[q_start:q_end],
                        r[q_start:q_end],
                        r_inv_norm[q_start:q_end],
                        para,
                    )
                except torch.OutOfMemoryError:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    half_q = max(query_chunk_size // 2, 1)
                    for sub_start in range(q_start, q_end, half_q):
                        sub_end = min(sub_start + half_q, q_end)
                        block_part = self._score_head_query_candidate_block(
                            h_block,
                            t[sub_start:sub_end],
                            dr[sub_start:sub_end],
                            r[sub_start:sub_end],
                            r_inv_norm[sub_start:sub_end],
                            para,
                        )
                        scores[sub_start:sub_end, cand_start:cand_end] = block_part
                    continue
                scores[q_start:q_end, cand_start:cand_end] = block

        return scores

    @torch.inference_mode()
    def _score_paired_triples(self, head_ids, relations, tail_ids) -> torch.Tensor:
        """Score one (head, relation, tail) triple per row — used for triple classification."""

        device = self.ent_embeddings.weight.device
        head_indices = _as_index_tensor(head_ids, self.entity_dict.entity_to_idx, device)
        relation_indices = _as_index_tensor(relations, self._relation_to_idx, device)
        tail_indices = _as_index_tensor(tail_ids, self.entity_dict.entity_to_idx, device)

        h = self.ent_embeddings(head_indices)
        r = self.rel_embeddings(relation_indices)
        t = self.ent_embeddings(tail_indices)
        dr = self.Dr(relation_indices)
        with _autocast_context(self.config):
            scores = DaBREncoder._calc(h, r, t, dr, self.para)
        return scores.float()

    @torch.inference_mode()
    def score_batch_from_indices(
        self,
        relation_indices: torch.Tensor,
        candidate_indices: torch.Tensor,
        mode: str = 'tail-batch',
        query_head_indices: torch.Tensor | None = None,
        query_tail_indices: torch.Tensor | None = None,
        query_chunk_size: int | None = None,
        candidate_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Fast link-prediction scoring using precomputed index tensors."""

        device = self.ent_embeddings.weight.device
        batch_mode = str(mode or 'tail-batch')
        if batch_mode not in {'head-batch', 'tail-batch'}:
            raise ValueError(f'mode {batch_mode} not supported')

        relation_indices = relation_indices.to(device=device, dtype=torch.long)
        candidate_indices = candidate_indices.to(device=device, dtype=torch.long)
        query_chunk_size, candidate_chunk_size = self._resolve_link_prediction_chunk_sizes(
            query_chunk_size, candidate_chunk_size,
        )

        if batch_mode == 'tail-batch':
            if query_head_indices is None:
                raise ValueError('query_head_indices is required for tail-batch scoring')
            head_ids = query_head_indices.to(device=device, dtype=torch.long)
            relations = relation_indices
            head_id_list = head_ids.detach().cpu().tolist()
            relation_list = relations.detach().cpu().tolist()
            query_cache = self.prepare_link_prediction_queries(head_id_list, relation_list)
            return self.score_link_prediction_candidates(
                query_cache,
                candidate_indices,
                query_chunk_size=query_chunk_size,
                candidate_chunk_size=candidate_chunk_size,
            )

        if query_tail_indices is None:
            raise ValueError('query_tail_indices is required for head-batch scoring')
        tail_ids = query_tail_indices.to(device=device, dtype=torch.long)
        tail_id_list = tail_ids.detach().cpu().tolist()
        relation_list = relation_indices.detach().cpu().tolist()
        query_cache = self.prepare_head_prediction_queries(tail_id_list, relation_list)
        return self.score_head_prediction_candidates(
            query_cache,
            candidate_indices,
            query_chunk_size=query_chunk_size,
            candidate_chunk_size=candidate_chunk_size,
        )

    def score_batch(
        self,
        head_ids,
        relations,
        tail_entity_ids,
        mode: str = 'tail-batch',
        query_tail_ids=None,
        query_chunk_size: int | None = None,
        candidate_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Score link-prediction queries against candidate entities.

        tail-batch (default): fix (head, relation), score candidate tails.
        head-batch: fix (relation, tail), score candidate heads via ``query_tail_ids``.
        """

        batch_mode = str(mode or 'tail-batch')
        if batch_mode not in {'head-batch', 'tail-batch'}:
            raise ValueError(f'mode {batch_mode} not supported')

        if batch_mode == 'head-batch':
            if query_tail_ids is None:
                raise ValueError('query_tail_ids is required for head-batch scoring')
            query_cache = self.prepare_head_prediction_queries(query_tail_ids, relations)
            return self.score_head_prediction_candidates(
                query_cache,
                tail_entity_ids,
                query_chunk_size=query_chunk_size,
                candidate_chunk_size=candidate_chunk_size,
            )

        if (
            not _is_entity_index_slice(tail_entity_ids)
            and len(head_ids) == len(relations) == len(tail_entity_ids)
        ):
            return self._score_paired_triples(head_ids, relations, tail_entity_ids)

        query_cache = self.prepare_link_prediction_queries(head_ids, relations)
        return self.score_link_prediction_candidates(
            query_cache,
            tail_entity_ids,
            query_chunk_size=query_chunk_size,
            candidate_chunk_size=candidate_chunk_size,
        )

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
