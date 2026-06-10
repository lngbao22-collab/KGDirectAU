"""Filtered 1-N negative sampler adapted from RotatE TrainDataset."""

from __future__ import annotations

import numpy as np
import torch


class FilteredSubsampler:
    """Filtered 1-N negative sampler for RotatE-style training."""

    def __init__(self, triples, nentity: int, num_negatives: int):
        self.nentity = int(nentity)
        self.num_negatives = int(num_negatives)
        self.count = self._count_frequency(triples)
        self.true_head, self.true_tail = self._build_filter_dicts(triples)

    @staticmethod
    def _normalize_triple(triple) -> tuple[int, int, int]:
        """Normalize a triple to (head, relation, tail) format and convert to integers."""

        if torch.is_tensor(triple):
            triple = triple.detach().cpu().tolist()
        return int(triple[0]), int(triple[1]), int(triple[2])

    @staticmethod
    def _count_frequency(triples, start: int = 4) -> dict[tuple[int, int], int]:
        """Count the frequency of (head, relation) and (tail, -relation-1) pairs in the training triples."""

        count = {}
        for triple in triples:
            head, relation, tail = FilteredSubsampler._normalize_triple(triple)
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def _build_filter_dicts(triples) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
        """Build dictionaries mapping (head, relation) to true tails and (relation, tail) to true heads for filtering."""

        true_head = {}
        true_tail = {}

        for triple in triples:
            head, relation, tail = FilteredSubsampler._normalize_triple(triple)
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)

            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail

    def _ensure_tensor_triples(self, batch_triples) -> torch.Tensor:
        """Convert batch triples to a tensor of shape [B, 3] if they are not already tensors."""

        if torch.is_tensor(batch_triples):
            return batch_triples.long()
        if isinstance(batch_triples, dict):
            if {"head_id", "relation", "tail_id"}.issubset(batch_triples.keys()):
                return torch.stack(
                    [
                        batch_triples["head_id"].long(),
                        batch_triples["relation"].long(),
                        batch_triples["tail_id"].long(),
                    ],
                    dim=-1,
                )
        return torch.tensor([self._normalize_triple(t) for t in batch_triples], dtype=torch.long)

    def _subsampling_weights(self, head: np.ndarray, relation: np.ndarray, tail: np.ndarray) -> torch.Tensor:
        """Compute sqrt-inverse-frequency subsampling weights for a batch."""

        weights = np.empty(head.shape[0], dtype=np.float64)
        for idx, (h, r, t) in enumerate(zip(head, relation, tail)):
            weights[idx] = self.count.get((int(h), int(r)), 4) + self.count.get((int(t), -int(r) - 1), 4)
        return torch.from_numpy(np.sqrt(1.0 / weights)).float()

    def _sample_filtered_negatives_row(self, key: tuple[int, int], filter_dict: dict, oversample: int) -> np.ndarray:
        """Draw filtered negatives for one row, resampling only when the first pass is short."""

        blocked = filter_dict.get(key, np.array([], dtype=np.int64))
        collected = []
        remaining = self.num_negatives
        attempts = 0
        while remaining > 0 and attempts < 4:
            candidate = np.random.randint(self.nentity, size=max(oversample, remaining * 2))
            valid = candidate[~np.isin(candidate, blocked, assume_unique=True)]
            if valid.size:
                collected.append(valid)
                remaining -= valid.size
            attempts += 1
        if not collected:
            return np.random.randint(self.nentity, size=self.num_negatives)
        return np.concatenate(collected)[: self.num_negatives]

    def sample(self, batch_triples, mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Sample filtered negatives and subsampling weights for a batch.

        Returns: positive_sample [B,3], negative_sample [B,num_neg], subsampling_weight [B]
        """

        positive_sample = self._ensure_tensor_triples(batch_triples)
        batch_size = positive_sample.size(0)

        head = positive_sample[:, 0].cpu().numpy()
        relation = positive_sample[:, 1].cpu().numpy()
        tail = positive_sample[:, 2].cpu().numpy()

        subsampling_weight = self._subsampling_weights(head, relation, tail)
        oversample = max(self.num_negatives * 2, 64)
        negative_rows = []
        if mode == "head-batch":
            for r, t in zip(relation, tail):
                negative_rows.append(self._sample_filtered_negatives_row((int(r), int(t)), self.true_head, oversample))
        elif mode == "tail-batch":
            for h, r in zip(head, relation):
                negative_rows.append(self._sample_filtered_negatives_row((int(h), int(r)), self.true_tail, oversample))
        else:
            raise ValueError(f"Training batch mode {mode} not supported")

        negative_sample = torch.from_numpy(np.stack(negative_rows, axis=0)).long()
        return positive_sample, negative_sample, subsampling_weight, mode


def build_sampler(args, train_triples, model):
    """Construct a filtered 1-N subsampler for adversarial RotatE-style training."""

    nentity = getattr(args, 'nentity', getattr(args, 'ent_total', None))
    if nentity is None and hasattr(model, 'entity_embedding'):
        nentity = model.entity_embedding.size(0)
    if nentity is None:
        raise ValueError('`nentity` or `ent_total` is required for FilteredSubsampler')
    num_neg = int(getattr(args, 'n_sample', 1))
    return FilteredSubsampler(train_triples, int(nentity), num_neg)
