"""Filtered 1-N negative sampler adapted from RotatE TrainDataset."""

import numpy as np
import torch


class FilteredSubsampler:
    """Filtered 1-N negative sampler for RotatE-style training."""

    def __init__(self, triples, nentity: int, num_negatives: int, num_negatives_h: int | None = None):
        self.nentity = int(nentity)
        self.num_negatives_tail = int(num_negatives)
        self.num_negatives_head = int(num_negatives if num_negatives_h is None else num_negatives_h)
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

    def _sample_filtered_negatives_row(
        self,
        key: tuple[int, int],
        filter_dict: dict,
        oversample: int,
        *,
        num_negatives: int,
    ) -> np.ndarray:
        """Draw filtered negatives for one row, resampling only when the first pass is short."""

        blocked = filter_dict.get(key)
        if blocked is None or blocked.size == 0:
            return np.random.randint(self.nentity, size=num_negatives, dtype=np.int64)

        pool_size = max(oversample, num_negatives + blocked.size, num_negatives * 2)
        candidates = np.random.randint(self.nentity, size=pool_size, dtype=np.int64)
        valid = candidates[~np.isin(candidates, blocked, assume_unique=True)]
        if valid.size >= num_negatives:
            return valid[:num_negatives]

        collected = [valid] if valid.size else []
        remaining = num_negatives - sum(part.size for part in collected)
        attempts = 0
        while remaining > 0 and attempts < 3:
            candidate = np.random.randint(self.nentity, size=max(oversample, remaining * 2), dtype=np.int64)
            extra = candidate[~np.isin(candidate, blocked, assume_unique=True)]
            if extra.size:
                collected.append(extra)
                remaining -= extra.size
            attempts += 1
        if not collected:
            return np.random.randint(self.nentity, size=num_negatives, dtype=np.int64)
        return np.concatenate(collected)[:num_negatives]

    def sample(self, batch_triples, mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Sample filtered negatives and subsampling weights for a batch.

        Returns: positive_sample [B,3], negative_sample [B,num_neg], subsampling_weight [B]
        """

        positive_sample = self._ensure_tensor_triples(batch_triples)
        batch_size = positive_sample.size(0)

        if positive_sample.is_cuda:
            head = positive_sample[:, 0].detach().cpu().numpy()
            relation = positive_sample[:, 1].detach().cpu().numpy()
            tail = positive_sample[:, 2].detach().cpu().numpy()
        else:
            head = positive_sample[:, 0].numpy()
            relation = positive_sample[:, 1].numpy()
            tail = positive_sample[:, 2].numpy()

        subsampling_weight = self._subsampling_weights(head, relation, tail)
        if mode == "head-batch":
            num_negatives = self.num_negatives_head
        elif mode == "tail-batch":
            num_negatives = self.num_negatives_tail
        else:
            raise ValueError(f"Training batch mode {mode} not supported")
        oversample = max(num_negatives * 2, 64)
        negative_sample = np.empty((batch_size, num_negatives), dtype=np.int64)
        if mode == "head-batch":
            for i, (r, t) in enumerate(zip(relation, tail)):
                negative_sample[i] = self._sample_filtered_negatives_row(
                    (int(r), int(t)), self.true_head, oversample, num_negatives=num_negatives
                )
        elif mode == "tail-batch":
            for i, (h, r) in enumerate(zip(head, relation)):
                negative_sample[i] = self._sample_filtered_negatives_row(
                    (int(h), int(r)), self.true_tail, oversample, num_negatives=num_negatives
                )
        else:
            raise ValueError(f"Training batch mode {mode} not supported")

        return positive_sample, torch.from_numpy(negative_sample).long(), subsampling_weight, mode


def build_sampler(args, train_triples, model):
    """Construct a filtered 1-N subsampler for adversarial RotatE-style training."""

    from models.builder import _resolve_nentity

    nentity = _resolve_nentity(args, model)
    n_sample_t = getattr(args, 'n_sample_t', None)
    n_sample_h = getattr(args, 'n_sample_h', None)
    n_sample = getattr(args, 'n_sample', None)
    if n_sample_t is not None or n_sample_h is not None:
        num_neg_t = int(n_sample_t if n_sample_t is not None else (n_sample or 1))
        num_neg_h = int(n_sample_h if n_sample_h is not None else (n_sample or 1))
        return FilteredSubsampler(train_triples, int(nentity), num_neg_t, num_negatives_h=num_neg_h)
    num_neg = int(n_sample or 1)
    return FilteredSubsampler(train_triples, int(nentity), num_neg)
