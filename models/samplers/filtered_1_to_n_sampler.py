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

    def sample(self, batch_triples, mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Sample filtered negatives and subsampling weights for a batch.

        Returns: positive_sample [B,3], negative_sample [B,num_neg], subsampling_weight [B]
        """

        positive_sample = self._ensure_tensor_triples(batch_triples)
        batch_size = positive_sample.size(0)

        head = positive_sample[:, 0].cpu().numpy()
        relation = positive_sample[:, 1].cpu().numpy()
        tail = positive_sample[:, 2].cpu().numpy()

        subsampling_weight = []
        for h, r, t in zip(head, relation, tail):
            weight = self.count[(int(h), int(r))] + self.count[(int(t), -int(r) - 1)]
            subsampling_weight.append(weight)
        subsampling_weight = torch.sqrt(1.0 / torch.tensor(subsampling_weight, dtype=torch.float))

        negative_samples = []
        for h, r, t in zip(head, relation, tail):
            negative_sample_list = []
            negative_sample_size = 0
            while negative_sample_size < self.num_negatives:
                candidate = np.random.randint(self.nentity, size=self.num_negatives * 2)
                if mode == "head-batch":
                    mask = np.in1d(candidate, self.true_head.get((int(r), int(t)), np.array([], dtype=np.int64)), assume_unique=True, invert=True)
                elif mode == "tail-batch":
                    mask = np.in1d(candidate, self.true_tail.get((int(h), int(r)), np.array([], dtype=np.int64)), assume_unique=True, invert=True)
                else:
                    raise ValueError(f"Training batch mode {mode} not supported")
                candidate = candidate[mask]
                negative_sample_list.append(candidate)
                negative_sample_size += candidate.size

            negative_sample = np.concatenate(negative_sample_list)[: self.num_negatives]
            negative_samples.append(torch.LongTensor(negative_sample))

        negative_sample = torch.stack(negative_samples, dim=0)
        return positive_sample, negative_sample, subsampling_weight, mode
