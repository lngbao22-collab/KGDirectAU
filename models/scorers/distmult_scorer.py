"""Pure DistMult scorer operating on raw tensors only."""

import torch

from base.model import KGEScorer


def build_scorer(args) -> 'DistMultScorer':
	return DistMultScorer(args)


class DistMultScorer(KGEScorer):
	"""DistMult score function with explicit 1-to-1 and 1-vs-All tensor paths."""

	bidirectional_score_batch = True

	def __init__(self, args=None):
		super().__init__()
		self.args = args

	def score_hrt(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return standard DistMult scores for matching batches of triples."""

		return torch.sum(h_emb * r_emb * t_emb, dim=-1)

	def score_rt(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return DistMult scores for head candidates with fixed (relation, tail)."""

		return self.score_hrt(h_emb, r_emb, t_emb)

	def score_hrt_candidates(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		"""Score many tail candidates per row: ``h_emb,r_emb`` are [B, D], ``t_emb`` is [B, C, D]."""

		query = h_emb * r_emb
		return torch.bmm(query.unsqueeze(1), t_emb.transpose(1, 2)).squeeze(1)

	def score_rt_candidates(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		"""Score many head candidates per row: ``h_emb`` is [B, C, D], ``r_emb,t_emb`` are [B, D]."""

		query = t_emb * r_emb
		return torch.bmm(query.unsqueeze(1), h_emb.transpose(1, 2)).squeeze(1)

	def score_hr_(self, h_emb: torch.Tensor, r_emb: torch.Tensor, all_t_embs: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all DistMult scores using LibKGE-style hr_ broadcasting."""

		return torch.mm(h_emb * r_emb, all_t_embs.t())

	def score_rt_(self, all_h_embs: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Return 1-vs-all DistMult head scores for each (relation, tail) query."""

		return torch.mm(t_emb * r_emb, all_h_embs.t())

	def build_query(self, h_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
		return h_emb * r_emb
