"""Pure cosine scorer and contrastive training state for SimKGC."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from base.kge_scorer import KGEScorer


class ContrastiveTrainingState(nn.Module):
	"""Training-only InfoNCE parameters and pre-batch memory (not used at LP eval)."""

	def __init__(self, args: Any, hidden_size: int):
		super().__init__()
		self.args = args
		info_nce_t = getattr(args, 'infonce_t', None)
		if info_nce_t is None:
			info_nce_t = getattr(args, 't', None)
		if info_nce_t is None:
			info_nce_t = 0.05
		info_nce_t = float(info_nce_t)
		self.log_inv_t = nn.Parameter(
			torch.tensor(1.0 / info_nce_t).log(),
			requires_grad=bool(getattr(args, 'finetune_t', True)),
		)
		self.add_margin = float(getattr(args, 'additive_margin', 0.0))
		self.batch_size = int(getattr(args, 'batch_size', 512))
		pre_batch = getattr(args, 'pre_batch', None)
		self.pre_batch = int(pre_batch if pre_batch is not None else 0)
		num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
		random_vector = torch.randn(num_pre_batch_vectors, hidden_size)
		self.register_buffer(
			'pre_batch_vectors',
			nn.functional.normalize(random_vector, dim=1),
			persistent=False,
		)
		self.offset = 0
		self.pre_batch_exs: list = [None for _ in range(num_pre_batch_vectors)]


def build_contrastive_state(args, hidden_size: int) -> ContrastiveTrainingState:
	return ContrastiveTrainingState(args, hidden_size)


class SimKGCScorer(KGEScorer):
	"""Cosine similarity scorer on L2-normalized query and entity vectors."""

	kgau_alignment_mode = 'cosine'

	def __init__(self, args=None):
		super().__init__()
		self.args = args

	def score_spo(
		self,
		q_emb: torch.Tensor,
		_r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		return torch.sum(q_emb * t_emb, dim=-1)

	def score_sp_(
		self,
		q_emb: torch.Tensor,
		_r_emb: torch.Tensor,
		all_t_embs: torch.Tensor,
	) -> torch.Tensor:
		return torch.mm(q_emb, all_t_embs.t())

	def build_query(self, q_emb: torch.Tensor, _r_emb: torch.Tensor) -> torch.Tensor:
		return q_emb

	def au_representations(
		self,
		q_emb: torch.Tensor,
		_r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		head_emb: torch.Tensor | None = None,
		**kwargs,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		if head_emb is None:
			head_emb = t_emb
		return q_emb, t_emb, head_emb


def build_scorer(args) -> SimKGCScorer:
	return SimKGCScorer(args)


def build_model(args):
	"""Backward-compatible factory delegating to the unified builder."""

	from models.builder import build_model as assemble_model

	return assemble_model(args)
