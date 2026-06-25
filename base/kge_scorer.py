"""Protocol and base class for pure-tensor KGE score functions."""

from __future__ import annotations

import torch
import torch.nn as nn


class KGEScorer(nn.Module):
	"""Pure tensor KGE logic — no embedding index lookups.

	Subclasses implement ``score_spo`` / ``score_sp_`` / ``score_po_`` for link
	prediction.  Optional hooks:

	* ``build_query`` — tail-prediction query vectors (cosine LP when enabled).
	* ``au_representations`` — (query, tail, head) vectors for KGAU training.
	* ``au_entity_embeddings`` — optional full-table entity vectors for uniformity.
	"""

	bidirectional_score_batch: bool = False
	kgau_alignment_mode: str | None = None

	def build_query(self, h_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
		"""Build tail-prediction query vectors (DistMult default: ``h * r``)."""

		return h_emb * r_emb

	def build_po_query(self, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		"""Build head-prediction query vectors (DistMult default: ``t * r``)."""

		return t_emb * r_emb

	def au_representations(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		*,
		predict_head: bool = False,
		**kwargs,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Return (query, align_target, head_entity) for alignment / uniformity losses.

		Tail prediction (default): align ``build_query(h, r)`` with the tail entity.
		Head prediction: align ``build_po_query(r, t)`` with the head entity (GB-Magic head-batch).
		"""

		if predict_head:
			return self.build_po_query(r_emb, t_emb), h_emb, h_emb
		return self.build_query(h_emb, r_emb), t_emb, h_emb
