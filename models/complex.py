"""ComplEx scorer and model (Hadamard ``score_emb``)."""

import torch

from base.model import KGEScorer, KGEModel


def build_scorer(args) -> 'ComplExScorer':
	return ComplExScorer(args)


class ComplExScorer(KGEScorer):
	"""ComplEx score function via a single Hadamard ``score_emb`` (Eq. 11).

	``combine`` modes: ``hrt``, ``hr_``, ``_rt``, ``hr_c``, ``_rt_c``.
	"""

	bidirectional_score_batch = True

	def __init__(self, args=None):
		super().__init__()
		self.args = args

	def supports_candidate_scoring(self) -> bool:
		return True

	def score_emb(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		combine: str,
		**kwargs,
	) -> torch.Tensor:
		"""Fast ComplEx scores via Hadamard products (Trouillon et al. Eq. 11)."""

		del kwargs
		n = r_emb.size(0)

		# Split relation and object into real (first half) and imaginary (second half).
		r_emb_re, r_emb_im = (part.contiguous() for part in r_emb.chunk(2, dim=-1))
		t_emb_re, t_emb_im = (part.contiguous() for part in t_emb.chunk(2, dim=-1))

		# Column blocks for each required combination.
		h_all = torch.cat((h_emb, h_emb), dim=-1)  # re, im, re, im
		r_all = torch.cat((r_emb_re, r_emb, -r_emb_im), dim=-1)  # re, re, im, -im
		t_all = torch.cat((t_emb, t_emb_im, t_emb_re), dim=-1)  # re, im, im, re

		if combine == 'hrt':
			out = (h_all * t_all * r_all).sum(dim=-1)
			return out.view(n, -1)
		if combine == 'hr_':
			return (h_all * r_all).mm(t_all.transpose(0, 1))
		if combine == '_rt':
			return (r_all * t_all).mm(h_all.transpose(0, 1))
		if combine == 'hr_c':
			# t_emb is [B, C, D]
			return (h_all * r_all).unsqueeze(1).bmm(t_all.transpose(1, 2)).squeeze(1)
		if combine == '_rt_c':
			# h_emb is [B, C, D]
			return (r_all * t_all).unsqueeze(1).bmm(h_all.transpose(1, 2)).squeeze(1)
		raise ValueError(f'cannot handle combine="{combine}"')


class ComplExModel(KGEModel):
	"""Bind lookup embedders to ``ComplExScorer`` (``scorers`` length 1 by default).

	KGAU encoders (ComplEx does **not** fold the relation into the target):

	* ``query_encoder(h, r)`` → complex product ``h ∘ r`` (tail prediction)
	* ``inverse_query_encoder(r, t)`` → ``conj(r) ∘ t`` (head prediction)
	* ``target_encoder`` → raw head or tail entity embedding
	"""

	target_uses_relation = False

	def __init__(
		self,
		ent_embedder,
		rel_embedder,
		scorers=None,
		args=None,
		aux_embedders=None,
	):
		if scorers is None:
			scorers = [ComplExScorer(args)]
		super().__init__(
			ent_embedder,
			rel_embedder,
			scorers=scorers,
			args=args,
			aux_embedders=aux_embedders,
		)

	def query_encoder(self, h: torch.Tensor, r: torch.Tensor, **kwargs) -> torch.Tensor:
		"""Tail-prediction query: ``h ∘ r`` in the concatenated complex space."""

		del kwargs
		h_re, h_im = torch.chunk(self.embed_h(h), 2, dim=-1)
		r_re, r_im = torch.chunk(self.embed_r(r), 2, dim=-1)
		return torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)

	def inverse_query_encoder(self, r: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
		"""Head-prediction query: ``conj(r) ∘ t``."""

		del kwargs
		r_re, r_im = torch.chunk(self.embed_r(r), 2, dim=-1)
		t_re, t_im = torch.chunk(self.embed_t(t), 2, dim=-1)
		return torch.cat([r_re * t_re + r_im * t_im, r_re * t_im - r_im * t_re], dim=-1)

	def target_encoder(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		*,
		predict_head: bool = False,
		**kwargs,
	) -> torch.Tensor:
		"""Alignment target: tail entity (tail-batch) or head entity (head-batch)."""

		del r, kwargs
		if predict_head:
			return self.embed_h(h)
		return self.embed_t(t)
