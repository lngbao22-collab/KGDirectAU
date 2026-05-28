"""Training strategy for listwise DistMult and ComplEx models."""

from typing import Callable, Optional

import torch
from torch.optim import Adam

from models.losses.infonce_loss import compute_listwise_loss


class SoftmaxStrategy(object):
	"""Listwise softmax training loop for KG encoders."""

	def __init__(self, encoder, args):
		self.encoder = encoder
		self.args = args
		lam = getattr(args, 'lam', getattr(args, 'weight_decay', 0.0))
		n_batch = getattr(args, 'n_batch', getattr(args, 'batch_size', 1))
		self.weight_decay = lam / max(n_batch, 1)
		self.optimizer = Adam(self.encoder.parameters(), weight_decay=self.weight_decay)
		self.device = next(self.encoder.parameters()).device

	def _iter_batches(self, src, rel, dst, batch_size) -> (torch.Tensor, torch.Tensor, torch.Tensor):
		"""Yield batches of triples from the provided tensors."""

		for start in range(0, len(src), batch_size):
			end = start + batch_size
			yield src[start:end], rel[start:end], dst[start:end]

	def pretrain(self, train_data, sampler, tester: Optional[Callable[[], float]] = None) -> float:
		"""Train the encoder with Bernoulli listwise negatives."""

		src, rel, dst = train_data
		n_train = len(src)
		n_epoch = getattr(self.args, 'n_epoch', getattr(self.args, 'epochs', 1))
		n_batch = getattr(self.args, 'n_batch', getattr(self.args, 'batch_size', 1))
		sample_freq = getattr(self.args, 'sample_freq', 1)
		best_perf = 0.0

		for epoch in range(n_epoch):
			if epoch % sample_freq == 0:
				src_corrupted, rel_corrupted, dst_corrupted = sampler.corrupt(src, rel, dst)
			else:
				src_corrupted, rel_corrupted, dst_corrupted = src, rel, dst

			epoch_loss = 0.0
			for ss, rs, ts in self._iter_batches(src_corrupted, rel_corrupted, dst_corrupted, n_batch):
				ss = ss.to(self.device)
				rs = rs.to(self.device)
				ts = ts.to(self.device)
				scores = self.encoder.forward(ss, rs, ts)
				truth = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
				loss = compute_listwise_loss(scores, truth)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				epoch_loss += loss.item() * ss.size(0)

			if tester is not None:
				perf = tester()
				if perf > best_perf:
					best_perf = perf

		return best_perf if tester is not None else epoch_loss / max(n_train, 1)