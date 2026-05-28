"""Abstract model interface shared by KG models."""

from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(nn.Module, ABC):
	"""Abstract base class for KG models. Defines the expected interface for forward passes and logit computation."""

	@abstractmethod
	def forward(self, *args, **kwargs):
		"""Run a forward pass and return model-specific outputs."""

	@abstractmethod
	def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
		"""Convert model outputs into logits/labels for the training objective."""