"""Standalone text embedder backed by a HuggingFace encoder."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEmbedder(nn.Module):
	"""Encode tokenized text inputs into dense CLS vectors."""

	def __init__(self, pretrained_model_name_or_path: str, args: Any | None = None):
		super().__init__()
		self.args = args
		self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path)

	def forward(self, text_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
		"""Return the pooled [CLS] representation for a tokenized text batch."""

		outputs = self.encoder(**dict(text_dict), return_dict=True)
		return outputs.last_hidden_state[:, 0, :]

