"""InfoNCE loss computation for models: SimKGC."""

from dataclasses import dataclass
import torch


@dataclass
class ModelOutput:
	"""Structured output from the model's forward pass, containing all necessary components for InfoNCE loss computation."""
	
	logits: torch.Tensor
	labels: torch.Tensor
	inv_t: torch.Tensor
	hr_vector: torch.Tensor
	tail_vector: torch.Tensor


def compute_infonce_logits(query_vec: torch.Tensor, candidate_vec: torch.Tensor, temp: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
	"""Compute the core InfoNCE logits shared across contrastive models."""

	logits = torch.mm(query_vec, candidate_vec.t())
	if margin > 0:
		logits.diagonal().sub_(margin)
	return logits * torch.exp(temp)


def compute_logits(model, output_dict: dict, batch_dict: dict) -> ModelOutput:
	"""Compute logits and labels for InfoNCE loss based on model outputs and batch information, applying necessary masking and adjustments."""
	
	hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
	batch_size = hr_vector.size(0)
	labels = torch.arange(batch_size).to(hr_vector.device)

	logits = compute_infonce_logits(
		query_vec=hr_vector,
		candidate_vec=tail_vector,
		temp=model.log_inv_t,
		margin=model.add_margin if model.training else 0.0,
	)

	return ModelOutput(
		logits=logits,
		labels=labels,
		inv_t=model.log_inv_t.detach().exp(),
		hr_vector=hr_vector.detach(),
		tail_vector=tail_vector.detach(),
    )
