"""Pointwise logistic (softplus) loss for DaBR."""

import torch.nn.functional as F
import torch


def compute_softplus_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Trouillon logistic loss for higher-is-better DaBR scores (labels +1 / -1).

    The encoder returns a plausibility score where **higher is better** (correct
    triples are close to 0, incorrect ones are much lower). Eq. 13 in the paper is
    ``softplus(-Y * phi)`` with the same higher-is-better convention, which is
    equivalent to ``softplus(-scores * labels)`` here.

    Positives (Y=+1): ``softplus(-s)`` — loss decreases as *s* increases.
    Negatives (Y=-1): ``softplus(+s)`` — loss decreases as *s* decreases.
    This matches filtered link prediction, which ranks candidates by descending score.
    """

    scores = scores.view(-1)
    labels = labels.view(-1).to(scores.device)
    return F.softplus(-scores * labels).mean()
