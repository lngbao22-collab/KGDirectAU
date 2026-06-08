"""Pointwise logistic (softplus) loss for DaBR."""

import torch.nn.functional as F
import torch


def compute_softplus_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """DaBR regularized logistic loss: Softplus(scores * labels), labels are +1 / -1.

    Matches the official DaBR objective ``mean(Softplus(score * y))`` (Eq. 13 in the
    paper). The encoder returns ``score = -dot(...) - para*||...||1``, so a correct
    triple already attains a high (close to 0) score; multiplying by the label and
    passing it straight through Softplus is what drives positives up and negatives
    down, consistent with the ``score > target`` ranking used at evaluation time.
    """

    scores = scores.view(-1)
    labels = labels.view(-1).to(scores.device)
    return F.softplus(scores * labels).mean()
