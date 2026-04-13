"""Loss helpers for packed next-token prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Compute next-token cross entropy with an explicit mask."""
    vocab = logits.size(-1)
    loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1), reduction="none")
    loss = loss * loss_mask.view(-1)
    denom = torch.clamp(loss_mask.sum(), min=1.0)
    return loss.sum() / denom
