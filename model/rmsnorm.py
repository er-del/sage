"""RMSNorm implementation used by SAGE."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root mean square normalization with float32 accumulation."""

    def __init__(self, dim: int, eps: float = 1.0e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension and cast back to the input dtype."""
        if x.ndim < 2:
            raise ValueError("RMSNorm expects at least 2 dimensions.")
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.eps)
        return (normalized.to(dtype=x.dtype)) * self.weight
