"""SwiGLU feed-forward module."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from model.config import ModelConfig


class SwiGLUMLP(nn.Module):
    """Bias-free SwiGLU feed-forward network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU and project back to the model width."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
