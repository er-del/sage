"""Transformer block for the dense SAGE model."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from model.attention import GQAAttention
from model.config import ModelConfig
from model.mlp import SwiGLUMLP
from model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with attention and SwiGLU."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = GQAAttention(config)
        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with residual connections."""
        attn_output, present = self.attn(self.norm1(hidden_states), cos, sin, past_key_value=past_key_value)
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states, present
