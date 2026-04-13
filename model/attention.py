"""Grouped-query attention with SDPA and KV-cache support."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from model.config import ModelConfig
from model.rope import apply_rope


def repeat_kv(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Expand KV heads to match the number of query heads."""
    if num_groups == 1:
        return x
    batch, kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, kv_heads, num_groups, seq_len, head_dim)
    return x.reshape(batch, kv_heads * num_groups, seq_len, head_dim)


class GQAAttention(nn.Module):
    """Fused-QKV grouped-query attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attn_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads
        qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(config.d_model, qkv_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute causal self-attention and return an updated KV cache."""
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q_end = self.num_heads * self.head_dim
        k_end = q_end + self.num_kv_heads * self.head_dim
        q, k, v = qkv.split((q_end, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim), dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q_rope, k_rope = apply_rope(q, repeat_kv(k, self.num_groups), cos, sin)
        k = k_rope[:, :: self.num_groups, :, :]

        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)

        expanded_k = repeat_kv(k, self.num_groups)
        expanded_v = repeat_kv(v, self.num_groups)
        attn_output = F.scaled_dot_product_attention(
            q_rope,
            expanded_k,
            expanded_v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=past_key_value is None,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model)
        return self.out_proj(attn_output), (k, v)
