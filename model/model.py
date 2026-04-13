"""Full dense decoder-only transformer model for SAGE."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from model.block import TransformerBlock
from model.config import ModelConfig
from model.rope import build_rope_cache
from model.rmsnorm import RMSNorm


class SageTransformer(nn.Module):
    """A dense Llama-style decoder-only transformer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        cos, sin = build_rope_cache(
            seq_len=config.context_length,
            head_dim=config.head_dim,
            base_frequency=config.rope_base_frequency,
            scaling_factor=config.rope_scaling_factor,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Apply scaled initialization to the model."""
        embed_std = 1.0 / math.sqrt(self.config.d_model)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=embed_std)
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            std = self.config.initializer_range
            if module is self.lm_head and self.config.tie_word_embeddings:
                continue
            if module.out_features == self.config.d_model:
                std = std / math.sqrt(2 * self.config.num_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Return logits and the updated KV cache."""
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        past_key_values = past_key_values or [None] * self.config.num_layers
        start = 0
        if past_key_values[0] is not None:
            start = past_key_values[0][0].size(-2)
        cos = self.rope_cos[start : start + seq_len].to(hidden_states.device)
        sin = self.rope_sin[start : start + seq_len].to(hidden_states.device)
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, past in zip(self.layers, past_key_values):
            hidden_states, present = layer(hidden_states, cos, sin, past_key_value=past)
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, presents
