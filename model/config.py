"""Model configuration for SAGE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Configuration for the dense SAGE decoder-only transformer."""

    name: str = "sage-1b"
    num_layers: int = 24
    d_model: int = 2048
    num_attn_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    ffn_hidden_dim: int = 5632
    vocab_size: int = 50_000
    context_length: int = 4096
    rope_base_frequency: int = 500_000
    rope_scaling_factor: float = 1.0
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    rms_norm_eps: float = 1.0e-5
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.num_attn_heads * self.head_dim != self.d_model:
            raise ValueError("num_attn_heads * head_dim must equal d_model.")
        if self.num_attn_heads % self.num_kv_heads != 0:
            raise ValueError("num_attn_heads must be divisible by num_kv_heads.")
        if self.ffn_hidden_dim % 256 != 0:
            raise ValueError("ffn_hidden_dim must be a multiple of 256.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        """Load a config from YAML."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a dict."""
        return asdict(self)
