"""KV-cache helpers for inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KVCache:
    """Stores per-layer key/value tensors."""

    entries: list[tuple[torch.Tensor, torch.Tensor]]

    @classmethod
    def empty(cls, num_layers: int) -> "KVCache":
        """Create an empty cache placeholder."""
        return cls(entries=[None] * num_layers)  # type: ignore[list-item]

    def append(self, layer_index: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store one layer's key/value pair."""
        self.entries[layer_index] = (key, value)
