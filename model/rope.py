"""Rotary positional embedding helpers."""

from __future__ import annotations

import torch


def _scaled_positions(seq_len: int, scaling_factor: float, device: torch.device) -> torch.Tensor:
    """Apply a simple YaRN-style position scaling factor."""
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    if scaling_factor > 1.0:
        positions = positions / scaling_factor
    return positions


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base_frequency: int = 500_000,
    scaling_factor: float = 1.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for RoPE."""
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE.")
    device = device or torch.device("cpu")
    positions = _scaled_positions(seq_len, scaling_factor, device)
    inv_freq = 1.0 / (base_frequency ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension in pairs."""
    even = x[..., ::2]
    odd = x[..., 1::2]
    rotated = torch.stack((-odd, even), dim=-1)
    return rotated.flatten(start_dim=-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    if q.shape != k.shape:
        raise ValueError("q and k must share the same shape for RoPE application.")
    seq_len = q.size(-2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0).repeat_interleave(2, dim=-1)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0).repeat_interleave(2, dim=-1)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out
