"""Distributed and strategy routing helpers."""

from __future__ import annotations

import os

import torch


def get_training_strategy(model_size_b: float) -> dict[str, object]:
    """Choose a training mode based on the visible hardware."""
    n_gpus = torch.cuda.device_count()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    n_nodes = max(1, world_size // max(n_gpus, 1)) if n_gpus else 1
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    if not has_cuda and not has_mps:
        return {"mode": "cpu", "backend": None, "tp": 1, "pp": 1, "zero": 0}
    if has_mps:
        return {"mode": "mps-single", "backend": None, "tp": 1, "pp": 1, "zero": 0}

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if n_nodes > 1:
        if model_size_b <= 1.0:
            return {"mode": "ddp", "backend": "nccl", "tp": 1, "pp": 1, "zero": 2}
        return {"mode": "fsdp", "backend": "nccl", "tp": 2, "pp": 1, "zero": 3}
    if n_gpus > 1:
        if model_size_b <= 1.0:
            return {"mode": "ddp", "backend": "nccl", "tp": 1, "pp": 1, "zero": 1}
        return {"mode": "fsdp", "backend": "nccl", "tp": 2, "pp": 1, "zero": 2}
    if vram_gb >= 40:
        return {"mode": "single", "backend": None, "tp": 1, "pp": 1, "zero": 0}
    if vram_gb >= 24:
        return {"mode": "single", "backend": None, "tp": 1, "pp": 1, "zero": 1}
    if vram_gb >= 16:
        return {"mode": "single", "backend": None, "tp": 1, "pp": 1, "zero": 2}
    return {"mode": "single", "backend": None, "tp": 1, "pp": 1, "zero": 3}
