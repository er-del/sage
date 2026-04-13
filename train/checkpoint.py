"""Checkpoint save, prune, and resume utilities."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.GradScaler | None,
    step: int,
    config: dict[str, object],
    output_dir: str,
    keep: int = 5,
) -> str:
    """Persist a resumable training checkpoint."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"ckpt_step_{step:07d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng_cpu": torch.get_rng_state(),
            "rng_gpu": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": config,
        },
        path,
    )
    _prune_old_checkpoints(output_dir, keep=keep)
    return str(path)


def load_latest_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scaler: torch.GradScaler | None,
    output_dir: str,
    device: str | torch.device,
) -> int:
    """Load the most recent checkpoint and return the step to resume from."""
    checkpoints = sorted(glob.glob(os.path.join(output_dir, "ckpt_step_*.pt")))
    if not checkpoints:
        return 0
    checkpoint = torch.load(checkpoints[-1], map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    torch.set_rng_state(checkpoint["rng_cpu"])
    if checkpoint.get("rng_gpu") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["rng_gpu"])
    return int(checkpoint["step"])


def _prune_old_checkpoints(output_dir: str, keep: int = 5) -> None:
    """Keep only the most recent checkpoints."""
    checkpoints = sorted(glob.glob(os.path.join(output_dir, "ckpt_step_*.pt")))
    for stale in checkpoints[:-keep]:
        os.remove(stale)
