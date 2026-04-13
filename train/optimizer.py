"""Optimizer and scheduler factories."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ScheduleConfig:
    """Training schedule settings."""

    peak_learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1.0e-8
    total_steps: int = 25_000


def create_optimizer(model: torch.nn.Module, config: ScheduleConfig) -> torch.optim.Optimizer:
    """Create an AdamW optimizer with correct weight-decay exclusions."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.peak_learning_rate,
        betas=config.betas,
        eps=config.adam_eps,
    )


def lr_lambda(current_step: int, config: ScheduleConfig) -> float:
    """Warm up linearly and then decay with cosine."""
    if current_step < config.warmup_steps:
        return float(current_step + 1) / float(max(1, config.warmup_steps))
    progress = (current_step - config.warmup_steps) / float(max(1, config.total_steps - config.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = config.min_learning_rate / config.peak_learning_rate
    return floor + (1.0 - floor) * cosine


def create_scheduler(optimizer: torch.optim.Optimizer, config: ScheduleConfig) -> torch.optim.lr_scheduler.LambdaLR:
    """Create the training LR scheduler."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, config))
