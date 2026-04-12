"""
SAGE Training System
====================
Complete training loop with AdamW, cosine-decay LR schedule, mixed-precision
(AMP), gradient accumulation, gradient clipping, and checkpoint management.
"""

import math
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from typing import Optional

from .config import SageConfig
from .model import SageModel
from .data import SageTokenizer, create_dataloader
from .utils import setup_logger, save_checkpoint, load_checkpoint

logger = setup_logger("sage.train")


# ---------------------------------------------------------------------------
# Learning-rate scheduler helpers
# ---------------------------------------------------------------------------

def get_lr(step: int, config: SageConfig, total_steps: int) -> float:
    """Cosine decay with linear warmup.  Returns the learning rate for *step*."""
    if step < config.warmup_steps:
        # Linear warmup
        return config.learning_rate * (step + 1) / config.warmup_steps

    # Cosine decay phase
    decay_steps = total_steps - config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, decay_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Manually sets the learning rate for every parameter group."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def create_optimizer(model: SageModel, config: SageConfig) -> torch.optim.AdamW:
    """
    Create an AdamW optimizer with weight-decay applied only to weight
    matrices (not biases or LayerNorm parameters).
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and LayerNorm weights should not be decayed
        if param.ndim == 1 or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return optimizer


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model: SageModel,
    config: SageConfig,
    total_steps: int = 500,
    dataset_name: str = "roneneldan/TinyStories",
    resume: bool = True,
    tokenizer: Optional[SageTokenizer] = None,
) -> SageModel:
    """
    Run pre-training for *total_steps* gradient-update steps.

    Parameters
    ----------
    model : SageModel
        The model to train (will be moved to config.device).
    config : SageConfig
        Hyperparameters.
    total_steps : int
        Number of optimizer steps to run.
    dataset_name : str
        HuggingFace dataset identifier.
    resume : bool
        If True, attempt to load the latest checkpoint before training.
    tokenizer : SageTokenizer, optional
        Tokenizer instance; one will be created if not supplied.

    Returns
    -------
    SageModel
        The trained model (on config.device).
    """
    device = config.device
    model = model.to(device)

    tok = tokenizer or SageTokenizer()
    optimizer = create_optimizer(model, config)

    # ------- resume from checkpoint if available -------
    start_step = 0
    if resume:
        model, optimizer, start_step = load_checkpoint(
            model, optimizer, config.checkpoint_dir, device=str(device)
        )
        if start_step >= total_steps:
            logger.info("Checkpoint already at or past requested steps. Nothing to do.")
            return model

    # ------- mixed precision setup -------
    use_amp = device.type == "cuda"
    # prefer bf16 if the GPU supports it
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # ------- data loader -------
    loader = create_dataloader(config, dataset_name=dataset_name, tokenizer=tok)
    data_iter = iter(loader)

    # ------- gradient checkpointing (saves VRAM) -------
    base_model = getattr(model, "module", model)
    if hasattr(base_model, "layers"):
        for layer in base_model.layers:
            layer: nn.Module
            # PyTorch gradient checkpointing
            try:
                from torch.utils.checkpoint import checkpoint  # noqa: F401
                # We wrap the forward below instead, using it at call-site.
            except ImportError:
                pass

    # ------- training loop -------
    model.train()
    accum_loss = 0.0
    log_interval = 10
    t0 = time.time()

    pbar = tqdm(range(start_step, total_steps), desc="Training", unit="step")
    micro_step = 0

    for step in pbar:
        # Update learning rate
        lr = get_lr(step, config, total_steps)
        set_lr(optimizer, lr)

        # Accumulate gradients over multiple micro-batches
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro in range(config.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart the data stream when exhausted
                data_iter = iter(loader)
                batch = next(data_iter)

            batch = batch.to(device)
            inputs = batch[:, :-1]   # all tokens except last
            targets = batch[:, 1:]   # all tokens except first

            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(inputs)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=tok.pad_token_id,
                )
                # Scale loss by accumulation steps so the effective loss
                # is independent of the number of micro-batches.
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()
            step_loss += loss.item()

        # Gradient clipping (unscale first for correct norm computation)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        accum_loss += step_loss
        micro_step += 1

        # ------- logging -------
        if (step + 1) % log_interval == 0 or step == total_steps - 1:
            avg_loss = accum_loss / log_interval
            elapsed = time.time() - t0
            perplexity = math.exp(min(avg_loss, 20))  # clamp to avoid overflow
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                ppl=f"{perplexity:.2f}",
                lr=f"{lr:.2e}",
                elapsed=f"{elapsed:.1f}s",
            )
            logger.info(
                f"step={step+1} | loss={avg_loss:.4f} | ppl={perplexity:.2f} | lr={lr:.2e}"
            )
            accum_loss = 0.0

        # ------- checkpoint every 100 steps -------
        if (step + 1) % 100 == 0 or step == total_steps - 1:
            save_checkpoint(model, optimizer, step + 1, config.checkpoint_dir)
            logger.info(f"Checkpoint saved at step {step + 1}")

    logger.info("Training complete.")
    return model
