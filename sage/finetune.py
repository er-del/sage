"""
SAGE Fine-Tuning
================
Provides two fine-tuning modes:

1. **Instruction tuning** — trains on instruction/response pairs with loss
   masked on the instruction portion.
2. **LoRA (Low-Rank Adaptation)** — injects small trainable matrices into
   attention layers while keeping the base model frozen.
"""

import math
import time
import copy
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
from typing import Optional, List

from .config import SageConfig
from .model import SageModel, CausalSelfAttention
from .data import SageTokenizer, create_instruction_batch
from .train import create_optimizer, get_lr, set_lr
from .utils import setup_logger, save_checkpoint

logger = setup_logger("sage.finetune")


# ===================================================================
# LoRA Implementation
# ===================================================================

class LoRALinear(nn.Module):
    """
    Wraps an existing ``nn.Linear`` with a low-rank adapter (A @ B).

    During fine-tuning only *A* and *B* are trained; the original weight
    is frozen.  After fine-tuning the adapter can be merged back into
    the original weight for zero-overhead inference.
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Low-rank matrices
        device, dtype = original.weight.device, original.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, device=device, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device, dtype=dtype))

        # Freeze the original weight
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """original(x) + x @ A @ B * scaling"""
        base_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

    def merge(self) -> nn.Linear:
        """Merge LoRA weights back into the original linear layer."""
        merged = copy.deepcopy(self.original)
        merged.weight.data += (self.lora_B.T @ self.lora_A.T).T * self.scaling
        merged.weight.requires_grad = True
        return merged


# ---------------------------------------------------------------------------
# LoRA injection / removal helpers
# ---------------------------------------------------------------------------

def inject_lora(model: SageModel, rank: int = 8, alpha: float = 16.0) -> SageModel:
    """
    Replace the Q, K, V, O projection layers in every attention block with
    LoRA-wrapped versions. Returns the same model (mutated in-place).
    """
    base_model = getattr(model, "module", model)
    for layer in base_model.layers:
        attn: CausalSelfAttention = layer.attn
        attn.wq = LoRALinear(attn.wq, rank=rank, alpha=alpha)
        attn.wk = LoRALinear(attn.wk, rank=rank, alpha=alpha)
        attn.wv = LoRALinear(attn.wv, rank=rank, alpha=alpha)
        attn.wo = LoRALinear(attn.wo, rank=rank, alpha=alpha)

    # Log trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA injected (rank={rank}).  Trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


def merge_lora(model: SageModel) -> SageModel:
    """
    Merge all LoRA adapters back into the base weights and replace the
    LoRALinear wrappers with plain nn.Linear modules.
    """
    base_model = getattr(model, "module", model)
    for layer in base_model.layers:
        attn: CausalSelfAttention = layer.attn
        for name in ("wq", "wk", "wv", "wo"):
            module = getattr(attn, name)
            if isinstance(module, LoRALinear):
                setattr(attn, name, module.merge())
    logger.info("LoRA weights merged into base model.")
    return model


# ===================================================================
# Instruction fine-tuning loop
# ===================================================================

def finetune_instruction(
    model: SageModel,
    config: SageConfig,
    samples: List[dict],
    total_steps: int = 200,
    use_lora: bool = True,
    lora_rank: int = 8,
    tokenizer: Optional[SageTokenizer] = None,
) -> SageModel:
    """
    Fine-tune the model on instruction/response pairs.

    Parameters
    ----------
    model : SageModel
    config : SageConfig
    samples : list[dict]
        Each dict must contain ``instruction`` and ``response`` string keys.
    total_steps : int
    use_lora : bool
        If True, inject LoRA adapters before training.
    lora_rank : int
    tokenizer : SageTokenizer, optional

    Returns
    -------
    SageModel — the fine-tuned model (LoRA merged if applicable).
    """
    device = config.device
    model = model.to(device)
    tok = tokenizer or SageTokenizer()

    if use_lora:
        model = inject_lora(model, rank=lora_rank)

    # ------- W&B Logging -------
    wandb.init(
        project=config.project_name,
        name=f"finetune-{time.strftime('%Y%m%d-%H%M')}",
        config=config.__dict__,
    )

    optimizer = create_optimizer(model, config)

    # AMP setup
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    model.train()
    pbar = tqdm(range(total_steps), desc="Fine-tuning", unit="step")
    accum_loss = 0.0

    for step in pbar:
        lr = get_lr(step, config, total_steps)
        set_lr(optimizer, lr)

        # Build a batch by sampling from the instruction dataset
        batch_size = min(config.batch_size, len(samples))
        import random
        batch_samples = random.choices(samples, k=batch_size)
        batch = create_instruction_batch(batch_samples, tok, max_len=config.max_seq_len)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(input_ids)
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = loss_mask[:, 1:].contiguous()

            # Compute per-token loss
            per_token_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )
            per_token_loss = per_token_loss.view(shift_labels.size())

            # Mask out instruction tokens so we only learn from responses
            masked_loss = (per_token_loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)

        scaler.scale(masked_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        accum_loss += masked_loss.item()

        if (step + 1) % 10 == 0:
            avg = accum_loss / 10
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}")
            logger.info(f"finetune step={step+1} | loss={avg:.4f}")
            wandb.log({
                "finetune/loss": avg,
                "finetune/lr": lr,
            }, step=step + 1)
            accum_loss = 0.0

    # Merge LoRA weights back for clean inference
    if use_lora:
        model = merge_lora(model)

    save_checkpoint(model, None, total_steps, config.checkpoint_dir, filename="sage_finetuned.pt")
    logger.info("Instruction fine-tuning complete. Checkpoint saved as sage_finetuned.pt")
    wandb.finish()
    return model


# ---------------------------------------------------------------------------
# Demo instruction samples (used when no dataset is provided)
# ---------------------------------------------------------------------------

DEMO_INSTRUCTION_SAMPLES = [
    {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
    {"instruction": "Explain gravity in simple terms.", "response": "Gravity is the force that pulls objects toward each other. The more mass an object has, the stronger its gravitational pull."},
    {"instruction": "Write a short poem about the ocean.", "response": "Waves crash upon the sandy shore,\nThe ocean's song forevermore.\nDeep blue stretching to the sky,\nSeagulls dance and clouds float by."},
    {"instruction": "What is 15 times 12?", "response": "15 times 12 equals 180."},
    {"instruction": "Summarize photosynthesis.", "response": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen, providing energy for the plant."},
    {"instruction": "Tell me a fun fact about space.", "response": "A day on Venus is longer than a year on Venus. It takes Venus 243 Earth days to rotate once on its axis but only 225 Earth days to orbit the Sun."},
    {"instruction": "How do airplanes fly?", "response": "Airplanes fly by generating lift through their wings. Air moves faster over the curved top of the wing than the flat bottom, creating lower pressure above and higher pressure below, which pushes the wing upward."},
    {"instruction": "What is machine learning?", "response": "Machine learning is a branch of artificial intelligence where computers learn patterns from data instead of being explicitly programmed, allowing them to make predictions or decisions."},
]
