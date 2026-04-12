"""
SAGE Optimization Layer
=======================
Post-training quantization (INT8), optional pruning, and knowledge-distillation
loss utilities.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Optional

from .model import SageModel
from .config import SageConfig
from .utils import setup_logger

logger = setup_logger("sage.optimize")


# ===================================================================
# INT8 Dynamic Quantization
# ===================================================================

def quantize_int8(model: SageModel) -> nn.Module:
    """
    Apply dynamic INT8 quantization to all Linear layers in the model.

    This reduces model size by ~2-4x and can speed up CPU inference.
    The model is moved to CPU before quantization because PyTorch's
    dynamic quantization only supports CPU tensors.

    Returns
    -------
    nn.Module — the quantized model (on CPU).
    """
    base_model = getattr(model, "module", model)
    base_model = base_model.cpu().eval()

    quantized = torch.quantization.quantize_dynamic(
        base_model,
        {nn.Linear},         # quantize all linear layers
        dtype=torch.qint8,
    )

    # Report size reduction
    orig_size = sum(p.numel() * p.element_size() for p in base_model.parameters())
    # Quantized parameters may not report element_size correctly,
    # so we estimate based on INT8 = 1 byte per weight.
    quant_size = sum(p.numel() for p in quantized.parameters())  # * 1 byte
    logger.info(
        f"Quantization complete.  "
        f"Original: {orig_size / 1e6:.1f} MB → Quantized: ~{quant_size / 1e6:.1f} MB (INT8)"
    )
    return quantized


# ===================================================================
# Weight Pruning
# ===================================================================

def prune_model(model: SageModel, amount: float = 0.3) -> SageModel:
    """
    Apply unstructured L1 pruning to all Linear layers, removing the
    *amount* fraction of weights with the smallest magnitude.

    Parameters
    ----------
    model : SageModel
    amount : float
        Fraction of weights to prune (0.0 – 1.0).

    Returns
    -------
    SageModel — the pruned model (pruning masks are permanent after this call).
    """
    pruned_count = 0
    total_count = 0

    base_model = getattr(model, "module", model)
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")  # make the pruning permanent
            pruned_count += (module.weight == 0).sum().item()
            total_count += module.weight.numel()

    sparsity = pruned_count / max(total_count, 1) * 100
    logger.info(
        f"Pruning complete.  {pruned_count:,} / {total_count:,} weights zeroed "
        f"({sparsity:.1f}% sparsity)"
    )
    return model


# ===================================================================
# Knowledge Distillation Loss
# ===================================================================

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Combined knowledge-distillation loss.

    ``L = alpha * KL(softmax(teacher/T), softmax(student/T)) * T^2
         + (1 - alpha) * CE(student, labels)``

    Parameters
    ----------
    student_logits : Tensor  [B, T, V]
    teacher_logits : Tensor  [B, T, V]
    labels : Tensor  [B, T]
    temperature : float
    alpha : float — weight for the distillation term (0 → pure CE, 1 → pure KD).
    ignore_index : int — label value to ignore in cross-entropy.

    Returns
    -------
    Tensor (scalar)
    """
    # Soft targets
    soft_student = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)

    kd_loss = torch.nn.functional.kl_div(
        soft_student.view(-1, soft_student.size(-1)),
        soft_teacher.view(-1, soft_teacher.size(-1)),
        reduction="batchmean",
    ) * (temperature ** 2)

    # Hard-label cross-entropy
    ce_loss = torch.nn.functional.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )

    return alpha * kd_loss + (1 - alpha) * ce_loss


# ===================================================================
# torch.compile wrapper (PyTorch 2.0+)
# ===================================================================

def try_compile(model: nn.Module) -> nn.Module:
    """
    Attempt to compile the model with ``torch.compile`` for faster
    execution.  Falls back gracefully if compilation is not available.
    """
    if hasattr(torch, "compile"):
        try:
            compiled = torch.compile(model)
            logger.info("Model compiled with torch.compile for accelerated execution.")
            return compiled
        except Exception as e:
            logger.warning(f"torch.compile failed ({e}). Using eager mode.")
    else:
        logger.info("torch.compile not available (requires PyTorch 2.0+). Using eager mode.")
    return model
