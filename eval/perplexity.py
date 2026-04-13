"""Validation perplexity evaluation."""

from __future__ import annotations

import math

import torch

from train.loss import masked_cross_entropy


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_batches: int = 16,
) -> dict[str, float]:
    """Evaluate average loss and perplexity on a validation loader."""
    model.eval()
    losses: list[float] = []
    for index, batch in enumerate(dataloader):
        if index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        if dtype is not None and device.type != "cpu":
            with torch.autocast(device_type=device.type, dtype=dtype):
                logits, _ = model(input_ids)
                loss = masked_cross_entropy(logits, labels, loss_mask)
        else:
            logits, _ = model(input_ids)
            loss = masked_cross_entropy(logits, labels, loss_mask)
        losses.append(float(loss))
    model.train()
    mean_loss = sum(losses) / max(len(losses), 1)
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20.0))}
