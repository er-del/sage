"""GPU-oriented FastAPI server for SAGE."""

from __future__ import annotations

from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from model.config import ModelConfig
from model.model import SageTransformer
from serve.control_plane import build_control_router
from serve.kv_cache import KVCache
from train.hardware import HardwareConfig


app = FastAPI(title="SAGE Server")
_MODEL: SageTransformer | None = None
_TOKENIZER = None


class GenerationRequest(BaseModel):
    """Request schema for text generation."""

    input_ids: list[int]
    max_new_tokens: int = 32


def get_model() -> SageTransformer:
    """Lazily create the model for server startup."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SageTransformer(ModelConfig())
        _MODEL.eval()
    return _MODEL


@app.get("/health")
def health() -> dict[str, object]:
    """Return basic health and hardware information."""
    hw = HardwareConfig(model_size_b=1.0, context_length=4096)
    return {"status": "ok", "hardware": hw.summary()}


@app.post("/generate")
def generate(request: GenerationRequest) -> dict[str, object]:
    """Generate continuation token ids from an input token list."""
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = torch.tensor([request.input_ids], dtype=torch.long, device=device)
    generated = list(request.input_ids)
    cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
    for _ in range(request.max_new_tokens):
        logits, cache = model(input_ids[:, -1:] if cache is not None else input_ids, past_key_values=cache)
        next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        generated.append(next_token)
        input_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
    return {"tokens": generated}


def _health_action(_: dict[str, object]) -> dict[str, object]:
    return health()


def _generate_action(args: dict[str, object]) -> dict[str, object]:
    request = GenerationRequest(
        input_ids=[int(item) for item in list(args.get("input_ids", []))],
        max_new_tokens=int(args.get("max_new_tokens", 32)),
    )
    return generate(request)


app.include_router(build_control_router({"health_check": _health_action, "generate": _generate_action}))
