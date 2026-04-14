"""GPU-oriented FastAPI server for SAGE."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from model.config import ModelConfig
from model.model import SageTransformer
from serve.control_plane import build_control_router, get_runtime_access_info
from train.checkpoint import load_latest_checkpoint
from train.hardware import HardwareConfig


app = FastAPI(title="SAGE Server")
_MODEL: SageTransformer | None = None
_TOKENIZER = None
_MODEL_DEVICE: torch.device | None = None
_MODEL_STATE: dict[str, Any] = {
    "model_config": None,
    "checkpoint_dir": None,
    "checkpoint_loaded": False,
    "checkpoint_step": 0,
    "tokenizer_path": None,
}
_LOGGER = logging.getLogger("uvicorn.error")


def _print_startup_banner() -> None:
    """Print the login details for the browser control UI."""
    access = get_runtime_access_info()
    local_url = (access["local_url"] or "http://127.0.0.1:8000").rstrip("/")
    public_url = access["public_url"]
    _LOGGER.info("SAGE local URL: %s/", local_url)
    if public_url:
        _LOGGER.info("SAGE public URL: %s/", public_url.rstrip("/"))
    _LOGGER.info("SAGE login password: %s", access["password"])


class GenerationRequest(BaseModel):
    """Request schema for text generation."""

    input_ids: list[int]
    max_new_tokens: int = 32


class ChatRequest(BaseModel):
    """Request schema for text generation through the tokenizer."""

    prompt: str
    max_new_tokens: int = 64


def get_generation_device() -> torch.device:
    """Pick the active inference device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_model_config_path() -> Path:
    configured = Path(os.environ.get("SAGE_MODEL_CONFIG", "configs/model/1b.yaml"))
    return configured if configured.exists() else Path("configs/model/1b.yaml")


def _resolve_checkpoint_dir() -> Path:
    return Path(os.environ.get("SAGE_CHECKPOINT_DIR", "runs/default"))


def _resolve_tokenizer_path() -> Path:
    return Path(os.environ.get("SAGE_TOKENIZER_MODEL", "tokenizer/tokenizer.model"))


def configure_runtime_paths(model_config: str | None, checkpoint_dir: str | None, tokenizer_model: str | None) -> None:
    """Configure runtime paths via environment variables."""
    if model_config:
        os.environ["SAGE_MODEL_CONFIG"] = model_config
    if checkpoint_dir:
        os.environ["SAGE_CHECKPOINT_DIR"] = checkpoint_dir
    if tokenizer_model:
        os.environ["SAGE_TOKENIZER_MODEL"] = tokenizer_model


def get_model() -> SageTransformer:
    """Lazily create the model for server startup."""
    global _MODEL, _MODEL_DEVICE
    if _MODEL is None:
        config_path = _resolve_model_config_path()
        config = ModelConfig.from_yaml(config_path) if config_path.exists() else ModelConfig()
        _MODEL = SageTransformer(config)
        checkpoint_dir = _resolve_checkpoint_dir()
        checkpoint_step = 0
        if checkpoint_dir.exists():
            checkpoint_step = load_latest_checkpoint(_MODEL, None, None, None, str(checkpoint_dir), device="cpu")
        _MODEL_STATE.update(
            {
                "model_config": str(config_path),
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_loaded": checkpoint_step > 0,
                "checkpoint_step": checkpoint_step,
            }
        )
        _MODEL.eval()
    device = get_generation_device()
    if _MODEL_DEVICE != device:
        _MODEL = _MODEL.to(device)
        _MODEL_DEVICE = device
    return _MODEL


def get_tokenizer():
    """Lazily load the SentencePiece tokenizer if present."""
    global _TOKENIZER
    if _TOKENIZER is None:
        tokenizer_path = _resolve_tokenizer_path()
        _MODEL_STATE["tokenizer_path"] = str(tokenizer_path)
        if not tokenizer_path.exists():
            return None
        from tokenizer.validate_tokenizer import load_processor

        _TOKENIZER = load_processor(str(tokenizer_path))
    return _TOKENIZER


def _generate_token_ids(input_ids: list[int], max_new_tokens: int) -> list[int]:
    """Run greedy decoding from input token ids."""
    model = get_model()
    device = get_generation_device()
    context_length = model.config.context_length
    generated = list(input_ids[-context_length:])
    with torch.inference_mode():
        for _ in range(max(0, int(max_new_tokens))):
            window = generated[-context_length:]
            tensor_ids = torch.tensor([window], dtype=torch.long, device=device)
            logits, _ = model(tensor_ids)
            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            generated.append(next_token)
    return generated


def chat_status() -> dict[str, object]:
    """Return whether text chat is configured for the current server."""
    tokenizer = get_tokenizer()
    checkpoint_dir = _MODEL_STATE["checkpoint_dir"] or str(_resolve_checkpoint_dir())
    checkpoint_loaded = bool(_MODEL_STATE["checkpoint_loaded"])
    if not checkpoint_loaded:
        checkpoint_loaded = any(Path(checkpoint_dir).glob("ckpt_step_*.pt"))
    checkpoint_step = int(_MODEL_STATE["checkpoint_step"] or 0)
    if checkpoint_step == 0 and checkpoint_loaded:
        latest = sorted(Path(checkpoint_dir).glob("ckpt_step_*.pt"))
        if latest:
            checkpoint_step = int(latest[-1].stem.split("_")[-1])
    available = tokenizer is not None
    warning = None
    if tokenizer is None:
        warning = "Tokenizer model not found. Train or place tokenizer/tokenizer.model before using browser chat."
    elif not checkpoint_loaded:
        warning = "No checkpoint loaded. Chat will run with randomly initialized model weights until you train or configure SAGE_CHECKPOINT_DIR."
    return {
        "available": available,
        "tokenizer_path": _MODEL_STATE["tokenizer_path"],
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_step": checkpoint_step,
        "warning": warning,
    }


@app.get("/health")
def health() -> dict[str, object]:
    """Return basic health and hardware information."""
    hw = HardwareConfig(model_size_b=1.0, context_length=4096)
    return {"status": "ok", "hardware": hw.summary(), "chat": chat_status()}


@app.post("/generate")
def generate(request: GenerationRequest) -> dict[str, object]:
    """Generate continuation token ids from an input token list."""
    return {"tokens": _generate_token_ids(request.input_ids, request.max_new_tokens)}


@app.get("/chat/status")
def get_chat_status() -> dict[str, object]:
    """Expose browser-chat readiness."""
    return chat_status()


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    """Generate text from a prompt using the local tokenizer."""
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return {
            "success": False,
            "detail": "Tokenizer model not found. Train the tokenizer first or set SAGE_TOKENIZER_MODEL.",
            **chat_status(),
        }
    prompt = request.prompt.strip()
    if not prompt:
        return {"success": False, "detail": "Prompt cannot be empty.", **chat_status()}
    prompt_ids = list(tokenizer.encode(prompt, out_type=int))
    generated = _generate_token_ids(prompt_ids, request.max_new_tokens)
    completion_ids = generated[len(prompt_ids) :]
    return {
        "success": True,
        "prompt": prompt,
        "response": tokenizer.decode(completion_ids),
        "input_ids": prompt_ids,
        "output_ids": generated,
        "new_token_ids": completion_ids,
        **chat_status(),
    }


def _health_action(_: dict[str, object]) -> dict[str, object]:
    return health()


def _generate_action(args: dict[str, object]) -> dict[str, object]:
    input_ids_raw = args.get("input_ids", [])
    if not isinstance(input_ids_raw, list):
        input_ids_raw = [input_ids_raw] if input_ids_raw is not None else []
    input_ids = [int(item) for item in input_ids_raw]  # type: ignore
    request = GenerationRequest(
        input_ids=input_ids,
        max_new_tokens=int(args.get("max_new_tokens", 32)),  # type: ignore
    )
    return generate(request)


app.include_router(build_control_router({"health_check": _health_action, "generate": _generate_action}))


@app.on_event("startup")
def _startup_banner() -> None:
    _print_startup_banner()
