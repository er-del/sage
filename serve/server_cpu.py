"""CPU and llama.cpp serving helpers."""

from __future__ import annotations

import shutil

from fastapi import FastAPI
from pydantic import BaseModel

from serve.control_plane import build_control_router


app = FastAPI(title="SAGE CPU Server")


class ChatRequest(BaseModel):
    """Request schema for the browser chat surface."""

    prompt: str
    max_new_tokens: int = 64


@app.get("/health")
def health() -> dict[str, object]:
    """Report llama.cpp availability for CPU serving."""
    return {"status": "ok", "llama_cpp_available": shutil.which("llama-server") is not None, "chat": chat_status()}


def chat_status() -> dict[str, object]:
    """Return chat readiness for the CPU server."""
    return {
        "available": False,
        "warning": "Browser chat is only wired to the PyTorch GPU server in this repo. Use serve.server:app for direct interaction.",
    }


@app.get("/chat/status")
def get_chat_status() -> dict[str, object]:
    """Expose browser-chat readiness."""
    return chat_status()


@app.post("/chat")
def chat(_: ChatRequest) -> dict[str, object]:
    """Return a clear error for CPU-only control-plane mode."""
    return {"success": False, "detail": chat_status()["warning"], **chat_status()}


def _health_action(_: dict[str, object]) -> dict[str, object]:
    return health()


app.include_router(build_control_router({"health_check": _health_action}))
