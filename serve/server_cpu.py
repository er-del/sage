"""CPU and llama.cpp serving helpers."""

from __future__ import annotations

import shutil

from fastapi import FastAPI

from serve.control_plane import build_control_router


app = FastAPI(title="SAGE CPU Server")


@app.get("/health")
def health() -> dict[str, object]:
    """Report llama.cpp availability for CPU serving."""
    return {"status": "ok", "llama_cpp_available": shutil.which("llama-server") is not None}


def _health_action(_: dict[str, object]) -> dict[str, object]:
    return health()


app.include_router(build_control_router({"health_check": _health_action}))
