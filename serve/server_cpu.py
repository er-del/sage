"""CPU and llama.cpp serving helpers."""

from __future__ import annotations

import shutil

from fastapi import FastAPI


app = FastAPI(title="SAGE CPU Server")


@app.get("/health")
def health() -> dict[str, object]:
    """Report llama.cpp availability for CPU serving."""
    return {"status": "ok", "llama_cpp_available": shutil.which("llama-server") is not None}
