"""Post-training quantization entry points."""

from __future__ import annotations

from pathlib import Path

import torch


def export_int8_state_dict(model: torch.nn.Module, output_path: str) -> str:
    """Save a dynamic-int8 quantized model state dict for CPU experiments."""
    quantized = torch.quantization.quantize_dynamic(model.cpu(), {torch.nn.Linear}, dtype=torch.qint8)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized.state_dict(), path)
    return str(path)


def gguf_conversion_command(checkpoint_dir: str, output_path: str) -> str:
    """Return a llama.cpp conversion command string."""
    return (
        f"python llama.cpp/convert_hf_to_gguf.py {checkpoint_dir} "
        f"--outfile {output_path} --outtype f16"
    )
