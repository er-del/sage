"""Hardware detection and runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
import platform

import torch

from train.distributed import get_training_strategy

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


@dataclass
class HardwareConfig:
    """Detect hardware and derive runtime decisions."""

    model_size_b: float
    context_length: int

    def __post_init__(self) -> None:
        self.device, self.dtype = self._detect_device_dtype()
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.vram_gb = self._get_vram()
        self.ram_gb = self._get_ram_gb()
        self.strategy = get_training_strategy(self.model_size_b)
        self.micro_batch = self._pick_micro_batch()
        self.grad_accum = self._pick_grad_accum()
        self.use_amp = self.device != "cpu"
        self.use_flash_attn = self.device == "cuda"
        self.use_qlora = False

    def _detect_device_dtype(self) -> tuple[str, torch.dtype]:
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return "cuda", dtype
        if torch.backends.mps.is_available():
            return "mps", torch.bfloat16
        return "cpu", torch.float32

    def _get_vram(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1e9

    def _get_ram_gb(self) -> float:
        if psutil is not None:
            return psutil.virtual_memory().total / 1e9
        if platform.system() == "Windows":
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong
            mem_kb = c_ulonglong()
            kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(mem_kb))
            return (mem_kb.value * 1024) / 1e9
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / 1e9
        return 0.0

    def _pick_micro_batch(self) -> int:
        if self.device == "cpu":
            return 1
        if self.vram_gb >= 80:
            return 8
        if self.vram_gb >= 40:
            return 4
        if self.vram_gb >= 24:
            return 2
        return 1

    def _pick_grad_accum(self) -> int:
        target_tokens = 2_000_000
        tokens_per_micro = self.micro_batch * self.context_length * max(self.n_gpus, 1)
        return max(1, target_tokens // max(tokens_per_micro, 1))

    def summary(self) -> dict[str, object]:
        """Return a JSON-safe hardware summary."""
        effective_batch = self.micro_batch * self.grad_accum * self.context_length * max(self.n_gpus, 1)
        return {
            "device": self.device,
            "dtype": str(self.dtype),
            "n_gpus": self.n_gpus,
            "vram_gb": round(self.vram_gb, 2),
            "ram_gb": round(self.ram_gb, 2),
            "strategy": self.strategy,
            "micro_batch": self.micro_batch,
            "grad_accum": self.grad_accum,
            "effective_batch_tokens": effective_batch,
            "use_flash_attn": self.use_flash_attn,
        }
