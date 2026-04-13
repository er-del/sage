import os
from dataclasses import dataclass, field
from typing import Any

@dataclass
class SageConfig:
    # Model dimensions corresponding to T4 (16GB VRAM) fit
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4   # GQA: must divide n_heads
    n_layers: int = 6
    d_ff: int = 2048
    
    # MoE (Mixture of Experts) config
    n_experts: int = 4
    num_experts_per_tok: int = 2
    
    # Vocabulary and sequence parameters
    vocab_size: int = 100277  # Default for tiktoken "cl100k_base"
    max_seq_len: int = 1024
    
    # Regularization
    dropout: float = 0.1
    
    # Training Loop defaults
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Checkpointing and path details
    checkpoint_dir: str = "checkpoints"
    project_name: str = "sage-v2"
    
    # Cache for device (set on first access)
    _device: Any = field(default=None, repr=False)
    
    @property
    def device(self):
        """Returns the best available device with CUDA compatibility checking."""
        if self._device is None:
            # Import here to avoid circular imports
            from .utils import get_compatible_device
            self._device = get_compatible_device()
        return self._device
