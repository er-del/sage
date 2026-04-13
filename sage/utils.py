import os
import logging
import torch
from typing import Optional, Tuple

def _get_logger(name: str) -> logging.Logger:
    """Simple logger getter to avoid circular imports."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.propagate = False
    return logger

def get_compatible_device() -> torch.device:
    """
    Returns the best available device with CUDA compatibility checking.
    
    Automatically detects GPU compute capability and falls back to CPU
    if the current PyTorch installation doesn't support the GPU.
    """
    logger = _get_logger("sage.device")
    
    # Check CUDA availability and compatibility
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability()
        major, minor = capability
        sm_version = f"sm_{major}{minor}"
        
        logger.info(f"Detected GPU: {gpu_name} (CUDA Capability: {sm_version})")
        
        # PyTorch 2.0+ minimum is sm_70, PyTorch 1.13 supports sm_60
        # Check if we can actually run a small operation
        try:
            test_tensor = torch.zeros(1).cuda()
            _ = test_tensor + test_tensor  # Simple operation to verify
            logger.info(f"✅ GPU is compatible with current PyTorch")
            return torch.device("cuda")
        except RuntimeError as e:
            if "no kernel image is available" in str(e).lower():
                logger.warning(f"⚠️  GPU {sm_version} not supported by current PyTorch")
                logger.warning(f"   Current PyTorch supports: {torch.cuda.get_arch_list() or 'sm_70+'}")
                logger.warning(f"   Install compatible PyTorch:")
                if major < 7:
                    logger.warning(f"   !pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121")
                else:
                    logger.warning(f"   !pip install torch --index-url https://download.pytorch.org/whl/cu118")
                logger.warning(f"   Falling back to CPU...")
            else:
                raise
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple Silicon (MPS)")
        return torch.device("mps")
    
    logger.info("Using CPU")
    return torch.device("cpu")

def setup_logger(name: str) -> logging.Logger:
    """Sets up a standardized logger for the SAGE system."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    # Prevent propagation to the root logger to avoid double printing
    logger.propagate = False
    return logger

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: Optional[torch.optim.Optimizer], 
    step: int, 
    checkpoint_dir: str, 
    filename: str = "sage_latest.pt"
) -> str:
    """Saves the model and optimizer state to a checkpoint file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    
    base_model = getattr(model, "module", model)
    checkpoint = {
        'step': step,
        'model_state_dict': base_model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
    torch.save(checkpoint, path)
    return path

def load_checkpoint(
    model: torch.nn.Module, 
    optimizer: Optional[torch.optim.Optimizer], 
    checkpoint_dir: str, 
    filename: str = "sage_latest.pt",
    device: str = "cpu"
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int]:
    """Loads a checkpoint and restores the model and optimizer states."""
    path = os.path.join(checkpoint_dir, filename)
    
    if not os.path.exists(path):
        logger = setup_logger("utils")
        logger.warning(f"No checkpoint found at {path}. Starting from scratch.")
        return model, optimizer, 0
        
    # Load to CPU first to avoid VRAM spikes, then the module will be moved later if needed
    checkpoint = torch.load(path, map_location=device)
    
    base_model = getattr(model, "module", model)
    base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    step = checkpoint.get('step', 0)
    
    logger = setup_logger("utils")
    logger.info(f"Loaded checkpoint from {path} at step {step}")
    
    return model, optimizer, step
