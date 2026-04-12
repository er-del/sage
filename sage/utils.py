import os
import logging
import torch
from typing import Optional, Tuple

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
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
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
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    step = checkpoint.get('step', 0)
    
    logger = setup_logger("utils")
    logger.info(f"Loaded checkpoint from {path} at step {step}")
    
    return model, optimizer, step
