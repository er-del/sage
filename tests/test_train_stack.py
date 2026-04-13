from pathlib import Path

import torch

from model.config import ModelConfig
from model.model import SageTransformer
from train.checkpoint import load_latest_checkpoint, save_checkpoint
from train.hardware import HardwareConfig
from train.optimizer import ScheduleConfig, create_optimizer, create_scheduler


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = ModelConfig(
        num_layers=2,
        d_model=64,
        num_attn_heads=4,
        num_kv_heads=2,
        head_dim=16,
        ffn_hidden_dim=256,
        vocab_size=128,
        context_length=32,
    )
    model = SageTransformer(config)
    schedule = ScheduleConfig(total_steps=8)
    optimizer = create_optimizer(model, schedule)
    scheduler = create_scheduler(optimizer, schedule)
    scaler = torch.GradScaler("cuda", enabled=False)
    path = save_checkpoint(model, optimizer, scheduler, scaler, 3, {"name": "test"}, str(tmp_path))
    assert Path(path).exists()
    resumed_step = load_latest_checkpoint(model, optimizer, scheduler, scaler, str(tmp_path), "cpu")
    assert resumed_step == 3


def test_hardware_summary_shape() -> None:
    summary = HardwareConfig(model_size_b=1.0, context_length=4096).summary()
    assert "device" in summary
    assert "effective_batch_tokens" in summary
