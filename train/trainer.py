"""Main training loop for SAGE."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
import yaml

from data.dataset import DatasetConfig, PackedDataset
from eval.perplexity import evaluate_perplexity
from model.config import ModelConfig
from model.model import SageTransformer
from train.checkpoint import load_latest_checkpoint, save_checkpoint
from train.hardware import HardwareConfig
from train.loss import masked_cross_entropy
from train.optimizer import ScheduleConfig, create_optimizer, create_scheduler


@dataclass
class TrainerConfig:
    """High-level trainer settings."""

    output_dir: str = "runs/default"
    checkpoint_interval: int = 1000
    log_interval: int = 10
    eval_interval: int = 1000
    total_steps: int = 25_000
    seed: int = 42
    use_wandb: bool = True


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack packed dataset examples into a batch."""
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


def create_dataloader(dataset: PackedDataset, batch_size: int) -> DataLoader:
    """Create the training DataLoader."""
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)


def train(
    model: SageTransformer,
    train_dataset: PackedDataset,
    validation_dataset: PackedDataset | None,
    model_config: ModelConfig,
    schedule_config: ScheduleConfig,
    trainer_config: TrainerConfig,
) -> dict[str, object]:
    """Run the training loop and return the final summary."""
    torch.manual_seed(trainer_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trainer_config.seed)
    hw = HardwareConfig(model_size_b=1.0, context_length=model_config.context_length)
    device = torch.device(hw.device)
    model = model.to(device)
    optimizer = create_optimizer(model, schedule_config)
    scheduler = create_scheduler(optimizer, schedule_config)
    scaler = torch.amp.GradScaler("cuda", enabled=(hw.device == "cuda" and hw.dtype == torch.float16))
    start_step = load_latest_checkpoint(model, optimizer, scheduler, scaler, trainer_config.output_dir, device)
    train_dataset.skip(start_step * hw.grad_accum)
    train_loader = create_dataloader(train_dataset, batch_size=hw.micro_batch)
    train_iter = iter(train_loader)

    Path(trainer_config.output_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(trainer_config.output_dir) / "metrics.jsonl"
    tokens_seen = start_step * hw.micro_batch * model_config.context_length
    last_log_time = time.perf_counter()
    wandb_run = _init_wandb(trainer_config, model_config, schedule_config, hw.summary())

    model.train()
    for step in range(start_step, trainer_config.total_steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(hw.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            if hw.use_amp:
                with torch.amp.autocast(device_type=hw.device, dtype=hw.dtype):
                    logits, _ = model(input_ids)
                    loss = masked_cross_entropy(logits, labels, loss_mask) / hw.grad_accum
            else:
                logits, _ = model(input_ids)
                loss = masked_cross_entropy(logits, labels, loss_mask) / hw.grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.item()
            tokens_seen += int(input_ids.numel())

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if (step + 1) % trainer_config.log_interval == 0:
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1.0e-6)
            tokens_per_second = (hw.micro_batch * hw.grad_accum * model_config.context_length) / elapsed
            metrics = {
                "step": step + 1,
                "loss": step_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "tokens_seen": tokens_seen,
                "tokens_per_second": tokens_per_second,
                "grad_norm": float(grad_norm),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            if wandb_run is not None:
                wandb_run.log(metrics, step=step + 1)
            last_log_time = now

        if (step + 1) % trainer_config.eval_interval == 0 and validation_dataset is not None:
            val_loader = create_dataloader(validation_dataset, batch_size=1)
            evaluation = evaluate_perplexity(model, val_loader, device=device, dtype=hw.dtype if hw.use_amp else None)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step + 1, **evaluation}) + "\n")
            if wandb_run is not None:
                wandb_run.log(evaluation, step=step + 1)

        if (step + 1) % trainer_config.checkpoint_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=step + 1,
                config={"model": model_config.to_dict(), "schedule": asdict(schedule_config), "trainer": asdict(trainer_config)},
                output_dir=trainer_config.output_dir,
            )

    if wandb_run is not None:
        wandb_run.finish()
    return {"output_dir": trainer_config.output_dir, "tokens_seen": tokens_seen, "hardware": hw.summary()}


def _init_wandb(
    trainer_config: TrainerConfig,
    model_config: ModelConfig,
    schedule_config: ScheduleConfig,
    hardware_summary: dict[str, object],
):
    """Start a wandb run when available and enabled."""
    if not trainer_config.use_wandb:
        return None
    try:
        import wandb
    except ImportError:
        return None
    return wandb.init(
        project="sage-llm",
        name=Path(trainer_config.output_dir).name,
        config={
            "model": model_config.to_dict(),
            "schedule": asdict(schedule_config),
            "trainer": asdict(trainer_config),
            "hardware": hardware_summary,
        },
        mode="offline",
    )


def build_argparser() -> argparse.ArgumentParser:
    """Build the trainer CLI."""
    parser = argparse.ArgumentParser(description="Train the SAGE dense language model.")
    parser.add_argument("--model-config", default="configs/model/1b.yaml")
    parser.add_argument("--schedule-config", default="configs/train/schedule.yaml")
    parser.add_argument("--train-shards", nargs="+", default=[])
    parser.add_argument("--validation-shards", nargs="*", default=[])
    parser.add_argument("--output-dir", default="runs/default")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entrypoint for local training runs."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    model_config = ModelConfig.from_yaml(args.model_config)
    schedule_payload = yaml.safe_load(Path(args.schedule_config).read_text(encoding="utf-8"))
    schedule = ScheduleConfig(
        peak_learning_rate=schedule_payload["peak_learning_rate"],
        min_learning_rate=schedule_payload["min_learning_rate"],
        warmup_steps=schedule_payload["warmup_steps"],
        weight_decay=schedule_payload["weight_decay"],
        betas=tuple(schedule_payload["betas"]),
        adam_eps=schedule_payload["adam_eps"],
        total_steps=args.steps or schedule_payload["total_steps"] if "total_steps" in schedule_payload else (args.steps or 25_000),
    )
    trainer_config = TrainerConfig(
        output_dir=args.output_dir,
        checkpoint_interval=schedule_payload.get("checkpoint_interval", 1000),
        log_interval=schedule_payload.get("log_interval", 10),
        eval_interval=schedule_payload.get("eval_interval", 1000),
        total_steps=args.steps or schedule_payload.get("total_steps", 25_000),
        seed=schedule_payload.get("seed", 42),
        use_wandb=not args.disable_wandb,
    )
    if not args.train_shards:
        print("No training shards provided. The trainer entrypoint is configured correctly but requires shard paths to run.")
        return
    train_dataset = PackedDataset(DatasetConfig(tuple(args.train_shards), model_config.context_length, split="train"))
    validation_dataset = None
    if args.validation_shards:
        validation_dataset = PackedDataset(DatasetConfig(tuple(args.validation_shards), model_config.context_length, split="validation"))
    model = SageTransformer(model_config)
    summary = train(model, train_dataset, validation_dataset, model_config, schedule, trainer_config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
