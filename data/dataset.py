"""Packed training dataset with deterministic resume support."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import IterableDataset

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional at import time
    pq = None


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for packing token streams into training batches."""

    shard_paths: tuple[str, ...]
    context_length: int
    split: str = "train"
    seed: int = 42


class PackedDataset(IterableDataset):
    """Iterate packed token sequences with document-boundary masks."""

    def __init__(self, config: DatasetConfig):
        super().__init__()
        self.config = config
        self._skip = 0

    def skip(self, n_batches: int) -> None:
        """Fast-forward the iterator by discarding the first n batches."""
        self._skip = max(0, int(n_batches))

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        skipped = 0
        for batch in self._generate():
            if skipped < self._skip:
                skipped += 1
                continue
            yield batch

    def _generate(self) -> Iterator[dict[str, torch.Tensor]]:
        token_buffer: list[int] = []
        boundary_buffer: list[int] = []
        for row in self._iter_rows():
            tokens = list(row["tokens"])
            if len(tokens) < 2:
                continue
            token_buffer.extend(tokens)
            boundary_buffer.extend([0] * (len(tokens) - 1) + [1])
            while len(token_buffer) >= self.config.context_length + 1:
                window_tokens = token_buffer[: self.config.context_length + 1]
                window_boundaries = boundary_buffer[: self.config.context_length + 1]
                yield pack_sequence(window_tokens, window_boundaries)
                token_buffer = token_buffer[self.config.context_length :]
                boundary_buffer = boundary_buffer[self.config.context_length :]

    def _iter_rows(self) -> Iterator[dict[str, object]]:
        if pq is None:
            raise ImportError("pyarrow is required to read parquet shards.")
        shard_paths = [Path(path) for path in self.config.shard_paths]
        rng = random.Random(self.config.seed)
        shard_paths = shard_paths[:]
        rng.shuffle(shard_paths)
        for path in shard_paths:
            table = pq.read_table(path, columns=["tokens", "split"])
            rows = table.to_pylist()
            for row in rows:
                if row["split"] != self.config.split:
                    continue
                yield row


def pack_sequence(tokens: list[int], boundaries: list[int]) -> dict[str, torch.Tensor]:
    """Turn one packed token window into model-ready tensors."""
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
    labels = torch.tensor(tokens[1:], dtype=torch.long)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    attention_document_mask = torch.tensor(boundaries[:-1], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "document_boundaries": attention_document_mask,
    }
