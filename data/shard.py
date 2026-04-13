"""Tokenization, manifesting, and Parquet sharding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional at import time
    pa = None
    pq = None


SCHEMA_COLUMNS = ("id", "text", "tokens", "domain_tag", "quality_tier", "lang", "token_count", "split")


@dataclass(frozen=True)
class ShardConfig:
    """Parameters for Parquet shard writing."""

    output_dir: str
    shard_size: int = 2048
    validation_ratio: float = 0.01
    test_ratio: float = 0.001


def assign_split(record_id: str, validation_ratio: float, test_ratio: float) -> str:
    """Assign a deterministic split from the content id."""
    value = int(record_id[:8], 16) / 0xFFFFFFFF
    if value < test_ratio:
        return "test"
    if value < test_ratio + validation_ratio:
        return "validation"
    return "train"


def build_manifest(shard_paths: Iterable[Path]) -> dict[str, object]:
    """Create a manifest describing shard files."""
    shard_paths = list(shard_paths)
    digest = hashlib.sha256()
    for path in shard_paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("utf-8"))
    return {
        "format": "parquet",
        "schema": list(SCHEMA_COLUMNS),
        "shards": [path.name for path in shard_paths],
        "dataset_hash": digest.hexdigest(),
    }


def write_shards(records: Iterable[dict[str, object]], tokenizer, config: ShardConfig) -> dict[str, object]:
    """Write tokenized records to Parquet shards and emit a manifest."""
    if pa is None or pq is None:
        raise ImportError("pyarrow is required to write parquet shards.")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer: list[dict[str, object]] = []
    shard_paths: list[Path] = []
    shard_index = 0
    for record in records:
        tokens = tokenizer.encode(str(record["text"]), out_type=int)
        row = {
            "id": str(record["id"]),
            "text": str(record["text"]),
            "tokens": tokens,
            "domain_tag": str(record["domain_tag"]),
            "quality_tier": str(record["quality_tier"]),
            "lang": str(record["lang"]),
            "token_count": len(tokens),
            "split": assign_split(str(record["id"]), config.validation_ratio, config.test_ratio),
        }
        buffer.append(row)
        if len(buffer) >= config.shard_size:
            shard_paths.append(_flush_shard(output_dir, shard_index, buffer))
            shard_index += 1
            buffer = []
    if buffer:
        shard_paths.append(_flush_shard(output_dir, shard_index, buffer))
    manifest = build_manifest(shard_paths)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _flush_shard(output_dir: Path, shard_index: int, rows: list[dict[str, object]]) -> Path:
    """Persist one Parquet shard."""
    table = pa.table({column: [row[column] for row in rows] for column in SCHEMA_COLUMNS})
    path = output_dir / f"shard-{shard_index:05d}.parquet"
    pq.write_table(table, path)
    return path
