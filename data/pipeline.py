"""End-to-end raw-corpus to Parquet shard pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm

from data.dedup import deduplicate_records
from data.filter import FilterConfig, filter_record
from data.ingest import SOURCE_REGISTRY, stream_source
from data.shard import ShardConfig, write_shards


def _select_sources(names: list[str] | None) -> tuple:
    if not names:
        return SOURCE_REGISTRY
    wanted = set(names)
    selected = tuple(spec for spec in SOURCE_REGISTRY if spec.name in wanted)
    missing = sorted(wanted - {spec.name for spec in selected})
    if missing:
        raise ValueError(f"Unknown sources: {', '.join(missing)}")
    return selected


def build_records(source_names: list[str] | None = None, limit_per_source: int | None = None) -> list[dict[str, object]]:
    """Load, filter, and deduplicate records from the configured raw sources."""
    records: list[dict[str, object]] = []
    for spec in _select_sources(source_names):
        source_records: list[dict[str, object]] = []
        for record in stream_source(spec):
            filtered = filter_record(record, FilterConfig())
            if filtered is None:
                continue
            source_records.append(filtered)
            if limit_per_source is not None and len(source_records) >= limit_per_source:
                break
        records.extend(source_records)
    return deduplicate_records(records)


def run_pipeline(
    tokenizer_model: str,
    output_dir: str = "data/processed",
    source_names: list[str] | None = None,
    shard_size: int = 2048,
    limit_per_source: int | None = None,
) -> dict[str, object]:
    """Create Parquet shards from the current raw JSONL corpora."""
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_model)
    records = build_records(source_names=source_names, limit_per_source=limit_per_source)
    manifest = write_shards(records, tokenizer, ShardConfig(output_dir=output_dir, shard_size=shard_size))
    summary = {
        "tokenizer_model": tokenizer_model,
        "output_dir": output_dir,
        "records": len(records),
        "sources": source_names or [spec.name for spec in SOURCE_REGISTRY],
        "manifest": manifest,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Filter, deduplicate, tokenize, and shard SAGE raw corpora.")
    parser.add_argument("--tokenizer-model", default="tokenizer/tokenizer.model", help="SentencePiece tokenizer model.")
    parser.add_argument("--output-dir", default="data/processed", help="Destination directory for parquet shards.")
    parser.add_argument("--sources", nargs="*", default=None, help="Subset of source names from data.ingest.SOURCE_REGISTRY.")
    parser.add_argument("--shard-size", type=int, default=2048, help="Rows per parquet shard.")
    parser.add_argument("--limit-per-source", type=int, default=None, help="Optional cap for smoke-testing.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_argparser().parse_args()
    summary = run_pipeline(
        tokenizer_model=args.tokenizer_model,
        output_dir=args.output_dir,
        source_names=args.sources,
        shard_size=args.shard_size,
        limit_per_source=args.limit_per_source,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
