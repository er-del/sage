import json
from pathlib import Path

from data.dataset import pack_sequence
from data.bootstrap import bootstrap_raw_corpora
from data.dedup import deduplicate_records
from data.filter import filter_record
from data.ingest import SourceSpec

import pytest


def test_filter_record_masks_pii() -> None:
    record = {
        "id": "1",
        "text": "Contact me at person@example.com. This document contains enough text to pass the minimum length requirement. " * 4,
        "license_category": "permissive",
        "domain_tag": "general",
        "quality_tier": "high",
    }
    filtered = filter_record(record)
    assert filtered is not None
    assert "[EMAIL]" in filtered["text"]


def test_deduplicate_records_removes_exact_duplicates() -> None:
    records = [
        {"text": "same text", "id": "1"},
        {"text": "same text", "id": "2"},
        {"text": "different text", "id": "3"},
    ]
    kept = deduplicate_records(records)
    assert len(kept) == 2


def test_pack_sequence_shapes() -> None:
    packed = pack_sequence([1, 2, 3, 4, 5], [0, 0, 1, 0, 1])
    assert packed["input_ids"].tolist() == [1, 2, 3, 4]
    assert packed["labels"].tolist() == [2, 3, 4, 5]
    assert packed["document_boundaries"].tolist() == [0, 0, 1, 0]


def test_bootstrap_raw_corpora_writes_jsonl(tmp_path: Path) -> None:
    summary = bootstrap_raw_corpora(output_dir=str(tmp_path), overwrite=True)
    assert summary["general_web"] > 0
    sample_path = tmp_path / "general_web.jsonl"
    first = json.loads(sample_path.read_text(encoding="utf-8").splitlines()[0])
    assert "text" in first
    assert len(first["text"]) >= 240


def test_pipeline_writes_manifest_from_bootstrap_data(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("sentencepiece")
    from data import pipeline
    from tokenizer.train_tokenizer import train_sentencepiece, write_training_text

    raw_dir = tmp_path / "raw"
    bootstrap_raw_corpora(output_dir=str(raw_dir), overwrite=True)
    training_text = tmp_path / "training.txt"
    write_training_text([str(path) for path in raw_dir.glob("*.jsonl")], str(training_text))
    prefix = tmp_path / "tokenizer"
    train_sentencepiece(str(training_text), str(prefix), vocab_size=512)

    registry = tuple(
        SourceSpec(
            name=path.stem,
            domain_tag="general",
            quality_tier="high",
            license_category="permissive",
            estimated_tokens=1_000,
            path=str(path),
        )
        for path in raw_dir.glob("*.jsonl")
    )
    monkeypatch.setattr(pipeline, "SOURCE_REGISTRY", registry)

    output_dir = tmp_path / "processed"
    summary = pipeline.run_pipeline(
        tokenizer_model=str(prefix) + ".model",
        output_dir=str(output_dir),
        shard_size=4,
    )
    manifest_path = output_dir / "manifest.json"
    assert summary["records"] > 0
    assert manifest_path.exists()
