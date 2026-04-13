"""Raw corpus ingestion utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class SourceSpec:
    """Describes one raw corpus source."""

    name: str
    domain_tag: str
    quality_tier: str
    license_category: str
    estimated_tokens: int
    path: str
    text_key: str = "text"


SOURCE_REGISTRY: tuple[SourceSpec, ...] = (
    SourceSpec("general_web", "general", "medium", "permissive", 20_000_000_000, "data/raw/general_web.jsonl"),
    SourceSpec("code", "code", "high", "permissive", 8_000_000_000, "data/raw/code.jsonl"),
    SourceSpec("math_science", "math", "high", "permissive", 4_000_000_000, "data/raw/math_science.jsonl"),
    SourceSpec("books_longform", "general", "high", "restricted", 5_000_000_000, "data/raw/books.jsonl"),
    SourceSpec("multilingual", "multilingual", "medium", "permissive", 3_000_000_000, "data/raw/multilingual.jsonl"),
    SourceSpec("synthetic", "reasoning", "high", "permissive", 1_000_000_000, "data/raw/synthetic.jsonl"),
)


def iter_jsonl(path: Path, text_key: str = "text") -> Iterator[dict[str, object]]:
    """Yield JSONL records from disk."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            text = payload.get(text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            yield payload


def stream_source(spec: SourceSpec) -> Iterator[dict[str, object]]:
    """Yield normalized records for one configured source."""
    path = Path(spec.path)
    if not path.exists():
        return iter(())
    return (
        {
            "id": stable_record_id(spec.name, record[spec.text_key]),
            "text": record[spec.text_key],
            "domain_tag": spec.domain_tag,
            "quality_tier": spec.quality_tier,
            "license_category": spec.license_category,
            "source_name": spec.name,
        }
        for record in iter_jsonl(path, spec.text_key)
    )


def stream_all_sources(sources: Iterable[SourceSpec] = SOURCE_REGISTRY) -> Iterator[dict[str, object]]:
    """Yield records from every source in the registry."""
    for spec in sources:
        yield from stream_source(spec)


def stable_record_id(source_name: str, text: str) -> str:
    """Hash a source+text pair into a stable content id."""
    digest = hashlib.sha256()
    digest.update(source_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()
