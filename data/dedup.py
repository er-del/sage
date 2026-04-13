"""Exact and near-duplicate detection helpers."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Iterable


TOKEN_RE = re.compile(r"\w+")


def exact_content_hash(text: str) -> str:
    """Return an exact content hash."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def shingles(text: str, n: int = 5) -> set[str]:
    """Build token shingles for near-duplicate detection."""
    tokens = TOKEN_RE.findall(text.lower())
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(left: str, right: str, n: int = 5) -> float:
    """Compute shingle-level Jaccard similarity."""
    left_set = shingles(left, n)
    right_set = shingles(right, n)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def deduplicate_records(records: Iterable[dict[str, object]], near_dup_threshold: float = 0.92) -> list[dict[str, object]]:
    """Drop exact and near-duplicate records."""
    exact_seen: set[str] = set()
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    kept: list[dict[str, object]] = []
    for record in records:
        text = str(record["text"])
        digest = exact_content_hash(text)
        if digest in exact_seen:
            continue
        signature = digest[:8]
        near_duplicate = False
        for candidate in buckets[signature]:
            if jaccard_similarity(text, str(candidate["text"])) >= near_dup_threshold:
                near_duplicate = True
                break
        if near_duplicate:
            continue
        exact_seen.add(digest)
        buckets[signature].append(record)
        kept.append(record)
    return kept
