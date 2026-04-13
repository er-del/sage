"""Long-context retrieval evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalProbe:
    """A synthetic retrieval probe for long-context checks."""

    prompt: str
    needle: str
    expected_index: int


def build_needle_in_haystack_probe(context_length: int) -> RetrievalProbe:
    """Create a deterministic retrieval prompt for smoke tests."""
    needle = "SAGE_LONG_CONTEXT_NEEDLE"
    haystack = ["token"] * max(context_length - 16, 16)
    insert_at = min(len(haystack) // 2, max(context_length // 4, 1))
    haystack.insert(insert_at, needle)
    prompt = " ".join(haystack)
    return RetrievalProbe(prompt=prompt, needle=needle, expected_index=insert_at)
