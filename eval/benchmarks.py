"""Benchmark harness registration for SAGE."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    """A normalized benchmark result."""

    name: str
    status: str
    score: float | None
    detail: str


BENCHMARKS = (
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "gsm8k",
    "math",
    "humaneval",
    "mbpp",
)


def run_registered_benchmarks(model, tokenizer=None) -> list[BenchmarkResult]:
    """Return a lightweight result set for the configured benchmarks."""
    _ = model
    _ = tokenizer
    return [
        BenchmarkResult(
            name=name,
            status="skipped",
            score=None,
            detail="Benchmark harness registered; dataset/task execution is external to unit tests.",
        )
        for name in BENCHMARKS
    ]
