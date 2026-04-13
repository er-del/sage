"""Checkpoint-to-checkpoint regression checks."""

from __future__ import annotations


def compare_metrics(previous: dict[str, float], current: dict[str, float], threshold: float = 0.005) -> dict[str, object]:
    """Flag metric drops larger than the configured threshold."""
    regressions: list[str] = []
    for key, prev_value in previous.items():
        curr_value = current.get(key)
        if curr_value is None:
            continue
        if curr_value < prev_value * (1.0 - threshold):
            regressions.append(key)
    return {"regressions": regressions, "passed": not regressions}
