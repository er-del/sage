"""CLI entrypoint for the registered benchmark suite."""

from __future__ import annotations

from model.config import ModelConfig
from model.model import SageTransformer

from eval.benchmarks import run_registered_benchmarks


def main() -> None:
    """Run the benchmark registry against a default model instance."""
    model = SageTransformer(ModelConfig())
    for result in run_registered_benchmarks(model):
        print(result)


if __name__ == "__main__":
    main()
