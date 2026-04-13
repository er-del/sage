"""Validation checks for the SentencePiece tokenizer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm


@dataclass(frozen=True)
class ValidationResult:
    """One tokenizer validation outcome."""

    name: str
    passed: bool
    detail: str


def load_processor(model_path: str) -> spm.SentencePieceProcessor:
    """Load a SentencePiece processor."""
    processor = spm.SentencePieceProcessor()
    processor.load(model_path)
    return processor


def validate_roundtrip(processor: spm.SentencePieceProcessor, text: str, name: str) -> ValidationResult:
    """Ensure encode->decode preserves the original string."""
    pieces = processor.encode(text, out_type=int)
    decoded = processor.decode(pieces)
    return ValidationResult(name, decoded == text, f"expected={text!r} got={decoded!r}")


def run_validation_suite(model_path: str) -> list[ValidationResult]:
    """Run the required tokenizer smoke tests."""
    processor = load_processor(model_path)
    samples = {
        "python": "def add(a, b):\n    return a += b if a == b else a != b\n",
        "latex": r"\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}",
        "whitespace": "if True:\n\tprint('tabs')\n    print('spaces')\n",
        "emoji": "Rare bytes: 😀 ⚙️ ∑",
        "multilingual": "English हिन्दी العربية 中文",
    }
    return [validate_roundtrip(processor, text, name) for name, text in samples.items()]


def validate_model_file(model_path: str) -> None:
    """Raise on validation failure."""
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)
    results = run_validation_suite(model_path)
    failed = [result for result in results if not result.passed]
    if failed:
        details = "\n".join(f"{item.name}: {item.detail}" for item in failed)
        raise AssertionError(f"Tokenizer validation failed:\n{details}")


def build_argparser() -> argparse.ArgumentParser:
    """Build the tokenizer validation CLI."""
    parser = argparse.ArgumentParser(description="Validate a SentencePiece tokenizer model.")
    parser.add_argument("model_path", nargs="?", default="tokenizer/tokenizer.model")
    return parser


def main() -> None:
    """CLI entrypoint for tokenizer validation."""
    args = build_argparser().parse_args()
    validate_model_file(args.model_path)
    print("tokenizer ok")


if __name__ == "__main__":
    main()
