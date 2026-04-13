"""SentencePiece tokenizer training for SAGE."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import sentencepiece as spm


DEFAULT_SPECIAL_TOKENS = ("<bos>", "<eos>", "<pad>", "<unk>", "[INST]", "[/INST]")


def write_training_text(corpus_paths: Iterable[str], output_path: str) -> str:
    """Concatenate corpus text into a plain-text file for SentencePiece."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as sink:
        for path in corpus_paths:
            with Path(path).open("r", encoding="utf-8") as source:
                for line in source:
                    line = line.strip()
                    if line:
                        sink.write(line)
                        sink.write("\n")
    return str(output)


def train_sentencepiece(input_path: str, model_prefix: str, vocab_size: int = 50_000) -> None:
    """Train a byte-fallback SentencePiece BPE model."""
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=0.9995,
        byte_fallback=True,
        bos_id=0,
        eos_id=1,
        pad_id=2,
        unk_id=3,
        user_defined_symbols=list(DEFAULT_SPECIAL_TOKENS[4:]),
        split_digits=False,
        split_by_unicode_script=False,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
    )


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Train the SAGE SentencePiece tokenizer.")
    parser.add_argument("--input", nargs="+", required=True, help="Plain-text corpus files.")
    parser.add_argument("--model-prefix", default="tokenizer/tokenizer", help="SentencePiece model prefix.")
    parser.add_argument("--vocab-size", type=int, default=50_000, help="Tokenizer vocabulary size.")
    parser.add_argument("--training-text", default="tokenizer/training_corpus.txt", help="Temporary combined text file.")
    return parser


def main() -> None:
    """Train the tokenizer from CLI arguments."""
    args = build_argparser().parse_args()
    training_text = write_training_text(args.input, args.training_text)
    train_sentencepiece(training_text, args.model_prefix, args.vocab_size)


if __name__ == "__main__":
    main()
