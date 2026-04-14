"""SentencePiece tokenizer training for SAGE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator

DEFAULT_SPECIAL_TOKENS = ("<bos>", "<eos>", "<pad>", "<unk>", "[INST]", "[/INST]")


def iter_training_text(corpus_paths: Iterable[str], text_key: str = "text") -> Iterator[str]:
    """Yield training lines from plain-text or JSONL corpus files."""
    for path in corpus_paths:
        source = Path(path)
        suffix = source.suffix.lower()
        with source.open("r", encoding="utf-8") as handle:
            if suffix == ".jsonl":
                for raw_line in handle:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    payload = json.loads(raw_line)
                    text = payload.get(text_key)
                    if isinstance(text, str) and text.strip():
                        yield text.strip()
                continue
            for raw_line in handle:
                text = raw_line.strip()
                if text:
                    yield text


def write_training_text(corpus_paths: Iterable[str], output_path: str, text_key: str = "text") -> str:
    """Concatenate corpus text into a plain-text file for SentencePiece."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as sink:
        for line in iter_training_text(corpus_paths, text_key=text_key):
            sink.write(line)
            sink.write("\n")
    return str(output)


def train_sentencepiece(input_path: str, model_prefix: str, vocab_size: int = 50_000) -> None:
    """Train a byte-fallback SentencePiece BPE model."""
    import sentencepiece as spm

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
        hard_vocab_limit=False,
    )


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Train the SAGE SentencePiece tokenizer.")
    parser.add_argument("--input", nargs="+", required=True, help="Plain-text or JSONL corpus files.")
    parser.add_argument("--model-prefix", default="tokenizer/tokenizer", help="SentencePiece model prefix.")
    parser.add_argument("--vocab-size", type=int, default=50_000, help="Tokenizer vocabulary size.")
    parser.add_argument("--training-text", default="tokenizer/training_corpus.txt", help="Temporary combined text file.")
    parser.add_argument("--text-key", default="text", help="JSONL field to read when --input contains .jsonl files.")
    return parser


def main() -> None:
    """Train the tokenizer from CLI arguments."""
    args = build_argparser().parse_args()
    training_text = write_training_text(args.input, args.training_text, text_key=args.text_key)
    train_sentencepiece(training_text, args.model_prefix, args.vocab_size)


if __name__ == "__main__":
    main()
