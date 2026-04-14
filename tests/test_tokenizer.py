from pathlib import Path
import json

import pytest


def test_validation_suite_roundtrip(tmp_path: Path) -> None:
    spm = pytest.importorskip("sentencepiece")
    from tokenizer.validate_tokenizer import run_validation_suite
    from tokenizer.train_tokenizer import train_sentencepiece

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "def add(a, b):\n    return a + b\n"
        "\\int_0^1 x^2 dx\n"
        "English हिन्दी العربية 中文\n"
        "😀 tabs\tand spaces\n",
        encoding="utf-8",
    )
    prefix = tmp_path / "spm"
    # Byte fallback adds 256 byte pieces plus meta tokens, so tiny vocab sizes
    # can fail even for small corpora. Use the real training helper and leave
    # slack above that floor.
    train_sentencepiece(str(corpus), str(prefix), vocab_size=512)
    results = run_validation_suite(str(prefix) + ".model")
    assert all(result.passed for result in results), results


def test_write_training_text_reads_jsonl(tmp_path: Path) -> None:
    from tokenizer.train_tokenizer import write_training_text

    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps({"text": "first training sample"}),
                json.dumps({"text": "second training sample"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "combined.txt"
    write_training_text([str(raw)], str(output))
    assert output.read_text(encoding="utf-8").splitlines() == ["first training sample", "second training sample"]
