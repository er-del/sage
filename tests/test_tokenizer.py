from pathlib import Path

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
