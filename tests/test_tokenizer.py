from pathlib import Path

import pytest


def test_validation_suite_roundtrip(tmp_path: Path) -> None:
    spm = pytest.importorskip("sentencepiece")
    from tokenizer.validate_tokenizer import run_validation_suite

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "def add(a, b):\n    return a + b\n"
        "\\int_0^1 x^2 dx\n"
        "English हिन्दी العربية 中文\n"
        "😀 tabs\tand spaces\n",
        encoding="utf-8",
    )
    prefix = tmp_path / "spm"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=300,
        character_coverage=0.9995,
        byte_fallback=True,
        bos_id=0,
        eos_id=1,
        pad_id=2,
        unk_id=3,
        user_defined_symbols=["[INST]", "[/INST]"],
        split_digits=False,
        split_by_unicode_script=False,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
    )
    results = run_validation_suite(str(prefix) + ".model")
    assert all(result.passed for result in results), results
