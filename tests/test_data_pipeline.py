from data.dataset import pack_sequence
from data.dedup import deduplicate_records
from data.filter import filter_record


def test_filter_record_masks_pii() -> None:
    record = {
        "id": "1",
        "text": "Contact me at person@example.com. This document contains enough text to pass the minimum length requirement. " * 4,
        "license_category": "permissive",
        "domain_tag": "general",
        "quality_tier": "high",
    }
    filtered = filter_record(record)
    assert filtered is not None
    assert "[EMAIL]" in filtered["text"]


def test_deduplicate_records_removes_exact_duplicates() -> None:
    records = [
        {"text": "same text", "id": "1"},
        {"text": "same text", "id": "2"},
        {"text": "different text", "id": "3"},
    ]
    kept = deduplicate_records(records)
    assert len(kept) == 2


def test_pack_sequence_shapes() -> None:
    packed = pack_sequence([1, 2, 3, 4, 5], [0, 0, 1, 0, 1])
    assert packed["input_ids"].tolist() == [1, 2, 3, 4]
    assert packed["labels"].tolist() == [2, 3, 4, 5]
    assert packed["document_boundaries"].tolist() == [0, 0, 1, 0]
