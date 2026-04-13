"""Corpus filtering, safety, and quality heuristics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3})?[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
HTML_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"[ \t]+")
NSFW_TERMS = {"porn", "explicit sex", "rape"}
HATE_TERMS = {"kill all", "ethnic cleansing"}
ALLOWED_LICENSES = {"permissive", "restricted"}
ALLOWED_LANGS = {"en", "es", "fr", "de", "hi", "zh", "ar", "pt"}


@dataclass(frozen=True)
class FilterConfig:
    """Policy controls for the filtering pipeline."""

    minimum_chars: int = 200
    maximum_chars: int = 200_000
    minimum_alpha_ratio: float = 0.45
    minimum_quality_score: float = 0.20
    language_confidence_threshold: float = 0.65


def normalize_text(text: str) -> str:
    """Strip tags and normalize whitespace."""
    text = HTML_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def detect_language(text: str) -> tuple[str, float]:
    """Use a light heuristic to assign a language code."""
    ascii_ratio = sum(ch.isascii() for ch in text) / max(len(text), 1)
    devanagari = sum("\u0900" <= ch <= "\u097f" for ch in text)
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    arabic = sum("\u0600" <= ch <= "\u06ff" for ch in text)
    if cjk > 8:
        return "zh", 0.95
    if arabic > 8:
        return "ar", 0.95
    if devanagari > 8:
        return "hi", 0.95
    if ascii_ratio > 0.9:
        return "en", 0.80
    return "unknown", 0.40


def quality_score(text: str) -> float:
    """Score text using length, punctuation, and alphabetic density."""
    if not text:
        return 0.0
    alpha_ratio = sum(ch.isalpha() for ch in text) / len(text)
    punct_ratio = sum(ch in ".,;:!?()[]{}" for ch in text) / len(text)
    line_count = text.count("\n") + 1
    score = min(len(text) / 4000.0, 1.0) * 0.4 + alpha_ratio * 0.4 + min(punct_ratio * 8.0, 1.0) * 0.2
    if line_count < 2 and len(text) > 10_000:
        score *= 0.85
    return round(score, 4)


def quality_tier(score: float) -> str:
    """Map a numeric score to a quality tier."""
    if score >= 0.70:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def strip_pii(text: str) -> str:
    """Mask basic email, phone, and SSN patterns."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = SSN_RE.sub("[SSN]", text)
    return text


def passes_safety_filter(text: str) -> bool:
    """Reject obviously unsafe content with simple keyword checks."""
    lower = text.lower()
    if any(term in lower for term in NSFW_TERMS):
        return False
    if any(term in lower for term in HATE_TERMS):
        return False
    return True


def license_allowed(category: str) -> bool:
    """Return whether the source license category is allowed."""
    return category in ALLOWED_LICENSES


def filter_record(record: dict[str, object], config: FilterConfig = FilterConfig()) -> dict[str, object] | None:
    """Apply the full filter pipeline to one record."""
    if not license_allowed(str(record.get("license_category", ""))):
        return None
    text = normalize_text(str(record.get("text", "")))
    if not (config.minimum_chars <= len(text) <= config.maximum_chars):
        return None
    lang, confidence = detect_language(text)
    if lang not in ALLOWED_LANGS or confidence < config.language_confidence_threshold:
        return None
    text = strip_pii(text)
    if not passes_safety_filter(text):
        return None
    score = quality_score(text)
    if score < config.minimum_quality_score:
        return None
    return {
        **record,
        "text": text,
        "lang": lang,
        "lang_confidence": confidence,
        "quality_score": score,
        "quality_tier": quality_tier(score),
        "token_count_estimate": max(1, len(text) // 4),
    }


def filter_corpus(records: Iterable[dict[str, object]], config: FilterConfig = FilterConfig()) -> list[dict[str, object]]:
    """Filter a corpus in memory."""
    kept: list[dict[str, object]] = []
    for record in records:
        filtered = filter_record(record, config)
        if filtered is not None:
            kept.append(filtered)
    return kept
