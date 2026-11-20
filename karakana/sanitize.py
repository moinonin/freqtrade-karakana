"""Post-collapse text sanitization.

Cleans up LLM output after a structural collapse is detected:
removes repetitive phrases, adds basic punctuation, fixes capitalization,
and truncates at the last coherent sentence boundary.

Usage:
    from karakana.sanitize import sanitize_collapsed_text

    clean = sanitize_collapsed_text(raw_output, collapse_chunk=15, chunk_size=12)
"""

from __future__ import annotations

import re
from collections import OrderedDict


def sanitize_collapsed_text(
    text: str,
    collapse_chunk: int | None = None,
    chunk_size: int = 12,
) -> str:
    """Clean up text produced after a structural collapse.

    Strategy:
    1. If collapse_chunk is known, keep only text before the collapse point.
    2. Remove duplicate/repetitive phrases.
    3. Add basic sentence punctuation where missing.
    4. Fix run-on capitalization.
    5. Remove trailing fragments (text that doesn't end at a sentence boundary).

    Parameters
    ----------
    text : str
        Raw generated text, possibly degraded.
    collapse_chunk : int or None
        The chunk index where collapse was flagged. If provided, text is
        truncated conservatively before this point.
    chunk_size : int
        Tokens per chunk (used to estimate the safe truncation point).

    Returns
    -------
    str
        Cleaned, readable text.
    """
    if not text or not text.strip():
        return ""

    text = text.strip()

    # ── 1. Truncate at collapse point ──────────────────────────────
    if collapse_chunk is not None and collapse_chunk > 0:
        safe_words = int(collapse_chunk * chunk_size * 0.75)
        words = text.split()
        if safe_words < len(words):
            text = " ".join(words[:safe_words])

    # ── 2. Remove repetitive phrases ───────────────────────────────
    text = _remove_repetition(text)

    # ── 3. Sentence boundary punctuation ───────────────────────────
    text = _add_punctuation(text)

    # ── 4. Fix capitalization ──────────────────────────────────────
    text = _fix_capitalization(text)

    # ── 5. Trim to last complete sentence ──────────────────────────
    text = _trim_to_sentence_boundary(text)

    return text


def _remove_repetition(text: str) -> str:
    """Remove duplicate consecutive phrases (common in collapse output)."""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return text

    seen: OrderedDict[str, bool] = OrderedDict()
    unique: list[str] = []

    for sent in sentences:
        key = sent.strip().lower()
        if unique and _similarity(key, unique[-1].strip().lower()) > 0.7:
            continue
        if key not in seen:
            seen[key] = True
            unique.append(sent)
        elif len(key.split()) > 15:
            break

    return " ".join(unique)


def _add_punctuation(text: str) -> str:
    """Insert periods at natural sentence boundaries.

    Only inserts a period when a lowercase word is immediately followed
    by a known sentence-starting word (pronouns, articles, conjunctions,
    question words, etc.) with no punctuation in between.
    This avoids false positives on proper nouns and acronyms.
    """
    sentence_starters = {
        "This",
        "These",
        "Those",
        "That",
        "It",
        "They",
        "We",
        "You",
        "I",
        "He",
        "She",
        "What",
        "When",
        "Where",
        "Which",
        "Why",
        "How",
        "Who",
        "However",
        "Therefore",
        "Moreover",
        "Furthermore",
        "Additionally",
        "Consequently",
        "Nevertheless",
        "Meanwhile",
        "Instead",
        "Otherwise",
        "There",
        "Here",
        "Now",
        "Then",
        "So",
        "But",
        "And",
        "If",
        "The",
        "A",
        "An",
        "One",
        "Two",
        "Three",
        "First",
        "Second",
        "Third",
        "Let",
        "Each",
        "Every",
        "Any",
        "Some",
        "All",
        "No",
        "Do",
        "Does",
        "Can",
        "Will",
        "Would",
        "Should",
        "Could",
        "May",
        "Might",
        "Once",
        "While",
        "After",
        "Before",
        "During",
        "Until",
        "Since",
        "In",
        "For",
        "As",
        "To",
        "Of",
        "On",
        "At",
        "By",
        "From",
        "With",
        "Without",
    }
    for word in sorted(sentence_starters, key=len, reverse=True):
        text = re.sub(
            rf"([a-z])\s+({word}\s)",
            lambda m, w=word: f"{m.group(1)}. {w} ",
            text,
        )

    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def _fix_capitalization(text: str) -> str:
    """Normalize capitalization of sentences."""
    sentences = _split_sentences(text)
    fixed = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent = sent[0].upper() + sent[1:] if len(sent) > 1 else sent.upper()
        fixed.append(sent)
    return " ".join(fixed)


def _trim_to_sentence_boundary(text: str) -> str:
    """Trim text at the last complete sentence boundary."""
    matches = list(re.finditer(r"[.!?]\s+[A-Z]|[.!?]$", text))
    if not matches:
        last_period = text.rfind(".")
        if last_period > len(text) * 0.3:
            return text[: last_period + 1]
        return text

    last_match = matches[-1]
    if text[last_match.start() : last_match.end()].strip()[-1] in ".!?":
        return text[: last_match.start() + 1].strip()
    return text[: last_match.end()].strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into rough sentence-like segments."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    result = []
    for part in parts:
        sub = re.split(r"(?<=[a-z])\s+(?=[A-Z][a-z])", part)
        result.extend(sub)
    return [p.strip() for p in result if p.strip()]


def _similarity(a: str, b: str) -> float:
    """Jaccard similarity between two strings (word-level)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def sanitize_from_report(
    text: str,
    report: object,
) -> str:
    """Clean up text using collapse metadata from a StreamingAlphaReport."""
    collapse_chunk = None
    if getattr(report, "collapsed", False):
        collapse_chunk = getattr(report, "collapse_chunk", None)
    return sanitize_collapsed_text(text, collapse_chunk=collapse_chunk)
