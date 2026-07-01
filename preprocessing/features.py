"""Feature extraction — computes text statistics without AI.

Extracted features include word count, character count, sentence count,
average word length, estimated reading time, and vocabulary richness.
"""

from __future__ import annotations

import math
import re
from typing import Any

from config import settings
from utils import get_logger

logger = get_logger(__name__)

# Rough reading speed: words per minute for silent reading.
_WPM: float = 238.0  # average adult silent reading speed

# Sentence-ending punctuation.
_SENTENCE_END_RE = re.compile(r"[.!?]+")


def extract_features(text: str, features: list[str] | None = None) -> dict[str, Any]:
    """Extract a dictionary of features from *text*.

    Args:
        text: Input text.
        features: List of feature names to compute.  Defaults to
                  :attr:`settings.FEATURES_EXTRACT`.

    Returns:
        Dictionary mapping feature names to computed values.  Unrecognised
        feature names are silently skipped.

    Raises:
        TypeError: If *text* is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    if features is None:
        features = settings.FEATURES_EXTRACT

    result: dict[str, Any] = {}
    words = _tokenize(text)
    chars = len(text)
    sentences = _count_sentences(text)

    for feature in features:
        try:
            value = _compute_feature(feature, text, words, chars, sentences)
            result[feature] = value
        except Exception as exc:
            logger.warning("Feature '%s' failed: %s", feature, exc)
            result[feature] = None

    logger.debug("Extracted %d features from %d chars", len(result), chars)
    return result


def _tokenize(text: str) -> list[str]:
    """Split text into words (alphanumeric sequences)."""
    return re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", text)


def _count_sentences(text: str) -> int:
    """Count roughly how many sentences are in *text*."""
    matches = _SENTENCE_END_RE.findall(text)
    count = len(matches) if matches else (1 if text.strip() else 0)
    return count


def _compute_feature(feature: str, text: str, words: list[str], chars: int, sentences: int) -> Any:
    """Dispatch a single feature computation by name."""
    word_count = len(words)

    if feature == "word_count":
        return word_count
    elif feature == "char_count":
        return chars
    elif feature == "char_count_no_spaces":
        return chars - text.count(" ")
    elif feature == "sentence_count":
        return sentences
    elif feature == "avg_word_length":
        return round(sum(len(w) for w in words) / word_count, 2) if word_count else 0.0
    elif feature == "reading_time_seconds":
        return round(word_count / _WPM * 60, 1) if word_count > 0 else 0.0
    elif feature == "vocabulary_count":
        return len({w.lower() for w in words})
    elif feature == "vocabulary_richness":
        vocab = len({w.lower() for w in words})
        return round(vocab / word_count, 4) if word_count else 0.0
    elif feature == "max_word_length":
        return max((len(w) for w in words), default=0)
    elif feature == "avg_sentence_length":
        return round(word_count / sentences, 1) if sentences else 0.0
    else:
        logger.debug("Unknown feature '%s' — skipping", feature)
        return None
