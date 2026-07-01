"""Text cleaning and Unicode normalization — pure string operations.

All functions operate on plain text and have no dependency on OCR or AI.
"""

from __future__ import annotations

import re
import unicodedata

from config import settings
from utils import get_logger

logger = get_logger(__name__)

# HTML tag pattern.
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Control characters (excluding common whitespace).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Multiple whitespace pattern.
_MULTI_WS_RE = re.compile(r"[ \t]+")

# Multiple newline pattern.
_MULTI_NL_RE = re.compile(r"\n{3,}")


def clean_text(
    text: str,
    strip_html: bool | None = None,
    collapse_whitespace: bool | None = None,
    remove_control_chars: bool | None = None,
    strip_punctuation: bool | None = None,
) -> str:
    """Clean a text string by removing HTML tags, control characters,
    and normalizing whitespace.

    Args:
        text: Input text.
        strip_html: Remove HTML tags.
        collapse_whitespace: Collapse multiple spaces/tabs into one.
        remove_control_chars: Strip non-printable control characters.
        strip_punctuation: Remove common punctuation characters.

    Returns:
        Cleaned text.

    Raises:
        TypeError: If *text* is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    original_length = len(text)

    strip_html = strip_html if strip_html is not None else settings.CLEAN_STRIP_HTML
    collapse_whitespace = collapse_whitespace if collapse_whitespace is not None else settings.CLEAN_COLLAPSE_WHITESPACE
    remove_control_chars = remove_control_chars if remove_control_chars is not None else settings.CLEAN_REMOVE_CONTROL_CHARS
    strip_punctuation = strip_punctuation if strip_punctuation is not None else settings.CLEAN_STRIP_PUNCTUATION

    if strip_html:
        text = _HTML_TAG_RE.sub("", text)
        logger.debug("Stripped HTML tags")

    if remove_control_chars:
        text = _CONTROL_CHARS_RE.sub("", text)
        logger.debug("Removed control characters")

    if collapse_whitespace:
        text = _MULTI_WS_RE.sub(" ", text)
        text = _MULTI_NL_RE.sub("\n\n", text)
        text = text.strip()
        logger.debug("Collapsed whitespace")

    if strip_punctuation:
        text = re.sub(r"[^\w\s]", "", text)
        logger.debug("Stripped punctuation")

    new_length = len(text)
    if new_length < original_length:
        logger.debug("clean_text: %d → %d chars (removed %d)", original_length, new_length, original_length - new_length)

    return text


def normalize_unicode(text: str, form: str | None = None) -> str:
    """Normalize Unicode in *text* to the specified form.

    Args:
        text: Input text.
        form: Normalization form — ``"NFC"``, ``"NFD"``, ``"NFKC"``,
              ``"NFKD"``.  Defaults to :attr:`settings.UNICODE_NORMALIZATION_FORM`.

    Returns:
        Unicode-normalized text.

    Raises:
        ValueError: If *form* is not a valid Unicode normalization form.
    """
    valid_forms = {"NFC", "NFD", "NFKC", "NFKD"}
    form = (form or settings.UNICODE_NORMALIZATION_FORM).upper()

    if form not in valid_forms:
        logger.warning("Invalid normalization form '%s', falling back to NFC", form)
        form = "NFC"

    result = unicodedata.normalize(form, text)
    if result != text:
        logger.debug("Unicode normalization %s applied (%d → %d chars)", form, len(text), len(result))

    return result
