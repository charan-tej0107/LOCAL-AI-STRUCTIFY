"""Text chunking — split text into overlapping chunks for downstream processing.

Supports three strategies:

* ``sentence``  — split on sentence boundaries, merge until *chunk_size*
* ``paragraph`` — split on paragraph boundaries (double newline)
* ``fixed``     — fixed-size character chunks with overlap
"""

from __future__ import annotations

import re
from typing import Literal

from config import settings
from utils import get_logger

logger = get_logger(__name__)

# Sentence boundary pattern (handles ., !, ? followed by whitespace/end).
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

# Paragraph boundary.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    method: Literal["sentence", "paragraph", "fixed"] | None = None,
    separator: str | None = None,
) -> list[str]:
    """Split *text* into a list of chunks.

    Args:
        text: Input text (assumed pre-cleaned).
        chunk_size: Maximum characters per chunk (default from config).
        overlap: Number of characters to overlap between consecutive
                 chunks (default from config).  Only used for ``fixed``
                 method.
        method: Chunking strategy (default from config).
        separator: Joiner for ``sentence`` and ``paragraph`` methods
                   (default from config).

    Returns:
        List of text chunks.  May be empty if *text* is empty.

    Raises:
        ValueError: If *method* is not recognised.
        TypeError: If *text* is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    if not text:
        return []

    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap if overlap is not None else settings.CHUNK_OVERLAP
    method = method or settings.CHUNK_METHOD
    separator = separator or settings.CHUNK_SEPARATOR

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")

    if method == "sentence":
        return _chunk_by_sentences(text, chunk_size, separator)
    elif method == "paragraph":
        return _chunk_by_paragraphs(text, chunk_size, separator)
    elif method == "fixed":
        return _chunk_fixed(text, chunk_size, overlap)
    else:
        raise ValueError(f"Unknown chunk method '{method}' — expected sentence, paragraph, or fixed")


def _chunk_by_sentences(text: str, chunk_size: int, separator: str) -> list[str]:
    """Chunk by sentence boundaries, merging until *chunk_size*."""
    sentences = _SENTENCE_RE.split(text)
    if not sentences:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        sent_len = len(sent) + (len(separator) if current else 0)

        if current_len + sent_len > chunk_size and current:
            chunks.append(separator.join(current))
            current = []
            current_len = 0

        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(separator.join(current))

    logger.debug("Sentence chunking: %d sentences → %d chunks", len(sentences), len(chunks))
    return chunks


def _chunk_by_paragraphs(text: str, chunk_size: int, separator: str) -> list[str]:
    """Chunk by paragraph boundaries, merging until *chunk_size*."""
    paragraphs = _PARAGRAPH_RE.split(text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not paragraphs:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + (len(separator) if current else 0)

        if current_len + para_len > chunk_size and current:
            chunks.append(separator.join(current))
            current = []
            current_len = 0

        current.append(para)
        current_len += para_len

    if current:
        chunks.append(separator.join(current))

    logger.debug("Paragraph chunking: %d paragraphs → %d chunks", len(paragraphs), len(chunks))
    return chunks


def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fixed-size character chunking with overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step

    logger.debug("Fixed chunking: %d chars → %d chunks (size=%d, overlap=%d)", len(text), len(chunks), chunk_size, overlap)
    return chunks
