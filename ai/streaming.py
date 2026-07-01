"""Streaming response handling for AI inference.

Provides a :class:`TokenCollector` that accumulates tokens yielded by a
stream generator and returns a completed :class:`AIResult`.
"""

from __future__ import annotations

import time
from typing import Generator

from ai.models import AIResult


def consume_stream(
    stream: Generator[str, None, None],
    model_name: str = "",
) -> AIResult:
    """Consume a token stream and assemble the final :class:`AIResult`.

    Args:
        stream: A generator yielding text tokens.
        model_name: Name of the model used (for the result).

    Returns:
        An :class:`AIResult` with the full text and metadata.
    """
    collector = TokenCollector(model_name=model_name)
    for token in stream:
        collector.add(token)
    return collector.result()


class TokenCollector:
    """Accumulates streaming tokens into an :class:`AIResult`.

    Usage::

        collector = TokenCollector(model_name="llama3")
        for token in response.iter_lines():
            collector.add(token.decode())
        result = collector.result()
    """

    def __init__(self, model_name: str = "") -> None:
        self._tokens: list[str] = []
        self._start = time.perf_counter()
        self._model_name = model_name
        self._error: str | None = None

    def add(self, token: str) -> None:
        """Append a single token to the accumulated output."""
        self._tokens.append(token)

    def set_error(self, message: str) -> None:
        """Mark the stream as failed with an error message."""
        self._error = message

    @property
    def full_text(self) -> str:
        """Return the complete accumulated text."""
        return "".join(self._tokens)

    @property
    def token_count(self) -> int:
        """Approximate token count (whitespace-split words)."""
        return len(self.full_text.split())

    def result(self) -> AIResult:
        """Build and return the final :class:`AIResult`."""
        elapsed = time.perf_counter() - self._start
        text = self.full_text
        return AIResult(
            success=self._error is None,
            text=text,
            model_used=self._model_name,
            tokens_out=self.token_count,
            tokens_in=0,
            processing_time_seconds=round(elapsed, 3),
            error=self._error or "",
        )
