"""Configurable retry with exponential backoff for AI inference calls."""

from __future__ import annotations

import time
from typing import Any, Callable

from config import settings
from utils import get_logger

logger = get_logger(__name__)

_PERMANENT_HTTP_STATUSES = frozenset({400, 401, 403, 404, 405, 406, 422})


class RetryHandler:
    """Wrap a callable with exponential-backoff retry logic.

    Args:
        max_attempts: Maximum number of attempts (Default from config).
        base_delay: Initial delay in seconds (Default from config).
        max_delay: Maximum delay cap in seconds (Default from config).
        backoff: Multiplier applied after each attempt (Default from config).

    Usage::

        handler = RetryHandler(max_attempts=3)
        result = handler.execute(my_fn, arg1, arg2, timeout=10)
    """

    def __init__(
        self,
        max_attempts: int | None = None,
        base_delay: float | None = None,
        max_delay: float | None = None,
        backoff: float | None = None,
    ) -> None:
        self.max_attempts = max_attempts or settings.LLM_RETRY_ATTEMPTS
        self.base_delay = base_delay or settings.RETRY_BASE_DELAY
        self.max_delay = max_delay or settings.RETRY_MAX_DELAY
        self.backoff = backoff or settings.RETRY_BACKOFF_FACTOR

    def execute(
        self,
        fn: Callable[..., Any],
        *args: Any,
        raise_on_failure: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute *fn* with retries.

        Args:
            fn: The callable to invoke.
            *args: Positional arguments for *fn*.
            raise_on_failure: If ``True``, re-raise the last exception
                after all attempts are exhausted.
            **kwargs: Keyword arguments for *fn*.

        Returns:
            The return value of *fn*, or ``None`` if all attempts failed
            and *raise_on_failure* is ``False``.
        """
        last_exc: Exception | None = None
        delay = self.base_delay

        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc

                # Do not retry permanent HTTP errors.
                if _is_permanent(exc):
                    logger.warning(
                        "Permanent failure on attempt %d/%d for %s: %s",
                        attempt,
                        self.max_attempts,
                        _name(fn),
                        exc,
                    )
                    raise

                if attempt < self.max_attempts:
                    retry_after = _parse_retry_after(exc)
                    actual_delay = retry_after if retry_after is not None else delay
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %.1fs…",
                        attempt,
                        self.max_attempts,
                        _name(fn),
                        exc,
                        actual_delay,
                    )
                    time.sleep(actual_delay)
                    delay = min(delay * self.backoff, self.max_delay)
                else:
                    logger.error(
                        "All %d attempts failed for %s: %s",
                        self.max_attempts,
                        _name(fn),
                        exc,
                    )

        if raise_on_failure and last_exc is not None:
            raise last_exc

        return None


def _name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", str(fn))


def _is_permanent(exc: Exception) -> bool:
    """Check if *exc* indicates a permanent (non-retryable) failure."""
    try:
        import httpx
    except ImportError:
        return False

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _PERMANENT_HTTP_STATUSES
    return False


def _parse_retry_after(exc: Exception) -> float | None:
    """Extract ``Retry-After`` header value from an HTTP response.

    Returns the number of seconds to wait, or ``None`` if the header
    is not present or cannot be parsed.
    """
    try:
        import httpx
    except ImportError:
        return None

    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        value = exc.response.headers.get("Retry-After")
        if value is not None:
            try:
                return float(value)
            except (ValueError, TypeError):
                pass
    return None
