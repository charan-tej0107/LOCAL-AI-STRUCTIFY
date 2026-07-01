"""Configurable retry with exponential backoff for AI inference calls."""

from __future__ import annotations

import time
from typing import Any, Callable

from config import settings
from utils import get_logger

logger = get_logger(__name__)


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
                if attempt < self.max_attempts:
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %.1fs…",
                        attempt,
                        self.max_attempts,
                        _name(fn),
                        exc,
                        delay,
                    )
                    time.sleep(delay)
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
