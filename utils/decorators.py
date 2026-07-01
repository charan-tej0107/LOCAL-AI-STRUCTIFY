"""Reusable decorators — retry, timing, logging, singleton, error handling.

All decorators preserve the wrapped function's metadata via
:func:`functools.wraps`.
"""

from __future__ import annotations

import asyncio
import functools
import time
import traceback
from typing import (
    Any,
    Callable,
    ParamSpec,
    Protocol,
    TypeVar,
    overload,
)

from utils.logger import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])


# ── sync retry ────────────────────────────────────────────────────────


def retry(
    attempts: int = 3,
    delay: float = 1.0,
    max_delay: float = 60.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry a synchronous function with exponential backoff.

    Args:
        attempts: Maximum number of tries.
        delay: Initial delay in seconds.
        max_delay: Maximum delay cap.
        backoff: Multiplier applied to *delay* after each failure.
        exceptions: Exception types that trigger a retry.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _delay = delay
            last_exc: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < attempts:
                        logger.warning(
                            "%s attempt %d/%d failed: %s — retrying in %.2fs",
                            func.__name__,
                            attempt,
                            attempts,
                            exc,
                            _delay,
                        )
                        time.sleep(_delay)
                        _delay = min(_delay * backoff, max_delay)
            raise RuntimeError(
                f"{func.__name__} failed after {attempts} attempts"
            ) from last_exc

        return wrapper

    return decorator


# ── async retry ───────────────────────────────────────────────────────


def retry_async(
    attempts: int = 3,
    delay: float = 1.0,
    max_delay: float = 60.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry an async function with exponential backoff."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _delay = delay
            last_exc: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)  # type: ignore[misc]
                except exceptions as exc:
                    last_exc = exc
                    if attempt < attempts:
                        logger.warning(
                            "%s attempt %d/%d failed: %s — retrying in %.2fs",
                            func.__name__,
                            attempt,
                            attempts,
                            exc,
                            _delay,
                        )
                        await asyncio.sleep(_delay)
                        _delay = min(_delay * backoff, max_delay)
            raise RuntimeError(
                f"{func.__name__} failed after {attempts} attempts"
            ) from last_exc

        return wrapper

    return decorator


# ── timing ────────────────────────────────────────────────────────────


def timing(func: Callable[P, R]) -> Callable[P, R]:
    """Log the execution duration of the wrapped function at DEBUG level."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.debug("%s took %.3f s", func.__name__, elapsed)

    return wrapper


# ── log_call ──────────────────────────────────────────────────────────


def log_call(func: Callable[P, R]) -> Callable[P, R]:
    """Log entry and exit (or exception) of the wrapped function."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger.debug("→ %s called", func.__name__)
        try:
            result = func(*args, **kwargs)
            logger.debug("← %s completed", func.__name__)
            return result
        except Exception:
            logger.exception("✗ %s raised", func.__name__)
            raise

    return wrapper


# ── singleton ─────────────────────────────────────────────────────────


class _SingletonProtocol(Protocol):
    _instance: Any | None


def singleton(cls: type[F]) -> type[F]:
    """Thread-unsafe singleton decorator.

    Usage::

        @singleton
        class DatabaseConnection:
            ...
    """
    original_init = cls.__init__

    @functools.wraps(original_init)  # type: ignore[arg-type]
    def new_init(self: _SingletonProtocol, *args: Any, **kwargs: Any) -> None:
        if not hasattr(cls, "_instance"):
            original_init(self, *args, **kwargs)
            cls._instance = self  # type: ignore[attr-defined]

    cls.__init__ = new_init  # type: ignore[assignment]

    def new_new(cls_: type, *args: Any, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        if not hasattr(cls_, "_instance"):
            instance = super(cls_, cls_).__new__(cls_)
            cls_._instance = instance  # type: ignore[attr-defined]
            return instance
        return cls_._instance  # type: ignore[attr-defined]

    cls.__new__ = new_new  # type: ignore[assignment]
    return cls


# ── handle_exceptions ─────────────────────────────────────────────────


@overload
def handle_exceptions(
    func: None = None,
    *,
    default_return: Any = None,
    log_level: str = "error",
    reraise: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


@overload
def handle_exceptions(
    func: Callable[P, R],
    *,
    default_return: Any = None,
    log_level: str = "error",
    reraise: bool = False,
) -> Callable[P, R]: ...


def handle_exceptions(
    func: Callable[P, R] | None = None,
    *,
    default_return: Any = None,
    log_level: str = "error",
    reraise: bool = False,
) -> Callable[..., Any]:
    """Catch exceptions, log them, and return a default value.

    Can be used with or without arguments::

        @handle_exceptions
        def risky() -> int: ...

        @handle_exceptions(default_return=[], log_level="warning")
        def fetch() -> list: ...
    """

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return f(*args, **kwargs)
            except Exception:
                log_fn = getattr(logger, log_level, logger.error)
                log_fn("Exception in %s\n%s", f.__name__, traceback.format_exc())
                if reraise:
                    raise
                return default_return  # type: ignore[return-value]

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
