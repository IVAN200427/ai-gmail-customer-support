import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic_ai.exceptions import (
    FallbackExceptionGroup,
    ModelAPIError,
    ModelQuotaExceededError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_ERRORS = (ModelAPIError, FallbackExceptionGroup, TimeoutError)
DEFAULT_ATTEMPTS = 4
BASE_DELAY_SECONDS = 2.0
MAX_RETRY_AFTER_SECONDS = 300.0


def _delay_seconds(attempt: int) -> float:
    return BASE_DELAY_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)


def _describe_error(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        details = "; ".join(_describe_error(sub) for sub in exc.exceptions)
        return f"{type(exc).__name__} ({details})"
    return f"{type(exc).__name__}: {exc}"


def _flatten(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _flatten(sub)]
    return [exc]


def _provider_retry_after(exc: BaseException) -> float | None:
    if not isinstance(exc, ModelAPIError):
        return None
    if exc.retry_after is not None:
        return exc.retry_after
    if isinstance(exc.body, dict):
        error = exc.body.get("error")
        if isinstance(error, dict):
            for detail in error.get("details", []):
                if not isinstance(detail, dict):
                    continue
                delay = detail.get("retryDelay")
                if isinstance(delay, str) and delay.endswith("s"):
                    try:
                        return float(delay[:-1])
                    except ValueError:
                        return None
    return None


def _is_short_retry(exc: BaseException) -> bool:
    if isinstance(exc, ModelQuotaExceededError):
        return False
    message = str(exc).lower()
    if "per day" in message or "daily" in message:
        return False
    retry_after = _provider_retry_after(exc)
    return retry_after is None or retry_after <= MAX_RETRY_AFTER_SECONDS


def _retry_delay(exc: BaseException, attempt: int) -> float | None:
    leaves = _flatten(exc)
    if not any(_is_short_retry(leaf) for leaf in leaves):
        return None
    retry_afters = [
        delay
        for leaf in leaves
        if _is_short_retry(leaf)
        if (delay := _provider_retry_after(leaf)) is not None
    ]
    provider_delay = min(retry_afters) if retry_afters else 0.0
    return max(provider_delay, _delay_seconds(attempt))


async def call_with_retry(
    operation: str,
    call: Callable[[], Awaitable[T]],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
) -> T:
    """Retry transient model failures, such as provider overload or a dropped connection."""
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except RETRYABLE_ERRORS as exc:
            delay = _retry_delay(exc, attempt)
            if delay is None:
                logger.error(
                    "%s failed and will not retry (quota or long provider delay): %s",
                    operation,
                    _describe_error(exc),
                )
                raise
            if attempt == attempts:
                logger.error(
                    "%s failed after %d attempts: %s",
                    operation,
                    attempts,
                    _describe_error(exc),
                )
                raise
            logger.warning(
                "%s failed on attempt %d/%d (%s). Retrying in %.1fs.",
                operation,
                attempt,
                attempts,
                _describe_error(exc),
                delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def call_with_retry_sync(
    operation: str,
    call: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
) -> T:
    """Sync version for the poller and other non-async callers."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except RETRYABLE_ERRORS as exc:
            delay = _retry_delay(exc, attempt)
            if delay is None:
                logger.error(
                    "%s failed and will not retry (quota or long provider delay): %s",
                    operation,
                    _describe_error(exc),
                )
                raise
            if attempt == attempts:
                logger.error(
                    "%s failed after %d attempts: %s",
                    operation,
                    attempts,
                    _describe_error(exc),
                )
                raise
            logger.warning(
                "%s failed on attempt %d/%d (%s). Retrying in %.1fs.",
                operation,
                attempt,
                attempts,
                _describe_error(exc),
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")
