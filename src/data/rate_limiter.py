"""
Token-bucket rate limiter for controlling outbound HTTP request rates.

Each data source (NSE, BSE, yfinance, news) has its own limiter instance
to avoid IP bans.

The :class:`RateLimiter` implements a true token-bucket algorithm:
tokens are added at a constant rate up to a maximum bucket capacity.
This allows short bursts while enforcing a long-term average.

API
---
- :meth:`RateLimiter.acquire` — non-blocking: consume one token or fail
- :meth:`RateLimiter.wait_and_acquire` — blocking: wait until a token is available
- :meth:`RateLimiter.remaining` — query available tokens without consuming

Factory helpers: :func:`create_nse_limiter`, :func:`create_bse_limiter`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synchronous token-bucket limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """
    Thread-safe token-bucket rate limiter.

    Tokens refill continuously at *max_requests* / *window_seconds* tokens per
    second, up to a maximum of *max_requests*.  The bucket starts full.

    Parameters
    ----------
    max_requests:
        Maximum number of requests (= bucket capacity) allowed per
        *window_seconds*.
    window_seconds:
        Rolling window duration in seconds (default 60).
    name:
        Identifier used in log messages.

    Examples
    --------
    ::

        limiter = create_nse_limiter()

        # Non-blocking: try to acquire
        if limiter.acquire():
            response = requests.get(url)

        # Blocking: wait for a token (use as context manager)
        with limiter:
            response = requests.get(url)
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
        name: str = "default",
    ) -> None:
        self._max_tokens: float = float(max_requests)
        self._refill_rate: float = max_requests / window_seconds  # tokens/second
        self._tokens: float = float(max_requests)  # bucket starts full
        self._last_refill: float = time.monotonic()
        self._name = name
        self._lock = Lock()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _refill(self) -> None:
        """Refill the bucket based on elapsed time (must be called under lock)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    # ── Public API ────────────────────────────────────────────────────────────

    def acquire(self) -> bool:
        """
        Non-blocking: consume one token if available.

        Returns
        -------
        bool:
            ``True`` if a token was consumed; ``False`` if the bucket is empty.
        """
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                logger.debug("[%s] Token acquired (%.1f remaining)", self._name, self._tokens)
                return True
            logger.debug("[%s] Bucket empty — acquire() returned False", self._name)
            return False

    def wait_and_acquire(self) -> None:
        """
        Blocking: wait until a token is available, then consume it.

        Sleeps in short increments to avoid busy-waiting.
        """
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    logger.debug("[%s] Token acquired after wait", self._name)
                    return
                # Time until 1 token becomes available
                wait = (1.0 - self._tokens) / self._refill_rate

            logger.debug("[%s] Rate limit — sleeping %.3fs", self._name, wait)
            time.sleep(min(wait, 0.05))  # poll at most every 50ms

    def remaining(self) -> int:
        """
        Return the approximate number of tokens currently available.

        The value is floored and may be slightly stale by the time it is read.
        """
        with self._lock:
            self._refill()
            return int(self._tokens)

    # ── Context manager (blocking) ────────────────────────────────────────────

    def __enter__(self) -> "RateLimiter":
        self.wait_and_acquire()
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"RateLimiter(name={self._name!r}, "
            f"max={int(self._max_tokens)}, "
            f"remaining={self.remaining()})"
        )


# ---------------------------------------------------------------------------
# Async token-bucket limiter
# ---------------------------------------------------------------------------


class AsyncRateLimiter:
    """
    Asyncio-compatible token-bucket rate limiter.

    Drop-in async replacement for :class:`RateLimiter`. Use with ``async with``.

    Parameters
    ----------
    max_requests:
        Bucket capacity.
    window_seconds:
        Refill window in seconds (default 60).
    name:
        Identifier used in log messages.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
        name: str = "default",
    ) -> None:
        self._max_tokens: float = float(max_requests)
        self._refill_rate: float = max_requests / window_seconds
        self._tokens: float = float(max_requests)
        self._last_refill: float = time.monotonic()
        self._name = name
        self._lock: Optional[asyncio.Lock] = None  # created lazily inside the event loop

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    async def acquire(self) -> None:
        """Suspend until a request token is available, then consume it."""
        async with self._get_lock():
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    logger.debug("[%s] Async token acquired", self._name)
                    return
                wait = (1.0 - self._tokens) / self._refill_rate
                logger.debug("[%s] Async rate limit — sleeping %.3fs", self._name, wait)
                await asyncio.sleep(min(wait, 0.05))

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, *_: object) -> None:
        pass

    def __repr__(self) -> str:
        return f"AsyncRateLimiter(name={self._name!r}, max={int(self._max_tokens)})"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_nse_limiter(name: str = "nse") -> RateLimiter:
    """
    Return a :class:`RateLimiter` configured for NSE API calls.

    Limit: 25 requests per 60 seconds.
    """
    return RateLimiter(max_requests=25, window_seconds=60.0, name=name)


def create_bse_limiter(name: str = "bse") -> RateLimiter:
    """
    Return a :class:`RateLimiter` configured for BSE API calls.

    Limit: 15 requests per 60 seconds.
    """
    return RateLimiter(max_requests=15, window_seconds=60.0, name=name)


def create_async_nse_limiter(name: str = "nse_async") -> AsyncRateLimiter:
    """Async variant of :func:`create_nse_limiter`."""
    return AsyncRateLimiter(max_requests=25, window_seconds=60.0, name=name)


def create_async_bse_limiter(name: str = "bse_async") -> AsyncRateLimiter:
    """Async variant of :func:`create_bse_limiter`."""
    return AsyncRateLimiter(max_requests=15, window_seconds=60.0, name=name)
