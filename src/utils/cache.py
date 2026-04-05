"""
In-memory TTL cache with thread-safe access and hit/miss statistics.

Provides a simple key-value cache where each entry expires after a
configured TTL.  Used to avoid hammering NSE/BSE APIs on every request.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _CacheEntry:
    """Internal cache entry with expiry timestamp."""

    value: Any
    expires_at: float  # monotonic time


class TTLCache(Generic[T]):
    """
    Thread-safe in-memory cache with per-key TTL and access statistics.

    Parameters
    ----------
    default_ttl:
        Default time-to-live in seconds for new entries.
    max_size:
        Maximum number of entries. Oldest-expiry entries are evicted when full.
        ``0`` means unlimited.

    Examples
    --------
    >>> cache: TTLCache[dict] = TTLCache(default_ttl=60)
    >>> cache.set("nifty50", {"price": 22000})
    >>> cached = cache.get("nifty50")
    >>> stats = cache.stats()
    >>> print(stats["hit_rate"])
    """

    def __init__(self, default_ttl: float = 60.0, max_size: int = 1000) -> None:
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._store: dict[str, _CacheEntry] = {}
        self._lock = Lock()
        self._hit_count: int = 0
        self._miss_count: int = 0

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[T]:
        """
        Return the cached value for *key*, or ``None`` if missing/expired.

        Parameters
        ----------
        key:
            Cache key.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._miss_count += 1
                logger.debug("Cache MISS (absent): %s", key)
                return None
            if time.monotonic() > entry.expires_at:
                del self._store[key]
                self._miss_count += 1
                logger.debug("Cache MISS (expired): %s", key)
                return None
            self._hit_count += 1
            logger.debug("Cache HIT: %s", key)
            return entry.value  # type: ignore[return-value]

    # ── Write ─────────────────────────────────────────────────────────────────

    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """
        Store *value* under *key* with optional custom *ttl*.

        Parameters
        ----------
        key:
            Cache key.
        value:
            Value to cache.
        ttl:
            Time-to-live in seconds. Uses ``default_ttl`` if not provided.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.monotonic() + effective_ttl
        with self._lock:
            if self._max_size > 0 and len(self._store) >= self._max_size:
                self._evict_one()
            self._store[key] = _CacheEntry(value=value, expires_at=expires_at)
            logger.debug("Cache SET: %s (ttl=%.1fs)", key, effective_ttl)

    def delete(self, key: str) -> None:
        """Remove *key* from the cache if it exists."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries (does not reset hit/miss counters)."""
        with self._lock:
            self._store.clear()

    # ── Read-through ──────────────────────────────────────────────────────────

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[float] = None,
    ) -> T:
        """
        Return the cached value for *key*; if missing, call *factory* and cache the result.

        Parameters
        ----------
        key:
            Cache key.
        factory:
            Zero-argument callable that produces the value when cache misses.
        ttl:
            Optional custom TTL.
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value, ttl)
        return value

    # ── Maintenance ───────────────────────────────────────────────────────────

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        This is a proactive cleanup; entries are also lazily evicted on
        :meth:`get`.  Useful to call periodically to reclaim memory.

        Returns
        -------
        int:
            Number of entries removed.
        """
        with self._lock:
            now = time.monotonic()
            expired = [k for k, e in self._store.items() if now > e.expires_at]
            for k in expired:
                del self._store[k]
        if expired:
            logger.debug("Cache cleanup: removed %d expired entries", len(expired))
        return len(expired)

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Return a snapshot of cache statistics.

        Returns
        -------
        dict:
            Keys:

            ``size``
                Number of entries currently stored (may include expired).
            ``max_size``
                Maximum allowed entries (0 = unlimited).
            ``hit_count``
                Cumulative cache hits since creation.
            ``miss_count``
                Cumulative cache misses since creation.
            ``hit_rate``
                ``hit_count / (hit_count + miss_count)``, or ``0.0`` if
                no requests yet.
            ``default_ttl``
                Default TTL in seconds.
        """
        with self._lock:
            total = self._hit_count + self._miss_count
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": self._hit_count / total if total > 0 else 0.0,
                "default_ttl": self._default_ttl,
            }

    def reset_stats(self) -> None:
        """Reset hit and miss counters to zero."""
        with self._lock:
            self._hit_count = 0
            self._miss_count = 0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evict_one(self) -> None:
        """Evict the entry with the soonest expiry (called under lock)."""
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k].expires_at)
        del self._store[oldest_key]
        logger.debug("Cache evicted: %s", oldest_key)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"TTLCache(size={s['size']}, hits={s['hit_count']}, "
            f"misses={s['miss_count']}, hit_rate={s['hit_rate']:.1%})"
        )
