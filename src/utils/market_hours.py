"""
Market open/close/holiday detection for Indian markets.

All comparisons are made in IST (Asia/Kolkata).

The module exposes both standalone functions (backward-compatible) and a
:class:`MarketHoursManager` class with richer functionality such as
timedelta-returning queries and a structured status dict.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from config.constants import MARKET_HOLIDAYS, MUHURAT_TRADING_DAYS, IST_TIMEZONE
from config.settings import settings

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)


class MarketSession(str, Enum):
    """Named trading session at any given moment."""

    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    OPEN = "open"
    POST_MARKET = "post_market"
    MUHURAT = "muhurat"
    HOLIDAY = "holiday"
    WEEKEND = "weekend"


def _parse_hhmm(hhmm: str) -> time:
    """Parse a ``"HH:MM"`` string into :class:`datetime.time`."""
    h, m = hhmm.split(":")
    return time(int(h), int(m))


# ---------------------------------------------------------------------------
# Standalone helpers (backward-compatible)
# ---------------------------------------------------------------------------

def now_ist() -> datetime:
    """Return the current datetime in IST."""
    return datetime.now(tz=_IST)


def today_ist() -> date:
    """Return today's date in IST."""
    return now_ist().date()


def is_holiday(d: date) -> bool:
    """
    Return ``True`` if *d* is a declared NSE/BSE trading holiday.

    Parameters
    ----------
    d:
        Date to check (in IST).
    """
    return d in MARKET_HOLIDAYS


def is_muhurat_day(d: date) -> bool:
    """Return ``True`` if *d* is a Muhurat (special Diwali) trading day."""
    return d in MUHURAT_TRADING_DAYS


def is_weekend(d: date) -> bool:
    """Return ``True`` if *d* is Saturday or Sunday."""
    return d.weekday() >= 5  # 5 = Saturday, 6 = Sunday


def is_trading_day(d: date) -> bool:
    """
    Return ``True`` if *d* is a regular or Muhurat trading day.

    Parameters
    ----------
    d:
        Date to evaluate.
    """
    if is_weekend(d):
        return False
    if is_holiday(d):
        # Muhurat trading is still valid even on Diwali holiday
        return is_muhurat_day(d)
    return True


def get_market_session(dt: Optional[datetime] = None) -> MarketSession:
    """
    Determine the current market session for *dt*.

    Parameters
    ----------
    dt:
        Datetime to evaluate (IST). Defaults to now.

    Returns
    -------
    MarketSession:
        The session enum value describing the current market state.
    """
    dt = dt or now_ist()
    # Normalise to IST
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_IST)
    else:
        dt = dt.astimezone(_IST)

    d = dt.date()
    t = dt.time()

    if is_weekend(d):
        return MarketSession.WEEKEND

    if is_holiday(d):
        if is_muhurat_day(d):
            return MarketSession.MUHURAT
        return MarketSession.HOLIDAY

    mh = settings.market_hours
    pre_open = _parse_hhmm(mh.pre_market_open)
    mkt_open = _parse_hhmm(mh.market_open)
    mkt_close = _parse_hhmm(mh.market_close)
    post_close = _parse_hhmm(mh.post_market_close)

    if t < pre_open:
        return MarketSession.CLOSED
    if pre_open <= t < mkt_open:
        return MarketSession.PRE_MARKET
    if mkt_open <= t < mkt_close:
        return MarketSession.OPEN
    if mkt_close <= t < post_close:
        return MarketSession.POST_MARKET
    return MarketSession.CLOSED


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """Return ``True`` only during the regular session (9:15–15:30 IST)."""
    return get_market_session(dt) == MarketSession.OPEN


def next_trading_day(from_date: Optional[date] = None) -> date:
    """
    Return the next trading day on or after *from_date* (inclusive).

    Parameters
    ----------
    from_date:
        Starting date. Defaults to today (IST).
    """
    d = from_date or today_ist()
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# MarketHoursManager
# ---------------------------------------------------------------------------


class MarketHoursManager:
    """
    High-level market hours helper for Indian exchanges.

    Wraps the standalone functions with richer return types and adds
    timedelta-based queries and a structured status dict.

    Parameters
    ----------
    clock:
        Optional zero-argument callable that returns the current IST datetime.
        Defaults to :func:`now_ist`.  Pass a fixed-time factory in tests to
        avoid depending on the system clock.

    Examples
    --------
    ::

        mgr = MarketHoursManager()
        if mgr.is_market_open():
            remaining = mgr.time_to_market_close()
            print(f"Market closes in {remaining}")
    """

    def __init__(self, clock: Optional[Callable[[], datetime]] = None) -> None:
        self._clock: Callable[[], datetime] = clock or now_ist

    # ── Internal ──────────────────────────────────────────────────────────────

    def _now(self) -> datetime:
        dt = self._clock()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=_IST)
        return dt.astimezone(_IST)

    @staticmethod
    def _make_ist(d: date, t: time) -> datetime:
        return datetime(d.year, d.month, d.day, t.hour, t.minute, 0, 0, tzinfo=_IST)

    # ── Session queries ───────────────────────────────────────────────────────

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Return ``True`` during the regular trading session (09:15–15:30 IST)."""
        return get_market_session(dt or self._now()) == MarketSession.OPEN

    def is_pre_market(self, dt: Optional[datetime] = None) -> bool:
        """Return ``True`` during the pre-market session (09:00–09:15 IST)."""
        return get_market_session(dt or self._now()) == MarketSession.PRE_MARKET

    def is_post_market(self, dt: Optional[datetime] = None) -> bool:
        """Return ``True`` during the post-market session (15:30–16:00 IST)."""
        return get_market_session(dt or self._now()) == MarketSession.POST_MARKET

    def get_session(self, dt: Optional[datetime] = None) -> MarketSession:
        """Return the :class:`MarketSession` for *dt* (defaults to now)."""
        return get_market_session(dt or self._now())

    # ── Time-to-event queries ──────────────────────────────────────────────────

    def time_to_market_open(self, dt: Optional[datetime] = None) -> Optional[timedelta]:
        """
        Return the timedelta until the next market open.

        Returns ``None`` if the market is currently open (or Muhurat session).

        For pre-market: time until today's 09:15.
        For post-market / closed / weekend / holiday: time until next trading
        day's 09:15.
        """
        now = dt or self._now()
        session = get_market_session(now)

        if session in (MarketSession.OPEN, MarketSession.MUHURAT):
            return None

        mh = settings.market_hours
        mkt_open_t = _parse_hhmm(mh.market_open)
        d = now.date()

        # Before market open on a trading day (CLOSED pre-09:00 or PRE_MARKET)
        if session in (MarketSession.PRE_MARKET, MarketSession.CLOSED) and is_trading_day(d):
            mkt_open_today = self._make_ist(d, mkt_open_t)
            if now < mkt_open_today:
                delta = mkt_open_today - now
                logger.debug("Time to market open: %s", delta)
                return delta

        # Post-market, evening closed, weekend, holiday → next trading day
        next_day = self.get_next_trading_day(d)
        target = self._make_ist(next_day, mkt_open_t)
        delta = target - now
        logger.debug("Time to next market open (%s): %s", next_day, delta)
        return delta

    def time_to_market_close(self, dt: Optional[datetime] = None) -> Optional[timedelta]:
        """
        Return the timedelta until market close (15:30 IST).

        Returns ``None`` if the market is currently closed (any non-OPEN session).
        """
        now = dt or self._now()
        if get_market_session(now) != MarketSession.OPEN:
            return None

        mh = settings.market_hours
        mkt_close_t = _parse_hhmm(mh.market_close)
        close_dt = self._make_ist(now.date(), mkt_close_t)
        delta = close_dt - now
        logger.debug("Time to market close: %s", delta)
        return delta

    # ── Day / expiry queries ──────────────────────────────────────────────────

    def is_holiday(self, d: Optional[date] = None) -> bool:
        """Return ``True`` if *d* (defaults to today IST) is a market holiday."""
        return is_holiday(d or self._now().date())

    def is_expiry_day(self, d: Optional[date] = None) -> bool:
        """
        Return ``True`` if *d* is an F&O expiry day (after holiday adjustment).

        A regular expiry is every Thursday.  When Thursday is a holiday the
        expiry moves to Wednesday (and further back if Wednesday is also a
        holiday).  This method returns ``True`` for the adjusted expiry date.

        Parameters
        ----------
        d:
            Date to check. Defaults to today (IST).
        """
        check = d or self._now().date()
        if not is_trading_day(check):
            return False

        # Thursday is a regular expiry day
        if check.weekday() == 3:
            return True

        # For Mon/Tue/Wed: check whether the Thursday of this week is a holiday
        # and whether check is the adjusted (moved-backward) expiry date.
        if check.weekday() < 3:
            days_to_thu = 3 - check.weekday()
            thu = check + timedelta(days=days_to_thu)
            # Walk Thursday backward until a trading day is found
            adj = thu
            while not is_trading_day(adj):
                adj -= timedelta(days=1)
            return adj == check

        return False

    def get_next_trading_day(self, from_date: Optional[date] = None) -> date:
        """
        Return the next trading day strictly after *from_date*.

        Parameters
        ----------
        from_date:
            Start date (exclusive). Defaults to today (IST).
        """
        d = (from_date or self._now().date()) + timedelta(days=1)
        while not is_trading_day(d):
            d += timedelta(days=1)
        return d

    # ── Status dict ───────────────────────────────────────────────────────────

    def get_market_status(self, dt: Optional[datetime] = None) -> dict:
        """
        Return a structured dict describing the current market state.

        Returns
        -------
        dict:
            Keys:

            ``status``
                Current :class:`MarketSession` value string.
            ``is_trading_day``
                Whether the current date is a regular/Muhurat trading day.
            ``is_holiday``
                Whether the current date is a declared holiday.
            ``is_weekend``
                Whether the current date is Saturday or Sunday.
            ``next_event``
                One of ``"market_open"``, ``"market_close"``,
                ``"next_trading_day"``.
            ``time_remaining``
                ``"HH:MM:SS"`` string until the next event, or ``None``.
            ``timestamp``
                ISO-formatted current datetime in IST.
        """
        now = dt or self._now()
        session = get_market_session(now)
        d = now.date()

        if session == MarketSession.OPEN:
            next_event = "market_close"
            time_rem = self.time_to_market_close(now)
        elif session == MarketSession.PRE_MARKET:
            next_event = "market_open"
            time_rem = self.time_to_market_open(now)
        elif session == MarketSession.POST_MARKET:
            next_event = "next_trading_day"
            time_rem = self.time_to_market_open(now)
        else:
            # CLOSED, WEEKEND, HOLIDAY, MUHURAT
            next_event = "market_open"
            time_rem = self.time_to_market_open(now)

        time_rem_str: Optional[str] = None
        if time_rem is not None:
            total_secs = max(0, int(time_rem.total_seconds()))
            h, remainder = divmod(total_secs, 3600)
            m, s = divmod(remainder, 60)
            time_rem_str = f"{h:02d}:{m:02d}:{s:02d}"

        return {
            "status": session.value,
            "is_trading_day": is_trading_day(d),
            "is_holiday": is_holiday(d),
            "is_weekend": is_weekend(d),
            "next_event": next_event,
            "time_remaining": time_rem_str,
            "timestamp": now.isoformat(),
        }

    def __repr__(self) -> str:
        now = self._now()
        session = get_market_session(now)
        return f"MarketHoursManager(session={session.value}, time={now.strftime('%H:%M:%S IST')})"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: Optional[MarketHoursManager] = None


def get_market_hours_manager() -> MarketHoursManager:
    """
    Return the process-wide :class:`MarketHoursManager` singleton.

    Uses the live system clock.  For testing, construct a
    :class:`MarketHoursManager` directly with a custom ``clock`` argument.
    """
    global _manager
    if _manager is None:
        _manager = MarketHoursManager()
    return _manager
