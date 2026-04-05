"""
IST timezone handling and F&O expiry date calculations.

All public functions that return dates account for the NSE/BSE rule that
when an expiry Thursday is a market holiday the expiry shifts to the
preceding Wednesday (and further back if that is also a holiday).
"""

from __future__ import annotations

import calendar
import logging
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE, WEEKLY_EXPIRY_INDICES

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)
_THURSDAY = 3  # weekday() index (0 = Monday)


# ---------------------------------------------------------------------------
# IST helpers
# ---------------------------------------------------------------------------

def get_ist_now() -> datetime:
    """Return the current datetime in IST (timezone-aware)."""
    return datetime.now(tz=_IST)


def to_ist(dt: datetime) -> datetime:
    """
    Convert *dt* to IST.

    Parameters
    ----------
    dt:
        Any timezone-aware or naive datetime.  Naive datetimes are assumed
        to already be in IST and are tagged accordingly.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_IST)
    return dt.astimezone(_IST)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _last_thursday_raw(year: int, month: int) -> date:
    """Last Thursday of *month* in *year* (no holiday adjustment)."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != _THURSDAY:
        d -= timedelta(days=1)
    return d


def _resolve_expiry(d: date) -> date:
    """
    Adjust *d* backward until it is a trading day.

    Implements the NSE rule: when the scheduled expiry Thursday is a holiday,
    the expiry moves to the nearest prior trading day.
    """
    from src.utils.market_hours import is_trading_day  # lazy to avoid circular import

    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# Backward-compatible standalone functions (no holiday adjustment)
# ---------------------------------------------------------------------------

def last_thursday_of_month(year: int, month: int) -> date:
    """
    Return the last Thursday of *month* in *year* (no holiday adjustment).

    Parameters
    ----------
    year:
        Calendar year.
    month:
        Calendar month (1–12).
    """
    return _last_thursday_raw(year, month)


def next_weekly_expiry(from_date: Optional[date] = None) -> date:
    """
    Return the next Thursday on or after *from_date* (no holiday adjustment).

    Weekly options (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX) expire
    on every Thursday.

    Parameters
    ----------
    from_date:
        Reference date. Defaults to today (IST).
    """
    from src.utils.market_hours import today_ist

    d = from_date or today_ist()
    days_until_thursday = (_THURSDAY - d.weekday()) % 7
    return d + timedelta(days=days_until_thursday)


def next_monthly_expiry(from_date: Optional[date] = None) -> date:
    """
    Return the next monthly F&O expiry (last Thursday of current/next month).

    If *from_date* is after the last Thursday of the current month, rolls over
    to the following month.  No holiday adjustment.

    Parameters
    ----------
    from_date:
        Reference date. Defaults to today (IST).
    """
    from src.utils.market_hours import today_ist

    d = from_date or today_ist()
    expiry = _last_thursday_raw(d.year, d.month)
    if d > expiry:
        if d.month == 12:
            expiry = _last_thursday_raw(d.year + 1, 1)
        else:
            expiry = _last_thursday_raw(d.year, d.month + 1)
    return expiry


def get_expiry_for_symbol(option_symbol: str, from_date: Optional[date] = None) -> date:
    """
    Return the upcoming expiry date for *option_symbol* (no holiday adjustment).

    Weekly expiry indices: next Thursday.
    All others: last Thursday of the month.

    Parameters
    ----------
    option_symbol:
        NSE option symbol (e.g. ``"NIFTY"``, ``"BANKNIFTY"``).
    from_date:
        Reference date.
    """
    if option_symbol.upper() in WEEKLY_EXPIRY_INDICES:
        return next_weekly_expiry(from_date)
    return next_monthly_expiry(from_date)


def days_to_expiry(option_symbol: str, from_date: Optional[date] = None) -> int:
    """
    Return the number of calendar days to the next expiry for *option_symbol*.
    """
    from src.utils.market_hours import today_ist

    ref = from_date or today_ist()
    return (get_expiry_for_symbol(option_symbol, ref) - ref).days


def format_ist(dt: date) -> str:
    """Format a date as ``"DD-Mon-YYYY"`` (e.g. ``"27-Jun-2024"``) for NSE API calls."""
    return dt.strftime("%d-%b-%Y")


# ---------------------------------------------------------------------------
# Enhanced functions with holiday adjustment
# ---------------------------------------------------------------------------

def is_weekly_expiry_index(index_id: str) -> bool:
    """
    Return ``True`` if *index_id* has weekly F&O expiry.

    Checks against :data:`~config.constants.WEEKLY_EXPIRY_INDICES` using
    both the registry ID and the index's ``option_symbol`` (lazy-loaded to
    avoid circular imports).

    Parameters
    ----------
    index_id:
        Registry ID (e.g. ``"NIFTY50"``, ``"BANKNIFTY"``).
    """
    uid = index_id.strip().upper()
    if uid in WEEKLY_EXPIRY_INDICES:
        return True
    # Check via registry's option_symbol (lazy import)
    try:
        from src.data.index_registry import get_registry  # noqa: PLC0415

        registry = get_registry()
        idx = registry.get_or_none(uid)
        if idx and idx.option_symbol and idx.option_symbol.upper() in WEEKLY_EXPIRY_INDICES:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def get_weekly_expiry(from_date: Optional[date] = None) -> date:
    """
    Return the next weekly expiry Thursday on or after *from_date*.

    Applies holiday adjustment: if Thursday is a holiday the expiry moves
    to the nearest prior trading day.

    Parameters
    ----------
    from_date:
        Reference date. Defaults to today (IST).
    """
    from src.utils.market_hours import today_ist

    d = from_date or today_ist()
    days_until = (_THURSDAY - d.weekday()) % 7
    thursday = d + timedelta(days=days_until)
    return _resolve_expiry(thursday)


def get_monthly_expiry(year: int, month: int) -> date:
    """
    Return the last Thursday of *month* in *year*, adjusted for holidays.

    Parameters
    ----------
    year:
        Calendar year.
    month:
        Calendar month (1–12).
    """
    return _resolve_expiry(_last_thursday_raw(year, month))


def get_current_expiry(index_id: str, from_date: Optional[date] = None) -> date:
    """
    Return the current (upcoming) expiry date for *index_id*.

    If today is itself the expiry day it is returned.  Holiday-adjusted.

    Parameters
    ----------
    index_id:
        Registry ID or NSE option symbol.
    from_date:
        Reference date. Defaults to today (IST).
    """
    from src.utils.market_hours import today_ist

    ref = from_date or today_ist()

    if is_weekly_expiry_index(index_id):
        return get_weekly_expiry(ref)

    # Monthly: last Thursday of current month (or next month if past it)
    monthly = get_monthly_expiry(ref.year, ref.month)
    if ref <= monthly:
        return monthly
    # Roll to next month
    if ref.month == 12:
        return get_monthly_expiry(ref.year + 1, 1)
    return get_monthly_expiry(ref.year, ref.month + 1)


def get_next_expiry(index_id: str, from_date: Optional[date] = None) -> date:
    """
    Return the expiry date **after** the current expiry for *index_id*.

    Holiday-adjusted.

    Parameters
    ----------
    index_id:
        Registry ID or NSE option symbol.
    from_date:
        Reference date. Defaults to today (IST).
    """
    current = get_current_expiry(index_id, from_date)

    if is_weekly_expiry_index(index_id):
        # Next Thursday after current
        return get_weekly_expiry(current + timedelta(days=1))

    # Next month's last Thursday
    if current.month == 12:
        return get_monthly_expiry(current.year + 1, 1)
    return get_monthly_expiry(current.year, current.month + 1)


def format_expiry_for_nse(d: date) -> str:
    """
    Format *d* as an NSE-style expiry string in UPPERCASE.

    Example: ``date(2024, 6, 27)`` → ``"27-JUN-2024"``.

    Parameters
    ----------
    d:
        Expiry date.
    """
    return d.strftime("%d-%b-%Y").upper()


# ---------------------------------------------------------------------------
# Trading-day counting utilities
# ---------------------------------------------------------------------------

def trading_days_between(start: date, end: date) -> int:
    """
    Count trading days from *start* to *end* (both inclusive).

    Parameters
    ----------
    start:
        First date of the range.
    end:
        Last date of the range (inclusive).

    Returns
    -------
    int:
        Number of trading days in ``[start, end]``.  Returns ``0`` if
        *start* > *end*.
    """
    from src.utils.market_hours import is_trading_day  # lazy to avoid circular

    if start > end:
        return 0
    count = 0
    d = start
    while d <= end:
        if is_trading_day(d):
            count += 1
        d += timedelta(days=1)
    return count


def get_last_n_trading_days(n: int, ref_date: Optional[date] = None) -> list[date]:
    """
    Return the *n* most recent trading days ending on *ref_date* (inclusive).

    Results are sorted oldest-first.

    Parameters
    ----------
    n:
        Number of trading days to return.
    ref_date:
        Upper bound date. Defaults to today (IST).

    Returns
    -------
    list[date]:
        *n* trading dates in ascending order.
    """
    from src.utils.market_hours import is_trading_day, today_ist

    end = ref_date or today_ist()
    result: list[date] = []
    d = end
    while len(result) < n:
        if is_trading_day(d):
            result.append(d)
        d -= timedelta(days=1)
    result.reverse()
    return result
