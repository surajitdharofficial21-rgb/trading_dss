"""
Event Calendar — track upcoming known events that cause volatility.

The system should PREPARE for scheduled events (RBI policy, budget, expiry,
earnings), not just react.  This module loads a JSON calendar of recurring
and one-off events and provides queries for upcoming events, event-day
detection, and regime modifiers that feed into position sizing and risk.
"""

from __future__ import annotations

import json
import logging
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional

from zoneinfo import ZoneInfo

from config.constants import IST_TIMEZONE, MARKET_HOLIDAYS
from config.settings import settings
from src.utils.market_hours import is_trading_day, is_holiday

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

# Severity ordering
_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NOISE": 4}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class UpcomingEvent:
    """A scheduled event with distance and impact metadata."""

    name: str
    expected_date: date | None = None
    expected_time: time | None = None
    impact_severity: str = "MEDIUM"
    affected_indices: list[str] = field(default_factory=list)
    hours_until_event: float | None = None
    is_today: bool = False
    is_within_caution_period: bool = False
    volatility_multiplier: float = 1.0
    note: str = ""


@dataclass
class EventRegimeModifier:
    """Risk-regime adjustments driven by upcoming events."""

    is_event_day: bool = False
    is_pre_event: bool = False
    is_expiry_day: bool = False
    volatility_multiplier: float = 1.0
    position_size_modifier: float = 1.0
    caution_level: str = "NORMAL"  # NORMAL / ELEVATED / HIGH / EXTREME
    active_events: list[str] = field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# EventCalendar
# ---------------------------------------------------------------------------


class EventCalendar:
    """Load and query the event calendar for upcoming market-moving events."""

    def __init__(self, calendar_path: Optional[Path] = None) -> None:
        path = calendar_path or (settings.config_dir / "events_calendar.json")
        self._recurring: list[dict] = []
        self._specific: list[dict] = []
        self._load(path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Events calendar not found at %s", path)
            return
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._recurring = data.get("recurring_events", [])
        self._specific = data.get("specific_events", [])
        logger.info(
            "Loaded event calendar: %d recurring, %d specific events",
            len(self._recurring),
            len(self._specific),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_upcoming_events(
        self,
        days_ahead: int = 7,
        index_id: str | None = None,
        reference_date: date | None = None,
        reference_time: datetime | None = None,
    ) -> list[UpcomingEvent]:
        """Return events in the next *days_ahead* days, optionally filtered."""
        ref = reference_date or date.today()
        now = reference_time or datetime.now(tz=_IST)
        # When a reference_date is given but no reference_time, synthesise a
        # coherent "now" so that is_today checks work against the reference.
        if reference_date and not reference_time:
            now = datetime(ref.year, ref.month, ref.day, 10, 0, tzinfo=_IST)
        end = ref + timedelta(days=days_ahead)

        events: list[UpcomingEvent] = []

        # --- Specific (one-off) events ---
        for ev in self._specific:
            ev_date = date.fromisoformat(ev["date"])
            if ref <= ev_date <= end:
                if index_id and not self._affects_index(ev, index_id):
                    continue
                events.append(self._build_upcoming(ev, ev_date, now))

        # --- Recurring events ---
        for ev in self._recurring:
            dates = self._resolve_recurring_dates(ev, ref, end)
            for d in dates:
                if index_id and not self._affects_index(ev, index_id):
                    continue
                events.append(self._build_upcoming(ev, d, now))

        # Sort by date (None dates last), then severity
        events.sort(
            key=lambda e: (
                e.expected_date or date.max,
                _SEVERITY_ORDER.get(e.impact_severity, 99),
            )
        )
        return events

    def is_event_day(
        self, d: date | None = None, index_id: str | None = None
    ) -> bool:
        """Return ``True`` if *d* has any HIGH or CRITICAL events."""
        d = d or date.today()
        events = self.get_upcoming_events(
            days_ahead=0, index_id=index_id, reference_date=d,
        )
        for ev in events:
            if ev.expected_date == d and ev.impact_severity in ("CRITICAL", "HIGH"):
                return True
        return False

    def get_regime_modifier(
        self,
        index_id: str,
        reference_date: date | None = None,
        reference_time: datetime | None = None,
    ) -> EventRegimeModifier:
        """Compute risk-regime adjustments for *index_id* right now."""
        ref = reference_date or date.today()
        now = reference_time or datetime.now(tz=_IST)

        # Look 2 days ahead to capture caution periods
        events = self.get_upcoming_events(
            days_ahead=2, index_id=index_id, reference_date=ref,
            reference_time=now
        )

        mod = EventRegimeModifier()

        # Check expiry
        # Thursday = weekday 3
        if ref.weekday() == 3 and is_trading_day(ref):
            mod.is_expiry_day = True
            # Check if it's monthly expiry (last Thursday of month)
            _, month_days = monthrange(ref.year, ref.month)
            next_thu = ref + timedelta(days=7)
            if next_thu.month != ref.month:
                # This is the last Thursday → monthly expiry
                mod.volatility_multiplier = max(mod.volatility_multiplier, 1.5)
            else:
                mod.volatility_multiplier = max(mod.volatility_multiplier, 1.2)

        max_severity = "NOISE"

        for ev in events:
            if ev.is_today:
                mod.is_event_day = True
                mod.active_events.append(ev.name)
                mod.volatility_multiplier = max(
                    mod.volatility_multiplier, ev.volatility_multiplier
                )
                if _SEVERITY_ORDER.get(ev.impact_severity, 99) < _SEVERITY_ORDER.get(max_severity, 99):
                    max_severity = ev.impact_severity

            if ev.is_within_caution_period and not ev.is_today:
                mod.is_pre_event = True
                mod.active_events.append(f"Pre-event: {ev.name}")
                # Pre-event gets partial volatility bump
                mod.volatility_multiplier = max(
                    mod.volatility_multiplier,
                    1.0 + (ev.volatility_multiplier - 1.0) * 0.5,
                )
                if _SEVERITY_ORDER.get(ev.impact_severity, 99) < _SEVERITY_ORDER.get(max_severity, 99):
                    max_severity = ev.impact_severity

        # Position size modifier
        if mod.is_event_day and max_severity in ("CRITICAL", "HIGH"):
            mod.position_size_modifier = 0.5
        elif mod.is_pre_event and max_severity in ("CRITICAL", "HIGH"):
            mod.position_size_modifier = 0.7
        elif mod.is_event_day:
            mod.position_size_modifier = 0.7
        elif mod.is_pre_event:
            mod.position_size_modifier = 0.85

        # Caution level
        if max_severity == "CRITICAL" and (mod.is_event_day or mod.is_pre_event):
            mod.caution_level = "EXTREME"
        elif max_severity == "HIGH" and mod.is_event_day:
            mod.caution_level = "HIGH"
        elif max_severity in ("HIGH", "CRITICAL") and mod.is_pre_event:
            mod.caution_level = "HIGH"
        elif mod.is_event_day or mod.is_pre_event:
            mod.caution_level = "ELEVATED"

        # Reasoning
        if mod.active_events:
            mod.reasoning = f"Caution: {', '.join(mod.active_events)}. Position size modifier: {mod.position_size_modifier}"
        else:
            mod.reasoning = "No significant scheduled events."

        return mod

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _affects_index(ev: dict, index_id: str) -> bool:
        affected = ev.get("affected_indices", [])
        return "ALL" in affected or index_id in affected

    def _build_upcoming(
        self, ev: dict, ev_date: date, now: datetime
    ) -> UpcomingEvent:
        ann_time_str = ev.get("announcement_time")
        ann_time: time | None = None
        if ann_time_str:
            h, m = ann_time_str.split(":")
            ann_time = time(int(h), int(m))

        # Compute hours until
        if ann_time:
            ev_dt = datetime(
                ev_date.year, ev_date.month, ev_date.day,
                ann_time.hour, ann_time.minute, tzinfo=_IST,
            )
        else:
            # Assume start-of-market-hours
            ev_dt = datetime(
                ev_date.year, ev_date.month, ev_date.day, 9, 15, tzinfo=_IST
            )

        hours_until = (ev_dt - now).total_seconds() / 3600.0

        caution_hours = ev.get("pre_event_caution_hours", 6)
        is_within_caution = 0 < hours_until <= caution_hours

        today = now.date()

        return UpcomingEvent(
            name=ev["name"],
            expected_date=ev_date,
            expected_time=ann_time,
            impact_severity=ev.get("impact_severity", "MEDIUM"),
            affected_indices=ev.get("affected_indices", []),
            hours_until_event=round(hours_until, 2),
            is_today=(ev_date == today),
            is_within_caution_period=is_within_caution,
            volatility_multiplier=ev.get("typical_volatility_increase", 1.0),
            note=ev.get("note", ""),
        )

    # ------------------------------------------------------------------
    # Recurring date resolution
    # ------------------------------------------------------------------

    def _resolve_recurring_dates(
        self, ev: dict, start: date, end: date
    ) -> list[date]:
        """Resolve approximate dates for a recurring event within [start, end]."""
        freq = ev.get("frequency", "")
        typical = ev.get("typical_dates", "")

        dates: list[date] = []

        if "Every Thursday" in typical or freq == "weekly":
            dates = self._thursdays_in_range(start, end)

        elif "Last Thursday" in typical:
            dates = self._last_thursdays_in_range(start, end)

        elif "February 1" in typical:
            # Annual event on Feb 1
            for year in range(start.year, end.year + 1):
                d = date(year, 2, 1)
                if start <= d <= end:
                    d = self._adjust_for_holiday(d)
                    dates.append(d)

        elif "First Friday" in typical:
            dates = self._first_weekday_in_range(start, end, weekday=4)

        elif "1st of each month" in typical:
            dates = self._day_of_month_in_range(start, end, day=1)

        elif "12th of each month" in typical or "12th" in typical:
            dates = self._day_of_month_in_range(start, end, day=12)

        elif "14th of each month" in typical or "14th" in typical:
            dates = self._day_of_month_in_range(start, end, day=14)

        elif "3rd-5th of each month" in typical:
            dates = self._day_of_month_in_range(start, end, day=4)

        elif "10th-14th of each month" in typical:
            dates = self._day_of_month_in_range(start, end, day=12)

        elif "15th-20th of each month" in typical:
            dates = self._day_of_month_in_range(start, end, day=17)

        elif freq == "bi-monthly" and typical:
            # RBI-style: specific months
            months = self._parse_month_names(typical)
            dates = self._specific_months_in_range(start, end, months, day=5)

        elif freq == "quarterly":
            # Approximate: mid-month of quarter-end months
            dates = self._quarterly_dates_in_range(start, end, typical)

        elif freq == "6-weekly":
            # Approximate: every 6 weeks from a reference
            dates = self._every_n_weeks(start, end, n=6)

        else:
            # Unknown frequency — log and skip
            logger.debug(
                "Cannot resolve recurring dates for '%s' (freq=%s, typical=%s). "
                "Exact date is estimated.",
                ev.get("name"),
                freq,
                typical,
            )

        return dates

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adjust_for_holiday(d: date) -> date:
        """If *d* is a holiday or weekend, move to next trading day."""
        while not is_trading_day(d):
            d += timedelta(days=1)
        return d

    @staticmethod
    def _thursdays_in_range(start: date, end: date) -> list[date]:
        dates: list[date] = []
        d = start
        while d <= end:
            if d.weekday() == 3:  # Thursday
                dates.append(d)
            d += timedelta(days=1)
        return dates

    @staticmethod
    def _last_thursdays_in_range(start: date, end: date) -> list[date]:
        dates: list[date] = []
        # Check each month in range
        current = date(start.year, start.month, 1)
        while current <= end:
            _, month_days = monthrange(current.year, current.month)
            last_day = date(current.year, current.month, month_days)
            # Walk backward to find last Thursday
            d = last_day
            while d.weekday() != 3:
                d -= timedelta(days=1)
            # Adjust if holiday
            while not is_trading_day(d):
                d -= timedelta(days=1)
            if start <= d <= end:
                dates.append(d)
            # Next month
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)
        return dates

    @staticmethod
    def _first_weekday_in_range(
        start: date, end: date, weekday: int
    ) -> list[date]:
        """Find first occurrence of *weekday* (0=Mon) each month in range."""
        dates: list[date] = []
        current = date(start.year, start.month, 1)
        while current <= end:
            d = current
            while d.weekday() != weekday:
                d += timedelta(days=1)
            if start <= d <= end:
                dates.append(d)
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)
        return dates

    @staticmethod
    def _day_of_month_in_range(
        start: date, end: date, day: int
    ) -> list[date]:
        dates: list[date] = []
        current = date(start.year, start.month, 1)
        while current <= end:
            _, month_days = monthrange(current.year, current.month)
            actual_day = min(day, month_days)
            d = date(current.year, current.month, actual_day)
            if start <= d <= end:
                dates.append(d)
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)
        return dates

    @staticmethod
    def _parse_month_names(text: str) -> list[int]:
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        months: list[int] = []
        for word in text.lower().replace(",", " ").split():
            if word in month_map:
                months.append(month_map[word])
        return sorted(set(months))

    def _specific_months_in_range(
        self, start: date, end: date, months: list[int], day: int
    ) -> list[date]:
        dates: list[date] = []
        for year in range(start.year, end.year + 1):
            for m in months:
                _, month_days = monthrange(year, m)
                d = date(year, m, min(day, month_days))
                d = self._adjust_for_holiday(d)
                if start <= d <= end:
                    dates.append(d)
        return dates

    @staticmethod
    def _quarterly_dates_in_range(
        start: date, end: date, typical: str
    ) -> list[date]:
        """Approximate quarterly dates (mid-month of typical quarter months)."""
        # Default quarters: end of Jan, Apr, Jul, Oct (for most Indian macro data)
        quarter_months = [1, 4, 7, 10]
        if "February" in typical:
            quarter_months = [2, 5, 8, 11]

        dates: list[date] = []
        for year in range(start.year, end.year + 1):
            for m in quarter_months:
                d = date(year, m, 15)
                if start <= d <= end:
                    dates.append(d)
        return dates

    @staticmethod
    def _every_n_weeks(
        start: date, end: date, n: int
    ) -> list[date]:
        """Generate dates every *n* weeks within range (rough approximation)."""
        # Start from a reference Wednesday (common for FOMC/ECB)
        ref = date(2024, 1, 31)  # known FOMC date
        dates: list[date] = []
        d = ref
        while d <= end:
            if d >= start:
                dates.append(d)
            d += timedelta(weeks=n)
        # Also walk backward from ref
        d = ref - timedelta(weeks=n)
        while d >= start:
            dates.append(d)
            d -= timedelta(weeks=n)
        dates.sort()
        return [d for d in dates if start <= d <= end]
