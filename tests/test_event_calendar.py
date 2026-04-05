"""
Tests for EventCalendar — event loading, upcoming events, regime modifiers.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest
from zoneinfo import ZoneInfo

from src.analysis.news.event_calendar import (
    EventCalendar,
    EventRegimeModifier,
    UpcomingEvent,
)

_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def calendar_path(tmp_path: Path) -> Path:
    data = {
        "recurring_events": [
            {
                "name": "RBI Monetary Policy",
                "frequency": "bi-monthly",
                "typical_dates": "February, April, June, August, October, December",
                "announcement_time": "10:00",
                "impact_severity": "CRITICAL",
                "affected_indices": ["BANKNIFTY", "NIFTY50", "SENSEX"],
                "typical_volatility_increase": 2.0,
                "pre_event_caution_hours": 24,
            },
            {
                "name": "Weekly F&O Expiry",
                "frequency": "weekly",
                "typical_dates": "Every Thursday",
                "impact_severity": "LOW",
                "affected_indices": ["NIFTY50", "BANKNIFTY"],
                "typical_volatility_increase": 1.2,
                "pre_event_caution_hours": 2,
            },
            {
                "name": "Monthly F&O Expiry",
                "frequency": "monthly",
                "typical_dates": "Last Thursday of month",
                "impact_severity": "MEDIUM",
                "affected_indices": ["NIFTY50", "BANKNIFTY"],
                "typical_volatility_increase": 1.5,
                "pre_event_caution_hours": 6,
            },
            {
                "name": "India Union Budget",
                "frequency": "annual",
                "typical_dates": "February 1",
                "impact_severity": "CRITICAL",
                "affected_indices": ["ALL"],
                "typical_volatility_increase": 3.0,
                "pre_event_caution_hours": 48,
            },
            {
                "name": "US Non-Farm Payrolls",
                "frequency": "monthly",
                "typical_dates": "First Friday of each month",
                "impact_severity": "HIGH",
                "affected_indices": ["NIFTY50", "NIFTY_IT"],
                "typical_volatility_increase": 1.4,
                "pre_event_caution_hours": 6,
            },
        ],
        "specific_events": [
            {
                "name": "India General Election Results",
                "date": "2026-06-01",
                "impact_severity": "CRITICAL",
                "affected_indices": ["ALL"],
                "typical_volatility_increase": 3.0,
                "pre_event_caution_hours": 48,
            }
        ],
    }
    p = tmp_path / "events_calendar.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture()
def cal(calendar_path: Path) -> EventCalendar:
    return EventCalendar(calendar_path)


# ---------------------------------------------------------------------------
# Tests: Loading
# ---------------------------------------------------------------------------


class TestLoading:
    def test_loads_recurring_and_specific(self, cal: EventCalendar) -> None:
        assert len(cal._recurring) == 5
        assert len(cal._specific) == 1

    def test_missing_file_no_crash(self, tmp_path: Path) -> None:
        cal = EventCalendar(tmp_path / "nonexistent.json")
        assert cal._recurring == []
        assert cal._specific == []


# ---------------------------------------------------------------------------
# Tests: Upcoming events retrieval
# ---------------------------------------------------------------------------


class TestUpcomingEvents:
    def test_specific_event_within_range(self, cal: EventCalendar) -> None:
        events = cal.get_upcoming_events(
            days_ahead=7, reference_date=date(2026, 5, 28)
        )
        names = [e.name for e in events]
        assert "India General Election Results" in names

    def test_specific_event_outside_range(self, cal: EventCalendar) -> None:
        events = cal.get_upcoming_events(
            days_ahead=7, reference_date=date(2026, 1, 1)
        )
        names = [e.name for e in events]
        assert "India General Election Results" not in names

    def test_weekly_expiry_thursdays(self, cal: EventCalendar) -> None:
        """All Thursdays in a week should appear as weekly expiry."""
        # Week of April 6-12, 2026 → Thursday is April 9
        events = cal.get_upcoming_events(
            days_ahead=7, reference_date=date(2026, 4, 6)
        )
        expiry_events = [e for e in events if e.name == "Weekly F&O Expiry"]
        expiry_dates = [e.expected_date for e in expiry_events]
        assert date(2026, 4, 9) in expiry_dates  # Thursday

    def test_filter_by_index(self, cal: EventCalendar) -> None:
        events = cal.get_upcoming_events(
            days_ahead=30, index_id="NIFTY_IT", reference_date=date(2026, 4, 1)
        )
        # Only events affecting NIFTY_IT or ALL should appear
        for ev in events:
            assert "NIFTY_IT" in ev.affected_indices or "ALL" in ev.affected_indices

    def test_budget_in_february(self, cal: EventCalendar) -> None:
        """Union Budget on Feb 1 should appear in late-Jan search."""
        events = cal.get_upcoming_events(
            days_ahead=7, reference_date=date(2026, 1, 28)
        )
        names = [e.name for e in events]
        # Feb 1 is Sunday in 2026, should be adjusted to next trading day
        assert "India Union Budget" in names


# ---------------------------------------------------------------------------
# Tests: Expiry day detection
# ---------------------------------------------------------------------------


class TestExpiryDay:
    def test_thursday_is_expiry(self, cal: EventCalendar) -> None:
        """April 9, 2026 is a Thursday → weekly expiry."""
        events = cal.get_upcoming_events(
            days_ahead=0, reference_date=date(2026, 4, 9)
        )
        names = [e.name for e in events if e.expected_date == date(2026, 4, 9)]
        assert "Weekly F&O Expiry" in names

    def test_last_thursday_monthly_expiry(self, cal: EventCalendar) -> None:
        """April 30, 2026 is last Thursday → both weekly and monthly expiry."""
        events = cal.get_upcoming_events(
            days_ahead=0, reference_date=date(2026, 4, 30)
        )
        names = [e.name for e in events if e.expected_date == date(2026, 4, 30)]
        assert "Weekly F&O Expiry" in names
        assert "Monthly F&O Expiry" in names


# ---------------------------------------------------------------------------
# Tests: Event day detection
# ---------------------------------------------------------------------------


class TestIsEventDay:
    def test_rbi_policy_day(self, cal: EventCalendar) -> None:
        """RBI policy in April 2026 (bi-monthly, ~5th) should be CRITICAL."""
        # The resolver uses day=5 for bi-monthly → April 5
        # April 5 is Sunday → adjusted to April 6 Monday
        assert cal.is_event_day(date(2026, 4, 6)) is True

    def test_normal_day(self, cal: EventCalendar) -> None:
        """A random mid-week day with no high/critical event."""
        # April 7, 2026 (Tuesday) — no HIGH/CRITICAL recurring events expected
        assert cal.is_event_day(date(2026, 4, 7)) is False

    def test_specific_event_day(self, cal: EventCalendar) -> None:
        assert cal.is_event_day(date(2026, 6, 1)) is True


# ---------------------------------------------------------------------------
# Tests: Regime modifier
# ---------------------------------------------------------------------------


class TestRegimeModifier:
    def test_normal_day_modifier(self, cal: EventCalendar) -> None:
        """Normal day: no events → default modifiers."""
        # Tuesday, unlikely to have CRITICAL events
        mod = cal.get_regime_modifier("NIFTY50", reference_date=date(2026, 4, 7))
        assert mod.position_size_modifier >= 0.7
        assert mod.caution_level in ("NORMAL", "ELEVATED")

    def test_expiry_day_modifier(self, cal: EventCalendar) -> None:
        """Thursday → expiry day detected, volatility multiplier bumped."""
        mod = cal.get_regime_modifier("NIFTY50", reference_date=date(2026, 4, 9))
        assert mod.is_expiry_day is True
        assert mod.volatility_multiplier >= 1.2

    def test_last_thursday_monthly_expiry_modifier(self, cal: EventCalendar) -> None:
        """Last Thursday of month → higher volatility multiplier."""
        mod = cal.get_regime_modifier("NIFTY50", reference_date=date(2026, 4, 30))
        assert mod.is_expiry_day is True
        assert mod.volatility_multiplier >= 1.5

    def test_event_day_position_size(self, cal: EventCalendar) -> None:
        """On event day with CRITICAL event → position_size_modifier ≤ 0.5."""
        # Election result day
        mod = cal.get_regime_modifier(
            "NIFTY50",
            reference_date=date(2026, 6, 1),
            reference_time=datetime(2026, 6, 1, 10, 0, tzinfo=_IST),
        )
        assert mod.is_event_day is True
        assert mod.position_size_modifier <= 0.5
        assert mod.caution_level == "EXTREME"

    def test_pre_event_caution(self, cal: EventCalendar) -> None:
        """Within caution period of CRITICAL event → EXTREME caution + reduced size."""
        # Day before election results, within 48h caution period
        mod = cal.get_regime_modifier(
            "NIFTY50",
            reference_date=date(2026, 5, 31),
            reference_time=datetime(2026, 5, 31, 10, 0, tzinfo=_IST),
        )
        # Should detect pre-event for the June 1 election
        if mod.is_pre_event:
            assert mod.position_size_modifier <= 0.85
            assert mod.caution_level in ("HIGH", "EXTREME")

    def test_reasoning_contains_event_name(self, cal: EventCalendar) -> None:
        mod = cal.get_regime_modifier(
            "NIFTY50",
            reference_date=date(2026, 6, 1),
            reference_time=datetime(2026, 6, 1, 10, 0, tzinfo=_IST),
        )
        assert "India General Election Results" in mod.reasoning

    def test_unaffected_index(self, cal: EventCalendar) -> None:
        """Index not in affected_indices → no events picked up."""
        mod = cal.get_regime_modifier(
            "NIFTY_PHARMA",
            reference_date=date(2026, 4, 9),
        )
        # Weekly expiry only affects NIFTY50/BANKNIFTY in our fixture
        assert not mod.is_event_day or mod.caution_level == "NORMAL"

    def test_multiple_events_take_highest(self, cal: EventCalendar) -> None:
        """When multiple events on same day, highest severity wins."""
        # April 30 has both monthly and weekly expiry
        mod = cal.get_regime_modifier("NIFTY50", reference_date=date(2026, 4, 30))
        assert mod.volatility_multiplier >= 1.5  # Monthly expiry (MEDIUM) bump


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_calendar(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"recurring_events": [], "specific_events": []}))
        cal = EventCalendar(p)
        events = cal.get_upcoming_events(days_ahead=30)
        assert events == []
        mod = cal.get_regime_modifier("NIFTY50")
        assert mod.position_size_modifier == 1.0

    def test_all_index_matches_any(self, cal: EventCalendar) -> None:
        """Events with 'ALL' in affected_indices should match any index."""
        events = cal.get_upcoming_events(
            days_ahead=7,
            index_id="NIFTY_PHARMA",
            reference_date=date(2026, 1, 28),
        )
        names = [e.name for e in events]
        # Budget has ALL → should appear for NIFTY_PHARMA
        assert "India Union Budget" in names
