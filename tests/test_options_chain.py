"""Tests for OptionsChainFetcher."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.data.options_chain import (
    OIBuildup,
    OISummary,
    OISpike,
    OptionStrike,
    OptionsChainData,
    OptionsChainFetcher,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPIRY = date(2024, 4, 4)
_EXPIRY2 = date(2024, 4, 11)
_NOW = datetime(2024, 4, 1, 10, 0, 0, tzinfo=timezone.utc)

MOCK_NSE_RESPONSE = {
    "records": {
        "underlyingValue": 22000.50,
        "expiryDates": ["04-Apr-2024", "11-Apr-2024"],
        "data": [
            {
                "strikePrice": 21900,
                "expiryDate": "04-Apr-2024",
                "CE": {"openInterest": 500, "changeinOpenInterest": -100,
                       "totalTradedVolume": 2000, "lastPrice": 150.0, "impliedVolatility": 18.0},
                "PE": {"openInterest": 2000, "changeinOpenInterest": 500,
                       "totalTradedVolume": 5000, "lastPrice": 20.0, "impliedVolatility": 14.0},
            },
            {
                "strikePrice": 22000,
                "expiryDate": "04-Apr-2024",
                "CE": {"openInterest": 1500, "changeinOpenInterest": 200,
                       "totalTradedVolume": 4000, "lastPrice": 80.0, "impliedVolatility": 16.0},
                "PE": {"openInterest": 1500, "changeinOpenInterest": 100,
                       "totalTradedVolume": 3500, "lastPrice": 75.0, "impliedVolatility": 16.0},
            },
            {
                "strikePrice": 22100,
                "expiryDate": "04-Apr-2024",
                "CE": {"openInterest": 3000, "changeinOpenInterest": 1000,
                       "totalTradedVolume": 6000, "lastPrice": 30.0, "impliedVolatility": 15.0},
                "PE": {"openInterest": 500, "changeinOpenInterest": -50,
                       "totalTradedVolume": 1000, "lastPrice": 130.0, "impliedVolatility": 17.0},
            },
            # second expiry — should be excluded when filtering for _EXPIRY
            {
                "strikePrice": 22000,
                "expiryDate": "11-Apr-2024",
                "CE": {"openInterest": 100, "changeinOpenInterest": 10,
                       "totalTradedVolume": 500, "lastPrice": 90.0, "impliedVolatility": 17.0},
                "PE": {"openInterest": 100, "changeinOpenInterest": 10,
                       "totalTradedVolume": 500, "lastPrice": 85.0, "impliedVolatility": 17.0},
            },
        ],
    }
}


def _make_strike(
    strike: float,
    ce_oi: int = 1000, ce_oi_change: int = 0, ce_volume: int = 0,
    ce_ltp: float = 100.0, ce_iv: float = 15.0,
    pe_oi: int = 1000, pe_oi_change: int = 0, pe_volume: int = 0,
    pe_ltp: float = 100.0, pe_iv: float = 15.0,
) -> OptionStrike:
    return OptionStrike(
        strike_price=strike,
        ce_oi=ce_oi, ce_oi_change=ce_oi_change, ce_volume=ce_volume,
        ce_ltp=ce_ltp, ce_iv=ce_iv,
        pe_oi=pe_oi, pe_oi_change=pe_oi_change, pe_volume=pe_volume,
        pe_ltp=pe_ltp, pe_iv=pe_iv,
    )


def _make_chain(
    strikes: list[OptionStrike],
    symbol: str = "NIFTY",
    expiry: date = _EXPIRY,
    spot: float = 22000.0,
    ts: datetime = _NOW,
) -> OptionsChainData:
    return OptionsChainData(
        index_id=symbol,
        spot_price=spot,
        timestamp=ts,
        expiry_date=expiry,
        strikes=tuple(strikes),
        available_expiries=(_EXPIRY, _EXPIRY2),
    )


@pytest.fixture()
def fetcher() -> OptionsChainFetcher:
    """Fetcher with mocked scraper; fetch_raw_chain patched per test."""
    scraper_mock = MagicMock()
    return OptionsChainFetcher(scraper_mock)


@pytest.fixture()
def fetcher_with_chain(fetcher: OptionsChainFetcher) -> OptionsChainFetcher:
    """Fetcher whose fetch_raw_chain returns the realistic NSE response."""
    with patch("src.data.options_chain.validate_options_chain") as mock_val:
        mock_val.return_value.is_valid = True
        mock_val.return_value.errors = []
        fetcher.fetch_raw_chain = MagicMock(return_value=MOCK_NSE_RESPONSE)
        yield fetcher


# ---------------------------------------------------------------------------
# TestDataclassFrozen
# ---------------------------------------------------------------------------


class TestDataclassFrozen:
    def test_option_strike_is_frozen(self) -> None:
        s = _make_strike(22000)
        with pytest.raises(Exception):
            s.ce_oi = 9999  # type: ignore[misc]

    def test_options_chain_data_is_frozen(self) -> None:
        chain = _make_chain([_make_strike(22000)])
        with pytest.raises(Exception):
            chain.spot_price = 0.0  # type: ignore[misc]

    def test_oi_summary_is_frozen(self) -> None:
        summary = OISummary(
            total_ce_oi=100, total_pe_oi=100,
            total_ce_oi_change=0, total_pe_oi_change=0,
            total_ce_volume=0, total_pe_volume=0,
            pcr=1.0, pcr_change=0.0,
            max_pain_strike=22000.0,
            highest_ce_oi_strike=22000.0,
            highest_pe_oi_strike=22000.0,
            top_5_ce_oi_strikes=(), top_5_pe_oi_strikes=(),
        )
        with pytest.raises(Exception):
            summary.pcr = 0.0  # type: ignore[misc]

    def test_oi_buildup_is_frozen(self) -> None:
        b = OIBuildup(22000, "CE", "LONG_BUILDUP", 100, 10.0, 5.0, "HIGH")
        with pytest.raises(Exception):
            b.significance = "LOW"  # type: ignore[misc]

    def test_oi_spike_is_frozen(self) -> None:
        sp = OISpike(22000, "CE", 1000, 1200, 20.0, True, _NOW)
        with pytest.raises(Exception):
            sp.current_oi = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestGetOptionsChain
# ---------------------------------------------------------------------------


class TestGetOptionsChain:
    def test_returns_chain_for_nearest_expiry(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        chain = fetcher_with_chain.get_options_chain("NIFTY")
        assert chain is not None
        assert chain.index_id == "NIFTY"
        assert chain.spot_price == 22000.50
        assert chain.expiry_date == _EXPIRY
        assert len(chain.available_expiries) == 2

    def test_filters_only_target_expiry_strikes(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        chain = fetcher_with_chain.get_options_chain("NIFTY")
        assert chain is not None
        # 4 data entries but only 3 belong to 04-Apr-2024
        assert len(chain.strikes) == 3

    def test_explicit_expiry_date_selects_correct_strikes(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        chain = fetcher_with_chain.get_options_chain("NIFTY", expiry_date=_EXPIRY2)
        assert chain is not None
        assert chain.expiry_date == _EXPIRY2
        assert len(chain.strikes) == 1  # only one row for 11-Apr-2024

    def test_strikes_sorted_ascending(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        chain = fetcher_with_chain.get_options_chain("NIFTY")
        assert chain is not None
        prices = [s.strike_price for s in chain.strikes]
        assert prices == sorted(prices)

    def test_strike_fields_parsed_correctly(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        chain = fetcher_with_chain.get_options_chain("NIFTY")
        assert chain is not None
        s = next(s for s in chain.strikes if s.strike_price == 21900)
        assert s.ce_oi == 500
        assert s.ce_oi_change == -100
        assert s.ce_ltp == 150.0
        assert s.pe_oi == 2000
        assert s.pe_ltp == 20.0
        assert s.pe_iv == 14.0

    def test_returns_none_on_fetch_failure(self, fetcher: OptionsChainFetcher) -> None:
        fetcher.fetch_raw_chain = MagicMock(return_value=None)
        assert fetcher.get_options_chain("NIFTY") is None

    def test_returns_none_when_validation_fails(self, fetcher: OptionsChainFetcher) -> None:
        fetcher.fetch_raw_chain = MagicMock(return_value=MOCK_NSE_RESPONSE)
        with patch("src.data.options_chain.validate_options_chain") as mock_val:
            mock_val.return_value.is_valid = False
            mock_val.return_value.errors = ["bad data"]
            result = fetcher.get_options_chain("NIFTY")
        assert result is None

    def test_returns_tuple_types(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        chain = fetcher_with_chain.get_options_chain("NIFTY")
        assert chain is not None
        assert isinstance(chain.strikes, tuple)
        assert isinstance(chain.available_expiries, tuple)


# ---------------------------------------------------------------------------
# TestIndexRegistryGuard
# ---------------------------------------------------------------------------


class TestIndexRegistryGuard:
    def test_non_fo_symbol_returns_none(self, fetcher: OptionsChainFetcher) -> None:
        # NIFTYIT has no options — should be blocked before hitting the API
        result = fetcher.fetch_raw_chain("NIFTYIT")
        assert result is None
        fetcher._scraper._call.assert_not_called()

    def test_valid_fo_symbol_calls_scraper(self, fetcher: OptionsChainFetcher) -> None:
        fetcher._scraper._call.return_value = MOCK_NSE_RESPONSE
        fetcher.fetch_raw_chain("NIFTY")
        fetcher._scraper._call.assert_called_once()


# ---------------------------------------------------------------------------
# TestGetAvailableExpiries
# ---------------------------------------------------------------------------


class TestGetAvailableExpiries:
    def test_returns_sorted_dates(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        expiries = fetcher_with_chain.get_available_expiries("NIFTY")
        assert expiries == sorted(expiries)

    def test_returns_correct_count(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        expiries = fetcher_with_chain.get_available_expiries("NIFTY")
        assert len(expiries) == 2
        assert expiries[0] == _EXPIRY
        assert expiries[1] == _EXPIRY2

    def test_returns_empty_on_fetch_failure(self, fetcher: OptionsChainFetcher) -> None:
        fetcher.fetch_raw_chain = MagicMock(return_value=None)
        assert fetcher.get_available_expiries("NIFTY") == []


# ---------------------------------------------------------------------------
# TestCalculateMaxPain
# ---------------------------------------------------------------------------


class TestCalculateMaxPain:
    def test_empty_strikes_returns_zero(self, fetcher: OptionsChainFetcher) -> None:
        assert fetcher.calculate_max_pain(()) == 0.0

    def test_known_values(self, fetcher: OptionsChainFetcher) -> None:
        # Manual calculation:
        # At 21900: CE loss=0, PE loss = 1500*(22000-21900) + 500*(22100-21900) = 150000+100000 = 250000
        # At 22000: CE loss = 500*(22000-21900)=50000, PE loss = 500*(22100-22000)=50000 → total=100000
        # At 22100: CE loss = 500*(22100-21900)+1500*(22100-22000)=100000+150000=250000, PE loss=0
        # Min loss at 22000 → max pain = 22000
        strikes = (
            _make_strike(21900, ce_oi=500, pe_oi=1500),
            _make_strike(22000, ce_oi=1500, pe_oi=1500),
            _make_strike(22100, ce_oi=3000, pe_oi=500),
        )
        assert fetcher.calculate_max_pain(strikes) == 22000.0

    def test_single_strike_returns_that_strike(self, fetcher: OptionsChainFetcher) -> None:
        strikes = (_make_strike(22000),)
        assert fetcher.calculate_max_pain(strikes) == 22000.0


# ---------------------------------------------------------------------------
# TestGetOISummary
# ---------------------------------------------------------------------------


class TestGetOISummary:
    def test_totals_correct(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        summary = fetcher_with_chain.get_oi_summary("NIFTY")
        assert summary is not None
        assert summary.total_ce_oi == 5000   # 500+1500+3000
        assert summary.total_pe_oi == 4000   # 2000+1500+500

    def test_pcr_correct(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        summary = fetcher_with_chain.get_oi_summary("NIFTY")
        assert summary is not None
        assert summary.pcr == round(4000 / 5000, 4)

    def test_highest_oi_strikes(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        summary = fetcher_with_chain.get_oi_summary("NIFTY")
        assert summary is not None
        assert summary.highest_ce_oi_strike == 22100  # CE OI=3000
        assert summary.highest_pe_oi_strike == 21900  # PE OI=2000

    def test_top5_are_tuples(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        summary = fetcher_with_chain.get_oi_summary("NIFTY")
        assert summary is not None
        assert isinstance(summary.top_5_ce_oi_strikes, tuple)
        assert isinstance(summary.top_5_pe_oi_strikes, tuple)

    def test_returns_none_on_fetch_failure(self, fetcher: OptionsChainFetcher) -> None:
        fetcher.fetch_raw_chain = MagicMock(return_value=None)
        assert fetcher.get_oi_summary("NIFTY") is None

    def test_zero_ce_oi_pcr_is_zero(self, fetcher: OptionsChainFetcher) -> None:
        chain = _make_chain([_make_strike(22000, ce_oi=0, pe_oi=1000)])
        fetcher.get_options_chain = MagicMock(return_value=chain)
        summary = fetcher.get_oi_summary("NIFTY")
        assert summary is not None
        assert summary.pcr == 0.0


# ---------------------------------------------------------------------------
# TestDetectOIBuildup
# ---------------------------------------------------------------------------


class TestDetectOIBuildup:
    def _chains(
        self,
        prev_ce_oi: int, prev_ce_ltp: float,
        prev_pe_oi: int, prev_pe_ltp: float,
        cur_ce_oi: int, cur_ce_ltp: float,
        cur_pe_oi: int, cur_pe_ltp: float,
    ) -> tuple[OptionsChainData, OptionsChainData]:
        prev = _make_chain([_make_strike(22000, ce_oi=prev_ce_oi, ce_ltp=prev_ce_ltp,
                                         pe_oi=prev_pe_oi, pe_ltp=prev_pe_ltp)])
        cur  = _make_chain([_make_strike(22000, ce_oi=cur_ce_oi, ce_ltp=cur_ce_ltp,
                                         pe_oi=cur_pe_oi, pe_ltp=cur_pe_ltp)])
        return cur, prev

    def test_long_buildup(self, fetcher: OptionsChainFetcher) -> None:
        cur, prev = self._chains(1000, 100, 1000, 100, 1200, 120, 1000, 100)
        blds = fetcher.detect_oi_buildup(cur, prev)
        ce = next(b for b in blds if b.option_type == "CE")
        assert ce.buildup_type == "LONG_BUILDUP"

    def test_short_buildup(self, fetcher: OptionsChainFetcher) -> None:
        cur, prev = self._chains(1000, 100, 1000, 100, 1200, 80, 1000, 100)
        blds = fetcher.detect_oi_buildup(cur, prev)
        ce = next(b for b in blds if b.option_type == "CE")
        assert ce.buildup_type == "SHORT_BUILDUP"

    def test_long_unwinding(self, fetcher: OptionsChainFetcher) -> None:
        cur, prev = self._chains(1000, 100, 1000, 100, 800, 80, 1000, 100)
        blds = fetcher.detect_oi_buildup(cur, prev)
        ce = next(b for b in blds if b.option_type == "CE")
        assert ce.buildup_type == "LONG_UNWINDING"

    def test_short_covering(self, fetcher: OptionsChainFetcher) -> None:
        cur, prev = self._chains(1000, 100, 1000, 100, 800, 120, 1000, 100)
        blds = fetcher.detect_oi_buildup(cur, prev)
        ce = next(b for b in blds if b.option_type == "CE")
        assert ce.buildup_type == "SHORT_COVERING"

    def test_neutral_not_included(self, fetcher: OptionsChainFetcher) -> None:
        # No OI change, no price change → NEUTRAL → excluded
        cur, prev = self._chains(1000, 100, 1000, 100, 1000, 100, 1000, 100)
        blds = fetcher.detect_oi_buildup(cur, prev)
        assert blds == []

    def test_significance_high(self, fetcher: OptionsChainFetcher) -> None:
        # >20% OI change
        cur, prev = self._chains(1000, 100, 1000, 100, 1300, 120, 1000, 100)
        blds = fetcher.detect_oi_buildup(cur, prev)
        ce = next(b for b in blds if b.option_type == "CE")
        assert ce.significance == "HIGH"

    def test_significance_medium(self, fetcher: OptionsChainFetcher) -> None:
        # 15% OI change
        cur, prev = self._chains(1000, 100, 1000, 100, 1150, 120, 1000, 100)
        blds = fetcher.detect_oi_buildup(cur, prev)
        ce = next(b for b in blds if b.option_type == "CE")
        assert ce.significance == "MEDIUM"

    def test_significance_low(self, fetcher: OptionsChainFetcher) -> None:
        # 5% OI change
        cur, prev = self._chains(1000, 100, 1000, 100, 1050, 120, 1000, 100)
        blds = fetcher.detect_oi_buildup(cur, prev)
        ce = next(b for b in blds if b.option_type == "CE")
        assert ce.significance == "LOW"

    def test_strike_not_in_previous_skipped(self, fetcher: OptionsChainFetcher) -> None:
        prev = _make_chain([_make_strike(21900)])
        cur  = _make_chain([_make_strike(21900), _make_strike(22000)])
        # 22000 has no previous — should be skipped, no KeyError
        blds = fetcher.detect_oi_buildup(cur, prev)
        strike_prices = {b.strike_price for b in blds}
        assert 22000 not in strike_prices

    def test_both_ce_and_pe_detected(self, fetcher: OptionsChainFetcher) -> None:
        cur, prev = self._chains(1000, 100, 1000, 100, 1200, 120, 800, 80)
        blds = fetcher.detect_oi_buildup(cur, prev)
        types = {b.option_type for b in blds}
        assert "CE" in types
        assert "PE" in types


# ---------------------------------------------------------------------------
# TestDetectOISpikes
# ---------------------------------------------------------------------------


class TestDetectOISpikes:
    def test_spike_above_threshold_detected(self, fetcher: OptionsChainFetcher) -> None:
        prev = _make_chain([_make_strike(22000, ce_oi=1000, pe_oi=1000)])
        cur  = _make_chain([_make_strike(22000, ce_oi=1200, pe_oi=1000)])  # 20% CE spike
        spikes = fetcher.detect_oi_spikes(cur, prev, threshold_pct=10.0)
        assert len(spikes) == 1
        assert spikes[0].option_type == "CE"
        assert spikes[0].change_pct == 20.0

    def test_below_threshold_not_detected(self, fetcher: OptionsChainFetcher) -> None:
        prev = _make_chain([_make_strike(22000, ce_oi=1000, pe_oi=1000)])
        cur  = _make_chain([_make_strike(22000, ce_oi=1050, pe_oi=1000)])  # 5% — below 10%
        spikes = fetcher.detect_oi_spikes(cur, prev, threshold_pct=10.0)
        assert spikes == []

    def test_is_new_position_true_for_oi_increase(self, fetcher: OptionsChainFetcher) -> None:
        prev = _make_chain([_make_strike(22000, ce_oi=1000)])
        cur  = _make_chain([_make_strike(22000, ce_oi=1200)])
        spikes = fetcher.detect_oi_spikes(cur, prev)
        assert spikes[0].is_new_position is True

    def test_is_new_position_false_for_oi_decrease(self, fetcher: OptionsChainFetcher) -> None:
        prev = _make_chain([_make_strike(22000, ce_oi=1000)])
        cur  = _make_chain([_make_strike(22000, ce_oi=700)])  # -30%
        spikes = fetcher.detect_oi_spikes(cur, prev)
        assert spikes[0].is_new_position is False
        assert spikes[0].change_pct == -30.0

    def test_zero_prev_oi_skipped(self, fetcher: OptionsChainFetcher) -> None:
        prev = _make_chain([_make_strike(22000, ce_oi=0)])
        cur  = _make_chain([_make_strike(22000, ce_oi=500)])
        spikes = fetcher.detect_oi_spikes(cur, prev)
        assert spikes == []

    def test_spike_timestamp_matches_current_chain(self, fetcher: OptionsChainFetcher) -> None:
        ts2 = datetime(2024, 4, 1, 11, 0, 0, tzinfo=timezone.utc)
        prev = _make_chain([_make_strike(22000, ce_oi=1000)])
        cur  = _make_chain([_make_strike(22000, ce_oi=1500)], ts=ts2)
        spikes = fetcher.detect_oi_spikes(cur, prev)
        assert spikes[0].timestamp == ts2


# ---------------------------------------------------------------------------
# TestMemorySnapshots
# ---------------------------------------------------------------------------


class TestMemorySnapshots:
    def test_first_fetch_stores_snapshot(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        fetcher_with_chain.get_options_chain("NIFTY")
        snaps = fetcher_with_chain.get_memory_snapshots("NIFTY", _EXPIRY)
        assert len(snaps) == 1

    def test_two_fetches_stores_two_snapshots(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        fetcher_with_chain.get_options_chain("NIFTY")
        fetcher_with_chain.get_options_chain("NIFTY")
        snaps = fetcher_with_chain.get_memory_snapshots("NIFTY", _EXPIRY)
        assert len(snaps) == 2

    def test_max_two_snapshots_retained(self, fetcher_with_chain: OptionsChainFetcher) -> None:
        for _ in range(5):
            fetcher_with_chain.get_options_chain("NIFTY")
        snaps = fetcher_with_chain.get_memory_snapshots("NIFTY", _EXPIRY)
        assert len(snaps) <= 2

    def test_unknown_key_returns_empty(self, fetcher: OptionsChainFetcher) -> None:
        assert fetcher.get_memory_snapshots("GHOST", _EXPIRY) == []


# ---------------------------------------------------------------------------
# TestSaveToDb
# ---------------------------------------------------------------------------


class TestSaveToDb:
    def test_inserts_both_ce_and_pe_rows(self, fetcher: OptionsChainFetcher) -> None:
        db = MagicMock()
        db.execute_many.return_value = None
        db.execute.return_value = None

        chain = _make_chain([
            _make_strike(21900, ce_oi=500, pe_oi=2000),
            _make_strike(22000, ce_oi=1500, pe_oi=1500),
        ])
        fetcher.save_to_db(chain, db)

        rows = db.execute_many.call_args[0][1]
        assert len(rows) == 4  # 2 strikes × 2 option types
        option_types = {r[4] for r in rows}
        assert option_types == {"CE", "PE"}

    def test_oi_aggregated_insert_called(self, fetcher: OptionsChainFetcher) -> None:
        db = MagicMock()
        chain = _make_chain([_make_strike(22000, ce_oi=1000, pe_oi=1000)])
        fetcher.save_to_db(chain, db)
        db.execute.assert_called_once()

    def test_does_not_call_fetch_raw_chain(self, fetcher: OptionsChainFetcher) -> None:
        """save_to_db must not trigger a network call."""
        db = MagicMock()
        fetcher.fetch_raw_chain = MagicMock()
        chain = _make_chain([_make_strike(22000)])
        fetcher.save_to_db(chain, db)
        fetcher.fetch_raw_chain.assert_not_called()

    def test_db_error_does_not_raise(self, fetcher: OptionsChainFetcher) -> None:
        db = MagicMock()
        db.execute_many.side_effect = Exception("db down")
        chain = _make_chain([_make_strike(22000)])
        # Should log error but not propagate
        fetcher.save_to_db(chain, db)

    def test_empty_strikes_skips_snapshot_insert(self, fetcher: OptionsChainFetcher) -> None:
        db = MagicMock()
        chain = _make_chain([])
        fetcher.save_to_db(chain, db)
        db.execute_many.assert_not_called()
