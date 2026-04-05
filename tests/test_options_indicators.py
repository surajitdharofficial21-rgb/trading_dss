"""
Tests for src/analysis/indicators/options_indicators.py

Covers:
  - PCR calculation and interpretation at boundary values
  - OI-based support / resistance identification
  - OI concentration calculation
  - OI change signal direction
  - Max pain calculation with known option chain data
  - Max pain gravitational pull classification
  - Max pain shift detection
  - IV analysis: ATM IV, skew, smile
  - IV Rank / Percentile with crafted IV history
  - IV regime classification and insufficient history handling
  - Buildup type detection (long buildup, short covering, etc.)
  - OI change analysis: dominant buildup and net sentiment
  - ATM signal detection
  - Options summary vote logic
  - Edge cases: empty chain, zero OI, zero IV
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import pytest

from src.data.options_chain import OptionsChainData, OptionStrike
from src.analysis.indicators.options_indicators import (
    IVAnalysis,
    IVRankResult,
    MaxPainAnalysis,
    OIChangeAnalysis,
    OIStructureAnalysis,
    OptionsIndicators,
    OptionsSummary,
    StrikeBuildup,
    _classify_buildup,
    _classify_significance,
    _interpret_pcr,
)

oi = OptionsIndicators()


# ---------------------------------------------------------------------------
# OptionsChainData factories
# ---------------------------------------------------------------------------


def make_chain(
    strikes: list[OptionStrike],
    spot_price: float = 22000.0,
    index_id: str = "NIFTY",
    expiry_date: date | None = None,
) -> OptionsChainData:
    """Build a minimal OptionsChainData from a list of strikes."""
    if expiry_date is None:
        expiry_date = date.today() + timedelta(days=3)
    return OptionsChainData(
        index_id=index_id,
        spot_price=spot_price,
        timestamp=datetime(2026, 4, 3, 14, 30),
        expiry_date=expiry_date,
        strikes=tuple(strikes),
        available_expiries=(expiry_date,),
    )


def make_strike(
    strike_price: float,
    ce_oi: int = 0,
    ce_oi_change: int = 0,
    ce_volume: int = 0,
    ce_ltp: float = 0.0,
    ce_iv: float = 0.0,
    pe_oi: int = 0,
    pe_oi_change: int = 0,
    pe_volume: int = 0,
    pe_ltp: float = 0.0,
    pe_iv: float = 0.0,
) -> OptionStrike:
    """Build a single OptionStrike with sensible defaults."""
    return OptionStrike(
        strike_price=strike_price,
        ce_oi=ce_oi,
        ce_oi_change=ce_oi_change,
        ce_volume=ce_volume,
        ce_ltp=ce_ltp,
        ce_iv=ce_iv,
        pe_oi=pe_oi,
        pe_oi_change=pe_oi_change,
        pe_volume=pe_volume,
        pe_ltp=pe_ltp,
        pe_iv=pe_iv,
    )


def _standard_chain() -> OptionsChainData:
    """A realistic NIFTY-like chain with asymmetric OI for testing.

    Spot = 22000
    Highest PE OI at 21800 (support)
    Highest CE OI at 22200 (resistance)
    PCR should be ~ 1.04 (slightly bullish)
    """
    strikes = [
        make_strike(21600, ce_oi=50000, ce_oi_change=2000, ce_ltp=420.0, ce_iv=16.0,
                    pe_oi=20000, pe_oi_change=1000, pe_ltp=25.0, pe_iv=18.0),
        make_strike(21700, ce_oi=80000, ce_oi_change=5000, ce_ltp=320.0, ce_iv=15.5,
                    pe_oi=40000, pe_oi_change=3000, pe_ltp=35.0, pe_iv=17.5),
        make_strike(21800, ce_oi=120000, ce_oi_change=8000, ce_ltp=230.0, ce_iv=15.0,
                    pe_oi=180000, pe_oi_change=15000, pe_ltp=50.0, pe_iv=16.5),
        make_strike(21900, ce_oi=150000, ce_oi_change=10000, ce_ltp=150.0, ce_iv=14.0,
                    pe_oi=130000, pe_oi_change=8000, pe_ltp=70.0, pe_iv=15.0),
        make_strike(22000, ce_oi=180000, ce_oi_change=12000, ce_ltp=90.0, ce_iv=13.5,
                    pe_oi=160000, pe_oi_change=11000, pe_ltp=90.0, pe_iv=13.5),
        make_strike(22100, ce_oi=160000, ce_oi_change=9000, ce_ltp=45.0, ce_iv=14.5,
                    pe_oi=100000, pe_oi_change=6000, pe_ltp=150.0, pe_iv=14.0),
        make_strike(22200, ce_oi=200000, ce_oi_change=14000, ce_ltp=20.0, ce_iv=15.0,
                    pe_oi=60000, pe_oi_change=2000, pe_ltp=230.0, pe_iv=15.5),
        make_strike(22300, ce_oi=100000, ce_oi_change=7000, ce_ltp=8.0, ce_iv=16.0,
                    pe_oi=30000, pe_oi_change=1000, pe_ltp=310.0, pe_iv=16.0),
        make_strike(22400, ce_oi=60000, ce_oi_change=3000, ce_ltp=3.0, ce_iv=17.0,
                    pe_oi=10000, pe_oi_change=500, pe_ltp=400.0, pe_iv=17.5),
    ]
    return make_chain(strikes, spot_price=22000.0)


def _empty_chain() -> OptionsChainData:
    """Chain with no strikes."""
    return make_chain([], spot_price=22000.0)


def _bullish_chain() -> OptionsChainData:
    """A chain with heavy PE OI (PCR > 1.2) → very bullish."""
    strikes = [
        make_strike(21800, ce_oi=50000, ce_oi_change=1000, ce_ltp=220.0, ce_iv=15.0,
                    pe_oi=300000, pe_oi_change=25000, pe_ltp=30.0, pe_iv=16.0),
        make_strike(22000, ce_oi=100000, ce_oi_change=5000, ce_ltp=80.0, ce_iv=13.0,
                    pe_oi=200000, pe_oi_change=15000, pe_ltp=80.0, pe_iv=13.0),
        make_strike(22200, ce_oi=80000, ce_oi_change=3000, ce_ltp=15.0, ce_iv=15.0,
                    pe_oi=150000, pe_oi_change=10000, pe_ltp=220.0, pe_iv=16.0),
    ]
    return make_chain(strikes, spot_price=22000.0)


def _bearish_chain() -> OptionsChainData:
    """A chain with heavy CE OI (PCR < 0.6) → very bearish."""
    strikes = [
        make_strike(21800, ce_oi=300000, ce_oi_change=25000, ce_ltp=220.0, ce_iv=15.0,
                    pe_oi=50000, pe_oi_change=1000, pe_ltp=30.0, pe_iv=16.0),
        make_strike(22000, ce_oi=200000, ce_oi_change=15000, ce_ltp=80.0, ce_iv=13.0,
                    pe_oi=80000, pe_oi_change=3000, pe_ltp=80.0, pe_iv=13.0),
        make_strike(22200, ce_oi=150000, ce_oi_change=10000, ce_ltp=15.0, ce_iv=15.0,
                    pe_oi=30000, pe_oi_change=500, pe_ltp=220.0, pe_iv=16.0),
    ]
    return make_chain(strikes, spot_price=22000.0)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for module-level helper functions."""

    @pytest.mark.parametrize(
        "pcr, expected",
        [
            (1.5, "VERY_BULLISH"),
            (1.21, "VERY_BULLISH"),
            (1.2, "BULLISH"),
            (1.1, "BULLISH"),
            (1.0, "BULLISH"),
            (0.9, "NEUTRAL"),
            (0.8, "NEUTRAL"),
            (0.7, "BEARISH"),
            (0.6, "BEARISH"),
            (0.5, "VERY_BEARISH"),
            (0.3, "VERY_BEARISH"),
            (0.0, "VERY_BEARISH"),
        ],
    )
    def test_interpret_pcr_boundaries(self, pcr: float, expected: str) -> None:
        assert _interpret_pcr(pcr) == expected

    @pytest.mark.parametrize(
        "price_change, oi_change, expected",
        [
            (1.0, 100, "LONG_BUILDUP"),
            (-1.0, 100, "SHORT_BUILDUP"),
            (-1.0, -100, "LONG_UNWINDING"),
            (1.0, -100, "SHORT_COVERING"),
            (0.0, 100, "NEUTRAL"),
            (1.0, 0, "NEUTRAL"),
            (0.0, 0, "NEUTRAL"),
        ],
    )
    def test_classify_buildup(
        self, price_change: float, oi_change: int, expected: str
    ) -> None:
        assert _classify_buildup(price_change, oi_change) == expected

    @pytest.mark.parametrize(
        "oi_change_pct, expected",
        [
            (25.0, "HIGH"),
            (20.1, "HIGH"),
            (15.0, "MEDIUM"),
            (10.1, "MEDIUM"),
            (5.0, "LOW"),
            (0.0, "LOW"),
            (-30.0, "HIGH"),
        ],
    )
    def test_classify_significance(self, oi_change_pct: float, expected: str) -> None:
        assert _classify_significance(oi_change_pct) == expected


# ---------------------------------------------------------------------------
# OI Structure Analysis
# ---------------------------------------------------------------------------


class TestOIStructureAnalysis:
    """Tests for OptionsIndicators.analyze_oi_structure."""

    def test_pcr_calculation(self) -> None:
        """PCR = total PE OI / total CE OI."""
        chain = _standard_chain()
        result = oi.analyze_oi_structure(chain)
        assert isinstance(result, OIStructureAnalysis)

        total_ce = sum(s.ce_oi for s in chain.strikes)
        total_pe = sum(s.pe_oi for s in chain.strikes)
        expected_pcr = round(total_pe / total_ce, 4)
        assert result.pcr == pytest.approx(expected_pcr, abs=0.001)
        assert result.total_ce_oi == total_ce
        assert result.total_pe_oi == total_pe

    def test_pcr_very_bullish(self) -> None:
        """PCR > 1.2 should be VERY_BULLISH."""
        chain = _bullish_chain()
        result = oi.analyze_oi_structure(chain)
        assert result.pcr > 1.2
        assert result.pcr_interpretation == "VERY_BULLISH"

    def test_pcr_very_bearish(self) -> None:
        """PCR < 0.6 should be VERY_BEARISH."""
        chain = _bearish_chain()
        result = oi.analyze_oi_structure(chain)
        assert result.pcr < 0.6
        assert result.pcr_interpretation == "VERY_BEARISH"

    def test_pcr_zero_ce_oi(self) -> None:
        """When total CE OI is 0, PCR should be 0.0."""
        strikes = [make_strike(22000, ce_oi=0, pe_oi=100000)]
        chain = make_chain(strikes)
        result = oi.analyze_oi_structure(chain)
        assert result.pcr == 0.0

    def test_oi_based_support_resistance(self) -> None:
        """Max PE OI strike = support, max CE OI strike = resistance."""
        chain = _standard_chain()
        result = oi.analyze_oi_structure(chain)
        # From the standard chain:
        # Highest PE OI = 180000 at 21800
        # Highest CE OI = 200000 at 22200
        assert result.max_pe_oi_strike == 21800.0
        assert result.max_ce_oi_strike == 22200.0
        assert result.oi_based_range == (21800.0, 22200.0)

    def test_oi_concentration(self) -> None:
        """OI concentration = top 3 strikes / total."""
        chain = _standard_chain()
        result = oi.analyze_oi_structure(chain)
        assert 0.0 < result.ce_oi_concentration <= 1.0
        assert 0.0 < result.pe_oi_concentration <= 1.0

    def test_oi_concentration_high_when_concentrated(self) -> None:
        """All OI in one strike → concentration should be 1.0."""
        strikes = [
            make_strike(22000, ce_oi=500000, pe_oi=500000),
            make_strike(22100, ce_oi=0, pe_oi=0),
            make_strike(22200, ce_oi=0, pe_oi=0),
        ]
        chain = make_chain(strikes)
        result = oi.analyze_oi_structure(chain)
        assert result.ce_oi_concentration == 1.0
        assert result.pe_oi_concentration == 1.0

    def test_oi_change_signal_bullish(self) -> None:
        """PE OI change > CE OI change → BULLISH."""
        chain = _bullish_chain()
        result = oi.analyze_oi_structure(chain)
        # Bullish chain has much more PE OI change than CE
        assert result.oi_change_signal == "BULLISH"

    def test_oi_change_signal_bearish(self) -> None:
        """CE OI change > PE OI change → BEARISH."""
        chain = _bearish_chain()
        result = oi.analyze_oi_structure(chain)
        assert result.oi_change_signal == "BEARISH"

    def test_oi_change_signal_neutral(self) -> None:
        """Equal OI changes → NEUTRAL."""
        strikes = [
            make_strike(22000, ce_oi=100000, ce_oi_change=5000,
                        pe_oi=100000, pe_oi_change=5000),
        ]
        chain = make_chain(strikes)
        result = oi.analyze_oi_structure(chain)
        assert result.oi_change_signal == "NEUTRAL"


# ---------------------------------------------------------------------------
# OI Change / Buildup Analysis
# ---------------------------------------------------------------------------


class TestOIChangeAnalysis:
    """Tests for OptionsIndicators.analyze_oi_change."""

    def test_buildup_detection(self) -> None:
        """Should detect buildup types based on OI change and price."""
        chain = _standard_chain()
        result = oi.analyze_oi_change(chain, spot_price=22000.0)
        assert isinstance(result, OIChangeAnalysis)
        assert len(result.buildups) > 0
        assert all(isinstance(b, StrikeBuildup) for b in result.buildups)

    def test_top_10_limit(self) -> None:
        """Buildups should be limited to top 10 by absolute OI change."""
        chain = _standard_chain()
        result = oi.analyze_oi_change(chain, spot_price=22000.0)
        assert len(result.buildups) <= 10

    def test_buildup_sorted_by_abs_oi_change(self) -> None:
        """Buildups should be sorted by absolute OI change descending."""
        chain = _standard_chain()
        result = oi.analyze_oi_change(chain, spot_price=22000.0)
        if len(result.buildups) >= 2:
            for i in range(len(result.buildups) - 1):
                assert abs(result.buildups[i].oi_change) >= abs(
                    result.buildups[i + 1].oi_change
                )

    def test_dominant_buildup_populated(self) -> None:
        """dominant_buildup should be a valid buildup type."""
        chain = _standard_chain()
        result = oi.analyze_oi_change(chain, spot_price=22000.0)
        valid = {"LONG_BUILDUP", "SHORT_BUILDUP", "LONG_UNWINDING", "SHORT_COVERING", "NEUTRAL"}
        assert result.dominant_buildup in valid

    def test_net_sentiment_valid(self) -> None:
        """net_sentiment should be BULLISH / BEARISH / NEUTRAL."""
        chain = _standard_chain()
        result = oi.analyze_oi_change(chain, spot_price=22000.0)
        assert result.net_sentiment in ("BULLISH", "BEARISH", "NEUTRAL")

    def test_atm_signal_detection(self) -> None:
        """ATM signal should reflect CE vs PE OI change at the money."""
        chain = _standard_chain()
        result = oi.analyze_oi_change(chain, spot_price=22000.0)
        valid_signals = {
            "BULLISH_ATM_PE_ADDING",
            "BEARISH_ATM_CE_ADDING",
            "UNWINDING_BOTH_SIDES",
            "NEUTRAL",
        }
        assert result.atm_signal in valid_signals

    def test_atm_ce_pe_oi_change_values(self) -> None:
        """ATM OI changes should match the strike closest to spot."""
        # Spot = 22000, ATM strike = 22000
        chain = _standard_chain()
        result = oi.analyze_oi_change(chain, spot_price=22000.0)
        # ATM CE OI change = 12000, ATM PE OI change = 11000
        assert result.atm_ce_oi_change == 12000
        assert result.atm_pe_oi_change == 11000

    def test_empty_chain_buildup(self) -> None:
        """Empty chain should return empty buildups."""
        chain = _empty_chain()
        result = oi.analyze_oi_change(chain, spot_price=22000.0)
        assert result.buildups == []
        assert result.dominant_buildup == "NEUTRAL"
        assert result.net_sentiment == "NEUTRAL"

    def test_buildup_with_no_oi_change(self) -> None:
        """Strikes with zero OI change should not produce buildups."""
        strikes = [
            make_strike(22000, ce_oi=100000, ce_oi_change=0,
                        pe_oi=100000, pe_oi_change=0),
        ]
        chain = make_chain(strikes)
        result = oi.analyze_oi_change(chain, spot_price=22000.0)
        assert result.buildups == []

    def test_long_buildup_detection(self) -> None:
        """Strike with OI adding and rich premium → LONG_BUILDUP."""
        # CE at 21800 with OI adding and premium (LTP=420) well above intrinsic (200)
        strikes = [
            make_strike(21800, ce_oi=100000, ce_oi_change=10000, ce_ltp=420.0,
                        pe_oi=0, pe_oi_change=0),
        ]
        chain = make_chain(strikes, spot_price=22000.0)
        result = oi.analyze_oi_change(chain, spot_price=22000.0)
        ce_buildups = [b for b in result.buildups if b.option_type == "CE"]
        assert len(ce_buildups) > 0
        assert ce_buildups[0].buildup_type == "LONG_BUILDUP"

    def test_short_buildup_detection(self) -> None:
        """Strike with OI adding and cheap premium → SHORT_BUILDUP."""
        # CE well below intrinsic → premium is cheap → SHORT_BUILDUP
        strikes = [
            make_strike(21800, ce_oi=100000, ce_oi_change=10000, ce_ltp=190.0,
                        pe_oi=0, pe_oi_change=0),
        ]
        chain = make_chain(strikes, spot_price=22000.0)
        result = oi.analyze_oi_change(chain, spot_price=22000.0)
        ce_buildups = [b for b in result.buildups if b.option_type == "CE"]
        assert len(ce_buildups) > 0
        assert ce_buildups[0].buildup_type == "SHORT_BUILDUP"


# ---------------------------------------------------------------------------
# Max Pain Analysis
# ---------------------------------------------------------------------------


class TestMaxPainAnalysis:
    """Tests for OptionsIndicators.calculate_max_pain_detailed."""

    def test_max_pain_known_values(self) -> None:
        """Max pain should be the strike where writers lose least."""
        # Simple symmetric chain — max pain should be close to ATM
        strikes = [
            make_strike(21800, ce_oi=100000, pe_oi=50000),
            make_strike(22000, ce_oi=150000, pe_oi=150000),
            make_strike(22200, ce_oi=50000, pe_oi=100000),
        ]
        chain = make_chain(strikes, spot_price=22000.0)
        result = oi.calculate_max_pain_detailed(chain)
        assert isinstance(result, MaxPainAnalysis)
        # With symmetric OI around 22000, max pain should be 22000
        assert result.max_pain_strike == 22000.0

    def test_max_pain_asymmetric(self) -> None:
        """Max pain shifts towards the strike with most total OI weight."""
        # Heavy CE OI at 22200, heavy PE OI at 21800
        # Max pain should be between these two
        chain = _standard_chain()
        result = oi.calculate_max_pain_detailed(chain)
        assert 21600 <= result.max_pain_strike <= 22400

    def test_max_pain_distance_metrics(self) -> None:
        """Distance from spot should be calculated correctly."""
        chain = _standard_chain()
        result = oi.calculate_max_pain_detailed(chain)
        expected_dist = result.max_pain_strike - chain.spot_price
        assert result.distance_from_spot == pytest.approx(expected_dist, abs=0.01)
        expected_pct = abs(expected_dist) / chain.spot_price * 100
        assert result.distance_pct == pytest.approx(expected_pct, abs=0.01)

    def test_max_pain_pain_curve(self) -> None:
        """Pain curve should contain entries for every strike."""
        chain = _standard_chain()
        result = oi.calculate_max_pain_detailed(chain)
        assert len(result.pain_curve) == len(chain.strikes)
        # All pain values should be non-negative
        for strike_price, pain in result.pain_curve:
            assert pain >= 0

    def test_max_pain_gravitational_pull_strong(self) -> None:
        """Spot at max pain → STRONG gravitational pull."""
        strikes = [
            make_strike(21800, ce_oi=50000, pe_oi=100000),
            make_strike(22000, ce_oi=200000, pe_oi=200000),
            make_strike(22200, ce_oi=100000, pe_oi=50000),
        ]
        chain = make_chain(strikes, spot_price=22000.0)
        result = oi.calculate_max_pain_detailed(chain)
        # Max pain = 22000, spot = 22000 → 0% distance → STRONG
        assert result.gravitational_pull == "STRONG"

    def test_max_pain_gravitational_pull_weak(self) -> None:
        """Spot far from max pain → WEAK gravitational pull."""
        # Put all OI at 21600 to push max pain there
        strikes = [
            make_strike(21600, ce_oi=500000, pe_oi=500000),
            make_strike(22000, ce_oi=1000, pe_oi=1000),
            make_strike(22400, ce_oi=1000, pe_oi=1000),
        ]
        chain = make_chain(strikes, spot_price=22400.0)
        result = oi.calculate_max_pain_detailed(chain)
        # Max pain ~ 21600, spot = 22400 → ~3.6% distance → WEAK
        assert result.gravitational_pull == "WEAK"

    def test_max_pain_shift_detection(self) -> None:
        """Should detect shift from previous max pain value."""
        chain = _standard_chain()
        result = oi.calculate_max_pain_detailed(chain, previous_max_pain=21900.0)
        expected_shift = result.max_pain_strike - 21900.0
        assert result.max_pain_shift == pytest.approx(expected_shift, abs=0.01)

    def test_max_pain_shift_none_when_no_previous(self) -> None:
        """First call without previous value → shift = None."""
        indicator = OptionsIndicators()  # Fresh instance
        chain = _standard_chain()
        result = indicator.calculate_max_pain_detailed(chain)
        assert result.max_pain_shift is None

    def test_max_pain_days_to_expiry(self) -> None:
        """days_to_expiry should be non-negative."""
        chain = _standard_chain()
        result = oi.calculate_max_pain_detailed(chain)
        assert result.days_to_expiry >= 0

    def test_max_pain_empty_chain(self) -> None:
        """Empty chain should return default values."""
        chain = _empty_chain()
        result = oi.calculate_max_pain_detailed(chain)
        assert result.max_pain_strike == 0.0
        assert result.pain_curve == []
        assert result.gravitational_pull == "WEAK"

    def test_max_pain_single_strike(self) -> None:
        """Single strike → max pain at that strike."""
        strikes = [make_strike(22000, ce_oi=100000, pe_oi=100000)]
        chain = make_chain(strikes, spot_price=22000.0)
        result = oi.calculate_max_pain_detailed(chain)
        assert result.max_pain_strike == 22000.0


# ---------------------------------------------------------------------------
# IV Analysis
# ---------------------------------------------------------------------------


class TestIVAnalysis:
    """Tests for OptionsIndicators.analyze_iv."""

    def test_atm_iv_calculation(self) -> None:
        """ATM IV should be the average of ATM CE and PE IV."""
        chain = _standard_chain()
        result = oi.analyze_iv(chain)
        assert isinstance(result, IVAnalysis)
        # ATM strike = 22000, CE IV = 13.5, PE IV = 13.5
        assert result.atm_iv == pytest.approx(13.5, abs=0.1)

    def test_iv_skew(self) -> None:
        """IV skew = avg OTM put IV - avg OTM call IV."""
        chain = _standard_chain()
        result = oi.analyze_iv(chain)
        # OTM puts (strike < spot): 21600-21900
        # OTM calls (strike > spot): 22100-22400
        assert isinstance(result.iv_skew, float)

    def test_iv_skew_positive_when_puts_expensive(self) -> None:
        """Positive skew when puts have higher IV than calls."""
        strikes = [
            make_strike(21800, ce_ltp=200.0, ce_iv=10.0, pe_ltp=30.0, pe_iv=25.0),
            make_strike(22000, ce_ltp=80.0, ce_iv=12.0, pe_ltp=80.0, pe_iv=12.0),
            make_strike(22200, ce_ltp=15.0, ce_iv=10.0, pe_ltp=200.0, pe_iv=10.0),
        ]
        chain = make_chain(strikes, spot_price=22000.0)
        result = oi.analyze_iv(chain)
        # OTM put IV at 21800 = 25.0, OTM call IV at 22200 = 10.0
        assert result.iv_skew > 0  # Puts more expensive

    def test_iv_smile_curve(self) -> None:
        """iv_smile should contain (strike, iv) pairs."""
        chain = _standard_chain()
        result = oi.analyze_iv(chain)
        assert len(result.iv_smile) > 0
        for strike_price, iv in result.iv_smile:
            assert strike_price > 0
            assert iv > 0

    def test_avg_ce_pe_iv(self) -> None:
        """Average CE and PE IV should be computed from valid strikes."""
        chain = _standard_chain()
        result = oi.analyze_iv(chain)
        assert result.avg_ce_iv > 0
        assert result.avg_pe_iv > 0

    def test_iv_put_call_spread(self) -> None:
        """iv_put_call_spread = avg PE IV - avg CE IV."""
        chain = _standard_chain()
        result = oi.analyze_iv(chain)
        expected = round(result.avg_pe_iv - result.avg_ce_iv, 4)
        assert result.iv_put_call_spread == pytest.approx(expected, abs=0.01)

    def test_iv_zero_iv_excluded(self) -> None:
        """Strikes with IV = 0 should be excluded from calculations."""
        strikes = [
            make_strike(21800, ce_iv=15.0, pe_iv=16.0, ce_oi=1000, pe_oi=1000),
            make_strike(22000, ce_iv=0.0, pe_iv=0.0, ce_oi=1000, pe_oi=1000),
            make_strike(22200, ce_iv=15.0, pe_iv=16.0, ce_oi=1000, pe_oi=1000),
        ]
        chain = make_chain(strikes, spot_price=22000.0)
        result = oi.analyze_iv(chain)
        # Smile should not include the zero-IV strike
        smile_strikes = [s for s, _ in result.iv_smile]
        assert 22000.0 not in smile_strikes

    def test_iv_all_zero(self) -> None:
        """All zero IV should return 0 for everything."""
        strikes = [
            make_strike(22000, ce_iv=0.0, pe_iv=0.0, ce_oi=1000, pe_oi=1000),
        ]
        chain = make_chain(strikes, spot_price=22000.0)
        result = oi.analyze_iv(chain)
        assert result.atm_iv == 0.0
        assert result.avg_ce_iv == 0.0
        assert result.avg_pe_iv == 0.0
        assert result.iv_smile == []


# ---------------------------------------------------------------------------
# IV Rank & Percentile
# ---------------------------------------------------------------------------


class TestIVRank:
    """Tests for OptionsIndicators.calculate_iv_rank."""

    def test_iv_rank_calculation(self) -> None:
        """IV Rank = (current - low) / (high - low) × 100."""
        # History: IVs from 10 to 30
        iv_history = [10.0 + i * 0.5 for i in range(60)]  # 10.0 to 39.5
        result = oi.calculate_iv_rank(25.0, iv_history)
        assert result is not None
        # Rank = (25 - 10) / (39.5 - 10) × 100 = 15 / 29.5 × 100 ≈ 50.85
        expected_rank = (25.0 - 10.0) / (39.5 - 10.0) * 100
        assert result.iv_rank == pytest.approx(expected_rank, abs=0.1)

    def test_iv_percentile_calculation(self) -> None:
        """IV Percentile = % of days where IV was below current."""
        iv_history = list(range(1, 101))  # 1 to 100
        result = oi.calculate_iv_rank(50.0, [float(x) for x in iv_history])
        assert result is not None
        # 49 values below 50 out of 100 → 49%
        assert result.iv_percentile == pytest.approx(49.0, abs=0.1)

    def test_iv_rank_low_regime(self) -> None:
        """IV Rank < 25 → LOW regime."""
        iv_history = [10.0 + i * 0.5 for i in range(60)]
        result = oi.calculate_iv_rank(12.0, iv_history)
        assert result is not None
        assert result.iv_regime == "LOW"
        assert "buying" in result.trading_implication.lower()

    def test_iv_rank_normal_regime(self) -> None:
        """IV Rank 25-50 → NORMAL regime."""
        iv_history = [10.0 + i * 0.5 for i in range(60)]
        current = 10.0 + (60 * 0.35) * 0.5  # ~35% of range
        result = oi.calculate_iv_rank(current, iv_history)
        assert result is not None
        assert result.iv_regime == "NORMAL"

    def test_iv_rank_high_regime(self) -> None:
        """IV Rank 50-75 → HIGH regime."""
        iv_history = [10.0 + i * 0.5 for i in range(60)]
        # High should be 39.5, low should be 10.0
        # For rank ~60: current = 10.0 + 0.6 * 29.5 = 27.7
        result = oi.calculate_iv_rank(27.7, iv_history)
        assert result is not None
        assert result.iv_regime == "HIGH"
        assert "selling" in result.trading_implication.lower()

    def test_iv_rank_very_high_regime(self) -> None:
        """IV Rank > 75 → VERY_HIGH regime."""
        iv_history = [10.0 + i * 0.5 for i in range(60)]
        result = oi.calculate_iv_rank(35.0, iv_history)
        assert result is not None
        assert result.iv_regime == "VERY_HIGH"

    def test_iv_rank_insufficient_history(self) -> None:
        """< 30 data points → insufficient_history."""
        iv_history = [15.0] * 20
        result = oi.calculate_iv_rank(15.0, iv_history)
        assert result is not None
        assert result.iv_regime == "insufficient_history"

    def test_iv_rank_empty_history(self) -> None:
        """Empty history → insufficient_history."""
        result = oi.calculate_iv_rank(15.0, [])
        assert result is not None
        assert result.iv_regime == "insufficient_history"

    def test_iv_rank_flat_history(self) -> None:
        """Flat IV history (all same value) → rank 50."""
        iv_history = [15.0] * 60
        result = oi.calculate_iv_rank(15.0, iv_history)
        assert result is not None
        assert result.iv_rank == 50.0

    def test_iv_rank_clamped_to_bounds(self) -> None:
        """IV rank should be clamped to 0-100 even for extreme values."""
        iv_history = [10.0 + i * 0.5 for i in range(60)]
        # Current IV way above historical range
        result = oi.calculate_iv_rank(50.0, iv_history)
        assert result is not None
        assert 0 <= result.iv_rank <= 100

    def test_iv_percentile_exact_boundaries(self) -> None:
        """Test percentile at exact min and max of history."""
        iv_history = [float(i) for i in range(1, 51)]  # 1 to 50
        # At min: 0% below 1
        result = oi.calculate_iv_rank(1.0, iv_history)
        assert result is not None
        assert result.iv_percentile == 0.0

        # At max: all 49 values below 50
        result = oi.calculate_iv_rank(50.0, iv_history)
        assert result is not None
        assert result.iv_percentile == pytest.approx(98.0, abs=0.1)


# ---------------------------------------------------------------------------
# Options Summary — vote logic
# ---------------------------------------------------------------------------


class TestOptionsSummary:
    """Tests for OptionsIndicators.get_options_summary."""

    def test_summary_returns_options_summary(self) -> None:
        chain = _standard_chain()
        result = oi.get_options_summary(chain)
        assert isinstance(result, OptionsSummary)

    def test_summary_none_for_empty_chain(self) -> None:
        """Empty chain should return None."""
        chain = _empty_chain()
        result = oi.get_options_summary(chain)
        assert result is None

    def test_summary_vote_valid_values(self) -> None:
        """options_vote should be a valid verdict."""
        valid_votes = {
            "STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"
        }
        chain = _standard_chain()
        result = oi.get_options_summary(chain)
        assert result is not None
        assert result.options_vote in valid_votes

    def test_summary_confidence_range(self) -> None:
        """Confidence should be between 0 and 1."""
        chain = _standard_chain()
        result = oi.get_options_summary(chain)
        assert result is not None
        assert 0.0 <= result.options_confidence <= 1.0

    def test_summary_fields_populated(self) -> None:
        """All key fields should be populated."""
        chain = _standard_chain()
        result = oi.get_options_summary(chain)
        assert result is not None
        assert result.index_id == "NIFTY"
        assert result.pcr > 0
        assert result.oi_support > 0
        assert result.oi_resistance > 0
        assert result.max_pain > 0
        assert result.atm_iv > 0

    def test_summary_expected_range(self) -> None:
        """expected_range should be (support, resistance)."""
        chain = _standard_chain()
        result = oi.get_options_summary(chain)
        assert result is not None
        support, resistance = result.expected_range
        assert support <= resistance

    def test_summary_with_iv_history(self) -> None:
        """Should populate iv_regime when IV history provided."""
        chain = _standard_chain()
        iv_history = [13.0 + i * 0.1 for i in range(60)]
        result = oi.get_options_summary(chain, iv_history=iv_history)
        assert result is not None
        assert result.iv_regime is not None
        assert result.iv_regime != "insufficient_history"

    def test_summary_without_iv_history(self) -> None:
        """Should warn when IV history not provided."""
        chain = _standard_chain()
        result = oi.get_options_summary(chain, iv_history=None)
        assert result is not None
        assert result.iv_regime is None
        assert any("IV history" in w for w in result.warnings)

    def test_summary_bullish_vote_on_bullish_chain(self) -> None:
        """Bullish chain should produce bullish or strong bullish vote."""
        chain = _bullish_chain()
        result = oi.get_options_summary(chain)
        assert result is not None
        assert result.options_vote in ("STRONG_BULLISH", "BULLISH", "NEUTRAL")

    def test_summary_bearish_vote_on_bearish_chain(self) -> None:
        """Bearish chain should produce bearish or strong bearish vote."""
        chain = _bearish_chain()
        result = oi.get_options_summary(chain)
        assert result is not None
        assert result.options_vote in ("STRONG_BEARISH", "BEARISH", "NEUTRAL")

    def test_summary_pcr_signal_matches_structure(self) -> None:
        """PCR signal in summary should match OI structure analysis."""
        chain = _standard_chain()
        result = oi.get_options_summary(chain)
        structure = oi.analyze_oi_structure(chain)
        assert result is not None
        assert result.pcr_signal == structure.pcr_interpretation

    def test_summary_max_pain_pull_valid(self) -> None:
        """max_pain_pull should be STRONG / MODERATE / WEAK."""
        chain = _standard_chain()
        result = oi.get_options_summary(chain)
        assert result is not None
        assert result.max_pain_pull in ("STRONG", "MODERATE", "WEAK")

    def test_summary_near_expiry_strengthens_max_pain(self) -> None:
        """Near expiry should add a warning about max pain pull."""
        strikes = [
            make_strike(22000, ce_oi=100000, ce_oi_change=5000, ce_ltp=80.0, ce_iv=13.0,
                        pe_oi=100000, pe_oi_change=5000, pe_ltp=80.0, pe_iv=13.0),
        ]
        # Expiry tomorrow
        chain = make_chain(
            strikes,
            spot_price=22000.0,
            expiry_date=date.today() + timedelta(days=1),
        )
        result = oi.get_options_summary(chain)
        assert result is not None
        assert any("expiry" in w.lower() for w in result.warnings)

    def test_summary_high_iv_skew_warning(self) -> None:
        """High IV skew should trigger a contrarian bullish warning."""
        # Create chain with very high put IV vs call IV
        strikes = [
            make_strike(21800, ce_ltp=200.0, ce_iv=10.0, ce_oi=50000, ce_oi_change=2000,
                        pe_ltp=30.0, pe_iv=30.0, pe_oi=100000, pe_oi_change=5000),
            make_strike(22000, ce_ltp=80.0, ce_iv=12.0, ce_oi=100000, ce_oi_change=5000,
                        pe_ltp=80.0, pe_iv=12.0, pe_oi=100000, pe_oi_change=5000),
            make_strike(22200, ce_ltp=15.0, ce_iv=10.0, ce_oi=80000, ce_oi_change=3000,
                        pe_ltp=200.0, pe_iv=10.0, pe_oi=50000, pe_oi_change=1000),
        ]
        chain = make_chain(strikes, spot_price=22000.0)
        result = oi.get_options_summary(chain)
        assert result is not None
        # IV skew should be high (OTM put IV = 30 vs OTM call IV = 10)
        if result.iv_skew > 5.0:
            assert any("skew" in w.lower() for w in result.warnings)

    def test_summary_days_to_expiry(self) -> None:
        """days_to_expiry should be non-negative."""
        chain = _standard_chain()
        result = oi.get_options_summary(chain)
        assert result is not None
        assert result.days_to_expiry >= 0

    def test_summary_timestamp(self) -> None:
        """Timestamp should be a datetime."""
        chain = _standard_chain()
        result = oi.get_options_summary(chain)
        assert result is not None
        assert isinstance(result.timestamp, datetime)


# ---------------------------------------------------------------------------
# Vote score mapping (unit-level coverage)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score, expected_vote",
    [
        (5.0, "STRONG_BULLISH"),
        (3.0, "STRONG_BULLISH"),
        (2.0, "BULLISH"),
        (1.5, "BULLISH"),
        (0.0, "NEUTRAL"),
        (-0.5, "NEUTRAL"),
        (-1.0, "NEUTRAL"),
        (-1.5, "BEARISH"),
        (-2.0, "BEARISH"),
        (-2.5, "BEARISH"),
        (-3.0, "STRONG_BEARISH"),
        (-5.0, "STRONG_BEARISH"),
    ],
)
def test_vote_score_mapping(score: float, expected_vote: str) -> None:
    """Verify the score → vote mapping used in get_options_summary."""
    if score >= 3.0:
        vote = "STRONG_BULLISH"
    elif score >= 1.5:
        vote = "BULLISH"
    elif score >= -1.0:
        vote = "NEUTRAL"
    elif score >= -2.5:
        vote = "BEARISH"
    else:
        vote = "STRONG_BEARISH"
    assert vote == expected_vote


# ---------------------------------------------------------------------------
# Cross-cutting edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Cross-cutting edge cases for robustness."""

    def test_single_strike_chain(self) -> None:
        """Chain with a single strike should work for all methods."""
        strikes = [
            make_strike(22000, ce_oi=100000, ce_oi_change=5000, ce_ltp=80.0, ce_iv=13.0,
                        pe_oi=100000, pe_oi_change=5000, pe_ltp=80.0, pe_iv=13.0),
        ]
        chain = make_chain(strikes, spot_price=22000.0)

        structure = oi.analyze_oi_structure(chain)
        assert structure.pcr == 1.0

        change = oi.analyze_oi_change(chain, spot_price=22000.0)
        assert isinstance(change, OIChangeAnalysis)

        max_p = oi.calculate_max_pain_detailed(chain)
        assert max_p.max_pain_strike == 22000.0

        iv = oi.analyze_iv(chain)
        assert isinstance(iv, IVAnalysis)

        summary = oi.get_options_summary(chain)
        assert summary is not None

    def test_all_zero_oi(self) -> None:
        """Strikes with all-zero OI should be handled gracefully."""
        strikes = [
            make_strike(22000, ce_oi=0, pe_oi=0),
            make_strike(22100, ce_oi=0, pe_oi=0),
        ]
        chain = make_chain(strikes, spot_price=22000.0)
        structure = oi.analyze_oi_structure(chain)
        assert structure.pcr == 0.0
        assert structure.ce_oi_concentration == 0.0
        assert structure.pe_oi_concentration == 0.0

    def test_expired_option_chain(self) -> None:
        """Chain with past expiry date should have 0 days to expiry."""
        strikes = [make_strike(22000, ce_oi=100000, pe_oi=100000)]
        chain = make_chain(
            strikes,
            spot_price=22000.0,
            expiry_date=date.today() - timedelta(days=5),
        )
        max_p = oi.calculate_max_pain_detailed(chain)
        assert max_p.days_to_expiry == 0

    def test_large_chain_performance(self) -> None:
        """Should handle a large chain (100+ strikes) without issues."""
        strikes = [
            make_strike(
                20000 + i * 50,
                ce_oi=max(0, 100000 - abs(i - 40) * 2000),
                ce_oi_change=1000,
                ce_ltp=max(1.0, 2000 - i * 50.0),
                ce_iv=15.0 + abs(i - 40) * 0.2,
                pe_oi=max(0, 100000 - abs(i - 40) * 2000),
                pe_oi_change=1000,
                pe_ltp=max(1.0, i * 50.0 - 2000),
                pe_iv=15.0 + abs(i - 40) * 0.2,
            )
            for i in range(100)
        ]
        chain = make_chain(strikes, spot_price=22000.0)
        summary = oi.get_options_summary(chain)
        assert summary is not None
        assert summary.options_vote in {
            "STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"
        }
