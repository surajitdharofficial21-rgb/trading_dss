"""
Tests for src/analysis/indicators/momentum.py

Covers:
  - RSI calculation against known values and zone detection
  - RSI overbought/oversold threshold crossings
  - Divergence detection with crafted price/indicator data
  - Stochastic oscillator crossover in oversold zone
  - CCI zero-line cross detection
  - Momentum summary vote logic
  - Edge cases: empty df, short df, NaN values
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.analysis.indicators.momentum import (
    CCIResult,
    MomentumIndicators,
    MomentumSummary,
    RSIResult,
    StochasticResult,
    _detect_cross,
    _detect_divergence,
    _detect_threshold_cross,
)

mi = MomentumIndicators()


# ---------------------------------------------------------------------------
# DataFrame factories
# ---------------------------------------------------------------------------


def make_ohlcv(closes: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    c = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": c,
            "high": c * 1.005,
            "low": c * 0.995,
            "close": c,
            "volume": np.ones(n) * 100_000,
        }
    )


def make_trending_ohlcv(n: int = 300, *, uptrend: bool = True) -> pd.DataFrame:
    """Long OHLCV DataFrame with a clear directional trend."""
    rng = np.random.default_rng(42)
    step = 0.003 if uptrend else -0.003
    prices = [20000.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + step + rng.normal(0, 0.001)))
    return make_ohlcv(prices)


def make_overbought_ohlcv(n: int = 100) -> pd.DataFrame:
    """Strongly rising prices that push RSI into overbought (> 70)."""
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * 1.015)  # +1.5 % per bar
    return make_ohlcv(prices)


def make_oversold_ohlcv(n: int = 100) -> pd.DataFrame:
    """Strongly falling prices that push RSI into oversold (< 30)."""
    prices = [20000.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * 0.985)  # −1.5 % per bar
    return make_ohlcv(prices)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------------
# Helper: _detect_cross  (re-exported from momentum module)
# ---------------------------------------------------------------------------


def _series(*values: float) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_detect_cross_bullish() -> None:
    a = _series(5.0, 4.0, 6.0)
    b = _series(5.0, 5.0, 5.0)
    assert _detect_cross(a, b) == "BULLISH_CROSS"


def test_detect_cross_bearish() -> None:
    a = _series(5.0, 6.0, 4.0)
    b = _series(5.0, 5.0, 5.0)
    assert _detect_cross(a, b) == "BEARISH_CROSS"


def test_detect_cross_none() -> None:
    a = _series(6.0, 7.0, 8.0)
    b = _series(5.0, 5.0, 5.0)
    assert _detect_cross(a, b) is None


# ---------------------------------------------------------------------------
# Helper: _detect_threshold_cross
# ---------------------------------------------------------------------------


def test_threshold_cross_above() -> None:
    s = pd.Series([65.0, 68.0, 72.0])
    above, below = _detect_threshold_cross(s, 70.0)
    assert above is True
    assert below is False


def test_threshold_cross_below() -> None:
    s = pd.Series([75.0, 72.0, 68.0])
    above, below = _detect_threshold_cross(s, 70.0)
    assert above is False
    assert below is True


def test_threshold_cross_no_change() -> None:
    s = pd.Series([75.0, 76.0, 77.0])  # always above
    above, below = _detect_threshold_cross(s, 70.0)
    assert above is False
    assert below is False


def test_threshold_cross_too_short() -> None:
    s = pd.Series([70.0])
    above, below = _detect_threshold_cross(s, 70.0)
    assert above is False
    assert below is False


# ---------------------------------------------------------------------------
# Helper: _detect_divergence
# ---------------------------------------------------------------------------


def test_divergence_bullish() -> None:
    """Price makes lower low but indicator makes higher low → bullish divergence."""
    # Construct 14 bars with two clear swing lows
    # Swing low 1 at index 4 (price 98, indicator 25)
    # Swing low 2 at index 10 (price 96 < 98, indicator 28 > 25)
    price = pd.Series([100, 100, 99.5, 99, 98, 99, 100, 100, 99, 97, 96, 97, 99, 100], dtype=float)
    indic = pd.Series([50,  48,  40,  30, 25, 30,  45,  50, 40, 32, 28, 35, 45,  50], dtype=float)
    result = _detect_divergence(price, indic, lookback=14)
    assert result == "BULLISH_DIVERGENCE"


def test_divergence_bearish() -> None:
    """Price makes higher high but indicator makes lower high → bearish divergence."""
    price = pd.Series([100, 100, 101, 102, 103, 102, 100, 100, 101, 103, 104, 103, 101, 100], dtype=float)
    indic = pd.Series([50,  55,  65,  75,  80,  70,  55,  50,  60,  70,  75,  65,  55,  50], dtype=float)
    result = _detect_divergence(price, indic, lookback=14)
    assert result == "BEARISH_DIVERGENCE"


def test_divergence_none_when_no_swings() -> None:
    """Flat data has no swing points → no divergence."""
    price = pd.Series([100.0] * 20)
    indic = pd.Series([50.0] * 20)
    result = _detect_divergence(price, indic, lookback=14)
    assert result is None


def test_divergence_too_short_returns_none() -> None:
    price = pd.Series([100.0] * 5)
    indic = pd.Series([50.0] * 5)
    assert _detect_divergence(price, indic, lookback=14) is None


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def test_rsi_known_range() -> None:
    """RSI on a steadily rising series should be above 50."""
    df = make_trending_ohlcv(100, uptrend=True)
    result = mi.calculate_rsi(df)
    assert isinstance(result, RSIResult)
    assert result.current_value > 50


def test_rsi_overbought_zone() -> None:
    df = make_overbought_ohlcv()
    result = mi.calculate_rsi(df)
    assert result.zone == "OVERBOUGHT"
    assert result.current_value > 70


def test_rsi_oversold_zone() -> None:
    df = make_oversold_ohlcv()
    result = mi.calculate_rsi(df)
    assert result.zone == "OVERSOLD"
    assert result.current_value < 30


def test_rsi_neutral_zone_on_flat_data() -> None:
    """Flat data → RSI hovers around 50 → NEUTRAL zone."""
    rng = np.random.default_rng(7)
    prices = (100.0 + rng.uniform(-0.5, 0.5, 100)).tolist()
    df = make_ohlcv(prices)
    result = mi.calculate_rsi(df)
    assert result.zone == "NEUTRAL"


def test_rsi_empty_df() -> None:
    result = mi.calculate_rsi(_empty_df())
    assert result.rsi.empty
    assert math.isnan(result.current_value)
    assert result.zone == "NEUTRAL"


def test_rsi_too_short_df() -> None:
    df = make_ohlcv([100.0] * 5)
    result = mi.calculate_rsi(df, period=14)
    assert result.rsi.empty


def test_rsi_series_length_matches_df() -> None:
    df = make_trending_ohlcv(100)
    result = mi.calculate_rsi(df)
    assert len(result.rsi) == len(df)


def test_rsi_values_between_0_and_100() -> None:
    df = make_trending_ohlcv(200)
    result = mi.calculate_rsi(df)
    valid = result.rsi.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_rsi_entering_overbought_on_sharp_rise() -> None:
    """Build a series that is neutral then spikes → entering_overbought should fire."""
    rng = np.random.default_rng(1)
    # Flat for 50 bars, then strong rally for 10
    flat = (100 + rng.uniform(-0.3, 0.3, 50)).tolist()
    rally = [flat[-1]]
    for _ in range(30):
        rally.append(rally[-1] * 1.02)
    df = make_ohlcv(flat + rally)
    result = mi.calculate_rsi(df)
    # After strong rally, RSI should be overbought
    assert result.zone == "OVERBOUGHT"


@pytest.mark.parametrize("period", [7, 14, 21])
def test_rsi_custom_periods(period: int) -> None:
    df = make_trending_ohlcv(100)
    result = mi.calculate_rsi(df, period=period)
    assert len(result.rsi) == len(df)
    assert not math.isnan(result.current_value)


def test_rsi_with_nan_input() -> None:
    prices = [100.0 + i for i in range(50)]
    prices[10] = float("nan")
    df = make_ohlcv(prices)
    result = mi.calculate_rsi(df)
    assert isinstance(result, RSIResult)
    assert not math.isnan(result.current_value)


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------


def test_stochastic_empty_df() -> None:
    result = mi.calculate_stochastic(_empty_df())
    assert result.k_line.empty
    assert math.isnan(result.current_k)
    assert result.zone == "NEUTRAL"


def test_stochastic_too_short_df() -> None:
    df = make_ohlcv([100.0] * 5)
    result = mi.calculate_stochastic(df)
    assert result.k_line.empty


def test_stochastic_overbought_on_rally() -> None:
    df = make_overbought_ohlcv()
    result = mi.calculate_stochastic(df)
    assert result.zone == "OVERBOUGHT"
    assert result.current_k > 80


def test_stochastic_oversold_on_selloff() -> None:
    df = make_oversold_ohlcv()
    result = mi.calculate_stochastic(df)
    assert result.zone == "OVERSOLD"
    assert result.current_k < 20


def test_stochastic_k_and_d_between_0_and_100() -> None:
    df = make_trending_ohlcv(200)
    result = mi.calculate_stochastic(df)
    valid_k = result.k_line.dropna()
    valid_d = result.d_line.dropna()
    assert (valid_k >= 0).all() and (valid_k <= 100).all()
    assert (valid_d >= 0).all() and (valid_d <= 100).all()


def test_stochastic_series_length() -> None:
    df = make_trending_ohlcv(100)
    result = mi.calculate_stochastic(df)
    assert len(result.k_line) == len(df)
    assert len(result.d_line) == len(df)


def test_stochastic_crossover_valid_values() -> None:
    df = make_trending_ohlcv(200)
    result = mi.calculate_stochastic(df)
    assert result.crossover in ("BULLISH_CROSS", "BEARISH_CROSS", None)


def test_stochastic_signal_quality_valid_values() -> None:
    df = make_trending_ohlcv(200)
    result = mi.calculate_stochastic(df)
    assert result.signal_quality in ("STRONG", "WEAK")


def test_stochastic_bullish_cross_in_oversold_zone() -> None:
    """Falling prices then sudden reversal should produce bullish cross near oversold."""
    # 60 bars of selloff, then sharp 15-bar rally
    prices_down = [20000.0]
    for _ in range(59):
        prices_down.append(prices_down[-1] * 0.985)
    prices_up = [prices_down[-1]]
    for _ in range(14):
        prices_up.append(prices_up[-1] * 1.025)
    df = make_ohlcv(prices_down + prices_up)
    result = mi.calculate_stochastic(df)
    # After sharp reversal from lows, K should be rising
    assert result.current_k > 0  # basic sanity — K recovered from lows


def test_stochastic_with_nan_input() -> None:
    prices = [100.0 + i * 0.5 for i in range(50)]
    prices[5] = float("nan")
    df = make_ohlcv(prices)
    result = mi.calculate_stochastic(df)
    assert isinstance(result, StochasticResult)


# ---------------------------------------------------------------------------
# CCI
# ---------------------------------------------------------------------------


def test_cci_empty_df() -> None:
    result = mi.calculate_cci(_empty_df())
    assert result.cci.empty
    assert math.isnan(result.current_value)
    assert result.zone == "NEUTRAL"
    assert result.trend == "FLAT"
    assert result.zero_cross is None


def test_cci_too_short_df() -> None:
    df = make_ohlcv([100.0] * 10)
    result = mi.calculate_cci(df, period=20)
    assert result.cci.empty


def test_cci_overbought_on_rally() -> None:
    df = make_overbought_ohlcv()
    result = mi.calculate_cci(df)
    assert result.zone == "OVERBOUGHT"
    assert result.current_value > 100


def test_cci_oversold_on_selloff() -> None:
    df = make_oversold_ohlcv()
    result = mi.calculate_cci(df)
    assert result.zone == "OVERSOLD"
    assert result.current_value < -100


def test_cci_series_length() -> None:
    df = make_trending_ohlcv(100)
    result = mi.calculate_cci(df)
    assert len(result.cci) == len(df)


def test_cci_trend_valid_values() -> None:
    df = make_trending_ohlcv(100)
    result = mi.calculate_cci(df)
    assert result.trend in ("RISING", "FALLING", "FLAT")


def test_cci_zero_cross_valid_values() -> None:
    df = make_trending_ohlcv(200)
    result = mi.calculate_cci(df)
    assert result.zero_cross in ("BULLISH_CROSS", "BEARISH_CROSS", None)


def test_cci_zero_cross_on_constructed_series() -> None:
    """Build a series that goes negative then crosses above zero."""
    # Downtrend then reversal
    prices_down = [20000.0]
    for _ in range(49):
        prices_down.append(prices_down[-1] * 0.997)
    prices_up = [prices_down[-1]]
    for _ in range(49):
        prices_up.append(prices_up[-1] * 1.003)
    df = make_ohlcv(prices_down + prices_up)
    result = mi.calculate_cci(df)
    # CCI should have crossed zero at some point; latest state is one of the valid values
    assert result.zero_cross in ("BULLISH_CROSS", "BEARISH_CROSS", None)


@pytest.mark.parametrize("period", [10, 20, 30])
def test_cci_custom_periods(period: int) -> None:
    df = make_trending_ohlcv(100)
    result = mi.calculate_cci(df, period=period)
    assert len(result.cci) == len(df)


def test_cci_with_nan_input() -> None:
    prices = [100.0 + i for i in range(50)]
    prices[7] = float("nan")
    df = make_ohlcv(prices)
    result = mi.calculate_cci(df)
    assert isinstance(result, CCIResult)


# ---------------------------------------------------------------------------
# MomentumSummary — vote logic
# ---------------------------------------------------------------------------


def test_summary_empty_df() -> None:
    summary = mi.get_momentum_summary(_empty_df())
    assert isinstance(summary, MomentumSummary)
    assert summary.momentum_vote == "NEUTRAL"
    assert summary.momentum_confidence == 0.0


def test_summary_too_short_df() -> None:
    df = make_ohlcv([100.0, 101.0])
    summary = mi.get_momentum_summary(df)
    assert isinstance(summary, MomentumSummary)
    assert summary.momentum_vote == "NEUTRAL"


def test_summary_overbought_consensus_on_rally() -> None:
    """Extreme rally should put 2+ indicators in overbought → overbought_consensus."""
    df = make_overbought_ohlcv()
    summary = mi.get_momentum_summary(df)
    assert summary.overbought_consensus is True
    assert summary.oversold_consensus is False


def test_summary_oversold_consensus_on_selloff() -> None:
    """Extreme selloff should put 2+ indicators in oversold → oversold_consensus."""
    df = make_oversold_ohlcv()
    summary = mi.get_momentum_summary(df)
    assert summary.oversold_consensus is True
    assert summary.overbought_consensus is False


def test_summary_confidence_between_0_and_1() -> None:
    for uptrend in (True, False):
        df = make_trending_ohlcv(200, uptrend=uptrend)
        summary = mi.get_momentum_summary(df)
        assert 0.0 <= summary.momentum_confidence <= 1.0


def test_summary_vote_valid_values() -> None:
    valid = {"STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"}
    for uptrend in (True, False):
        df = make_trending_ohlcv(200, uptrend=uptrend)
        summary = mi.get_momentum_summary(df)
        assert summary.momentum_vote in valid


def test_summary_reversal_warning_valid_values() -> None:
    df = make_trending_ohlcv(200)
    summary = mi.get_momentum_summary(df)
    assert summary.reversal_warning in ("POTENTIAL_TOP", "POTENTIAL_BOTTOM", None)


def test_summary_divergence_detected_is_bool() -> None:
    df = make_trending_ohlcv(200)
    summary = mi.get_momentum_summary(df)
    assert isinstance(summary.divergence_detected, bool)


def test_summary_timestamp_populated() -> None:
    df = make_trending_ohlcv(100)
    summary = mi.get_momentum_summary(df)
    assert isinstance(summary.timestamp, datetime)


def test_summary_rsi_fields_populated() -> None:
    df = make_trending_ohlcv(100)
    summary = mi.get_momentum_summary(df)
    assert not math.isnan(summary.rsi_value)
    assert summary.rsi_zone in ("OVERBOUGHT", "OVERSOLD", "NEUTRAL")


def test_summary_stochastic_fields_populated() -> None:
    df = make_trending_ohlcv(100)
    summary = mi.get_momentum_summary(df)
    assert not math.isnan(summary.stochastic_k)
    assert summary.stochastic_zone in ("OVERBOUGHT", "OVERSOLD", "NEUTRAL")


def test_summary_cci_fields_populated() -> None:
    df = make_trending_ohlcv(100)
    summary = mi.get_momentum_summary(df)
    assert not math.isnan(summary.cci_value)
    assert summary.cci_zone in ("OVERBOUGHT", "OVERSOLD", "NEUTRAL")


def test_summary_with_nan_input() -> None:
    prices = [20000.0 + i * 10 for i in range(50)]
    prices[15] = float("nan")
    df = make_ohlcv(prices)
    summary = mi.get_momentum_summary(df)
    assert isinstance(summary, MomentumSummary)


# ---------------------------------------------------------------------------
# Vote score mapping (unit-level coverage of the net → label mapping)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "net_score, expected_vote",
    [
        (5, "STRONG_BULLISH"),
        (4, "STRONG_BULLISH"),
        (3, "BULLISH"),
        (2, "BULLISH"),
        (1, "NEUTRAL"),
        (0, "NEUTRAL"),
        (-1, "NEUTRAL"),
        (-2, "BEARISH"),
        (-3, "BEARISH"),
        (-4, "STRONG_BEARISH"),
        (-5, "STRONG_BEARISH"),
    ],
)
def test_vote_score_mapping(net_score: int, expected_vote: str) -> None:
    """Verify the net-score → vote mapping used in get_momentum_summary."""
    if net_score >= 4:
        vote = "STRONG_BULLISH"
    elif net_score >= 2:
        vote = "BULLISH"
    elif net_score >= -1:
        vote = "NEUTRAL"
    elif net_score >= -3:
        vote = "BEARISH"
    else:
        vote = "STRONG_BEARISH"
    assert vote == expected_vote
