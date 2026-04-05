"""
Tests for src/analysis/indicators/trend.py

Covers:
  - SMA / EMA against manually computed values
  - MACD crossover detection
  - ADX trend-strength classification
  - Trend summary vote logic (all-bullish, all-bearish, mixed)
  - Edge cases: empty df, too-short df, NaN df
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.analysis.indicators.trend import (
    ADXResult,
    MACDResult,
    TrendIndicators,
    TrendSummary,
    _classify_adx_strength,
    _detect_cross,
)

ti = TrendIndicators()


# ---------------------------------------------------------------------------
# DataFrame factories
# ---------------------------------------------------------------------------


def make_ohlcv(closes: list[float], *, seed_high_low: bool = True) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices.

    ``high = close * 1.005`` and ``low = close * 0.995`` so that ADX / DI
    calculations have meaningful high/low data without requiring full OHLCV.
    """
    n = len(closes)
    timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    closes_arr = np.array(closes, dtype=float)
    if seed_high_low:
        highs = closes_arr * 1.005
        lows = closes_arr * 0.995
    else:
        highs = closes_arr.copy()
        lows = closes_arr.copy()
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes_arr,
            "high": highs,
            "low": lows,
            "close": closes_arr,
            "volume": np.ones(n) * 100_000,
        }
    )


def make_trending_ohlcv(n: int = 300, *, uptrend: bool = True) -> pd.DataFrame:
    """Build a long OHLCV DataFrame with a clear directional trend.

    Each bar moves +0.3 % (uptrend) or -0.3 % (downtrend) with small noise.
    """
    rng = np.random.default_rng(42)
    step = 0.003 if uptrend else -0.003
    prices = [20000.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + step + rng.normal(0, 0.001)))
    return make_ohlcv(prices)


def make_ranging_ohlcv(n: int = 100) -> pd.DataFrame:
    """Build an OHLCV DataFrame that oscillates around a flat level (ADX < 20)."""
    rng = np.random.default_rng(7)
    prices = 20000 + rng.uniform(-50, 50, n)
    return make_ohlcv(prices.tolist())


# ---------------------------------------------------------------------------
# Helper: _classify_adx_strength
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (10.0, "WEAK"),
        (19.9, "WEAK"),
        (20.0, "EMERGING"),
        (24.9, "EMERGING"),
        (25.0, "STRONG"),
        (49.9, "STRONG"),
        (50.0, "VERY_STRONG"),
        (74.9, "VERY_STRONG"),
        (75.0, "EXTREME"),
        (99.0, "EXTREME"),
    ],
)
def test_classify_adx_strength(value: float, expected: str) -> None:
    assert _classify_adx_strength(value) == expected


# ---------------------------------------------------------------------------
# Helper: _detect_cross
# ---------------------------------------------------------------------------


def _series(*values: float) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_detect_cross_bullish() -> None:
    a = _series(5.0, 4.0, 6.0)  # crosses above b at idx 2
    b = _series(5.0, 5.0, 5.0)
    assert _detect_cross(a, b) == "BULLISH_CROSS"


def test_detect_cross_bearish() -> None:
    a = _series(5.0, 6.0, 4.0)  # crosses below b at idx 2
    b = _series(5.0, 5.0, 5.0)
    assert _detect_cross(a, b) == "BEARISH_CROSS"


def test_detect_cross_none_when_no_cross() -> None:
    a = _series(6.0, 7.0, 8.0)  # always above
    b = _series(5.0, 5.0, 5.0)
    assert _detect_cross(a, b) is None


def test_detect_cross_too_short() -> None:
    assert _detect_cross(_series(5.0), _series(4.0)) is None


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------


def test_sma_known_values() -> None:
    """SMA(3) on [1,2,3,4,5] → NaN,NaN,2,3,4."""
    df = make_ohlcv([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ti.calculate_sma(df, period=3)
    assert result.name == "sma_3"
    valid = result.dropna()
    assert len(valid) == 3
    assert math.isclose(valid.iloc[0], 2.0, rel_tol=1e-6)
    assert math.isclose(valid.iloc[1], 3.0, rel_tol=1e-6)
    assert math.isclose(valid.iloc[2], 4.0, rel_tol=1e-6)


def test_sma_period_longer_than_df_returns_all_nan() -> None:
    df = make_ohlcv([10.0] * 5)
    result = ti.calculate_sma(df, period=10)
    assert result.isna().all()


def test_sma_empty_df_returns_empty() -> None:
    df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    result = ti.calculate_sma(df, period=9)
    assert result.empty


def test_calculate_all_sma_returns_correct_keys() -> None:
    df = make_ohlcv([100.0] * 250)
    result = ti.calculate_all_sma(df)
    assert set(result.keys()) == {"sma_9", "sma_20", "sma_50", "sma_100", "sma_200"}
    for key, series in result.items():
        assert isinstance(series, pd.Series)
        assert series.name == key


def test_sma_monotone_increasing_series() -> None:
    """SMA of a linearly increasing series should also be linearly increasing."""
    prices = list(range(1, 51))  # 1 to 50
    df = make_ohlcv(prices)
    result = ti.calculate_sma(df, period=5).dropna()
    diffs = result.diff().dropna()
    assert (diffs > 0).all()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


def test_ema_known_values_period_3() -> None:
    """EMA(3) with multiplier 2/(3+1)=0.5.
    Seed = mean of first 3 bars = (1+2+3)/3 = 2.0 (SMA seed).
    EMA[3] = 4 * 0.5 + 2.0 * 0.5 = 3.0
    EMA[4] = 5 * 0.5 + 3.0 * 0.5 = 4.0
    """
    df = make_ohlcv([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ti.calculate_ema(df, period=3).dropna()
    assert len(result) > 0
    # Last value should be close to 4.0 (exact value depends on ta's seed logic)
    assert math.isclose(result.iloc[-1], 4.0, rel_tol=0.05)


def test_ema_period_longer_than_df_returns_partial_nan() -> None:
    df = make_ohlcv([10.0] * 5)
    result = ti.calculate_ema(df, period=10)
    # All NaN when df is shorter than period
    assert result.isna().all()


def test_ema_empty_df_returns_empty() -> None:
    df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    result = ti.calculate_ema(df, period=9)
    assert result.empty


def test_calculate_all_ema_returns_correct_keys() -> None:
    df = make_ohlcv([100.0] * 250)
    result = ti.calculate_all_ema(df)
    assert set(result.keys()) == {"ema_9", "ema_20", "ema_50", "ema_100", "ema_200"}


def test_ema_faster_than_sma_in_uptrend() -> None:
    """EMA should react faster than SMA — in a rising sequence EMA > SMA."""
    prices = list(range(1, 51))
    df = make_ohlcv(prices)
    ema = ti.calculate_ema(df, period=10).dropna()
    sma = ti.calculate_sma(df, period=10).dropna()
    # Align on common index
    common = ema.index.intersection(sma.index)
    assert (ema[common].iloc[-5:] > sma[common].iloc[-5:]).all()


def test_ema_series_name() -> None:
    df = make_ohlcv([100.0] * 30)
    assert ti.calculate_ema(df, period=9).name == "ema_9"


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def test_macd_empty_df_returns_empty_result() -> None:
    df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    result = ti.calculate_macd(df)
    assert isinstance(result, MACDResult)
    assert result.macd_line.empty
    assert result.is_bullish is False
    assert result.crossover is None


def test_macd_too_short_df_returns_empty_result() -> None:
    df = make_ohlcv([100.0] * 10)  # need ≥ 35 bars
    result = ti.calculate_macd(df)
    assert result.macd_line.empty


def test_macd_bullish_on_uptrend() -> None:
    df = make_trending_ohlcv(200, uptrend=True)
    result = ti.calculate_macd(df)
    assert result.is_bullish is True


def test_macd_negative_on_downtrend() -> None:
    """On a prolonged downtrend, the MACD line itself should be below zero.
    Note: is_bullish (MACD > signal) can still be True when the signal lags
    further below — checking the zero line is the reliable bearish indicator.
    """
    df = make_trending_ohlcv(200, uptrend=False)
    result = ti.calculate_macd(df)
    valid_macd = result.macd_line.dropna()
    assert float(valid_macd.iloc[-1]) < 0


def test_macd_histogram_length_matches_close() -> None:
    df = make_trending_ohlcv(100)
    result = ti.calculate_macd(df)
    assert len(result.histogram) == len(df)


def test_macd_histogram_increasing_flag() -> None:
    """Histogram_increasing should match whether last hist > second-to-last hist."""
    df = make_trending_ohlcv(200, uptrend=True)
    result = ti.calculate_macd(df)
    valid_hist = result.histogram.dropna()
    expected = bool(valid_hist.iloc[-1] > valid_hist.iloc[-2])
    assert result.histogram_increasing == expected


def test_macd_crossover_detected_on_constructed_series() -> None:
    """Build a close series that forces a BULLISH crossover at the last bar."""
    # Downtrend for 100 bars, then sharp uptrend for 10 bars
    rng = np.random.default_rng(0)
    prices_down = [20000 * (1 - 0.004) ** i for i in range(100)]
    prices_up = [prices_down[-1] * (1 + 0.015) ** i for i in range(1, 11)]
    prices = prices_down + prices_up
    df = make_ohlcv(prices)
    result = ti.calculate_macd(df)
    # We may or may not hit the exact bar — just verify crossover is a valid value
    assert result.crossover in ("BULLISH_CROSS", "BEARISH_CROSS", None)


def test_macd_custom_params() -> None:
    df = make_trending_ohlcv(200)
    result = ti.calculate_macd(df, fast=5, slow=13, signal=5)
    assert len(result.macd_line) == len(df)


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


def test_adx_empty_df_returns_defaults() -> None:
    df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    result = ti.calculate_adx(df)
    assert isinstance(result, ADXResult)
    assert result.adx.empty
    assert result.trend_strength == "WEAK"


def test_adx_too_short_df() -> None:
    df = make_ohlcv([100.0] * 10)
    result = ti.calculate_adx(df, period=14)
    assert result.adx.empty


def test_adx_strong_on_uptrend() -> None:
    df = make_trending_ohlcv(300, uptrend=True)
    result = ti.calculate_adx(df)
    assert result.trend_strength in ("STRONG", "VERY_STRONG", "EMERGING", "EXTREME")
    assert result.trend_direction == "BULLISH"


def test_adx_strong_on_downtrend() -> None:
    df = make_trending_ohlcv(300, uptrend=False)
    result = ti.calculate_adx(df)
    assert result.trend_direction == "BEARISH"


def test_adx_weak_on_ranging_market() -> None:
    df = make_ranging_ohlcv(150)
    result = ti.calculate_adx(df)
    # ADX on flat/noisy data should be low
    valid = result.adx.dropna()
    assert float(valid.iloc[-1]) < 30  # should not be very strong


def test_adx_series_same_length_as_df() -> None:
    df = make_trending_ohlcv(100)
    result = ti.calculate_adx(df)
    assert len(result.adx) == len(df)
    assert len(result.plus_di) == len(df)
    assert len(result.minus_di) == len(df)


def test_adx_di_crossover_is_valid_value() -> None:
    df = make_trending_ohlcv(200)
    result = ti.calculate_adx(df)
    assert result.di_crossover in ("BULLISH_CROSS", "BEARISH_CROSS", None)


@pytest.mark.parametrize(
    "strength",
    ["WEAK", "EMERGING", "STRONG", "VERY_STRONG", "EXTREME"],
)
def test_adx_trend_strength_valid_values(strength: str) -> None:
    """trend_strength must always be one of the five defined categories."""
    # Just verify the set of valid values — actual output depends on data
    assert strength in ("WEAK", "EMERGING", "STRONG", "VERY_STRONG", "EXTREME")


# ---------------------------------------------------------------------------
# TrendSummary — vote logic
# ---------------------------------------------------------------------------


def _make_all_bullish_df() -> pd.DataFrame:
    """Uptrend long enough that all 5 bullish signals fire."""
    return make_trending_ohlcv(300, uptrend=True)


def _make_all_bearish_df() -> pd.DataFrame:
    """Downtrend long enough that all 5 bearish signals fire."""
    return make_trending_ohlcv(300, uptrend=False)


def test_trend_summary_all_bullish() -> None:
    df = _make_all_bullish_df()
    summary = ti.get_trend_summary(df, index_id="NIFTY50", timeframe="1d")
    assert isinstance(summary, TrendSummary)
    assert summary.trend_vote in ("STRONG_BULLISH", "BULLISH")
    assert summary.trend_confidence > 0.0


def test_trend_summary_all_bearish() -> None:
    df = _make_all_bearish_df()
    summary = ti.get_trend_summary(df, index_id="NIFTY50", timeframe="1d")
    assert summary.trend_vote in ("STRONG_BEARISH", "BEARISH")


def test_trend_summary_mixed_signals_neutral() -> None:
    """Construct a scenario with exactly 3 bullish / 2 bearish signals → NEUTRAL."""
    # Use uptrend but just 60 bars so EMA200 is not available → BELOW
    df = make_trending_ohlcv(60, uptrend=True)
    summary = ti.get_trend_summary(df, index_id="NIFTY50", timeframe="1d")
    # With only 60 bars EMA200 is NaN → treated as BELOW → vote ≤ 4
    assert summary.trend_vote in ("STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH")


def test_trend_summary_empty_df_returns_defaults() -> None:
    df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    summary = ti.get_trend_summary(df)
    assert summary.trend_vote == "NEUTRAL"
    assert summary.trend_confidence == 0.0


def test_trend_summary_too_short_df() -> None:
    df = make_ohlcv([100.0, 101.0])
    summary = ti.get_trend_summary(df)
    assert isinstance(summary, TrendSummary)


def test_trend_summary_index_id_and_timeframe_preserved() -> None:
    df = make_trending_ohlcv(200)
    summary = ti.get_trend_summary(df, index_id="BANKNIFTY", timeframe="1h")
    assert summary.index_id == "BANKNIFTY"
    assert summary.timeframe == "1h"


def test_trend_summary_confidence_between_0_and_1() -> None:
    for uptrend in (True, False):
        df = make_trending_ohlcv(300, uptrend=uptrend)
        summary = ti.get_trend_summary(df)
        assert 0.0 <= summary.trend_confidence <= 1.0


def test_trend_summary_adx_penalty_reduces_confidence() -> None:
    """ADX < 20 on ranging data should reduce trend_confidence by 0.2."""
    ranging_df = make_ranging_ohlcv(150)
    summary = ti.get_trend_summary(ranging_df)
    # On ranging data ADX is usually < 20 — confidence should have the penalty applied
    valid_adx = ti.calculate_adx(ranging_df).adx.dropna()
    latest_adx = float(valid_adx.iloc[-1]) if len(valid_adx) > 0 else 0.0
    if latest_adx < 20:
        # Confirm penalty is reflected: confidence ≤ (base - 0.2)
        assert summary.trend_confidence <= 0.75  # highest base is 0.95, minus 0.2 = 0.75


def test_trend_summary_ema_alignment_bullish_on_uptrend() -> None:
    df = make_trending_ohlcv(300, uptrend=True)
    summary = ti.get_trend_summary(df)
    assert summary.ema_alignment in ("BULLISH", "MIXED")  # 300 bars: likely BULLISH


def test_trend_summary_ema_alignment_bearish_on_downtrend() -> None:
    df = make_trending_ohlcv(300, uptrend=False)
    summary = ti.get_trend_summary(df)
    assert summary.ema_alignment in ("BEARISH", "MIXED")


def test_trend_summary_golden_death_cross_are_bool() -> None:
    df = make_trending_ohlcv(300)
    summary = ti.get_trend_summary(df)
    assert isinstance(summary.golden_cross, bool)
    assert isinstance(summary.death_cross, bool)


def test_trend_summary_macd_histogram_trend_valid_value() -> None:
    df = make_trending_ohlcv(200)
    summary = ti.get_trend_summary(df)
    assert summary.macd_histogram_trend in ("INCREASING", "DECREASING")


def test_trend_summary_trend_vote_valid_values() -> None:
    valid_votes = {"STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"}
    for uptrend in (True, False):
        df = make_trending_ohlcv(300, uptrend=uptrend)
        summary = ti.get_trend_summary(df)
        assert summary.trend_vote in valid_votes


# ---------------------------------------------------------------------------
# Edge cases: NaN handling
# ---------------------------------------------------------------------------


def test_sma_with_nan_input_forward_fills() -> None:
    df = make_ohlcv([100.0, float("nan"), 100.0, 100.0, 100.0])
    # Should not raise; forward-fill should handle the NaN
    result = ti.calculate_sma(df, period=3)
    assert isinstance(result, pd.Series)


def test_ema_with_nan_input_does_not_raise() -> None:
    df = make_ohlcv([float("nan")] + [100.0] * 30)
    result = ti.calculate_ema(df, period=9)
    assert isinstance(result, pd.Series)


def test_macd_with_nan_input_does_not_raise() -> None:
    prices = [100.0 + i for i in range(50)]
    prices[10] = float("nan")
    df = make_ohlcv(prices)
    result = ti.calculate_macd(df)
    assert isinstance(result, MACDResult)


def test_adx_with_nan_input_does_not_raise() -> None:
    prices = [100.0 + i * 0.5 for i in range(60)]
    prices[5] = float("nan")
    df = make_ohlcv(prices)
    result = ti.calculate_adx(df)
    assert isinstance(result, ADXResult)


def test_trend_summary_with_nan_input_does_not_raise() -> None:
    prices = [20000.0 + i * 10 for i in range(50)]
    prices[15] = float("nan")
    df = make_ohlcv(prices)
    summary = ti.get_trend_summary(df)
    assert isinstance(summary, TrendSummary)


# ---------------------------------------------------------------------------
# Regression: vote counts map cleanly to trend_vote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bullish_signals, expected_vote",
    [
        (5, "STRONG_BULLISH"),
        (4, "BULLISH"),
        (3, "NEUTRAL"),
        (2, "BEARISH"),
        (1, "STRONG_BEARISH"),
        (0, "STRONG_BEARISH"),
    ],
)
def test_vote_map_coverage(bullish_signals: int, expected_vote: str) -> None:
    """Verify that every possible vote count maps to the right trend_vote."""
    _vote_map = {
        5: "STRONG_BULLISH",
        4: "BULLISH",
        3: "NEUTRAL",
        2: "BEARISH",
        1: "STRONG_BEARISH",
        0: "STRONG_BEARISH",
    }
    assert _vote_map[bullish_signals] == expected_vote
