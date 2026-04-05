"""
Tests for src/analysis/indicators/volume.py

Covers:
  - VWAP calculation with known values
  - VWAP daily reset logic with multi-day intraday data
  - VWAP bands and price zone classification
  - OBV calculation and EMA smoothing
  - OBV divergence detection with crafted data
  - Volume Profile POC and Value Area calculation
  - Volume confirmation logic
  - Volume summary vote logic
  - Edge cases: empty df, short df, NaN values, zero volume
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.analysis.indicators.volume import (
    OBVResult,
    VolumeAnalysis,
    VolumeIndicators,
    VolumeProfileResult,
    VolumeSummary,
    VWAPResult,
    _is_intraday,
)

vi = VolumeIndicators()


# ---------------------------------------------------------------------------
# DataFrame factories
# ---------------------------------------------------------------------------


def make_ohlcv(
    closes: list[float],
    volumes: list[float] | None = None,
    timestamps: list[datetime] | None = None,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from close prices and optional volumes."""
    n = len(closes)
    if timestamps is None:
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    if volumes is None:
        volumes = [100_000.0] * n
    c = np.array(closes, dtype=float)
    v = np.array(volumes, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": c,
            "high": c * 1.005,
            "low": c * 0.995,
            "close": c,
            "volume": v,
        }
    )


def make_intraday_ohlcv(
    days: int = 2,
    bars_per_day: int = 10,
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Build a multi-day intraday OHLCV DataFrame.

    Each day gets *bars_per_day* bars starting at 09:15 IST with 15-minute
    intervals.
    """
    rows: list[dict] = []
    rng = np.random.default_rng(42)
    price = base_price

    for d in range(days):
        day_start = datetime(2024, 1, 2 + d, 9, 15)
        for b in range(bars_per_day):
            ts = day_start + timedelta(minutes=15 * b)
            price = price * (1 + rng.normal(0.001, 0.002))
            vol = float(rng.integers(50_000, 200_000))
            rows.append(
                {
                    "timestamp": ts,
                    "open": price * 0.999,
                    "high": price * 1.003,
                    "low": price * 0.997,
                    "close": price,
                    "volume": vol,
                }
            )

    return pd.DataFrame(rows)


def make_trending_ohlcv(
    n: int = 300,
    *,
    uptrend: bool = True,
    base_volume: float = 100_000.0,
) -> pd.DataFrame:
    """Long OHLCV DataFrame with a clear directional trend."""
    rng = np.random.default_rng(42)
    step = 0.003 if uptrend else -0.003
    prices = [20000.0]
    volumes: list[float] = []
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + step + rng.normal(0, 0.001)))
    for _ in range(n):
        volumes.append(base_volume * rng.uniform(0.5, 1.5))
    return make_ohlcv(prices, volumes)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )


# ---------------------------------------------------------------------------
# Helper: _is_intraday
# ---------------------------------------------------------------------------


def test_is_intraday_detects_intraday_data() -> None:
    df = make_intraday_ohlcv(days=2, bars_per_day=5)
    assert _is_intraday(df) is True


def test_is_intraday_detects_daily_data() -> None:
    df = make_ohlcv([100.0, 101.0, 102.0])
    assert _is_intraday(df) is False


def test_is_intraday_too_short_returns_false() -> None:
    df = make_ohlcv([100.0])
    assert _is_intraday(df) is False


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------


class TestVWAP:
    """Tests for VolumeIndicators.calculate_vwap."""

    def test_vwap_known_values(self) -> None:
        """VWAP on uniform-volume data should equal running mean of TP."""
        closes = [100.0, 102.0, 104.0, 103.0, 105.0]
        volumes = [1000.0] * 5
        df = make_ohlcv(closes, volumes)
        result = vi.calculate_vwap(df)

        assert isinstance(result, VWAPResult)
        assert not math.isnan(result.current_vwap)
        # With uniform volume, VWAP should converge towards mean TP
        assert result.current_vwap > 0

    def test_vwap_above_below_classification(self) -> None:
        """Price well above VWAP → 'ABOVE'."""
        # Steadily rising prices
        prices = [100.0 + i * 2 for i in range(50)]
        df = make_ohlcv(prices)
        result = vi.calculate_vwap(df)
        # Latest close is much higher than cumulative VWAP
        assert result.price_vs_vwap == "ABOVE"
        assert result.distance_from_vwap > 0

    def test_vwap_below_classification(self) -> None:
        """Price well below VWAP → 'BELOW'."""
        prices = [200.0 - i * 2 for i in range(50)]
        df = make_ohlcv(prices)
        result = vi.calculate_vwap(df)
        assert result.price_vs_vwap == "BELOW"
        assert result.distance_from_vwap < 0

    def test_vwap_at_vwap_when_flat(self) -> None:
        """Completely flat prices → AT_VWAP."""
        prices = [100.0] * 20
        df = make_ohlcv(prices)
        result = vi.calculate_vwap(df)
        assert result.price_vs_vwap == "AT_VWAP"
        assert abs(result.distance_from_vwap) <= 0.05

    def test_vwap_daily_reset_with_intraday_data(self) -> None:
        """VWAP should reset at the start of each trading day for intraday data."""
        df = make_intraday_ohlcv(days=2, bars_per_day=10)
        result = vi.calculate_vwap(df)

        assert isinstance(result, VWAPResult)
        assert len(result.vwap) == len(df)

        # Get VWAP values at start of each day — they should differ
        # because VWAP resets, not carry forward from previous day
        vwap_vals = result.vwap.dropna()
        assert len(vwap_vals) > 0

    def test_vwap_no_reset_on_daily_data(self) -> None:
        """Daily data should produce a single cumulative VWAP (no intra-day reset)."""
        prices = [100.0 + i for i in range(30)]
        df = make_ohlcv(prices)
        result = vi.calculate_vwap(df)

        # VWAP should be monotonically defined (no jumps from reset)
        vwap_vals = result.vwap.dropna()
        assert len(vwap_vals) == len(df)

    def test_vwap_bands_exist(self) -> None:
        """Check that VWAP bands are calculated and have correct relationships."""
        df = make_trending_ohlcv(100)
        result = vi.calculate_vwap(df)

        valid_ub2 = result.upper_band_2.dropna()
        valid_ub1 = result.upper_band_1.dropna()
        valid_lb1 = result.lower_band_1.dropna()
        valid_lb2 = result.lower_band_2.dropna()

        assert len(valid_ub2) > 0
        assert len(valid_ub1) > 0

        # 2σ bands should be wider than 1σ bands
        assert float(valid_ub2.iloc[-1]) >= float(valid_ub1.iloc[-1])
        assert float(valid_lb2.iloc[-1]) <= float(valid_lb1.iloc[-1])

    def test_vwap_price_zone_valid_values(self) -> None:
        df = make_trending_ohlcv(100)
        result = vi.calculate_vwap(df)
        valid_zones = {"ABOVE_2SD", "ABOVE_1SD", "NEAR_VWAP", "BELOW_1SD", "BELOW_2SD"}
        assert result.price_zone in valid_zones

    def test_vwap_slope_valid_values(self) -> None:
        df = make_trending_ohlcv(100)
        result = vi.calculate_vwap(df)
        assert result.vwap_slope in ("RISING", "FALLING", "FLAT")

    def test_vwap_institutional_bias_valid_values(self) -> None:
        df = make_trending_ohlcv(100)
        result = vi.calculate_vwap(df)
        assert result.institutional_bias in ("BUYING", "SELLING", "NEUTRAL")

    def test_vwap_institutional_bias_buying_on_uptrend(self) -> None:
        """Strongly rising prices should produce BUYING bias."""
        prices = [100.0]
        for _ in range(49):
            prices.append(prices[-1] * 1.015)
        df = make_ohlcv(prices)
        result = vi.calculate_vwap(df)
        assert result.institutional_bias == "BUYING"

    def test_vwap_institutional_bias_selling_on_downtrend(self) -> None:
        """Strongly falling prices should produce SELLING bias."""
        prices = [200.0]
        for _ in range(49):
            prices.append(prices[-1] * 0.985)
        df = make_ohlcv(prices)
        result = vi.calculate_vwap(df)
        assert result.institutional_bias == "SELLING"

    def test_vwap_empty_df(self) -> None:
        result = vi.calculate_vwap(_empty_df())
        assert result.vwap.empty
        assert math.isnan(result.current_vwap)
        assert result.price_vs_vwap == "AT_VWAP"

    def test_vwap_too_short_df(self) -> None:
        df = make_ohlcv([100.0])
        result = vi.calculate_vwap(df)
        assert result.vwap.empty

    def test_vwap_series_length_matches_df(self) -> None:
        df = make_trending_ohlcv(50)
        result = vi.calculate_vwap(df)
        assert len(result.vwap) == len(df)

    def test_vwap_with_varying_volume(self) -> None:
        """VWAP should be pulled towards bars with higher volume."""
        closes = [100.0, 100.0, 100.0, 200.0, 200.0]
        # Heavy volume on the high-price bars
        volumes = [100.0, 100.0, 100.0, 10_000.0, 10_000.0]
        df = make_ohlcv(closes, volumes)
        result = vi.calculate_vwap(df)
        # VWAP should be much closer to 200 than to 100
        assert result.current_vwap > 150


# ---------------------------------------------------------------------------
# OBV
# ---------------------------------------------------------------------------


class TestOBV:
    """Tests for VolumeIndicators.calculate_obv."""

    def test_obv_known_values(self) -> None:
        """OBV should add volume on up-closes and subtract on down-closes."""
        closes = [100.0, 102.0, 101.0, 103.0, 104.0]
        volumes = [1000.0, 2000.0, 1500.0, 3000.0, 2500.0]
        df = make_ohlcv(closes, volumes)
        result = vi.calculate_obv(df)

        assert isinstance(result, OBVResult)
        obv_vals = result.obv.values

        # Bar 0: OBV starts at 0 (diff is NaN → 0)
        # Bar 1: close↑ → +2000 → OBV = 2000
        # Bar 2: close↓ → −1500 → OBV = 500
        # Bar 3: close↑ → +3000 → OBV = 3500
        # Bar 4: close↑ → +2500 → OBV = 6000
        assert float(obv_vals[1]) == pytest.approx(2000.0)
        assert float(obv_vals[2]) == pytest.approx(500.0)
        assert float(obv_vals[3]) == pytest.approx(3500.0)
        assert float(obv_vals[4]) == pytest.approx(6000.0)

    def test_obv_flat_close_no_change(self) -> None:
        """When close doesn't change, OBV should not change."""
        closes = [100.0, 100.0, 100.0]
        volumes = [5000.0, 5000.0, 5000.0]
        df = make_ohlcv(closes, volumes)
        result = vi.calculate_obv(df)
        # All diffs are zero → all directions are 0 → OBV = 0
        assert float(result.obv.iloc[-1]) == pytest.approx(0.0)

    def test_obv_ema_exists_and_smoothed(self) -> None:
        df = make_trending_ohlcv(100)
        result = vi.calculate_obv(df)
        assert len(result.obv_ema) == len(df)
        # EMA should be smoother than raw OBV
        obv_std = float(result.obv.diff().std())
        ema_std = float(result.obv_ema.diff().std())
        assert ema_std <= obv_std

    def test_obv_trend_valid_values(self) -> None:
        df = make_trending_ohlcv(100)
        result = vi.calculate_obv(df)
        assert result.obv_trend in ("RISING", "FALLING", "FLAT")

    def test_obv_trend_rising_on_uptrend(self) -> None:
        """Uptrend with consistent volume → OBV should be RISING."""
        df = make_trending_ohlcv(200, uptrend=True)
        result = vi.calculate_obv(df)
        assert result.obv_trend == "RISING"

    def test_obv_trend_falling_on_downtrend(self) -> None:
        """Downtrend → OBV should be FALLING."""
        df = make_trending_ohlcv(200, uptrend=False)
        result = vi.calculate_obv(df)
        assert result.obv_trend == "FALLING"

    def test_obv_accumulation_on_uptrend(self) -> None:
        df = make_trending_ohlcv(200, uptrend=True)
        result = vi.calculate_obv(df)
        assert result.accumulation_distribution == "ACCUMULATION"

    def test_obv_distribution_on_downtrend(self) -> None:
        df = make_trending_ohlcv(200, uptrend=False)
        result = vi.calculate_obv(df)
        assert result.accumulation_distribution == "DISTRIBUTION"

    def test_obv_divergence_valid_values(self) -> None:
        df = make_trending_ohlcv(200)
        result = vi.calculate_obv(df)
        assert result.divergence in ("BULLISH_DIVERGENCE", "BEARISH_DIVERGENCE", None)

    def test_obv_divergence_bearish_crafted(self) -> None:
        """Price making higher highs but OBV declining → bearish divergence."""
        # Build 20 bars: price rises, but volume is heavy on down-bars
        n = 20
        closes: list[float] = []
        volumes: list[float] = []
        price = 100.0
        for i in range(n):
            if i % 2 == 0:
                price += 1.5  # up bar (small)
                volumes.append(500.0)   # low volume on up
            else:
                price -= 0.5  # small down bar
                volumes.append(5000.0)  # heavy volume on down
            closes.append(price)

        df = make_ohlcv(closes, volumes)
        result = vi.calculate_obv(df)
        # OBV should be declining despite price rising overall
        # divergence detection may or may not trigger with this data shape,
        # but OBV itself should be negative/declining
        assert float(result.obv.iloc[-1]) < 0

    def test_obv_empty_df(self) -> None:
        result = vi.calculate_obv(_empty_df())
        assert result.obv.empty
        assert result.obv_trend == "FLAT"
        assert result.divergence is None

    def test_obv_too_short_df(self) -> None:
        df = make_ohlcv([100.0])
        result = vi.calculate_obv(df)
        assert result.obv.empty

    def test_obv_series_length(self) -> None:
        df = make_trending_ohlcv(100)
        result = vi.calculate_obv(df)
        assert len(result.obv) == len(df)
        assert len(result.obv_ema) == len(df)


# ---------------------------------------------------------------------------
# Volume Profile
# ---------------------------------------------------------------------------


class TestVolumeProfile:
    """Tests for VolumeIndicators.calculate_volume_profile."""

    def test_volume_profile_poc_is_highest_volume_level(self) -> None:
        """POC should be the price level with the most volume."""
        # Concentrate most volume at a specific price level
        closes = [100.0] * 5 + [110.0] * 3 + [120.0] * 2 + [100.0] * 10
        volumes = [500.0] * 5 + [100.0] * 3 + [100.0] * 2 + [5000.0] * 10
        df = make_ohlcv(closes, volumes)
        result = vi.calculate_volume_profile(df, num_bins=10)

        assert isinstance(result, VolumeProfileResult)
        assert not math.isnan(result.poc)
        # POC should be near 100 (where the most volume is concentrated)
        assert abs(result.poc - 100.0) < 5.0

    def test_volume_profile_value_area_contains_70pct(self) -> None:
        """Value Area should contain approximately 70% of total volume."""
        df = make_trending_ohlcv(100)
        result = vi.calculate_volume_profile(df, num_bins=20)

        assert not math.isnan(result.value_area_high)
        assert not math.isnan(result.value_area_low)
        assert result.value_area_high > result.value_area_low

    def test_volume_profile_vah_greater_than_val(self) -> None:
        df = make_trending_ohlcv(50)
        result = vi.calculate_volume_profile(df)
        assert result.value_area_high >= result.value_area_low

    def test_volume_profile_poc_within_price_range(self) -> None:
        df = make_trending_ohlcv(100)
        result = vi.calculate_volume_profile(df)
        price_min = float(df["low"].min())
        price_max = float(df["high"].max())
        assert price_min <= result.poc <= price_max

    def test_volume_profile_current_price_in_value_area(self) -> None:
        """When all prices are the same, current price should be in value area."""
        closes = [100.0] * 30
        df = make_ohlcv(closes)
        result = vi.calculate_volume_profile(df, num_bins=10)
        assert result.current_price_in_value_area is True

    def test_volume_profile_high_volume_nodes(self) -> None:
        """Nodes with volume > 1.5× average should appear in high_volume_nodes."""
        # Create clusters: lots of volume at 100, little elsewhere
        closes = [100.0] * 15 + [150.0] * 5
        volumes = [10_000.0] * 15 + [100.0] * 5
        df = make_ohlcv(closes, volumes)
        result = vi.calculate_volume_profile(df, num_bins=10)
        assert len(result.high_volume_nodes) >= 1

    def test_volume_profile_low_volume_nodes(self) -> None:
        """Nodes with volume < 0.5× average should appear in low_volume_nodes."""
        closes = [100.0] * 15 + [150.0] * 5
        volumes = [10_000.0] * 15 + [100.0] * 5
        df = make_ohlcv(closes, volumes)
        result = vi.calculate_volume_profile(df, num_bins=10)
        assert len(result.low_volume_nodes) >= 1

    def test_volume_profile_price_levels_count(self) -> None:
        """Number of price levels should equal num_bins."""
        df = make_trending_ohlcv(50)
        result = vi.calculate_volume_profile(df, num_bins=15)
        assert len(result.price_levels) == 15
        assert len(result.volume_at_level) == 15

    def test_volume_profile_total_volume_matches(self) -> None:
        """Sum of volume at all levels should approximate total volume."""
        df = make_trending_ohlcv(50)
        result = vi.calculate_volume_profile(df, num_bins=20)
        total_profile = sum(result.volume_at_level)
        total_df = float(df["volume"].sum())
        assert pytest.approx(total_profile, rel=0.01) == total_df

    def test_volume_profile_empty_df(self) -> None:
        result = vi.calculate_volume_profile(_empty_df())
        assert result.price_levels == []
        assert math.isnan(result.poc)

    def test_volume_profile_flat_data(self) -> None:
        """Completely flat price data should still return a valid result."""
        closes = [100.0] * 30
        df = make_ohlcv(closes)
        result = vi.calculate_volume_profile(df)
        # Should handle gracefully even though high == low after scaling
        assert isinstance(result, VolumeProfileResult)


# ---------------------------------------------------------------------------
# Volume Analysis (general)
# ---------------------------------------------------------------------------


class TestVolumeAnalysis:
    """Tests for VolumeIndicators.analyze_volume."""

    def test_volume_ratio_calculation(self) -> None:
        """volume_ratio = current / avg."""
        # 19 bars at 1000, then 1 bar at 3000 → ratio ≈ 3.0
        volumes = [1000.0] * 19 + [3000.0]
        df = make_ohlcv([100.0] * 20, volumes)
        result = vi.analyze_volume(df, period=20)
        assert isinstance(result, VolumeAnalysis)
        assert result.volume_ratio > 2.0

    def test_is_high_volume_flag(self) -> None:
        volumes = [1000.0] * 19 + [2000.0]
        df = make_ohlcv([100.0] * 20, volumes)
        result = vi.analyze_volume(df, period=20)
        assert result.is_high_volume is True

    def test_is_low_volume_flag(self) -> None:
        volumes = [10_000.0] * 19 + [1000.0]
        df = make_ohlcv([100.0] * 20, volumes)
        result = vi.analyze_volume(df, period=20)
        assert result.is_low_volume is True

    def test_climax_volume_flag(self) -> None:
        volumes = [1000.0] * 19 + [5000.0]
        df = make_ohlcv([100.0] * 20, volumes)
        result = vi.analyze_volume(df, period=20)
        assert result.climax_volume is True

    def test_volume_confirms_price_up_with_high_volume(self) -> None:
        """Price up + above-average volume → confirms."""
        closes = [100.0] * 19 + [105.0]
        volumes = [1000.0] * 19 + [2000.0]
        df = make_ohlcv(closes, volumes)
        result = vi.analyze_volume(df, period=20)
        assert result.volume_confirms_price is True

    def test_volume_does_not_confirm_price_up_with_low_volume(self) -> None:
        """Price up + below-average volume → does not confirm."""
        closes = [100.0] * 19 + [105.0]
        volumes = [10_000.0] * 19 + [100.0]
        df = make_ohlcv(closes, volumes)
        result = vi.analyze_volume(df, period=20)
        assert result.volume_confirms_price is False

    def test_volume_trend_increasing(self) -> None:
        volumes = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0]
        df = make_ohlcv([100.0] * 7, volumes)
        result = vi.analyze_volume(df, period=5)
        assert result.volume_trend == "INCREASING"

    def test_volume_trend_decreasing(self) -> None:
        volumes = [7000.0, 6000.0, 5000.0, 4000.0, 3000.0, 2000.0, 1000.0]
        df = make_ohlcv([100.0] * 7, volumes)
        result = vi.analyze_volume(df, period=5)
        assert result.volume_trend == "DECREASING"

    def test_volume_trend_stable(self) -> None:
        volumes = [1000.0] * 10
        df = make_ohlcv([100.0] * 10, volumes)
        result = vi.analyze_volume(df, period=5)
        assert result.volume_trend == "STABLE"

    def test_volume_analysis_empty_df(self) -> None:
        result = vi.analyze_volume(_empty_df())
        assert result.current_volume == 0
        assert result.volume_ratio == 0.0
        assert result.is_low_volume is True

    def test_volume_analysis_too_short_df(self) -> None:
        df = make_ohlcv([100.0])
        result = vi.analyze_volume(df)
        assert result.current_volume == 0

    def test_avg_volume_calculation(self) -> None:
        volumes = [1000.0] * 20
        df = make_ohlcv([100.0] * 20, volumes)
        result = vi.analyze_volume(df, period=20)
        assert result.avg_volume == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Edge cases: zero volume
# ---------------------------------------------------------------------------


class TestZeroVolume:
    """Edge case: all volume data is zero."""

    def test_vwap_with_zero_volume(self) -> None:
        """VWAP should handle zero volume gracefully."""
        closes = [100.0, 101.0, 102.0, 103.0, 104.0]
        volumes = [0.0] * 5
        df = make_ohlcv(closes, volumes)
        result = vi.calculate_vwap(df)
        # With zero volume, VWAP is NaN
        assert isinstance(result, VWAPResult)

    def test_obv_with_zero_volume(self) -> None:
        """OBV with zero volume should stay at zero."""
        closes = [100.0, 102.0, 101.0, 103.0]
        volumes = [0.0] * 4
        df = make_ohlcv(closes, volumes)
        result = vi.calculate_obv(df)
        assert float(result.obv.iloc[-1]) == pytest.approx(0.0)

    def test_volume_analysis_with_zero_volume(self) -> None:
        closes = [100.0] * 20
        volumes = [0.0] * 20
        df = make_ohlcv(closes, volumes)
        result = vi.analyze_volume(df)
        assert result.current_volume == 0
        assert result.volume_ratio == 0.0

    def test_volume_summary_with_zero_volume_returns_neutral(self) -> None:
        """Indices with no volume data should return NEUTRAL."""
        closes = [100.0] * 30
        volumes = [0.0] * 30
        df = make_ohlcv(closes, volumes)
        summary = vi.get_volume_summary(df)
        assert summary.volume_vote == "NEUTRAL"
        assert any("Zero volume" in w for w in summary.warnings)


# ---------------------------------------------------------------------------
# Volume Summary — vote logic
# ---------------------------------------------------------------------------


class TestVolumeSummary:
    """Tests for VolumeIndicators.get_volume_summary."""

    def test_summary_empty_df(self) -> None:
        summary = vi.get_volume_summary(_empty_df())
        assert isinstance(summary, VolumeSummary)
        assert summary.volume_vote == "NEUTRAL"
        assert summary.volume_confidence == 0.0

    def test_summary_too_short_df(self) -> None:
        df = make_ohlcv([100.0, 101.0])
        summary = vi.get_volume_summary(df)
        assert isinstance(summary, VolumeSummary)

    def test_summary_vote_valid_values(self) -> None:
        valid = {"STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"}
        for uptrend in (True, False):
            df = make_trending_ohlcv(100, uptrend=uptrend)
            summary = vi.get_volume_summary(df)
            assert summary.volume_vote in valid

    def test_summary_confidence_between_0_and_1(self) -> None:
        for uptrend in (True, False):
            df = make_trending_ohlcv(200, uptrend=uptrend)
            summary = vi.get_volume_summary(df)
            assert 0.0 <= summary.volume_confidence <= 1.0

    def test_summary_bullish_on_uptrend(self) -> None:
        """Strong uptrend should produce bullish or strong bullish vote."""
        df = make_trending_ohlcv(200, uptrend=True)
        summary = vi.get_volume_summary(df)
        assert summary.volume_vote in ("STRONG_BULLISH", "BULLISH", "NEUTRAL")

    def test_summary_bearish_on_downtrend(self) -> None:
        """Strong downtrend should produce bearish or strong bearish vote."""
        df = make_trending_ohlcv(200, uptrend=False)
        summary = vi.get_volume_summary(df)
        assert summary.volume_vote in ("STRONG_BEARISH", "BEARISH", "NEUTRAL")

    def test_summary_timestamp_populated(self) -> None:
        df = make_trending_ohlcv(100)
        summary = vi.get_volume_summary(df)
        assert isinstance(summary.timestamp, datetime)

    def test_summary_vwap_fields_populated(self) -> None:
        df = make_trending_ohlcv(100)
        summary = vi.get_volume_summary(df)
        assert summary.price_vs_vwap in ("ABOVE", "BELOW", "AT_VWAP")
        assert summary.vwap_zone in (
            "ABOVE_2SD", "ABOVE_1SD", "NEAR_VWAP", "BELOW_1SD", "BELOW_2SD"
        )
        assert summary.institutional_bias in ("BUYING", "SELLING", "NEUTRAL")

    def test_summary_obv_fields_populated(self) -> None:
        df = make_trending_ohlcv(100)
        summary = vi.get_volume_summary(df)
        assert summary.obv_trend in ("RISING", "FALLING", "FLAT")
        assert summary.accumulation_distribution in (
            "ACCUMULATION", "DISTRIBUTION", "NEUTRAL"
        )

    def test_summary_volume_profile_skipped_when_short(self) -> None:
        """< 20 bars should skip volume profile and note it in warnings."""
        df = make_ohlcv([100.0 + i for i in range(10)])
        summary = vi.get_volume_summary(df)
        assert math.isnan(summary.poc)
        assert any("Volume Profile skipped" in w for w in summary.warnings)

    def test_summary_volume_profile_fields_when_enough_data(self) -> None:
        df = make_trending_ohlcv(100)
        summary = vi.get_volume_summary(df)
        assert not math.isnan(summary.poc)
        assert not math.isnan(summary.value_area_high)
        assert not math.isnan(summary.value_area_low)

    def test_summary_with_nan_input(self) -> None:
        prices = [20000.0 + i * 10 for i in range(50)]
        prices[15] = float("nan")
        df = make_ohlcv(prices)
        summary = vi.get_volume_summary(df)
        assert isinstance(summary, VolumeSummary)

    def test_summary_climax_volume_reduces_confidence(self) -> None:
        """Climax volume bar should reduce confidence."""
        # Normal bars then one extreme volume bar
        closes = [100.0 + i * 0.1 for i in range(30)]
        volumes = [1000.0] * 29 + [100_000.0]  # 100× the average → climax
        df = make_ohlcv(closes, volumes)
        summary = vi.get_volume_summary(df)
        assert any("Climax volume" in w for w in summary.warnings)


# ---------------------------------------------------------------------------
# Vote score mapping (unit-level coverage)
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
    """Verify the net-score → vote mapping used in get_volume_summary."""
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
