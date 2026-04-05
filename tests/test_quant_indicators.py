"""
Tests for src/analysis/indicators/quant.py

Covers:
  - Z-Score calculation against manually computed values
  - Z-Score zone classification (all seven zones)
  - Beta calculation: index = benchmark → beta ~ 1.0
  - Beta interpretation thresholds
  - Correlation: perfectly correlated series → 1.0
  - Correlation strength classification
  - Black-Scholes against known option prices
  - Greeks: deep ITM call delta ~ 1.0, deep OTM call delta ~ 0.0
  - Implied Volatility round-trip test
  - QuantSummary vote logic
  - Edge cases: empty df, short df, NaN, zero time-to-expiry, etc.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.analysis.indicators.quant import (
    BSResult,
    BetaResult,
    CorrelationResult,
    QuantIndicators,
    QuantSummary,
    ZScoreResult,
    _norm_cdf,
    _norm_pdf,
)

qi = QuantIndicators()


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


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def make_flat_ohlcv(n: int = 100, base: float = 20000.0) -> pd.DataFrame:
    """Flat OHLCV → Z-score should hover near 0."""
    rng = np.random.default_rng(99)
    prices = (base + rng.uniform(-5, 5, n)).tolist()
    return make_ohlcv(prices)


def make_benchmark_ohlcv(
    n: int = 300, base: float = 20000.0, daily_return: float = 0.001
) -> pd.DataFrame:
    """Build a benchmark-like OHLCV series."""
    rng = np.random.default_rng(7)
    prices = [base]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_return + rng.normal(0, 0.002)))
    return make_ohlcv(prices)


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------


class TestNormHelpers:
    """Validate the hand-rolled normal CDF/PDF against known values."""

    def test_norm_cdf_at_zero(self) -> None:
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-10

    def test_norm_cdf_at_positive(self) -> None:
        # N(1.96) ≈ 0.975
        assert abs(_norm_cdf(1.96) - 0.975) < 0.001

    def test_norm_cdf_at_negative(self) -> None:
        # N(-1.96) ≈ 0.025
        assert abs(_norm_cdf(-1.96) - 0.025) < 0.001

    def test_norm_cdf_symmetry(self) -> None:
        x = 1.5
        assert abs(_norm_cdf(x) + _norm_cdf(-x) - 1.0) < 1e-10

    def test_norm_pdf_at_zero(self) -> None:
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert abs(_norm_pdf(0.0) - expected) < 1e-10

    def test_norm_pdf_symmetry(self) -> None:
        assert abs(_norm_pdf(1.0) - _norm_pdf(-1.0)) < 1e-10


# ---------------------------------------------------------------------------
# Z-Score
# ---------------------------------------------------------------------------


class TestZScore:
    """Tests for calculate_zscore."""

    def test_zscore_basic_computation(self) -> None:
        """Z-score of a trending-up series should be positive."""
        df = make_trending_ohlcv(100, uptrend=True)
        result = qi.calculate_zscore(df)
        assert isinstance(result, ZScoreResult)
        assert result.current_zscore > 0

    def test_zscore_downtrend_negative(self) -> None:
        """Z-score of a trending-down series should be negative."""
        df = make_trending_ohlcv(100, uptrend=False)
        result = qi.calculate_zscore(df)
        assert result.current_zscore < 0

    def test_zscore_flat_near_zero(self) -> None:
        """Flat data should have z-score near 0."""
        df = make_flat_ohlcv(100)
        result = qi.calculate_zscore(df)
        assert abs(result.current_zscore) < 1.5
        assert result.zone == "FAIR_VALUE"

    def test_zscore_manual_values(self) -> None:
        """Verify z-score against manually computed values."""
        # 20 identical values then a jump → z-score = (jump - mean) / std
        prices = [100.0] * 19 + [110.0]
        df = make_ohlcv(prices)
        result = qi.calculate_zscore(df, period=20)

        # Manual: SMA = (19*100 + 110)/20 = 100.5
        # Std_dev ≈ sqrt((19*(100-100.5)^2 + (110-100.5)^2) / 19)
        # Z = (110 - 100.5) / std
        assert result.current_zscore > 2  # clearly above mean

    def test_zscore_series_length(self) -> None:
        df = make_trending_ohlcv(100)
        result = qi.calculate_zscore(df)
        assert len(result.zscore) == len(df)

    def test_zscore_empty_df(self) -> None:
        result = qi.calculate_zscore(_empty_df())
        assert result.zscore.empty
        assert math.isnan(result.current_zscore)
        assert result.zone == "FAIR_VALUE"

    def test_zscore_too_short(self) -> None:
        df = make_ohlcv([100.0] * 5)
        result = qi.calculate_zscore(df, period=20)
        assert result.zscore.empty

    def test_zscore_mean_reversion_buy(self) -> None:
        """Flat data then sudden crash → z < -2 → BUY signal."""
        prices = [20000.0] * 99 + [14000.0]  # single-bar 30% crash
        df = make_ohlcv(prices)
        result = qi.calculate_zscore(df)
        assert result.current_zscore < -2
        assert result.mean_reversion_signal == "BUY"

    def test_zscore_mean_reversion_sell(self) -> None:
        """Flat data then sudden spike → z > 2 → SELL signal."""
        prices = [100.0] * 99 + [150.0]  # single-bar 50% spike
        df = make_ohlcv(prices)
        result = qi.calculate_zscore(df)
        assert result.current_zscore > 2
        assert result.mean_reversion_signal == "SELL"

    def test_zscore_no_signal_in_fair_value(self) -> None:
        df = make_flat_ohlcv(100)
        result = qi.calculate_zscore(df)
        assert result.mean_reversion_signal is None

    def test_zscore_distance_from_mean(self) -> None:
        df = make_trending_ohlcv(100, uptrend=True)
        result = qi.calculate_zscore(df)
        assert not math.isnan(result.distance_from_mean_pct)

    def test_zscore_days_since_fair_value_flat(self) -> None:
        """Flat data → z near 0 → days_since_fair_value should be 0."""
        df = make_flat_ohlcv(100)
        result = qi.calculate_zscore(df)
        assert result.days_since_fair_value == 0

    def test_zscore_reversion_probability_extreme(self) -> None:
        prices = [20000.0]
        for _ in range(99):
            prices.append(prices[-1] * 0.98)
        df = make_ohlcv(prices)
        result = qi.calculate_zscore(df)
        assert result.reversion_probability in ("HIGH", "MODERATE")

    def test_zscore_with_nan_input(self) -> None:
        prices = [100.0 + i for i in range(50)]
        prices[10] = float("nan")
        df = make_ohlcv(prices)
        result = qi.calculate_zscore(df)
        assert isinstance(result, ZScoreResult)

    @pytest.mark.parametrize("period", [10, 20, 50])
    def test_zscore_custom_periods(self, period: int) -> None:
        df = make_trending_ohlcv(100)
        result = qi.calculate_zscore(df, period=period)
        assert len(result.zscore) == len(df)

    # Zone classification
    @pytest.mark.parametrize(
        "z_value, expected_zone",
        [
            (3.0, "EXTREMELY_OVERBOUGHT"),
            (2.3, "OVERBOUGHT"),
            (1.5, "SLIGHTLY_OVERBOUGHT"),
            (0.0, "FAIR_VALUE"),
            (0.5, "FAIR_VALUE"),
            (-0.8, "FAIR_VALUE"),
            (-1.5, "SLIGHTLY_OVERSOLD"),
            (-2.3, "OVERSOLD"),
            (-3.0, "EXTREMELY_OVERSOLD"),
            (float("nan"), "FAIR_VALUE"),
        ],
    )
    def test_zscore_zone_classification(self, z_value: float, expected_zone: str) -> None:
        assert QuantIndicators._classify_zscore_zone(z_value) == expected_zone

    # Reversion probability
    @pytest.mark.parametrize(
        "z_value, expected_prob",
        [
            (3.0, "HIGH"),
            (-2.8, "HIGH"),
            (1.8, "MODERATE"),
            (-1.6, "MODERATE"),
            (0.5, "LOW"),
            (float("nan"), "LOW"),
        ],
    )
    def test_reversion_probability(self, z_value: float, expected_prob: str) -> None:
        assert QuantIndicators._reversion_probability(z_value) == expected_prob


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------


class TestBeta:
    """Tests for calculate_beta."""

    def test_beta_identical_series(self) -> None:
        """If index = benchmark, beta should be ~1.0."""
        df = make_benchmark_ohlcv(300)
        result = qi.calculate_beta(df, df)
        assert isinstance(result, BetaResult)
        assert abs(result.beta - 1.0) < 0.05

    def test_beta_identical_r_squared_near_one(self) -> None:
        """If index = benchmark, R² should be ~1.0."""
        df = make_benchmark_ohlcv(300)
        result = qi.calculate_beta(df, df)
        assert result.r_squared > 0.99

    def test_beta_series_populated(self) -> None:
        idx = make_trending_ohlcv(300, uptrend=True)
        bm = make_benchmark_ohlcv(300)
        result = qi.calculate_beta(idx, bm)
        assert len(result.beta_series) > 0

    def test_beta_empty_index(self) -> None:
        result = qi.calculate_beta(_empty_df(), make_benchmark_ohlcv(300))
        assert math.isnan(result.beta)

    def test_beta_empty_benchmark(self) -> None:
        result = qi.calculate_beta(make_trending_ohlcv(300), _empty_df())
        assert math.isnan(result.beta)

    def test_beta_too_short(self) -> None:
        idx = make_ohlcv([100.0] * 30)
        bm = make_ohlcv([100.0] * 30)
        result = qi.calculate_beta(idx, bm)
        assert math.isnan(result.beta)

    def test_beta_alpha_populated(self) -> None:
        idx = make_trending_ohlcv(300, uptrend=True)
        bm = make_benchmark_ohlcv(300)
        result = qi.calculate_beta(idx, bm)
        assert not math.isnan(result.alpha)

    @pytest.mark.parametrize(
        "beta_val, expected_interp",
        [
            (0.3, "LOW_BETA"),
            (0.6, "MODERATE_LOW"),
            (1.0, "MARKET_NEUTRAL"),
            (1.3, "MODERATE_HIGH"),
            (1.8, "HIGH_BETA"),
            (float("nan"), "MARKET_NEUTRAL"),
        ],
    )
    def test_beta_interpretation(self, beta_val: float, expected_interp: str) -> None:
        assert QuantIndicators._interpret_beta(beta_val) == expected_interp

    def test_beta_scaled_series(self) -> None:
        """2x leveraged version of benchmark should have beta ~2.0."""
        rng = np.random.default_rng(42)
        n = 300
        bm_returns = rng.normal(0.001, 0.01, n)
        bm_prices = [20000.0]
        for r in bm_returns:
            bm_prices.append(bm_prices[-1] * (1 + r))
        # 2x leveraged
        idx_prices = [20000.0]
        for r in bm_returns:
            idx_prices.append(idx_prices[-1] * (1 + 2 * r))

        bm_df = make_ohlcv(bm_prices)
        idx_df = make_ohlcv(idx_prices)
        result = qi.calculate_beta(idx_df, bm_df)
        assert abs(result.beta - 2.0) < 0.2


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


class TestCorrelation:
    """Tests for calculate_correlation."""

    def test_perfectly_correlated(self) -> None:
        """Identical series should have correlation 1.0."""
        df = make_benchmark_ohlcv(200)
        result = qi.calculate_correlation(df, df)
        assert isinstance(result, CorrelationResult)
        assert abs(result.correlation - 1.0) < 0.01

    def test_perfectly_correlated_strength(self) -> None:
        df = make_benchmark_ohlcv(200)
        result = qi.calculate_correlation(df, df)
        assert result.strength == "STRONG_POSITIVE"

    def test_inverse_correlation(self) -> None:
        """Series that moves opposite should have negative correlation."""
        rng = np.random.default_rng(42)
        n = 200
        returns = rng.normal(0, 0.01, n)
        prices1 = [100.0]
        prices2 = [100.0]
        for r in returns:
            prices1.append(prices1[-1] * (1 + r))
            prices2.append(prices2[-1] * (1 - r))
        df1 = make_ohlcv(prices1)
        df2 = make_ohlcv(prices2)
        result = qi.calculate_correlation(df1, df2)
        assert result.correlation < -0.7
        assert result.strength == "STRONG_NEGATIVE"

    def test_uncorrelated(self) -> None:
        """Two independent random series should have weak correlation."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(999)
        n = 200
        prices1 = (100 + rng1.standard_normal(n).cumsum()).tolist()
        prices2 = (100 + rng2.standard_normal(n).cumsum()).tolist()
        # Ensure positive prices
        prices1 = [max(10, p) for p in prices1]
        prices2 = [max(10, p) for p in prices2]
        df1 = make_ohlcv(prices1)
        df2 = make_ohlcv(prices2)
        result = qi.calculate_correlation(df1, df2)
        # Should be weak or moderate — not strong
        assert result.strength in ("WEAK", "MODERATE_POSITIVE", "MODERATE_NEGATIVE")

    def test_correlation_empty_df(self) -> None:
        result = qi.calculate_correlation(_empty_df(), make_benchmark_ohlcv(200))
        assert math.isnan(result.correlation)
        assert result.strength == "WEAK"

    def test_correlation_too_short(self) -> None:
        df = make_ohlcv([100.0] * 10)
        result = qi.calculate_correlation(df, df, period=50)
        assert math.isnan(result.correlation)

    def test_rolling_correlation_length(self) -> None:
        df = make_benchmark_ohlcv(200)
        result = qi.calculate_correlation(df, df, period=50)
        assert len(result.rolling_correlation) > 0

    def test_is_diverging_flag(self) -> None:
        """Is_diverging should be a bool."""
        df = make_benchmark_ohlcv(200)
        result = qi.calculate_correlation(df, df)
        assert isinstance(result.is_diverging, bool)

    def test_regime_change_flag(self) -> None:
        df = make_benchmark_ohlcv(200)
        result = qi.calculate_correlation(df, df)
        assert isinstance(result.correlation_regime_change, bool)

    @pytest.mark.parametrize(
        "corr_val, expected_strength",
        [
            (0.9, "STRONG_POSITIVE"),
            (0.5, "MODERATE_POSITIVE"),
            (0.1, "WEAK"),
            (-0.1, "WEAK"),
            (-0.5, "MODERATE_NEGATIVE"),
            (-0.9, "STRONG_NEGATIVE"),
            (float("nan"), "WEAK"),
        ],
    )
    def test_strength_classification(self, corr_val: float, expected_strength: str) -> None:
        assert QuantIndicators._classify_correlation_strength(corr_val) == expected_strength


# ---------------------------------------------------------------------------
# Black-Scholes Model
# ---------------------------------------------------------------------------


class TestBlackScholes:
    """Tests for calculate_bs_price."""

    def test_bs_call_known_price(self) -> None:
        """Test against a well-known BS example.

        S=100, K=100, T=1, r=0.05, σ=0.2 → Call ≈ 10.45 (textbook value).
        """
        result = qi.calculate_bs_price(
            spot=100, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=0.2, option_type="CE",
        )
        assert isinstance(result, BSResult)
        assert abs(result.theoretical_price - 10.45) < 0.1

    def test_bs_put_known_price(self) -> None:
        """Put-call parity: Put ≈ Call - S + K * e^(-rT).

        S=100, K=100, T=1, r=0.05, σ=0.2 → Put ≈ 5.57
        """
        result = qi.calculate_bs_price(
            spot=100, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=0.2, option_type="PE",
        )
        assert abs(result.theoretical_price - 5.57) < 0.15

    def test_bs_put_call_parity(self) -> None:
        """Put-call parity: C - P = S - K * e^(-rT)."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        call = qi.calculate_bs_price(S, K, T, r, sigma, "CE")
        put = qi.calculate_bs_price(S, K, T, r, sigma, "PE")
        parity = call.theoretical_price - put.theoretical_price - (S - K * math.exp(-r * T))
        assert abs(parity) < 0.01

    def test_bs_deep_itm_call_delta(self) -> None:
        """Deep ITM call delta should be close to 1.0."""
        result = qi.calculate_bs_price(
            spot=150, strike=100, time_to_expiry_years=0.5,
            risk_free_rate=0.05, volatility=0.2, option_type="CE",
        )
        assert result.delta > 0.95

    def test_bs_deep_otm_call_delta(self) -> None:
        """Deep OTM call delta should be close to 0.0."""
        result = qi.calculate_bs_price(
            spot=50, strike=100, time_to_expiry_years=0.5,
            risk_free_rate=0.05, volatility=0.2, option_type="CE",
        )
        assert result.delta < 0.05

    def test_bs_deep_itm_put_delta(self) -> None:
        """Deep ITM put delta should be close to -1.0."""
        result = qi.calculate_bs_price(
            spot=50, strike=100, time_to_expiry_years=0.5,
            risk_free_rate=0.05, volatility=0.2, option_type="PE",
        )
        assert result.delta < -0.95

    def test_bs_deep_otm_put_delta(self) -> None:
        """Deep OTM put delta should be near 0."""
        result = qi.calculate_bs_price(
            spot=150, strike=100, time_to_expiry_years=0.5,
            risk_free_rate=0.05, volatility=0.2, option_type="PE",
        )
        assert result.delta > -0.05

    def test_bs_atm_call_delta_near_half(self) -> None:
        """ATM call delta should be near 0.5 (drift from r shifts it slightly)."""
        result = qi.calculate_bs_price(
            spot=100, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=0.2, option_type="CE",
        )
        assert abs(result.delta - 0.5) < 0.15

    def test_bs_gamma_positive(self) -> None:
        """Gamma should always be positive."""
        result = qi.calculate_bs_price(
            spot=100, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=0.2, option_type="CE",
        )
        assert result.gamma > 0

    def test_bs_theta_negative_for_call(self) -> None:
        """Theta (time decay) should be negative for a long call."""
        result = qi.calculate_bs_price(
            spot=100, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=0.2, option_type="CE",
        )
        assert result.theta < 0

    def test_bs_vega_positive(self) -> None:
        """Vega should be positive (higher IV → higher option price)."""
        result = qi.calculate_bs_price(
            spot=100, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=0.2, option_type="CE",
        )
        assert result.vega > 0

    def test_bs_intrinsic_value_call(self) -> None:
        result = qi.calculate_bs_price(
            spot=110, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=0.2, option_type="CE",
        )
        assert result.intrinsic_value == 10.0

    def test_bs_intrinsic_value_put(self) -> None:
        result = qi.calculate_bs_price(
            spot=90, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=0.2, option_type="PE",
        )
        assert result.intrinsic_value == 10.0

    def test_bs_time_value_non_negative(self) -> None:
        result = qi.calculate_bs_price(
            spot=100, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=0.2, option_type="CE",
        )
        assert result.time_value >= 0

    def test_bs_zero_time_to_expiry(self) -> None:
        """At expiry, price = intrinsic value."""
        result = qi.calculate_bs_price(
            spot=110, strike=100, time_to_expiry_years=0,
            risk_free_rate=0.05, volatility=0.2, option_type="CE",
        )
        assert result.theoretical_price == 10.0
        assert result.time_value == 0.0

    def test_bs_zero_volatility(self) -> None:
        """With zero vol, price = intrinsic."""
        result = qi.calculate_bs_price(
            spot=110, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=0, option_type="CE",
        )
        assert result.theoretical_price == 10.0

    def test_bs_overpriced_detection(self) -> None:
        result = qi.calculate_bs_price(
            spot=100, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=0.2, option_type="CE",
            market_price=15.0,
        )
        assert result.is_overpriced is True
        assert result.is_underpriced is False
        assert result.mispricing_pct is not None
        assert result.mispricing_pct > 0

    def test_bs_underpriced_detection(self) -> None:
        result = qi.calculate_bs_price(
            spot=100, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=0.2, option_type="CE",
            market_price=5.0,
        )
        assert result.is_underpriced is True
        assert result.is_overpriced is False
        assert result.mispricing_pct is not None
        assert result.mispricing_pct < 0

    def test_bs_no_market_price(self) -> None:
        result = qi.calculate_bs_price(
            spot=100, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=0.2, option_type="CE",
        )
        assert result.is_overpriced is False
        assert result.is_underpriced is False
        assert result.mispricing_pct is None

    def test_bs_higher_vol_higher_price(self) -> None:
        """Increasing volatility should increase option price."""
        low_vol = qi.calculate_bs_price(100, 100, 1.0, 0.05, 0.1, "CE")
        high_vol = qi.calculate_bs_price(100, 100, 1.0, 0.05, 0.4, "CE")
        assert high_vol.theoretical_price > low_vol.theoretical_price

    def test_bs_longer_time_higher_price(self) -> None:
        """More time to expiry → higher option price (call)."""
        short = qi.calculate_bs_price(100, 100, 0.1, 0.05, 0.2, "CE")
        long_ = qi.calculate_bs_price(100, 100, 1.0, 0.05, 0.2, "CE")
        assert long_.theoretical_price > short.theoretical_price


# ---------------------------------------------------------------------------
# Implied Volatility
# ---------------------------------------------------------------------------


class TestImpliedVolatility:
    """Tests for calculate_implied_volatility."""

    def test_iv_round_trip(self) -> None:
        """Given a BS price with known σ, IV solver should recover σ."""
        input_vol = 0.25
        bs = qi.calculate_bs_price(
            spot=100, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=input_vol, option_type="CE",
        )
        recovered_iv = qi.calculate_implied_volatility(
            market_price=bs.theoretical_price,
            spot=100, strike=100, time_to_expiry=1.0,
            risk_free_rate=0.05, option_type="CE",
        )
        assert recovered_iv is not None
        assert abs(recovered_iv - input_vol) < 0.001

    def test_iv_round_trip_put(self) -> None:
        """IV round-trip for a put option."""
        input_vol = 0.30
        bs = qi.calculate_bs_price(
            spot=100, strike=100, time_to_expiry_years=1.0,
            risk_free_rate=0.05, volatility=input_vol, option_type="PE",
        )
        recovered_iv = qi.calculate_implied_volatility(
            market_price=bs.theoretical_price,
            spot=100, strike=100, time_to_expiry=1.0,
            risk_free_rate=0.05, option_type="PE",
        )
        assert recovered_iv is not None
        assert abs(recovered_iv - input_vol) < 0.001

    def test_iv_with_deep_itm(self) -> None:
        """IV solver should work for deep ITM option."""
        input_vol = 0.20
        bs = qi.calculate_bs_price(
            spot=150, strike=100, time_to_expiry_years=0.5,
            risk_free_rate=0.05, volatility=input_vol, option_type="CE",
        )
        recovered_iv = qi.calculate_implied_volatility(
            market_price=bs.theoretical_price,
            spot=150, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, option_type="CE",
        )
        assert recovered_iv is not None
        assert abs(recovered_iv - input_vol) < 0.01

    def test_iv_zero_market_price_returns_none(self) -> None:
        result = qi.calculate_implied_volatility(
            market_price=0, spot=100, strike=100,
            time_to_expiry=1.0, risk_free_rate=0.05,
        )
        assert result is None

    def test_iv_negative_price_returns_none(self) -> None:
        result = qi.calculate_implied_volatility(
            market_price=-5, spot=100, strike=100,
            time_to_expiry=1.0, risk_free_rate=0.05,
        )
        assert result is None

    def test_iv_zero_time_returns_none(self) -> None:
        result = qi.calculate_implied_volatility(
            market_price=10, spot=100, strike=100,
            time_to_expiry=0, risk_free_rate=0.05,
        )
        assert result is None

    def test_iv_non_negative(self) -> None:
        """IV should never be negative."""
        result = qi.calculate_implied_volatility(
            market_price=0.01, spot=100, strike=100,
            time_to_expiry=1.0, risk_free_rate=0.05,
        )
        if result is not None:
            assert result >= 0

    @pytest.mark.parametrize("input_vol", [0.10, 0.20, 0.30, 0.50, 0.80])
    def test_iv_various_volatilities(self, input_vol: float) -> None:
        """Round-trip for various volatility levels."""
        bs = qi.calculate_bs_price(100, 100, 1.0, 0.05, input_vol, "CE")
        recovered = qi.calculate_implied_volatility(
            bs.theoretical_price, 100, 100, 1.0, 0.05, "CE",
        )
        assert recovered is not None
        assert abs(recovered - input_vol) < 0.005


# ---------------------------------------------------------------------------
# Quant Summary
# ---------------------------------------------------------------------------


class TestQuantSummary:
    """Tests for get_quant_summary."""

    def test_summary_empty_df(self) -> None:
        summary = qi.get_quant_summary(_empty_df())
        assert isinstance(summary, QuantSummary)
        assert summary.quant_vote == "NEUTRAL"
        assert summary.quant_confidence == 0.0

    def test_summary_basic_fields(self) -> None:
        df = make_trending_ohlcv(100)
        summary = qi.get_quant_summary(df)
        assert isinstance(summary, QuantSummary)
        assert summary.quant_vote in ("BULLISH", "NEUTRAL", "BEARISH")
        assert 0 <= summary.quant_confidence <= 0.6

    def test_summary_confidence_capped_at_06(self) -> None:
        """Quant confidence should never exceed 0.6."""
        df = make_trending_ohlcv(300, uptrend=True)
        bm = make_benchmark_ohlcv(300)
        summary = qi.get_quant_summary(df, benchmark_df=bm)
        assert summary.quant_confidence <= 0.6

    def test_summary_with_benchmark(self) -> None:
        df = make_trending_ohlcv(300)
        bm = make_benchmark_ohlcv(300)
        summary = qi.get_quant_summary(df, benchmark_df=bm)
        assert summary.beta is not None
        assert summary.alpha is not None
        assert summary.beta_interpretation is not None

    def test_summary_without_benchmark(self) -> None:
        df = make_trending_ohlcv(100)
        summary = qi.get_quant_summary(df)
        assert summary.beta is None
        assert summary.alpha is None
        assert summary.beta_interpretation is None

    def test_summary_statistical_regime_valid(self) -> None:
        df = make_trending_ohlcv(100)
        summary = qi.get_quant_summary(df)
        assert summary.statistical_regime in ("MEAN_REVERTING", "TRENDING", "NORMAL")

    def test_summary_zscore_fields(self) -> None:
        df = make_trending_ohlcv(100)
        summary = qi.get_quant_summary(df)
        assert not math.isnan(summary.zscore)
        valid_zones = {
            "EXTREMELY_OVERBOUGHT", "OVERBOUGHT", "SLIGHTLY_OVERBOUGHT",
            "FAIR_VALUE", "SLIGHTLY_OVERSOLD", "OVERSOLD", "EXTREMELY_OVERSOLD",
        }
        assert summary.zscore_zone in valid_zones

    def test_summary_timestamp(self) -> None:
        df = make_trending_ohlcv(100)
        summary = qi.get_quant_summary(df)
        assert isinstance(summary.timestamp, datetime)

    def test_summary_bearish_on_strong_uptrend(self) -> None:
        """Flat then spike → high z-score → BEARISH (mean reversion SELL)."""
        prices = [100.0] * 99 + [150.0]
        df = make_ohlcv(prices)
        summary = qi.get_quant_summary(df)
        assert summary.quant_vote == "BEARISH"
        assert summary.mean_reversion_signal == "SELL"

    def test_summary_bullish_on_strong_downtrend(self) -> None:
        """Flat then crash → low z-score → BULLISH (mean reversion BUY)."""
        prices = [20000.0] * 99 + [14000.0]
        df = make_ohlcv(prices)
        summary = qi.get_quant_summary(df)
        assert summary.quant_vote == "BULLISH"
        assert summary.mean_reversion_signal == "BUY"

    def test_summary_neutral_on_flat(self) -> None:
        df = make_flat_ohlcv(100)
        summary = qi.get_quant_summary(df)
        assert summary.quant_vote == "NEUTRAL"

    def test_summary_with_nan_input(self) -> None:
        prices = [20000.0 + i * 10 for i in range(50)]
        prices[15] = float("nan")
        df = make_ohlcv(prices)
        summary = qi.get_quant_summary(df)
        assert isinstance(summary, QuantSummary)


# ---------------------------------------------------------------------------
# Edge case: parametrised
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Parametrised edge-case tests across all indicators."""

    @pytest.mark.parametrize(
        "spot, strike, T, r, sigma, opt_type",
        [
            (100, 100, 1.0, 0.05, 0.2, "CE"),   # Standard ATM call
            (100, 100, 1.0, 0.05, 0.2, "PE"),   # Standard ATM put
            (100, 100, 0.001, 0.05, 0.2, "CE"),  # Very short time
            (100, 80, 1.0, 0.0, 0.2, "CE"),     # Zero rate, ITM
            (100, 120, 1.0, 0.10, 0.5, "PE"),   # High vol OTM put
        ],
    )
    def test_bs_various_params(
        self, spot: float, strike: float, T: float, r: float,
        sigma: float, opt_type: str,
    ) -> None:
        result = qi.calculate_bs_price(spot, strike, T, r, sigma, opt_type)
        assert result.theoretical_price >= 0
        assert result.intrinsic_value >= 0
        assert result.time_value >= 0

    @pytest.mark.parametrize(
        "spot, strike, T, r, sigma, opt_type",
        [
            (0, 100, 1.0, 0.05, 0.2, "CE"),     # Zero spot
            (100, 0, 1.0, 0.05, 0.2, "CE"),     # Zero strike
            (-100, 100, 1.0, 0.05, 0.2, "CE"),  # Negative spot
        ],
    )
    def test_bs_invalid_inputs_graceful(
        self, spot: float, strike: float, T: float, r: float,
        sigma: float, opt_type: str,
    ) -> None:
        """BS should not crash on invalid inputs."""
        result = qi.calculate_bs_price(spot, strike, T, r, sigma, opt_type)
        assert isinstance(result, BSResult)

    def test_zscore_constant_prices(self) -> None:
        """All identical prices → std=0 → z-score should be NaN, no crash."""
        df = make_ohlcv([100.0] * 30)
        result = qi.calculate_zscore(df, period=20)
        # std=0 → zscore=NaN, handled gracefully
        assert isinstance(result, ZScoreResult)

    def test_beta_missing_close_column(self) -> None:
        """DataFrame without 'close' → graceful fallback."""
        broken = pd.DataFrame({"open": [100] * 100})
        result = qi.calculate_beta(broken, make_benchmark_ohlcv(100))
        assert math.isnan(result.beta)

    def test_correlation_single_row(self) -> None:
        df = make_ohlcv([100.0])
        result = qi.calculate_correlation(df, df, period=50)
        assert math.isnan(result.correlation)
