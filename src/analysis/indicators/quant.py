"""
Quantitative & mathematical indicators for trading analysis.

Implements Z-Score, Beta, Correlation, Black-Scholes pricing (with Greeks),
Implied Volatility (Newton-Raphson), and a composite quant summary.
All methods are pure functions — no database calls or side effects.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normal distribution helpers (replaces scipy.stats.norm)
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """Cumulative distribution function for the standard normal distribution.

    Uses ``math.erf`` — equivalent to ``scipy.stats.norm.cdf(x)`` but
    avoids adding *scipy* as a dependency.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Probability density function for the standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ZScoreResult:
    """Result of Z-Score calculation.

    The Z-Score measures how many standard deviations the current price is
    from its rolling mean.  Extreme values suggest mean-reversion
    opportunities.
    """

    zscore: pd.Series                   # Z-score values
    current_zscore: float               # Latest z-score
    zone: str                           # EXTREMELY_OVERBOUGHT / OVERBOUGHT /
                                        # SLIGHTLY_OVERBOUGHT / FAIR_VALUE /
                                        # SLIGHTLY_OVERSOLD / OVERSOLD /
                                        # EXTREMELY_OVERSOLD
    mean_reversion_signal: Optional[str]  # BUY / SELL / None
    distance_from_mean_pct: float       # How far price is from the mean (%)
    days_since_fair_value: int          # Bars since z-score was in -1..1
    reversion_probability: str          # HIGH / MODERATE / LOW


@dataclass
class BetaResult:
    """Result of Beta calculation against a benchmark.

    Beta measures the systematic risk of an asset relative to the market.
    Alpha (Jensen's alpha) captures the excess return beyond what beta
    predicts.
    """

    beta: float                         # Current beta value
    beta_series: pd.Series              # Rolling beta over time
    interpretation: str                 # LOW_BETA / MODERATE_LOW /
                                        # MARKET_NEUTRAL / MODERATE_HIGH /
                                        # HIGH_BETA
    alpha: float                        # Jensen's alpha (annualised)
    r_squared: float                    # Goodness-of-fit (0-1)


@dataclass
class CorrelationResult:
    """Result of Pearson correlation between two return series."""

    correlation: float                  # Current correlation (-1 to +1)
    rolling_correlation: pd.Series      # Rolling correlation over time
    strength: str                       # STRONG_POSITIVE / MODERATE_POSITIVE /
                                        # WEAK / MODERATE_NEGATIVE /
                                        # STRONG_NEGATIVE
    is_diverging: bool                  # Correlation decreasing over last 20 bars
    correlation_regime_change: bool     # Correlation crossed 0 recently


@dataclass
class BSResult:
    """Result of Black-Scholes option pricing with Greeks."""

    theoretical_price: float
    intrinsic_value: float              # max(0, S-K) for CE, max(0, K-S) for PE
    time_value: float                   # theoretical - intrinsic

    # Greeks
    delta: float                        # dP/dS
    gamma: float                        # d²P/dS²
    theta: float                        # dP/dt  (per day, negative)
    vega: float                         # dP/dσ  (per 1 % IV change)
    rho: float                          # dP/dr

    # Pricing analysis
    is_overpriced: bool                 # market_price > theoretical
    is_underpriced: bool                # market_price < theoretical
    mispricing_pct: Optional[float]     # (market - theoretical) / theoretical * 100


@dataclass
class QuantSummary:
    """Composite quantitative assessment from Z-Score, Beta, and statistical regime."""

    timestamp: datetime

    # Z-Score
    zscore: float
    zscore_zone: str
    mean_reversion_signal: Optional[str]

    # Beta (if benchmark provided)
    beta: Optional[float]
    alpha: Optional[float]
    beta_interpretation: Optional[str]

    # Statistical regime
    statistical_regime: str             # MEAN_REVERTING / TRENDING / NORMAL

    # Quant vote
    quant_vote: str                     # BULLISH / NEUTRAL / BEARISH
    quant_confidence: float


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy of *df* with NaNs forward-filled.

    Logs a warning when NaN values are detected in the input.
    """
    if df.empty:
        return df.copy()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    has_nan = df[list(required)].isna().any().any()
    if has_nan:
        logger.warning("Input DataFrame contains NaN values — forward-filling before calculation.")
    clean = df.copy()
    clean[list(required)] = clean[list(required)].ffill()
    return clean


def _safe_log(x: float) -> float:
    """Return ``ln(x)`` guarding against non-positive values."""
    if x <= 0:
        return float("-inf")
    return math.log(x)


# ---------------------------------------------------------------------------
# QuantIndicators
# ---------------------------------------------------------------------------


class QuantIndicators:
    """Pure-function quantitative indicator calculations.

    All public methods accept DataFrames / scalars and return either a
    ``pd.Series`` or a result dataclass.  No state is stored on the instance.
    """

    # ------------------------------------------------------------------
    # Z-Score
    # ------------------------------------------------------------------

    def calculate_zscore(
        self, df: pd.DataFrame, period: int = 20
    ) -> ZScoreResult:
        """Calculate Z-Score of the close price.

        Z-Score = (close − SMA) / σ

        Zones:
          > 2.5  → EXTREMELY_OVERBOUGHT
          > 2    → OVERBOUGHT
          > 1    → SLIGHTLY_OVERBOUGHT
          -1..1  → FAIR_VALUE
          < -1   → SLIGHTLY_OVERSOLD
          < -2   → OVERSOLD
          < -2.5 → EXTREMELY_OVERSOLD

        Args:
            df: OHLCV DataFrame.
            period: Rolling window (default 20).

        Returns:
            ZScoreResult with full series and latest-bar summary fields.
        """
        clean = _prepare_df(df)

        if clean.empty or len(clean) < period:
            logger.warning(
                "Insufficient data for Z-Score calculation: need %d rows, got %d.",
                period,
                len(clean) if not clean.empty else 0,
            )
            empty = pd.Series(dtype=float)
            return ZScoreResult(
                zscore=empty,
                current_zscore=float("nan"),
                zone="FAIR_VALUE",
                mean_reversion_signal=None,
                distance_from_mean_pct=float("nan"),
                days_since_fair_value=0,
                reversion_probability="LOW",
            )

        close = clean["close"]
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std(ddof=1)

        # Guard against zero std
        zscore_series = (close - sma) / std.replace(0, np.nan)

        valid = zscore_series.dropna()
        current_z = float(valid.iloc[-1]) if len(valid) > 0 else float("nan")

        # Zone classification (order matters — check extremes first)
        zone = self._classify_zscore_zone(current_z)

        # Mean reversion signal
        if not math.isnan(current_z) and current_z < -2:
            mean_reversion_signal: Optional[str] = "BUY"
        elif not math.isnan(current_z) and current_z > 2:
            mean_reversion_signal = "SELL"
        else:
            mean_reversion_signal = None

        # Distance from mean as percentage
        latest_close = float(close.iloc[-1])
        valid_sma = sma.dropna()
        if len(valid_sma) > 0 and float(valid_sma.iloc[-1]) != 0:
            latest_sma = float(valid_sma.iloc[-1])
            distance_from_mean_pct = round(
                (latest_close - latest_sma) / latest_sma * 100, 4
            )
        else:
            distance_from_mean_pct = float("nan")

        # Days since fair value (-1 to 1)
        days_since_fair_value = self._days_since_fair_value(zscore_series)

        # Reversion probability
        reversion_probability = self._reversion_probability(current_z)

        return ZScoreResult(
            zscore=zscore_series,
            current_zscore=round(current_z, 4) if not math.isnan(current_z) else current_z,
            zone=zone,
            mean_reversion_signal=mean_reversion_signal,
            distance_from_mean_pct=distance_from_mean_pct,
            days_since_fair_value=days_since_fair_value,
            reversion_probability=reversion_probability,
        )

    @staticmethod
    def _classify_zscore_zone(z: float) -> str:
        """Map a z-score value to its zone label."""
        if math.isnan(z):
            return "FAIR_VALUE"
        if z > 2.5:
            return "EXTREMELY_OVERBOUGHT"
        if z > 2:
            return "OVERBOUGHT"
        if z > 1:
            return "SLIGHTLY_OVERBOUGHT"
        if z < -2.5:
            return "EXTREMELY_OVERSOLD"
        if z < -2:
            return "OVERSOLD"
        if z < -1:
            return "SLIGHTLY_OVERSOLD"
        return "FAIR_VALUE"

    @staticmethod
    def _days_since_fair_value(zscore_series: pd.Series) -> int:
        """Count bars since z-score was last in [-1, 1]."""
        valid = zscore_series.dropna()
        if len(valid) == 0:
            return 0
        fair_mask = (valid >= -1) & (valid <= 1)
        if not fair_mask.any():
            return len(valid)
        # Find the last True index
        last_fair_idx = fair_mask[::-1].idxmax()
        last_fair_pos = valid.index.get_loc(last_fair_idx)
        return len(valid) - 1 - last_fair_pos

    @staticmethod
    def _reversion_probability(z: float) -> str:
        """Classify reversion probability from z-score magnitude."""
        if math.isnan(z):
            return "LOW"
        abs_z = abs(z)
        if abs_z > 2.5:
            return "HIGH"
        if abs_z > 1.5:
            return "MODERATE"
        return "LOW"

    # ------------------------------------------------------------------
    # Beta
    # ------------------------------------------------------------------

    def calculate_beta(
        self,
        index_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        period: int = 252,
    ) -> BetaResult:
        """Calculate Beta and Alpha against a benchmark.

        Beta = cov(index_returns, benchmark_returns) / var(benchmark_returns)
        Alpha (Jensen's) = mean(index_returns) − beta × mean(benchmark_returns),
        annualised.

        Args:
            index_df: OHLCV DataFrame for the target index.
            benchmark_df: OHLCV DataFrame for the benchmark (e.g. NIFTY 50).
            period: Rolling window for beta calculation (default 252).

        Returns:
            BetaResult with beta, alpha, r_squared, and interpretation.
        """
        empty_result = BetaResult(
            beta=float("nan"),
            beta_series=pd.Series(dtype=float),
            interpretation="MARKET_NEUTRAL",
            alpha=float("nan"),
            r_squared=float("nan"),
        )

        # Validate inputs
        for label, df in [("index", index_df), ("benchmark", benchmark_df)]:
            if df.empty or "close" not in df.columns:
                logger.warning("Cannot compute Beta: %s DataFrame is empty or missing 'close'.", label)
                return empty_result

        # Align series on index
        idx_close = index_df["close"].dropna()
        bm_close = benchmark_df["close"].dropna()

        min_required = 60
        if len(idx_close) < min_required or len(bm_close) < min_required:
            logger.warning(
                "Not enough data for Beta: need >= %d rows, got index=%d, benchmark=%d.",
                min_required,
                len(idx_close),
                len(bm_close),
            )
            return empty_result

        # Compute daily returns — use pct_change to match lengths
        idx_returns = idx_close.pct_change().dropna()
        bm_returns = bm_close.pct_change().dropna()

        # Align on common index
        common_idx = idx_returns.index.intersection(bm_returns.index)
        if len(common_idx) < min_required:
            # Fall back to positional alignment if indices don't overlap
            min_len = min(len(idx_returns), len(bm_returns))
            if min_len < min_required:
                logger.warning("Not enough overlapping data for Beta calculation.")
                return empty_result
            idx_r = idx_returns.iloc[-min_len:].reset_index(drop=True)
            bm_r = bm_returns.iloc[-min_len:].reset_index(drop=True)
        else:
            idx_r = idx_returns.loc[common_idx]
            bm_r = bm_returns.loc[common_idx]

        # Rolling beta
        effective_period = min(period, len(idx_r))
        rolling_cov = idx_r.rolling(window=effective_period).cov(bm_r)
        rolling_var = bm_r.rolling(window=effective_period).var()

        beta_series = rolling_cov / rolling_var.replace(0, np.nan)

        valid_beta = beta_series.dropna()
        current_beta = float(valid_beta.iloc[-1]) if len(valid_beta) > 0 else float("nan")

        # Interpretation
        interpretation = self._interpret_beta(current_beta)

        # Jensen's alpha (annualised)
        mean_idx_r = float(idx_r.mean())
        mean_bm_r = float(bm_r.mean())
        if not math.isnan(current_beta):
            alpha = (mean_idx_r - current_beta * mean_bm_r) * 252
        else:
            alpha = float("nan")

        # R-squared
        if len(idx_r) >= 2 and len(bm_r) >= 2:
            corr = float(idx_r.corr(bm_r))
            r_squared = corr ** 2 if not math.isnan(corr) else float("nan")
        else:
            r_squared = float("nan")

        return BetaResult(
            beta=round(current_beta, 4) if not math.isnan(current_beta) else current_beta,
            beta_series=beta_series,
            interpretation=interpretation,
            alpha=round(alpha, 6) if not math.isnan(alpha) else alpha,
            r_squared=round(r_squared, 4) if not math.isnan(r_squared) else r_squared,
        )

    @staticmethod
    def _interpret_beta(beta: float) -> str:
        """Classify beta into an interpretation label."""
        if math.isnan(beta):
            return "MARKET_NEUTRAL"
        if beta < 0.5:
            return "LOW_BETA"
        if beta < 0.8:
            return "MODERATE_LOW"
        if beta <= 1.2:
            return "MARKET_NEUTRAL"
        if beta <= 1.5:
            return "MODERATE_HIGH"
        return "HIGH_BETA"

    # ------------------------------------------------------------------
    # Correlation Coefficient
    # ------------------------------------------------------------------

    def calculate_correlation(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        period: int = 50,
    ) -> CorrelationResult:
        """Calculate Pearson correlation of returns between two instruments.

        Args:
            df1: OHLCV DataFrame for the first instrument.
            df2: OHLCV DataFrame for the second instrument.
            period: Rolling window (default 50).

        Returns:
            CorrelationResult with current correlation, rolling series, and
            regime-change detection.
        """
        empty_result = CorrelationResult(
            correlation=float("nan"),
            rolling_correlation=pd.Series(dtype=float),
            strength="WEAK",
            is_diverging=False,
            correlation_regime_change=False,
        )

        for label, df in [("df1", df1), ("df2", df2)]:
            if df.empty or "close" not in df.columns:
                logger.warning(
                    "Cannot compute Correlation: %s is empty or missing 'close'.", label,
                )
                return empty_result

        r1 = df1["close"].pct_change().dropna()
        r2 = df2["close"].pct_change().dropna()

        # Align
        common_idx = r1.index.intersection(r2.index)
        if len(common_idx) >= period:
            r1 = r1.loc[common_idx]
            r2 = r2.loc[common_idx]
        else:
            min_len = min(len(r1), len(r2))
            if min_len < period:
                logger.warning("Not enough data for correlation: need %d, got %d.", period, min_len)
                return empty_result
            r1 = r1.iloc[-min_len:].reset_index(drop=True)
            r2 = r2.iloc[-min_len:].reset_index(drop=True)

        rolling_corr = r1.rolling(window=period).corr(r2)

        valid = rolling_corr.dropna()
        current_corr = float(valid.iloc[-1]) if len(valid) > 0 else float("nan")

        # Strength
        strength = self._classify_correlation_strength(current_corr)

        # Diverging: correlation decreasing over last 20 bars
        is_diverging = False
        if len(valid) >= 20:
            tail = valid.iloc[-20:]
            slope = float(np.polyfit(range(len(tail)), tail.values, 1)[0])
            is_diverging = slope < -0.005  # meaningful decrease

        # Regime change: correlation crossed 0 recently (last 5 bars)
        correlation_regime_change = False
        if len(valid) >= 5:
            tail = valid.iloc[-5:]
            signs = np.sign(tail.values)
            if any(signs[i] != signs[i + 1] and signs[i] != 0 and signs[i + 1] != 0
                   for i in range(len(signs) - 1)):
                correlation_regime_change = True

        return CorrelationResult(
            correlation=round(current_corr, 4) if not math.isnan(current_corr) else current_corr,
            rolling_correlation=rolling_corr,
            strength=strength,
            is_diverging=is_diverging,
            correlation_regime_change=correlation_regime_change,
        )

    @staticmethod
    def _classify_correlation_strength(corr: float) -> str:
        """Map a correlation coefficient to a strength label."""
        if math.isnan(corr):
            return "WEAK"
        if corr > 0.7:
            return "STRONG_POSITIVE"
        if corr > 0.3:
            return "MODERATE_POSITIVE"
        if corr < -0.7:
            return "STRONG_NEGATIVE"
        if corr < -0.3:
            return "MODERATE_NEGATIVE"
        return "WEAK"

    # ------------------------------------------------------------------
    # Black-Scholes Model
    # ------------------------------------------------------------------

    def calculate_bs_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str = "CE",
        market_price: Optional[float] = None,
    ) -> BSResult:
        """Calculate Black-Scholes option price and Greeks.

        Uses the standard Black-Scholes formula for European options:
            d1 = (ln(S/K) + (r + σ²/2) * T) / (σ * √T)
            d2 = d1 − σ * √T
            Call = S * N(d1) − K * e^(-rT) * N(d2)
            Put  = K * e^(-rT) * N(-d2) − S * N(-d1)

        Args:
            spot: Current underlying price (S).
            strike: Option strike price (K).
            time_to_expiry_years: Time to expiry in years (T).
            risk_free_rate: Risk-free interest rate (decimal, e.g. 0.065).
            volatility: Annualised volatility (decimal, e.g. 0.20 for 20%).
            option_type: ``"CE"`` for call, ``"PE"`` for put.
            market_price: Market price of the option (optional, for mispricing).

        Returns:
            BSResult with theoretical price, Greeks, and mispricing analysis.
        """
        is_call = option_type.upper() == "CE"

        # Intrinsic value
        if is_call:
            intrinsic = max(0.0, spot - strike)
        else:
            intrinsic = max(0.0, strike - spot)

        # Edge case: zero or negative time to expiry → return intrinsic
        if time_to_expiry_years <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
            return BSResult(
                theoretical_price=intrinsic,
                intrinsic_value=intrinsic,
                time_value=0.0,
                delta=1.0 if is_call and spot > strike else (-1.0 if not is_call and strike > spot else 0.0),
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                rho=0.0,
                is_overpriced=market_price > intrinsic if market_price is not None else False,
                is_underpriced=market_price < intrinsic if market_price is not None else False,
                mispricing_pct=self._mispricing_pct(market_price, intrinsic) if market_price is not None else None,
            )

        T = time_to_expiry_years
        r = risk_free_rate
        sigma = volatility
        S = spot
        K = strike
        sqrt_T = math.sqrt(T)

        d1 = (_safe_log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        discount = math.exp(-r * T)

        if is_call:
            price = S * _norm_cdf(d1) - K * discount * _norm_cdf(d2)
            delta = _norm_cdf(d1)
            rho = K * T * discount * _norm_cdf(d2) / 100  # per 1% rate change
        else:
            price = K * discount * _norm_cdf(-d2) - S * _norm_cdf(-d1)
            delta = _norm_cdf(d1) - 1.0
            rho = -K * T * discount * _norm_cdf(-d2) / 100

        # Greeks common to both
        gamma = _norm_pdf(d1) / (S * sigma * sqrt_T)
        theta = (
            -(S * _norm_pdf(d1) * sigma) / (2 * sqrt_T)
            - r * K * discount * (_norm_cdf(d2) if is_call else _norm_cdf(-d2))
            * (1 if is_call else -1)
        ) / 365.0  # per calendar day

        # Correct theta for put
        if not is_call:
            theta = (
                -(S * _norm_pdf(d1) * sigma) / (2 * sqrt_T)
                + r * K * discount * _norm_cdf(-d2)
            ) / 365.0

        vega = S * _norm_pdf(d1) * sqrt_T / 100  # per 1% IV change

        time_value = max(0.0, price - intrinsic)

        # Mispricing analysis
        if market_price is not None:
            is_overpriced = market_price > price
            is_underpriced = market_price < price
            mispricing_pct = self._mispricing_pct(market_price, price)
        else:
            is_overpriced = False
            is_underpriced = False
            mispricing_pct = None

        return BSResult(
            theoretical_price=round(price, 4),
            intrinsic_value=round(intrinsic, 4),
            time_value=round(time_value, 4),
            delta=round(delta, 6),
            gamma=round(gamma, 6),
            theta=round(theta, 6),
            vega=round(vega, 6),
            rho=round(rho, 6),
            is_overpriced=is_overpriced,
            is_underpriced=is_underpriced,
            mispricing_pct=round(mispricing_pct, 4) if mispricing_pct is not None else None,
        )

    @staticmethod
    def _mispricing_pct(market_price: float, theoretical: float) -> float:
        """Calculate mispricing percentage."""
        if theoretical == 0:
            return float("inf") if market_price > 0 else 0.0
        return (market_price - theoretical) / theoretical * 100

    # ------------------------------------------------------------------
    # Implied Volatility (reverse Black-Scholes)
    # ------------------------------------------------------------------

    def calculate_implied_volatility(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: str = "CE",
        max_iterations: int = 100,
        tolerance: float = 0.0001,
    ) -> Optional[float]:
        """Reverse-engineer implied volatility from a market price using Newton-Raphson.

        Finds the σ such that BS_price(σ) = market_price.

        Args:
            market_price: Observed option price in the market.
            spot: Current underlying price.
            strike: Option strike price.
            time_to_expiry: Time to expiry in years.
            risk_free_rate: Risk-free rate (decimal).
            option_type: ``"CE"`` or ``"PE"``.
            max_iterations: Max Newton-Raphson iterations (default 100).
            tolerance: Convergence tolerance (default 0.0001).

        Returns:
            Implied volatility as a decimal (e.g. 0.20 for 20%), or ``None``
            if the solver does not converge.
        """
        if market_price <= 0 or spot <= 0 or strike <= 0 or time_to_expiry <= 0:
            return None

        # Initial guess
        sigma = 0.3  # 30%

        for _ in range(max_iterations):
            try:
                bs = self.calculate_bs_price(
                    spot, strike, time_to_expiry, risk_free_rate, sigma, option_type
                )
                diff = bs.theoretical_price - market_price
                vega_raw = self._raw_vega(spot, strike, time_to_expiry, risk_free_rate, sigma)

                if abs(diff) < tolerance:
                    return max(0.0, round(sigma, 6))

                if abs(vega_raw) < 1e-12:
                    # Vega too small — switch to bisection-style nudge
                    sigma += 0.01 if diff < 0 else -0.01
                else:
                    sigma -= diff / vega_raw

                # Keep sigma in sane bounds
                if sigma <= 0.001:
                    sigma = 0.001
                if sigma > 10.0:
                    return None  # diverging

            except (ValueError, ZeroDivisionError, OverflowError):
                return None

        return None  # didn't converge

    @staticmethod
    def _raw_vega(
        spot: float,
        strike: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Return the raw vega (dP/dσ) without the ÷100 scaling.

        This is needed for Newton-Raphson where we iterate on σ directly.
        """
        if T <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
            return 0.0
        sqrt_T = math.sqrt(T)
        d1 = (_safe_log(spot / strike) + (r + sigma ** 2 / 2) * T) / (sigma * sqrt_T)
        return spot * _norm_pdf(d1) * sqrt_T

    # ------------------------------------------------------------------
    # Quant Summary
    # ------------------------------------------------------------------

    def get_quant_summary(
        self,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> QuantSummary:
        """Produce a composite quant summary from Z-Score, Beta, and stat regime.

        Vote logic:
          - Z-score < -2  → mean reversion BUY signal (bullish)
          - Z-score > 2   → mean reversion SELL signal (bearish)
          - Z-score between -1 and 1 → neutral
          - High beta + market trending up → amplified bullish
          - High beta + market trending down → amplified bearish
          - Positive alpha → confidence boost
          - Max confidence = 0.6 (quant signals are supplementary)

        Args:
            df: OHLCV DataFrame for the target instrument.
            benchmark_df: OHLCV DataFrame for the benchmark (optional).

        Returns:
            QuantSummary with all fields populated.
        """
        now = datetime.utcnow()
        empty_summary = QuantSummary(
            timestamp=now,
            zscore=float("nan"),
            zscore_zone="FAIR_VALUE",
            mean_reversion_signal=None,
            beta=None,
            alpha=None,
            beta_interpretation=None,
            statistical_regime="NORMAL",
            quant_vote="NEUTRAL",
            quant_confidence=0.0,
        )

        clean = _prepare_df(df)
        if clean.empty or len(clean) < 2:
            logger.warning("DataFrame too short for quant summary.")
            return empty_summary

        timestamp = (
            pd.Timestamp(clean["timestamp"].iloc[-1]).to_pydatetime()
            if "timestamp" in clean.columns
            else now
        )

        # --- Z-Score -------------------------------------------------------
        zscore_result = self.calculate_zscore(clean)
        current_z = zscore_result.current_zscore

        # --- Beta (optional) -----------------------------------------------
        beta_val: Optional[float] = None
        alpha_val: Optional[float] = None
        beta_interp: Optional[str] = None
        beta_result: Optional[BetaResult] = None

        if benchmark_df is not None and not benchmark_df.empty:
            beta_result = self.calculate_beta(clean, benchmark_df)
            if not math.isnan(beta_result.beta):
                beta_val = beta_result.beta
                alpha_val = beta_result.alpha
                beta_interp = beta_result.interpretation

        # --- Statistical regime --------------------------------------------
        statistical_regime = self._determine_statistical_regime(zscore_result)

        # --- Voting --------------------------------------------------------
        quant_vote, quant_confidence = self._compute_quant_vote(
            zscore_result=zscore_result,
            beta_result=beta_result,
            benchmark_df=benchmark_df,
        )

        return QuantSummary(
            timestamp=timestamp,
            zscore=current_z,
            zscore_zone=zscore_result.zone,
            mean_reversion_signal=zscore_result.mean_reversion_signal,
            beta=beta_val,
            alpha=alpha_val,
            beta_interpretation=beta_interp,
            statistical_regime=statistical_regime,
            quant_vote=quant_vote,
            quant_confidence=quant_confidence,
        )

    @staticmethod
    def _determine_statistical_regime(zscore_result: ZScoreResult) -> str:
        """Classify the statistical regime from z-score behaviour."""
        z = zscore_result.current_zscore
        if math.isnan(z):
            return "NORMAL"

        abs_z = abs(z)
        days_since = zscore_result.days_since_fair_value

        #  Extreme z-score with HIGH reversion probability → MEAN_REVERTING
        if abs_z > 2 and zscore_result.reversion_probability == "HIGH":
            return "MEAN_REVERTING"

        # Persistent deviation from fair value → TRENDING
        if abs_z > 1 and days_since >= 10:
            return "TRENDING"

        # Z-score near 0
        if abs_z <= 1:
            return "NORMAL"

        return "NORMAL"

    def _compute_quant_vote(
        self,
        zscore_result: ZScoreResult,
        beta_result: Optional[BetaResult],
        benchmark_df: Optional[pd.DataFrame],
    ) -> tuple[str, float]:
        """Return ``(quant_vote, quant_confidence)`` from quant signals."""
        z = zscore_result.current_zscore
        if math.isnan(z):
            return "NEUTRAL", 0.0

        # Base vote from z-score
        if z < -2:
            vote = "BULLISH"
            confidence = 0.45
        elif z > 2:
            vote = "BEARISH"
            confidence = 0.45
        elif -1 <= z <= 1:
            vote = "NEUTRAL"
            confidence = 0.2
        else:
            # Mild deviation: slight lean
            vote = "BEARISH" if z > 1 else "BULLISH"
            confidence = 0.25

        # Beta amplification
        if beta_result is not None and not math.isnan(beta_result.beta):
            # Determine benchmark trend: are last 20 days of benchmark returns positive?
            bm_trending_up = False
            bm_trending_down = False
            if benchmark_df is not None and not benchmark_df.empty and "close" in benchmark_df.columns:
                bm_close = benchmark_df["close"].dropna()
                if len(bm_close) >= 20:
                    recent_return = (float(bm_close.iloc[-1]) - float(bm_close.iloc[-20])) / float(bm_close.iloc[-20])
                    bm_trending_up = recent_return > 0.02
                    bm_trending_down = recent_return < -0.02

            if beta_result.beta > 1.2:
                if bm_trending_up and vote == "BULLISH":
                    confidence = min(0.6, confidence + 0.1)
                elif bm_trending_down and vote == "BEARISH":
                    confidence = min(0.6, confidence + 0.1)

            # Positive alpha → confidence boost
            if not math.isnan(beta_result.alpha) and beta_result.alpha > 0:
                confidence = min(0.6, confidence + 0.05)

        # Cap at 0.6
        confidence = min(0.6, round(confidence, 2))

        return vote, confidence
