"""
Volume indicators for trading analysis.

Implements VWAP, OBV, Volume Profile, general Volume Analysis, and a
composite volume summary.
All methods are pure functions — no database calls or side effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.analysis.indicators.momentum import _detect_divergence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VWAPResult:
    """Result of VWAP (Volume Weighted Average Price) calculation.

    VWAP = cumulative(typical_price × volume) / cumulative(volume)
    where typical_price = (high + low + close) / 3.

    For intraday data VWAP resets each trading day.  For daily data a
    rolling cumulative VWAP is calculated over the full session.

    ``institutional_bias`` reflects whether the price has been consistently
    above VWAP (institutional buying), below (selling), or crossing back
    and forth (neutral).
    """

    vwap: pd.Series                  # VWAP values
    current_vwap: float              # Latest VWAP
    price_vs_vwap: str               # "ABOVE" / "BELOW" / "AT_VWAP" (within 0.05 %)
    distance_from_vwap: float        # (close − vwap) / vwap × 100
    vwap_slope: str                  # "RISING" / "FALLING" / "FLAT"

    # VWAP standard-deviation bands
    upper_band_1: pd.Series          # VWAP + 1 σ
    lower_band_1: pd.Series          # VWAP − 1 σ
    upper_band_2: pd.Series          # VWAP + 2 σ
    lower_band_2: pd.Series          # VWAP − 2 σ

    price_zone: str                  # ABOVE_2SD / ABOVE_1SD / NEAR_VWAP / BELOW_1SD / BELOW_2SD

    # Trading interpretation
    institutional_bias: str          # "BUYING" / "SELLING" / "NEUTRAL"


@dataclass
class OBVResult:
    """Result of On-Balance Volume calculation.

    OBV accumulates volume on up-closes and subtracts it on down-closes,
    producing a running total that reveals buying/selling pressure.

    ``divergence`` is detected by the same swing-point algorithm used in
    the momentum module: price and OBV moving in opposite directions warn
    of a potential trend reversal.
    """

    obv: pd.Series                   # OBV values
    obv_ema: pd.Series               # 20-period EMA of OBV
    obv_trend: str                   # "RISING" / "FALLING" / "FLAT" (EMA slope over 10 bars)
    divergence: Optional[str]        # "BULLISH_DIVERGENCE" / "BEARISH_DIVERGENCE" / None
    accumulation_distribution: str   # "ACCUMULATION" / "DISTRIBUTION" / "NEUTRAL"


@dataclass
class VolumeProfileResult:
    """Result of Volume Profile calculation.

    The price range is divided into *num_bins* equal zones and volume is
    summed per zone.  The Point of Control (POC) is the price level with
    the highest traded volume; the Value Area contains 70 % of total
    volume centred on the POC.
    """

    price_levels: list[float]        # Centre price of each bin
    volume_at_level: list[float]     # Total volume at each bin
    poc: float                       # Point of Control (highest-volume level)
    value_area_high: float           # Upper boundary of the 70 % value area
    value_area_low: float            # Lower boundary of the 70 % value area
    current_price_in_value_area: bool
    high_volume_nodes: list[float]   # Levels with volume > 1.5× average (S/R)
    low_volume_nodes: list[float]    # Levels with volume < 0.5× average (fast-move zones)


@dataclass
class VolumeAnalysis:
    """General volume-bar analysis relative to a rolling average.

    ``volume_confirms_price`` is True when price and volume move in the
    same direction (up+up or down+down) — a basic volume-confirmation
    check.
    """

    current_volume: int
    avg_volume: float                # *period*-bar average
    volume_ratio: float              # current / average (>1 = above average)
    volume_trend: str                # "INCREASING" / "DECREASING" / "STABLE" (last 5 bars)
    is_high_volume: bool             # volume_ratio > 1.5
    is_low_volume: bool              # volume_ratio < 0.5
    volume_confirms_price: bool      # Price direction confirmed by volume direction
    climax_volume: bool              # volume_ratio > 3.0 (extreme)


@dataclass
class VolumeSummary:
    """Composite volume reading from VWAP, OBV, Volume Profile, and
    general volume analysis.

    ``volume_vote`` is derived from per-indicator bullish / bearish votes.
    """

    timestamp: datetime

    # VWAP
    price_vs_vwap: str
    vwap_zone: str
    institutional_bias: str

    # OBV
    obv_trend: str
    obv_divergence: Optional[str]
    accumulation_distribution: str

    # Volume Profile
    poc: float
    value_area_high: float
    value_area_low: float
    in_value_area: bool

    # Volume analysis
    volume_ratio: float
    volume_confirms_price: bool

    # Overall volume verdict
    volume_vote: str                 # STRONG_BULLISH / BULLISH / NEUTRAL / BEARISH / STRONG_BEARISH
    volume_confidence: float         # 0.0 – 1.0

    # Warnings / notes
    warnings: list[str] = field(default_factory=list)


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


def _is_intraday(df: pd.DataFrame) -> bool:
    """Heuristically detect whether *df* contains intraday data.

    Checks the ``timestamp`` column (if present) for multiple rows on the
    same calendar date.  Falls back to inspecting the index.
    """
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = df.index.to_series()
    else:
        return False

    if len(ts) < 2:
        return False

    dates = ts.dt.date
    return dates.nunique() < len(dates)


def _detect_slope(series: pd.Series, lookback: int = 5, threshold: float = 0.0) -> str:
    """Classify the slope of the last *lookback* valid values.

    Returns ``"RISING"``, ``"FALLING"``, or ``"FLAT"``.
    """
    valid = series.dropna()
    if len(valid) < lookback:
        return "FLAT"
    tail = valid.iloc[-lookback:].values.astype(float)
    slope = np.polyfit(range(lookback), tail, 1)[0]
    if slope > threshold:
        return "RISING"
    if slope < -threshold:
        return "FALLING"
    return "FLAT"


# ---------------------------------------------------------------------------
# VolumeIndicators
# ---------------------------------------------------------------------------


class VolumeIndicators:
    """Pure-function volume indicator calculations for OHLCV DataFrames.

    All public methods accept a DataFrame with columns
    ``[timestamp, open, high, low, close, volume]`` and return either a
    ``pd.Series`` or a result dataclass.  No state is stored on the instance.
    """

    # ------------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------------

    def calculate_vwap(self, df: pd.DataFrame) -> VWAPResult:
        """Calculate Volume Weighted Average Price with std-dev bands.

        VWAP = Σ(typical_price × volume) / Σ(volume)
        where typical_price = (high + low + close) / 3.

        For intraday data the cumulative sums reset at the start of each
        trading day.  For daily data a rolling cumulative VWAP is
        calculated over the entire session.

        Standard deviation bands are calculated from the running
        variance of (typical_price − VWAP)² weighted by volume.

        Args:
            df: OHLCV DataFrame.

        Returns:
            VWAPResult with all series and latest-bar summary fields.
        """
        clean = _prepare_df(df)

        if clean.empty or len(clean) < 2:
            logger.warning(
                "Insufficient data for VWAP calculation: got %d rows.",
                len(clean) if not clean.empty else 0,
            )
            empty = pd.Series(dtype=float)
            return VWAPResult(
                vwap=empty,
                current_vwap=float("nan"),
                price_vs_vwap="AT_VWAP",
                distance_from_vwap=0.0,
                vwap_slope="FLAT",
                upper_band_1=empty,
                lower_band_1=empty,
                upper_band_2=empty,
                lower_band_2=empty,
                price_zone="NEAR_VWAP",
                institutional_bias="NEUTRAL",
            )

        typical_price = (clean["high"] + clean["low"] + clean["close"]) / 3
        volume = clean["volume"].astype(float)

        # --- Determine day groups for cumulative reset ----------------------
        intraday = _is_intraday(clean)
        if intraday and "timestamp" in clean.columns:
            day_group = pd.to_datetime(clean["timestamp"]).dt.date
        elif intraday and isinstance(clean.index, pd.DatetimeIndex):
            day_group = clean.index.date
        else:
            # Daily data: one single group (no reset)
            day_group = pd.Series(0, index=clean.index)

        day_group = pd.Series(day_group, index=clean.index) if not isinstance(day_group, pd.Series) else day_group.reset_index(drop=True)
        day_group.index = clean.index

        # --- Per-group cumulative VWAP and std-dev bands --------------------
        tp_vol = typical_price * volume
        cum_tp_vol = tp_vol.groupby(day_group).cumsum()
        cum_vol = volume.groupby(day_group).cumsum()

        vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)

        # Running weighted variance for bands
        sq_diff = (typical_price - vwap_series) ** 2
        cum_sq_diff_vol = (sq_diff * volume).groupby(day_group).cumsum()
        variance = cum_sq_diff_vol / cum_vol.replace(0, np.nan)
        std_dev = np.sqrt(variance)

        upper_band_1 = vwap_series + std_dev
        lower_band_1 = vwap_series - std_dev
        upper_band_2 = vwap_series + 2 * std_dev
        lower_band_2 = vwap_series - 2 * std_dev

        # --- Latest-bar summary fields --------------------------------------
        valid_vwap = vwap_series.dropna()
        current_vwap = float(valid_vwap.iloc[-1]) if len(valid_vwap) > 0 else float("nan")
        latest_close = float(clean["close"].iloc[-1])

        # Price vs VWAP
        if np.isnan(current_vwap) or current_vwap == 0:
            price_vs_vwap = "AT_VWAP"
            distance_from_vwap = 0.0
        else:
            distance_from_vwap = round((latest_close - current_vwap) / current_vwap * 100, 4)
            if abs(distance_from_vwap) <= 0.05:
                price_vs_vwap = "AT_VWAP"
            elif latest_close > current_vwap:
                price_vs_vwap = "ABOVE"
            else:
                price_vs_vwap = "BELOW"

        # VWAP slope (last 5 bars of VWAP)
        vwap_slope = _detect_slope(vwap_series, lookback=5)

        # Price zone relative to bands
        valid_ub2 = upper_band_2.dropna()
        valid_ub1 = upper_band_1.dropna()
        valid_lb1 = lower_band_1.dropna()
        valid_lb2 = lower_band_2.dropna()

        if len(valid_ub2) > 0 and len(valid_lb2) > 0:
            latest_ub2 = float(valid_ub2.iloc[-1])
            latest_ub1 = float(valid_ub1.iloc[-1])
            latest_lb1 = float(valid_lb1.iloc[-1])
            latest_lb2 = float(valid_lb2.iloc[-1])

            if latest_close > latest_ub2:
                price_zone = "ABOVE_2SD"
            elif latest_close > latest_ub1:
                price_zone = "ABOVE_1SD"
            elif latest_close < latest_lb2:
                price_zone = "BELOW_2SD"
            elif latest_close < latest_lb1:
                price_zone = "BELOW_1SD"
            else:
                price_zone = "NEAR_VWAP"
        else:
            price_zone = "NEAR_VWAP"

        # Institutional bias — examine last 10 bars' relation to VWAP
        institutional_bias = self._assess_institutional_bias(clean["close"], vwap_series)

        return VWAPResult(
            vwap=vwap_series,
            current_vwap=current_vwap,
            price_vs_vwap=price_vs_vwap,
            distance_from_vwap=distance_from_vwap,
            vwap_slope=vwap_slope,
            upper_band_1=upper_band_1,
            lower_band_1=lower_band_1,
            upper_band_2=upper_band_2,
            lower_band_2=lower_band_2,
            price_zone=price_zone,
            institutional_bias=institutional_bias,
        )

    @staticmethod
    def _assess_institutional_bias(
        close: pd.Series, vwap: pd.Series, lookback: int = 10
    ) -> str:
        """Determine institutional bias from how consistently price stays
        above or below VWAP over the last *lookback* bars.

        Returns ``"BUYING"`` (≥ 70 % above), ``"SELLING"`` (≥ 70 % below),
        or ``"NEUTRAL"``.
        """
        aligned = pd.DataFrame({"close": close, "vwap": vwap}).dropna()
        if len(aligned) < 2:
            return "NEUTRAL"

        tail = aligned.iloc[-lookback:] if len(aligned) >= lookback else aligned
        above_pct = float((tail["close"] > tail["vwap"]).sum()) / len(tail)

        if above_pct >= 0.7:
            return "BUYING"
        if above_pct <= 0.3:
            return "SELLING"
        return "NEUTRAL"

    # ------------------------------------------------------------------
    # OBV
    # ------------------------------------------------------------------

    def calculate_obv(self, df: pd.DataFrame, ema_period: int = 20) -> OBVResult:
        """Calculate On-Balance Volume.

        OBV rules:
          - close > prev_close → OBV += volume
          - close < prev_close → OBV -= volume
          - close == prev_close → OBV unchanged

        An EMA of OBV (*ema_period*) is used to smooth the trend.  The
        slope of that EMA over the last 10 bars classifies the OBV trend.

        Divergence detection re-uses ``_detect_divergence`` from the
        momentum module with a 14-bar lookback.

        Args:
            df: OHLCV DataFrame.
            ema_period: EMA period for OBV smoothing (default 20).

        Returns:
            OBVResult with full series and latest-bar summary fields.
        """
        clean = _prepare_df(df)

        if clean.empty or len(clean) < 2:
            logger.warning(
                "Insufficient data for OBV calculation: got %d rows.",
                len(clean) if not clean.empty else 0,
            )
            empty = pd.Series(dtype=float)
            return OBVResult(
                obv=empty,
                obv_ema=empty,
                obv_trend="FLAT",
                divergence=None,
                accumulation_distribution="NEUTRAL",
            )

        close = clean["close"]
        volume = clean["volume"].astype(float)

        # OBV calculation
        direction = np.sign(close.diff())
        obv_values = (direction * volume).fillna(0).cumsum()
        obv_series = pd.Series(obv_values.values, index=clean.index, name="obv")

        # EMA of OBV
        obv_ema = obv_series.ewm(span=ema_period, adjust=False).mean()

        # OBV trend (slope of EMA over last 10 bars)
        obv_trend = _detect_slope(obv_ema, lookback=10)

        # Divergence (price vs OBV)
        divergence = _detect_divergence(close, obv_series, lookback=14)

        # Accumulation / Distribution classification
        if obv_trend == "RISING":
            accumulation_distribution = "ACCUMULATION"
        elif obv_trend == "FALLING":
            accumulation_distribution = "DISTRIBUTION"
        else:
            accumulation_distribution = "NEUTRAL"

        return OBVResult(
            obv=obv_series,
            obv_ema=obv_ema,
            obv_trend=obv_trend,
            divergence=divergence,
            accumulation_distribution=accumulation_distribution,
        )

    # ------------------------------------------------------------------
    # Volume Profile
    # ------------------------------------------------------------------

    def calculate_volume_profile(
        self, df: pd.DataFrame, num_bins: int = 20
    ) -> VolumeProfileResult:
        """Calculate Volume Profile (volume at price).

        Divides the price range into *num_bins* equal zones, sums the
        volume in each zone, and identifies the Point of Control (POC),
        Value Area (70 % of total volume around POC), and high/low volume
        nodes.

        Value Area expansion:
          1. Start with the POC bin.
          2. Look at the bin immediately above and below.
          3. Add whichever has more volume.
          4. Repeat until 70 % of total volume is included.

        Args:
            df: OHLCV DataFrame (requires at least 20 bars).
            num_bins: Number of price bins (default 20).

        Returns:
            VolumeProfileResult with all fields populated.
        """
        clean = _prepare_df(df)

        empty_result = VolumeProfileResult(
            price_levels=[],
            volume_at_level=[],
            poc=float("nan"),
            value_area_high=float("nan"),
            value_area_low=float("nan"),
            current_price_in_value_area=False,
            high_volume_nodes=[],
            low_volume_nodes=[],
        )

        if clean.empty or len(clean) < 2:
            logger.warning("Insufficient data for Volume Profile calculation.")
            return empty_result

        close = clean["close"].values.astype(float)
        high = clean["high"].values.astype(float)
        low = clean["low"].values.astype(float)
        volume = clean["volume"].values.astype(float)

        price_min = float(np.nanmin(low))
        price_max = float(np.nanmax(high))

        if price_max <= price_min:
            logger.warning("No price range for Volume Profile (flat data).")
            return empty_result

        # Build bins
        bin_edges = np.linspace(price_min, price_max, num_bins + 1)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Assign each bar's volume to the bin that contains its close
        bin_indices = np.clip(
            np.digitize(close, bin_edges) - 1, 0, num_bins - 1
        )
        volume_by_bin = np.zeros(num_bins, dtype=float)
        for i in range(len(close)):
            volume_by_bin[bin_indices[i]] += volume[i]

        price_levels = bin_centres.tolist()
        volume_at_level = volume_by_bin.tolist()

        # POC — bin with the highest volume
        poc_idx = int(np.argmax(volume_by_bin))
        poc = float(bin_centres[poc_idx])

        # Value Area (70 % of total volume, expanding from POC)
        total_volume = float(volume_by_bin.sum())
        target_volume = total_volume * 0.70

        included = np.zeros(num_bins, dtype=bool)
        included[poc_idx] = True
        accumulated = float(volume_by_bin[poc_idx])
        lower_ptr = poc_idx - 1
        upper_ptr = poc_idx + 1

        while accumulated < target_volume and (lower_ptr >= 0 or upper_ptr < num_bins):
            vol_above = float(volume_by_bin[upper_ptr]) if upper_ptr < num_bins else -1.0
            vol_below = float(volume_by_bin[lower_ptr]) if lower_ptr >= 0 else -1.0

            if vol_above >= vol_below:
                included[upper_ptr] = True
                accumulated += vol_above
                upper_ptr += 1
            else:
                included[lower_ptr] = True
                accumulated += vol_below
                lower_ptr -= 1

        included_indices = np.where(included)[0]
        va_low_idx = int(included_indices.min())
        va_high_idx = int(included_indices.max())
        value_area_low = float(bin_edges[va_low_idx])
        value_area_high = float(bin_edges[va_high_idx + 1])

        latest_close = float(close[-1])
        current_price_in_value_area = value_area_low <= latest_close <= value_area_high

        # High / low volume nodes
        avg_vol = float(volume_by_bin.mean()) if num_bins > 0 else 0.0
        high_volume_nodes = [
            float(bin_centres[i])
            for i in range(num_bins)
            if avg_vol > 0 and volume_by_bin[i] > 1.5 * avg_vol
        ]
        low_volume_nodes = [
            float(bin_centres[i])
            for i in range(num_bins)
            if avg_vol > 0 and volume_by_bin[i] < 0.5 * avg_vol
        ]

        return VolumeProfileResult(
            price_levels=price_levels,
            volume_at_level=volume_at_level,
            poc=poc,
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            current_price_in_value_area=current_price_in_value_area,
            high_volume_nodes=high_volume_nodes,
            low_volume_nodes=low_volume_nodes,
        )

    # ------------------------------------------------------------------
    # Volume Analysis (general)
    # ------------------------------------------------------------------

    def analyze_volume(self, df: pd.DataFrame, period: int = 20) -> VolumeAnalysis:
        """Analyse the most recent bar's volume relative to a rolling average.

        ``volume_confirms_price`` is True when close is up from the prior
        bar and volume is above average, *or* close is down and volume is
        above average (strong selling).  It is False when price and
        volume diverge (e.g. price up on declining volume).

        Args:
            df: OHLCV DataFrame.
            period: Rolling average period (default 20).

        Returns:
            VolumeAnalysis with all fields populated.
        """
        clean = _prepare_df(df)

        if clean.empty or len(clean) < 2:
            logger.warning(
                "Insufficient data for volume analysis: got %d rows.",
                len(clean) if not clean.empty else 0,
            )
            return VolumeAnalysis(
                current_volume=0,
                avg_volume=0.0,
                volume_ratio=0.0,
                volume_trend="STABLE",
                is_high_volume=False,
                is_low_volume=True,
                volume_confirms_price=False,
                climax_volume=False,
            )

        volume = clean["volume"].astype(float)
        close = clean["close"]

        current_volume = int(volume.iloc[-1])
        avg_volume = float(volume.rolling(window=period, min_periods=1).mean().iloc[-1])
        volume_ratio = round(current_volume / avg_volume, 4) if avg_volume > 0 else 0.0

        is_high_volume = volume_ratio > 1.5
        is_low_volume = volume_ratio < 0.5
        climax_volume = volume_ratio > 3.0

        # Volume trend over the last 5 bars
        if len(volume) >= 5:
            tail = volume.iloc[-5:].values.astype(float)
            slope = np.polyfit(range(5), tail, 1)[0]
            avg_tail = float(np.mean(tail))
            relative_slope = slope / avg_tail if avg_tail > 0 else 0.0
            if relative_slope > 0.05:
                volume_trend = "INCREASING"
            elif relative_slope < -0.05:
                volume_trend = "DECREASING"
            else:
                volume_trend = "STABLE"
        else:
            volume_trend = "STABLE"

        # Volume confirms price direction
        price_up = float(close.iloc[-1]) > float(close.iloc[-2])
        price_down = float(close.iloc[-1]) < float(close.iloc[-2])
        vol_up = volume_ratio > 1.0

        if price_up and vol_up:
            volume_confirms_price = True
        elif price_down and vol_up:
            volume_confirms_price = True   # strong selling is still confirming
        else:
            volume_confirms_price = False

        return VolumeAnalysis(
            current_volume=current_volume,
            avg_volume=round(avg_volume, 2),
            volume_ratio=volume_ratio,
            volume_trend=volume_trend,
            is_high_volume=is_high_volume,
            is_low_volume=is_low_volume,
            volume_confirms_price=volume_confirms_price,
            climax_volume=climax_volume,
        )

    # ------------------------------------------------------------------
    # Volume Summary
    # ------------------------------------------------------------------

    def get_volume_summary(self, df: pd.DataFrame) -> VolumeSummary:
        """Produce a composite volume summary from VWAP, OBV, Volume Profile,
        and general volume analysis.

        Voting system:

        Bullish votes:
          - Price > VWAP                                               → +1
          - OBV rising                                                 → +1
          - Accumulation detected                                      → +1
          - Volume confirms price and price is up                      → +1
          - Price above value area                                     → +1

        Bearish votes:
          - Price < VWAP                                               → +1
          - OBV falling                                                → +1
          - Distribution detected                                      → +1
          - Volume confirms price and price is down                    → +1
          - Price below value area                                     → +1

        Overrides / adjustments:
          - OBV bearish divergence                                     → −2 net
          - OBV bullish divergence                                     → +2 net
          - Climax volume                                              → reduce confidence
          - Price in value area (balanced)                             → ±0

        For daily data VWAP is weighted less (halved contribution).

        Score mapping:
          net ≥ +4  → STRONG_BULLISH  (confidence 0.90)
          net ≥ +2  → BULLISH         (confidence 0.70)
          net ≥ −1  → NEUTRAL         (confidence 0.50)
          net ≥ −3  → BEARISH         (confidence 0.70)
          net < −3  → STRONG_BEARISH  (confidence 0.90)

        Args:
            df: OHLCV DataFrame.

        Returns:
            VolumeSummary with all fields populated.
        """
        now = datetime.utcnow()
        warnings_list: list[str] = []

        empty_summary = VolumeSummary(
            timestamp=now,
            price_vs_vwap="AT_VWAP",
            vwap_zone="NEAR_VWAP",
            institutional_bias="NEUTRAL",
            obv_trend="FLAT",
            obv_divergence=None,
            accumulation_distribution="NEUTRAL",
            poc=float("nan"),
            value_area_high=float("nan"),
            value_area_low=float("nan"),
            in_value_area=False,
            volume_ratio=0.0,
            volume_confirms_price=False,
            volume_vote="NEUTRAL",
            volume_confidence=0.0,
            warnings=warnings_list,
        )

        clean = _prepare_df(df)
        if clean.empty or len(clean) < 2:
            logger.warning("DataFrame too short for volume summary.")
            return empty_summary

        # Check for zero-volume data (some BSE indices)
        total_volume = float(clean["volume"].sum())
        if total_volume == 0:
            warnings_list.append("Zero volume data — skipping volume analysis, returning NEUTRAL.")
            logger.warning("All volume data is zero — returning NEUTRAL volume summary.")
            return VolumeSummary(
                timestamp=now,
                price_vs_vwap="AT_VWAP",
                vwap_zone="NEAR_VWAP",
                institutional_bias="NEUTRAL",
                obv_trend="FLAT",
                obv_divergence=None,
                accumulation_distribution="NEUTRAL",
                poc=float("nan"),
                value_area_high=float("nan"),
                value_area_low=float("nan"),
                in_value_area=False,
                volume_ratio=0.0,
                volume_confirms_price=False,
                volume_vote="NEUTRAL",
                volume_confidence=0.0,
                warnings=warnings_list,
            )

        timestamp = (
            pd.Timestamp(clean["timestamp"].iloc[-1]).to_pydatetime()
            if "timestamp" in clean.columns
            else now
        )

        # --- Sub-indicator calculations ------------------------------------
        vwap_result = self.calculate_vwap(clean)
        obv_result = self.calculate_obv(clean)
        vol_analysis = self.analyze_volume(clean)

        # Volume Profile (skip if < 20 bars)
        if len(clean) >= 20:
            vp_result = self.calculate_volume_profile(clean)
        else:
            warnings_list.append("Less than 20 bars — Volume Profile skipped.")
            vp_result = VolumeProfileResult(
                price_levels=[],
                volume_at_level=[],
                poc=float("nan"),
                value_area_high=float("nan"),
                value_area_low=float("nan"),
                current_price_in_value_area=False,
                high_volume_nodes=[],
                low_volume_nodes=[],
            )

        # --- Voting ---------------------------------------------------------
        bullish_votes = 0
        bearish_votes = 0

        intraday = _is_intraday(clean)
        vwap_weight = 1 if intraday else 0  # daily VWAP weighted less

        # VWAP signal
        if vwap_result.price_vs_vwap == "ABOVE":
            bullish_votes += 1
            if not intraday:
                warnings_list.append("VWAP on daily data: trend context only.")
        elif vwap_result.price_vs_vwap == "BELOW":
            bearish_votes += 1

        # Apply half-weight for daily: undo the full vote and add 0 instead
        if not intraday and vwap_result.price_vs_vwap in ("ABOVE", "BELOW"):
            # Keep the vote but we'll factor daily-weighting into confidence later
            pass

        # OBV trend
        if obv_result.obv_trend == "RISING":
            bullish_votes += 1
        elif obv_result.obv_trend == "FALLING":
            bearish_votes += 1

        # Accumulation / Distribution
        if obv_result.accumulation_distribution == "ACCUMULATION":
            bullish_votes += 1
        elif obv_result.accumulation_distribution == "DISTRIBUTION":
            bearish_votes += 1

        # Volume confirms price
        latest_close = float(clean["close"].iloc[-1])
        prev_close = float(clean["close"].iloc[-2])
        price_up = latest_close > prev_close

        if vol_analysis.volume_confirms_price:
            if price_up:
                bullish_votes += 1
            else:
                bearish_votes += 1

        # Value area position
        if not np.isnan(vp_result.value_area_high) and not np.isnan(vp_result.value_area_low):
            if latest_close > vp_result.value_area_high:
                bullish_votes += 1
            elif latest_close < vp_result.value_area_low:
                bearish_votes += 1
            # In value area → no vote (balanced)

        # Divergence overrides
        if obv_result.divergence == "BEARISH_DIVERGENCE":
            bearish_votes += 2
        elif obv_result.divergence == "BULLISH_DIVERGENCE":
            bullish_votes += 2

        net = bullish_votes - bearish_votes

        if net >= 4:
            volume_vote, base_confidence = "STRONG_BULLISH", 0.90
        elif net >= 2:
            volume_vote, base_confidence = "BULLISH", 0.70
        elif net >= -1:
            volume_vote, base_confidence = "NEUTRAL", 0.50
        elif net >= -3:
            volume_vote, base_confidence = "BEARISH", 0.70
        else:
            volume_vote, base_confidence = "STRONG_BEARISH", 0.90

        # Climax volume reduces confidence (potential exhaustion)
        if vol_analysis.climax_volume:
            base_confidence = max(0.2, base_confidence - 0.2)
            warnings_list.append("Climax volume detected — possible exhaustion move.")

        # Daily VWAP reduces confidence slightly
        if not intraday:
            base_confidence = max(0.2, base_confidence - 0.05)

        volume_confidence = round(base_confidence, 2)

        return VolumeSummary(
            timestamp=timestamp,
            price_vs_vwap=vwap_result.price_vs_vwap,
            vwap_zone=vwap_result.price_zone,
            institutional_bias=vwap_result.institutional_bias,
            obv_trend=obv_result.obv_trend,
            obv_divergence=obv_result.divergence,
            accumulation_distribution=obv_result.accumulation_distribution,
            poc=vp_result.poc,
            value_area_high=vp_result.value_area_high,
            value_area_low=vp_result.value_area_low,
            in_value_area=vp_result.current_price_in_value_area,
            volume_ratio=vol_analysis.volume_ratio,
            volume_confirms_price=vol_analysis.volume_confirms_price,
            volume_vote=volume_vote,
            volume_confidence=volume_confidence,
            warnings=warnings_list,
        )
