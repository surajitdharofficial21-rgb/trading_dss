"""
Market regime classification.

Identifies the current market regime (trending/ranging, bull/bear)
using price structure, volatility, and breadth indicators.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.data.vix_data import VIXSnapshot, VIXRegime

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Classification of the current broad market regime."""

    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class RegimeSnapshot:
    """
    Current market regime assessment.

    Attributes
    ----------
    regime:
        Classified regime.
    trend_score:
        Price trend score (positive = uptrend).
    volatility_score:
        Normalised volatility reading.
    breadth_score:
        Market breadth indicator (advances/declines ratio minus 1).
    vix_regime:
        VIX-based fear classification.
    confidence:
        Confidence in this regime classification (0–1).
    summary:
        Human-readable description.
    """

    regime: MarketRegime
    trend_score: float
    volatility_score: float
    breadth_score: float
    vix_regime: Optional[VIXRegime]
    confidence: float
    summary: str


class RegimeDetector:
    """
    Classifies market regime from index price data and VIX.

    Parameters
    ----------
    trend_ema_short:
        Short EMA period used for trend detection (default 20).
    trend_ema_long:
        Long EMA period (default 50).
    """

    def __init__(
        self,
        trend_ema_short: int = 20,
        trend_ema_long: int = 50,
    ) -> None:
        self._ema_short = trend_ema_short
        self._ema_long = trend_ema_long

    def detect(
        self,
        benchmark_df: pd.DataFrame,
        vix_snapshot: Optional[VIXSnapshot] = None,
        advances: Optional[int] = None,
        declines: Optional[int] = None,
    ) -> RegimeSnapshot:
        """
        Classify the current market regime.

        Parameters
        ----------
        benchmark_df:
            Daily OHLCV DataFrame for the benchmark index (typically NIFTY 50).
            Needs at least 60 rows.
        vix_snapshot:
            Latest VIX reading. ``None`` if unavailable.
        advances:
            Number of advancing stocks today (for breadth).
        declines:
            Number of declining stocks today.

        Returns
        -------
        RegimeSnapshot:
            Classified regime with supporting scores.
        """
        close = benchmark_df["close"]

        # ── Trend score ──────────────────────────────────────────────────────
        if len(close) >= self._ema_long:
            ema_s = close.ewm(span=self._ema_short, adjust=False).mean().iloc[-1]
            ema_l = close.ewm(span=self._ema_long, adjust=False).mean().iloc[-1]
            current = close.iloc[-1]
            trend_score = float(
                (current - ema_l) / ema_l * 100
                + (ema_s - ema_l) / ema_l * 50
            )
        else:
            trend_score = 0.0

        # ── Volatility score ─────────────────────────────────────────────────
        if len(close) >= 21:
            vol_20d = close.pct_change().tail(20).std() * np.sqrt(252) * 100
            volatility_score = float(vol_20d)
        else:
            volatility_score = 15.0  # neutral assumption

        # ── Breadth score ────────────────────────────────────────────────────
        if advances is not None and declines is not None and (advances + declines) > 0:
            breadth_score = (advances - declines) / (advances + declines)
        else:
            breadth_score = 0.0

        # ── VIX adjustment ───────────────────────────────────────────────────
        vix_regime = vix_snapshot.regime if vix_snapshot else None

        # ── Regime classification ────────────────────────────────────────────
        if vix_snapshot and vix_snapshot.value >= 25:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = 0.8
            summary = f"High volatility environment — VIX at {vix_snapshot.value:.1f}"
        elif trend_score > 3 and breadth_score > 0.3:
            regime = MarketRegime.STRONG_BULL
            confidence = min(0.9, 0.6 + abs(trend_score) / 20)
            summary = "Strong uptrend with broad market participation"
        elif trend_score > 1:
            regime = MarketRegime.BULL
            confidence = 0.65
            summary = "Moderate uptrend"
        elif trend_score < -3 and breadth_score < -0.3:
            regime = MarketRegime.STRONG_BEAR
            confidence = min(0.9, 0.6 + abs(trend_score) / 20)
            summary = "Strong downtrend with broad market selling"
        elif trend_score < -1:
            regime = MarketRegime.BEAR
            confidence = 0.65
            summary = "Moderate downtrend"
        else:
            regime = MarketRegime.NEUTRAL
            confidence = 0.5
            summary = "Consolidating / range-bound market"

        return RegimeSnapshot(
            regime=regime,
            trend_score=round(trend_score, 4),
            volatility_score=round(volatility_score, 4),
            breadth_score=round(breadth_score, 4),
            vix_regime=vix_regime,
            confidence=round(confidence, 4),
            summary=summary,
        )
