"""
Signal generation brain.

Aggregates technical, options, news, and VIX signals into a unified
trading signal for any given index.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config.constants import (
    CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_HIGH, CONFIDENCE_VERY_HIGH
)
from config.settings import settings

logger = logging.getLogger(__name__)


class SignalDirection(str, Enum):
    """Direction of the generated signal."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalStrength(str, Enum):
    """Strength classification of the signal."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class TradingSignal:
    """
    A composite trading signal for a single index.

    Attributes
    ----------
    index_id:
        Registry ID of the index.
    direction:
        Overall signal direction.
    strength:
        Signal strength classification.
    confidence:
        Numeric confidence score [0, 1].
    technical_score:
        Contribution from technical indicators (−1 to +1).
    options_score:
        Contribution from options analysis (−1 to +1).
    news_score:
        Contribution from news sentiment (−1 to +1).
    vix_adjustment:
        Risk adjustment based on VIX regime (factor applied to confidence).
    reasons:
        Human-readable list of factors contributing to this signal.
    """

    index_id: str
    direction: SignalDirection
    strength: SignalStrength
    confidence: float
    technical_score: float
    options_score: float
    news_score: float
    vix_adjustment: float
    reasons: list[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        """Return True when confidence exceeds the alert threshold."""
        return self.confidence >= settings.telegram.alert_min_confidence


@dataclass
class SignalInputs:
    """
    All raw signals fed into the decision engine for one index.

    Parameters are all optional; missing values are treated as neutral (0).
    """

    index_id: str
    rsi: Optional[float] = None
    ema_short: Optional[float] = None
    ema_long: Optional[float] = None
    current_price: Optional[float] = None
    volume_vs_avg: Optional[float] = None
    pcr_oi: Optional[float] = None
    news_sentiment: Optional[float] = None
    vix_value: Optional[float] = None
    anomalies_count: int = 0


class DecisionEngine:
    """
    Converts raw market signals into a composite :class:`TradingSignal`.

    Weights for each signal component are configurable; defaults reflect
    typical importance in Indian index trading.

    Parameters
    ----------
    technical_weight:
        Weight for technical indicator score (default 0.40).
    options_weight:
        Weight for options chain score (default 0.35).
    news_weight:
        Weight for news sentiment score (default 0.25).
    """

    def __init__(
        self,
        technical_weight: float = 0.40,
        options_weight: float = 0.35,
        news_weight: float = 0.25,
    ) -> None:
        self._w_tech = technical_weight
        self._w_opts = options_weight
        self._w_news = news_weight
        self._thresholds = settings.thresholds

    # ── Score computation ─────────────────────────────────────────────────────

    def _technical_score(self, inputs: SignalInputs) -> tuple[float, list[str]]:
        """Return a score in [−1, 1] from technical indicators."""
        score = 0.0
        reasons: list[str] = []
        t = self._thresholds

        if inputs.rsi is not None:
            if inputs.rsi < t.rsi_oversold:
                score += 0.4
                reasons.append(f"RSI oversold ({inputs.rsi:.1f})")
            elif inputs.rsi > t.rsi_overbought:
                score -= 0.4
                reasons.append(f"RSI overbought ({inputs.rsi:.1f})")

        if inputs.ema_short is not None and inputs.ema_long is not None:
            if inputs.ema_short > inputs.ema_long:
                score += 0.3
                reasons.append("EMA short > EMA long (uptrend)")
            else:
                score -= 0.3
                reasons.append("EMA short < EMA long (downtrend)")

        if inputs.current_price is not None and inputs.ema_short is not None:
            if inputs.current_price > inputs.ema_short:
                score += 0.2
                reasons.append("Price above short EMA")
            else:
                score -= 0.2
                reasons.append("Price below short EMA")

        if inputs.volume_vs_avg is not None and inputs.volume_vs_avg > t.volume_spike_multiplier:
            reasons.append(f"Volume spike ({inputs.volume_vs_avg:.1f}x avg)")

        return max(-1.0, min(1.0, score)), reasons

    def _options_score(self, inputs: SignalInputs) -> tuple[float, list[str]]:
        """Return a score in [−1, 1] from options chain analysis."""
        score = 0.0
        reasons: list[str] = []
        t = self._thresholds

        if inputs.pcr_oi is not None:
            if inputs.pcr_oi < t.pcr_extreme_low:
                score += 0.6
                reasons.append(f"PCR low ({inputs.pcr_oi:.2f}) — bearish options positioning")
            elif inputs.pcr_oi > t.pcr_extreme_high:
                score -= 0.6
                reasons.append(f"PCR high ({inputs.pcr_oi:.2f}) — heavy put loading")

        if inputs.anomalies_count > 0:
            reasons.append(f"{inputs.anomalies_count} OI anomalies detected")

        return max(-1.0, min(1.0, score)), reasons

    def _news_score(self, inputs: SignalInputs) -> tuple[float, list[str]]:
        """Return a score in [−1, 1] from news sentiment."""
        score = inputs.news_sentiment or 0.0
        reasons: list[str] = []
        t = self._thresholds

        if score >= t.sentiment_bullish_threshold:
            reasons.append(f"Positive news sentiment ({score:.2f})")
        elif score <= t.sentiment_bearish_threshold:
            reasons.append(f"Negative news sentiment ({score:.2f})")

        return max(-1.0, min(1.0, score)), reasons

    def _vix_adjustment(self, vix: Optional[float]) -> tuple[float, list[str]]:
        """Return a confidence multiplier [0.5, 1.0] based on VIX level."""
        reasons: list[str] = []
        t = self._thresholds

        if vix is None:
            return 1.0, reasons
        if vix >= t.vix_extreme_threshold:
            reasons.append(f"Extreme fear: VIX {vix:.1f} — confidence reduced")
            return 0.5, reasons
        if vix >= t.vix_high_threshold:
            reasons.append(f"Elevated VIX ({vix:.1f}) — confidence reduced")
            return 0.75, reasons
        return 1.0, reasons

    # ── Main interface ────────────────────────────────────────────────────────

    def generate_signal(self, inputs: SignalInputs) -> TradingSignal:
        """
        Generate a composite trading signal from *inputs*.

        Parameters
        ----------
        inputs:
            All available signal inputs for one index.

        Returns
        -------
        TradingSignal:
            Fully populated composite signal.
        """
        tech_score, tech_reasons = self._technical_score(inputs)
        opts_score, opts_reasons = self._options_score(inputs)
        news_score, news_reasons = self._news_score(inputs)
        vix_adj, vix_reasons = self._vix_adjustment(inputs.vix_value)

        composite = (
            self._w_tech * tech_score
            + self._w_opts * opts_score
            + self._w_news * news_score
        )
        confidence = abs(composite) * vix_adj
        confidence = max(0.0, min(1.0, confidence))

        if composite > 0.1:
            direction = SignalDirection.BULLISH
        elif composite < -0.1:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        if confidence >= CONFIDENCE_VERY_HIGH:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= CONFIDENCE_HIGH:
            strength = SignalStrength.STRONG
        elif confidence >= CONFIDENCE_MEDIUM:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        all_reasons = tech_reasons + opts_reasons + news_reasons + vix_reasons

        return TradingSignal(
            index_id=inputs.index_id,
            direction=direction,
            strength=strength,
            confidence=round(confidence, 4),
            technical_score=round(tech_score, 4),
            options_score=round(opts_score, 4),
            news_score=round(news_score, 4),
            vix_adjustment=vix_adj,
            reasons=all_reasons,
        )
