"""
Unusual market activity detection.

Detects volume spikes, OI anomalies, and unusual price moves
relative to recent history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Classification of a detected anomaly."""

    VOLUME_SPIKE = "volume_spike"
    OI_SPIKE = "oi_spike"
    PRICE_GAP = "price_gap"
    UNUSUAL_MOVE = "unusual_move"
    VIX_SPIKE = "vix_spike"


@dataclass
class Anomaly:
    """
    A single detected anomaly.

    Attributes
    ----------
    index_id:
        Registry ID of the affected index.
    anomaly_type:
        Classification of the anomaly.
    value:
        Observed value (volume, OI, price, etc.).
    baseline:
        Expected value (rolling mean or threshold).
    magnitude:
        How many times the baseline was exceeded.
    description:
        Human-readable description.
    """

    index_id: str
    anomaly_type: AnomalyType
    value: float
    baseline: float
    magnitude: float
    description: str


class AnomalyDetector:
    """
    Detects unusual activity in price, volume, and options data.

    Parameters
    ----------
    volume_spike_multiplier:
        Flag volume when > this multiple of rolling 20-day mean.
        Defaults to ``settings.thresholds.volume_spike_multiplier``.
    oi_spike_threshold:
        Flag OI change % above this value.
        Defaults to ``settings.thresholds.oi_spike_threshold``.
    """

    def __init__(
        self,
        volume_spike_multiplier: Optional[float] = None,
        oi_spike_threshold: Optional[float] = None,
    ) -> None:
        t = settings.thresholds
        self._vol_multiplier = volume_spike_multiplier or t.volume_spike_multiplier
        self._oi_threshold = oi_spike_threshold or t.oi_spike_threshold

    def detect_volume_spike(
        self,
        ohlcv_df: pd.DataFrame,
        index_id: str,
        lookback: int = 20,
    ) -> Optional[Anomaly]:
        """
        Detect if the latest bar has unusually high volume.

        Parameters
        ----------
        ohlcv_df:
            Daily OHLCV DataFrame with ``volume`` column, at least
            ``lookback + 1`` rows.
        index_id:
            Registry ID (for labelling).
        lookback:
            Number of prior bars used to compute baseline volume.
        """
        if len(ohlcv_df) < lookback + 1:
            return None

        baseline = ohlcv_df["volume"].iloc[-lookback - 1:-1].mean()
        current = float(ohlcv_df["volume"].iloc[-1])

        if current > self._vol_multiplier * baseline:
            magnitude = current / baseline
            return Anomaly(
                index_id=index_id,
                anomaly_type=AnomalyType.VOLUME_SPIKE,
                value=current,
                baseline=baseline,
                magnitude=magnitude,
                description=(
                    f"{index_id}: volume {current:,.0f} is "
                    f"{magnitude:.1f}x the {lookback}-day average ({baseline:,.0f})"
                ),
            )
        return None

    def detect_oi_spike(
        self,
        chain_df: pd.DataFrame,
        index_id: str,
        column: str = "ce_chg_oi",
    ) -> list[Anomaly]:
        """
        Detect strikes with abnormal OI build-up in an options chain.

        Parameters
        ----------
        chain_df:
            Options chain DataFrame.
        index_id:
            Registry ID.
        column:
            Column to inspect (``"ce_chg_oi"`` or ``"pe_chg_oi"``).
        """
        if column not in chain_df.columns or chain_df.empty:
            return []

        mean_oi = chain_df[column].abs().mean()
        anomalies = []
        for _, row in chain_df[chain_df[column].abs() > self._oi_threshold * mean_oi].iterrows():
            value = float(row[column])
            magnitude = abs(value) / mean_oi if mean_oi > 0 else 0.0
            anomalies.append(Anomaly(
                index_id=index_id,
                anomaly_type=AnomalyType.OI_SPIKE,
                value=value,
                baseline=mean_oi,
                magnitude=magnitude,
                description=(
                    f"{index_id}: strike {row.get('strike')} — "
                    f"{column} change of {value:,.0f} is {magnitude:.1f}x mean"
                ),
            ))
        return anomalies

    def detect_price_gap(
        self,
        ohlcv_df: pd.DataFrame,
        index_id: str,
        gap_threshold_pct: float = 0.5,
    ) -> Optional[Anomaly]:
        """
        Detect a price gap between yesterday's close and today's open.

        Parameters
        ----------
        ohlcv_df:
            Daily OHLCV with at least 2 rows.
        index_id:
            Registry ID.
        gap_threshold_pct:
            Minimum gap size as % of previous close to report.
        """
        if len(ohlcv_df) < 2:
            return None
        prev_close = float(ohlcv_df["close"].iloc[-2])
        today_open = float(ohlcv_df["open"].iloc[-1])
        gap_pct = abs(today_open - prev_close) / prev_close * 100

        if gap_pct >= gap_threshold_pct:
            direction = "up" if today_open > prev_close else "down"
            return Anomaly(
                index_id=index_id,
                anomaly_type=AnomalyType.PRICE_GAP,
                value=today_open,
                baseline=prev_close,
                magnitude=gap_pct,
                description=(
                    f"{index_id}: gap {direction} of {gap_pct:.2f}% "
                    f"(prev close {prev_close:.2f} → open {today_open:.2f})"
                ),
            )
        return None
