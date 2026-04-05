"""
Options chain analysis: OI interpretation, PCR, and Max Pain calculation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class OIAnalysis:
    """
    Result of an open-interest analysis pass.

    Attributes
    ----------
    pcr_oi:
        Put-Call Ratio based on open interest (total PE OI / total CE OI).
    pcr_volume:
        Put-Call Ratio based on volume.
    max_pain:
        Strike price at which aggregate option buyer loss is maximum.
    atm_strike:
        At-the-money strike closest to current underlying.
    ce_oi_spikes:
        Strikes with unusual CE OI build-up.
    pe_oi_spikes:
        Strikes with unusual PE OI build-up.
    sentiment:
        Derived sentiment: ``"bullish"`` | ``"bearish"`` | ``"neutral"``.
    """

    pcr_oi: float
    pcr_volume: float
    max_pain: float
    atm_strike: float
    ce_oi_spikes: list[float]
    pe_oi_spikes: list[float]
    sentiment: str


def calculate_pcr(chain_df: pd.DataFrame) -> tuple[float, float]:
    """
    Calculate Put-Call Ratio from an options chain DataFrame.

    Parameters
    ----------
    chain_df:
        DataFrame with columns ``ce_oi``, ``pe_oi``, ``ce_volume``, ``pe_volume``.

    Returns
    -------
    tuple[float, float]:
        ``(pcr_oi, pcr_volume)`` — ratios of total PE to total CE.
    """
    total_ce_oi = chain_df["ce_oi"].sum()
    total_pe_oi = chain_df["pe_oi"].sum()
    total_ce_vol = chain_df.get("ce_volume", pd.Series(dtype=float)).sum()
    total_pe_vol = chain_df.get("pe_volume", pd.Series(dtype=float)).sum()

    pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0.0
    pcr_volume = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0.0
    return round(pcr_oi, 4), round(pcr_volume, 4)


def calculate_max_pain(chain_df: pd.DataFrame) -> float:
    """
    Calculate Max Pain strike.

    Max Pain is the strike at which total option writer profit (= buyer loss)
    is maximised on expiry.

    Parameters
    ----------
    chain_df:
        Options chain with columns ``strike``, ``ce_oi``, ``pe_oi``.

    Returns
    -------
    float:
        Max Pain strike price.
    """
    strikes = chain_df["strike"].sort_values().unique()
    losses: dict[float, float] = {}

    for expiry_strike in strikes:
        ce_loss = sum(
            max(expiry_strike - s, 0) * oi
            for s, oi in zip(chain_df["strike"], chain_df["ce_oi"])
        )
        pe_loss = sum(
            max(s - expiry_strike, 0) * oi
            for s, oi in zip(chain_df["strike"], chain_df["pe_oi"])
        )
        losses[expiry_strike] = ce_loss + pe_loss

    if not losses:
        return 0.0
    return float(min(losses, key=losses.__getitem__))


def find_oi_spikes(
    chain_df: pd.DataFrame,
    column: str,
    multiplier: float | None = None,
) -> list[float]:
    """
    Find strikes with abnormally high OI compared to the mean.

    Parameters
    ----------
    chain_df:
        Options chain DataFrame.
    column:
        ``"ce_oi"`` or ``"pe_oi"``.
    multiplier:
        Flag strike if OI > multiplier × mean. Defaults to
        ``settings.thresholds.oi_spike_threshold``.

    Returns
    -------
    list[float]:
        Strike prices where OI is anomalously high.
    """
    threshold = multiplier or settings.thresholds.oi_spike_threshold
    mean_oi = chain_df[column].mean()
    spikes = chain_df[chain_df[column] > threshold * mean_oi]["strike"]
    return sorted(spikes.tolist())


def analyse_chain(
    chain_df: pd.DataFrame,
    underlying_value: float,
) -> OIAnalysis:
    """
    Run a full options chain analysis.

    Parameters
    ----------
    chain_df:
        Near-expiry options chain DataFrame (output of
        :meth:`~src.data.options_chain.OptionsChainFetcher.get_near_expiry`).
    underlying_value:
        Current spot price of the underlying index.

    Returns
    -------
    OIAnalysis:
        Comprehensive options analysis result.
    """
    if chain_df.empty:
        raise ValueError("Cannot analyse an empty options chain")

    # ATM strike = strike closest to current underlying
    atm = float(chain_df["strike"].sub(underlying_value).abs().idxmin())
    atm_strike = float(chain_df.loc[atm, "strike"])

    pcr_oi, pcr_volume = calculate_pcr(chain_df)
    max_pain = calculate_max_pain(chain_df)
    ce_spikes = find_oi_spikes(chain_df, "ce_oi")
    pe_spikes = find_oi_spikes(chain_df, "pe_oi")

    thresholds = settings.thresholds
    if pcr_oi > thresholds.pcr_extreme_high:
        sentiment = "bearish"
    elif pcr_oi < thresholds.pcr_extreme_low:
        sentiment = "bullish"
    else:
        sentiment = "neutral"

    return OIAnalysis(
        pcr_oi=pcr_oi,
        pcr_volume=pcr_volume,
        max_pain=max_pain,
        atm_strike=atm_strike,
        ce_oi_spikes=ce_spikes,
        pe_oi_spikes=pe_spikes,
        sentiment=sentiment,
    )
