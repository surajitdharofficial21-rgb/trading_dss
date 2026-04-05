"""
Sector rotation and relative strength analysis across all Indian indices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.data.index_registry import IndexRegistry

logger = logging.getLogger(__name__)


@dataclass
class RelativeStrengthScore:
    """
    Relative strength of one index versus a benchmark.

    Attributes
    ----------
    index_id:
        Index being scored.
    benchmark_id:
        Benchmark index (usually ``"NIFTY50"``).
    rs_1d:
        1-day relative return vs benchmark.
    rs_5d:
        5-day relative return vs benchmark.
    rs_20d:
        20-day relative return vs benchmark.
    trend:
        ``"outperforming"`` | ``"underperforming"`` | ``"inline"``.
    """

    index_id: str
    benchmark_id: str
    rs_1d: float
    rs_5d: float
    rs_20d: float
    trend: str


class SectorAnalyzer:
    """
    Compute sector rotation signals and relative strength scores.

    Parameters
    ----------
    registry:
        Loaded :class:`~src.data.index_registry.IndexRegistry`.
    benchmark_id:
        Index ID to use as the market benchmark (default ``"NIFTY50"``).
    """

    def __init__(
        self,
        registry: IndexRegistry,
        benchmark_id: str = "NIFTY50",
    ) -> None:
        self._registry = registry
        self._benchmark_id = benchmark_id

    def relative_strength(
        self,
        prices: dict[str, pd.Series],
        periods: tuple[int, int, int] = (1, 5, 20),
    ) -> list[RelativeStrengthScore]:
        """
        Compute relative strength scores for all available sectoral indices.

        Parameters
        ----------
        prices:
            Mapping of ``{index_id: close_series}``. Must include benchmark.
        periods:
            Lookback periods for RS calculation (default: 1, 5, 20 days).

        Returns
        -------
        list[RelativeStrengthScore]:
            One entry per index, sorted by 20-day RS descending.
        """
        if self._benchmark_id not in prices:
            logger.error("Benchmark %s not in price data", self._benchmark_id)
            return []

        bench = prices[self._benchmark_id]
        p1, p5, p20 = periods
        results: list[RelativeStrengthScore] = []

        for index_id, series in prices.items():
            if index_id == self._benchmark_id:
                continue

            # Align on common dates
            aligned = pd.concat([series, bench], axis=1, keys=["idx", "bench"]).dropna()
            if len(aligned) < p20 + 1:
                continue

            def _rs(n: int) -> float:
                idx_ret = (aligned["idx"].iloc[-1] / aligned["idx"].iloc[-(n + 1)] - 1) * 100
                bench_ret = (aligned["bench"].iloc[-1] / aligned["bench"].iloc[-(n + 1)] - 1) * 100
                return round(idx_ret - bench_ret, 4)

            rs_1d = _rs(p1)
            rs_5d = _rs(p5)
            rs_20d = _rs(p20)

            # Determine trend from 20-day and 5-day RS
            if rs_20d > 0.5 and rs_5d > 0:
                trend = "outperforming"
            elif rs_20d < -0.5 and rs_5d < 0:
                trend = "underperforming"
            else:
                trend = "inline"

            results.append(RelativeStrengthScore(
                index_id=index_id,
                benchmark_id=self._benchmark_id,
                rs_1d=rs_1d,
                rs_5d=rs_5d,
                rs_20d=rs_20d,
                trend=trend,
            ))

        results.sort(key=lambda r: r.rs_20d, reverse=True)
        return results

    def sector_heatmap(
        self, prices: dict[str, pd.Series], period_days: int = 5
    ) -> pd.DataFrame:
        """
        Return a DataFrame of percentage returns over *period_days* for all sectoral indices.

        Parameters
        ----------
        prices:
            Mapping of ``{index_id: close_series}``.
        period_days:
            Lookback period.

        Returns
        -------
        pd.DataFrame:
            Columns: ``index_id``, ``display_name``, ``sector_category``, ``return_pct``.
        """
        rows = []
        for index_id, series in prices.items():
            if len(series) < period_days + 1:
                continue
            definition = self._registry.get_or_none(index_id)
            if definition is None or not definition.is_active:
                continue
            ret_pct = (series.iloc[-1] / series.iloc[-(period_days + 1)] - 1) * 100
            rows.append({
                "index_id": index_id,
                "display_name": definition.display_name,
                "sector_category": definition.sector_category,
                "return_pct": round(float(ret_pct), 4),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("return_pct", ascending=False).reset_index(drop=True)
        return df
