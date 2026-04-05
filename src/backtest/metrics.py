"""
Performance metrics for backtesting results.

All metrics follow standard quantitative finance definitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceMetrics:
    """
    Comprehensive set of strategy performance metrics.

    Attributes
    ----------
    total_return_pct:
        Total return of the strategy (%).
    cagr_pct:
        Compounded Annual Growth Rate (%).
    sharpe_ratio:
        Risk-adjusted return (annualised, assuming rf=0).
    sortino_ratio:
        Downside risk-adjusted return.
    max_drawdown_pct:
        Maximum peak-to-trough drawdown (%).
    win_rate_pct:
        Percentage of winning trades.
    profit_factor:
        Gross profit / Gross loss.
    avg_win_pct:
        Average winning trade return (%).
    avg_loss_pct:
        Average losing trade return (%).
    total_trades:
        Total number of completed trades.
    calmar_ratio:
        CAGR / Max Drawdown (absolute).
    """

    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    total_trades: int
    calmar_ratio: float


def calculate_metrics(
    equity_curve: pd.Series,
    trade_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Compute performance metrics from an equity curve.

    Parameters
    ----------
    equity_curve:
        Series of portfolio values indexed by date.
    trade_returns:
        Optional per-trade return series. If provided, enables
        win rate and profit factor calculations.
    risk_free_rate:
        Annualised risk-free rate as fraction (default 0.0).

    Returns
    -------
    PerformanceMetrics:
        Complete set of performance metrics.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        raise ValueError("equity_curve must have at least 2 data points")

    initial = float(equity_curve.iloc[0])
    final = float(equity_curve.iloc[-1])
    total_return = (final / initial - 1) * 100

    n_days = len(equity_curve)
    years = n_days / _TRADING_DAYS_PER_YEAR
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    daily_returns = equity_curve.pct_change().dropna()
    daily_rf = risk_free_rate / _TRADING_DAYS_PER_YEAR
    excess_returns = daily_returns - daily_rf

    sharpe = (
        float(excess_returns.mean() / excess_returns.std() * np.sqrt(_TRADING_DAYS_PER_YEAR))
        if excess_returns.std() > 0 else 0.0
    )

    downside = excess_returns[excess_returns < 0]
    sortino = (
        float(excess_returns.mean() / downside.std() * np.sqrt(_TRADING_DAYS_PER_YEAR))
        if len(downside) > 0 and downside.std() > 0 else 0.0
    )

    # Max drawdown
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max * 100
    max_dd = float(drawdown.min())

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Per-trade stats
    if trade_returns is not None and len(trade_returns) > 0:
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns <= 0]
        win_rate = len(wins) / len(trade_returns) * 100
        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_win = float(wins.mean() * 100) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean() * 100) if len(losses) > 0 else 0.0
        total_trades = len(trade_returns)
    else:
        win_rate = profit_factor = avg_win = avg_loss = 0.0
        total_trades = 0

    return PerformanceMetrics(
        total_return_pct=round(total_return, 4),
        cagr_pct=round(cagr, 4),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown_pct=round(max_dd, 4),
        win_rate_pct=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
        avg_win_pct=round(avg_win, 4),
        avg_loss_pct=round(avg_loss, 4),
        total_trades=total_trades,
        calmar_ratio=round(calmar, 4),
    )
