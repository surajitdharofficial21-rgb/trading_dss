"""
Strategy replay engine.

Replays a sequence of signals against historical OHLCV data,
tracking trades and equity curve without look-ahead bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from config.constants import (
    STT_EQUITY_INTRADAY, EXCHANGE_TRANSACTION_CHARGE_NSE, GST_RATE,
    DEFAULT_BROKERAGE_FLAT, STAMP_DUTY_INTRADAY,
)
from src.backtest.metrics import PerformanceMetrics, calculate_metrics

logger = logging.getLogger(__name__)

# Strategy function signature: receives OHLCV slice, returns +1/0/-1
StrategyFn = Callable[[pd.DataFrame], int]


@dataclass
class Trade:
    """A single completed trade."""

    entry_date: str
    exit_date: str
    direction: int          # +1 long, -1 short
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    net_pnl: float
    return_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Full results of a backtest run."""

    index_id: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    trades: list[Trade]
    equity_curve: pd.Series
    metrics: PerformanceMetrics


class Backtester:
    """
    Event-driven backtester for index strategies.

    Iterates bar-by-bar through historical data, calling the strategy
    function on each bar to get a signal, then executes paper trades
    at the next bar's open (no look-ahead).

    Parameters
    ----------
    initial_capital:
        Starting capital in ₹.
    quantity:
        Fixed number of units/lots per trade.
    commission_per_trade:
        Fixed commission per trade leg in ₹ (default ₹20).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        quantity: int = 1,
        commission_per_trade: float = DEFAULT_BROKERAGE_FLAT,
    ) -> None:
        self._capital = initial_capital
        self._quantity = quantity
        self._commission = commission_per_trade

    def run(
        self,
        index_id: str,
        ohlcv: pd.DataFrame,
        strategy: StrategyFn,
        warmup_bars: int = 50,
    ) -> BacktestResult:
        """
        Run a backtest of *strategy* on *ohlcv*.

        Parameters
        ----------
        index_id:
            Label for the result.
        ohlcv:
            Historical OHLCV with lowercase columns, DatetimeIndex.
        strategy:
            Callable that takes the OHLCV slice up to bar ``i`` and
            returns: ``+1`` (buy), ``-1`` (sell), ``0`` (hold/no trade).
        warmup_bars:
            Number of initial bars given to the strategy without trading,
            to allow indicator initialisation.

        Returns
        -------
        BacktestResult:
            Trades, equity curve, and performance metrics.
        """
        capital = self._capital
        equity: list[float] = [capital]
        dates: list[str] = [str(ohlcv.index[0].date())]
        trades: list[Trade] = []

        position: int = 0
        entry_price: float = 0.0
        entry_date: str = ""

        for i in range(warmup_bars, len(ohlcv) - 1):
            bar_slice = ohlcv.iloc[: i + 1]
            signal = strategy(bar_slice)
            next_bar = ohlcv.iloc[i + 1]
            exec_price = float(next_bar["open"])
            exec_date = str(next_bar.name.date())  # type: ignore[union-attr]

            # Close existing position on opposing signal
            if position != 0 and signal != 0 and signal != position:
                pnl = (exec_price - entry_price) * position * self._quantity
                costs = self._commission * 2
                net = pnl - costs
                capital += net
                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=exec_date,
                    direction=position,
                    entry_price=entry_price,
                    exit_price=exec_price,
                    quantity=self._quantity,
                    gross_pnl=round(pnl, 2),
                    net_pnl=round(net, 2),
                    return_pct=round(pnl / (entry_price * self._quantity) * 100, 4),
                    exit_reason="signal_reversal",
                ))
                position = 0

            # Open new position
            if position == 0 and signal != 0:
                position = signal
                entry_price = exec_price
                entry_date = exec_date

            equity.append(capital)
            dates.append(exec_date)

        # Close any open position at the last bar
        if position != 0:
            last = ohlcv.iloc[-1]
            close_price = float(last["close"])
            pnl = (close_price - entry_price) * position * self._quantity
            net = pnl - self._commission * 2
            capital += net
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=str(last.name.date()),  # type: ignore[union-attr]
                direction=position,
                entry_price=entry_price,
                exit_price=close_price,
                quantity=self._quantity,
                gross_pnl=round(pnl, 2),
                net_pnl=round(net, 2),
                return_pct=round(pnl / (entry_price * self._quantity) * 100, 4),
                exit_reason="end_of_data",
            ))

        equity_series = pd.Series(equity, index=pd.to_datetime(dates))
        trade_returns = pd.Series([t.return_pct / 100 for t in trades]) if trades else None

        metrics = calculate_metrics(equity_series, trade_returns)

        return BacktestResult(
            index_id=index_id,
            start_date=str(ohlcv.index[0].date()),
            end_date=str(ohlcv.index[-1].date()),
            initial_capital=self._capital,
            final_capital=round(capital, 2),
            trades=trades,
            equity_curve=equity_series,
            metrics=metrics,
        )
