"""
Backtest API endpoints.

POST /api/v1/backtest/run    — run a backtest for an index
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_index_registry
from src.data.index_registry import IndexRegistry

router = APIRouter()
logger = logging.getLogger(__name__)


class BacktestRequest(BaseModel):
    """Request body for a backtest run."""

    index_id: str = Field(..., description="Index registry ID")
    start_date: str = Field(..., description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="End date YYYY-MM-DD")
    strategy: str = Field(default="ema_crossover", description="Strategy name")
    initial_capital: float = Field(default=100_000.0, ge=1_000.0)
    quantity: int = Field(default=1, ge=1)


@router.post("/run", summary="Run a backtest")
async def run_backtest(
    request: BacktestRequest,
    registry: IndexRegistry = Depends(get_index_registry),
) -> dict:
    """
    Run a backtest for the requested index and strategy.

    Downloads historical data via yfinance and runs the named strategy
    through the backtesting engine.
    """
    defn = registry.get_or_none(request.index_id.upper())
    if defn is None:
        raise HTTPException(status_code=404, detail=f"Index not found: {request.index_id!r}")
    if not defn.yahoo_symbol:
        raise HTTPException(
            status_code=400,
            detail=f"Index {request.index_id} has no Yahoo symbol — cannot download historical data",
        )

    try:
        from datetime import date as date_type
        start = date_type.fromisoformat(request.start_date)
        end = date_type.fromisoformat(request.end_date) if request.end_date else date_type.today()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date format: {exc}")

    from src.data.historical_data import HistoricalDataFetcher
    from src.backtest.backtester import Backtester
    from src.analysis.technical import ema as compute_ema

    try:
        fetcher = HistoricalDataFetcher(registry)
        ohlcv = fetcher.fetch(request.index_id.upper(), start, end)

        def ema_crossover_strategy(df):
            if len(df) < 50:
                return 0
            short_ema = compute_ema(df["close"], 20).iloc[-1]
            long_ema = compute_ema(df["close"], 50).iloc[-1]
            if short_ema > long_ema:
                return 1
            elif short_ema < long_ema:
                return -1
            return 0

        bt = Backtester(
            initial_capital=request.initial_capital,
            quantity=request.quantity,
        )
        result = bt.run(request.index_id.upper(), ohlcv, ema_crossover_strategy)

        return {
            "index_id": result.index_id,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_trades": result.metrics.total_trades,
            "metrics": {
                "total_return_pct": result.metrics.total_return_pct,
                "cagr_pct": result.metrics.cagr_pct,
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "sortino_ratio": result.metrics.sortino_ratio,
                "max_drawdown_pct": result.metrics.max_drawdown_pct,
                "win_rate_pct": result.metrics.win_rate_pct,
                "profit_factor": result.metrics.profit_factor,
                "calmar_ratio": result.metrics.calmar_ratio,
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Backtest failed for %s: %s", request.index_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))
