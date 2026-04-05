"""
CLI backtest runner.

Usage:
    python scripts/run_backtest.py --index NIFTY50 --start 2022-01-01 --end 2024-01-01
    python scripts/run_backtest.py --index BANKNIFTY --start 2023-01-01 --capital 500000
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging
from config.settings import settings
from src.data.index_registry import IndexRegistry
from src.data.historical_data import HistoricalDataFetcher
from src.backtest.backtester import Backtester
from src.analysis.technical import ema


def ema_crossover(short: int = 20, long: int = 50):
    """Return an EMA crossover strategy for given periods."""
    def strategy(df):
        if len(df) < long:
            return 0
        s = ema(df["close"], short).iloc[-1]
        l = ema(df["close"], long).iloc[-1]
        return 1 if s > l else -1
    return strategy


def main() -> None:
    setup_logging(log_dir=settings.logging.log_dir)

    parser = argparse.ArgumentParser(description="Run a backtest")
    parser.add_argument("--index", required=True, help="Index ID (e.g. NIFTY50)")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default today)")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Initial capital ₹")
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--short-ema", type=int, default=20)
    parser.add_argument("--long-ema", type=int, default=50)
    args = parser.parse_args()

    registry = IndexRegistry.from_file(settings.indices_config_path)
    fetcher = HistoricalDataFetcher(registry)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()

    print(f"Downloading {args.index} history from {start} to {end}…")
    ohlcv = fetcher.fetch(args.index.upper(), start, end)
    print(f"  {len(ohlcv)} bars downloaded")

    bt = Backtester(initial_capital=args.capital, quantity=args.quantity)
    result = bt.run(
        args.index.upper(), ohlcv,
        ema_crossover(args.short_ema, args.long_ema),
    )

    m = result.metrics
    print(f"\n{'─' * 50}")
    print(f"  BACKTEST RESULTS — {result.index_id}")
    print(f"  Period : {result.start_date} → {result.end_date}")
    print(f"{'─' * 50}")
    print(f"  Capital      : ₹{result.initial_capital:>12,.2f} → ₹{result.final_capital:>12,.2f}")
    print(f"  Total Return : {m.total_return_pct:>+.2f}%")
    print(f"  CAGR         : {m.cagr_pct:>+.2f}%")
    print(f"  Sharpe Ratio : {m.sharpe_ratio:>.3f}")
    print(f"  Sortino      : {m.sortino_ratio:>.3f}")
    print(f"  Max Drawdown : {m.max_drawdown_pct:.2f}%")
    print(f"  Win Rate     : {m.win_rate_pct:.1f}%")
    print(f"  Profit Factor: {m.profit_factor:.2f}")
    print(f"  Total Trades : {m.total_trades}")
    print(f"{'─' * 50}\n")


if __name__ == "__main__":
    main()
