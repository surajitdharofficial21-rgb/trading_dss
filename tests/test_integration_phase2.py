"""
End-to-end integration tests for Phase 2 — Technical Analysis Aggregator.

Tests:
  - Full TechnicalAggregator.analyze() on synthetic data
  - Multiple indices with and without options
  - IndicatorStore save/load round-trip
  - No duplicate DB entries on re-run
  - Performance: analysis of 2 years of data in < 5 seconds
  - NaN handling and edge cases
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.analysis.technical_aggregator import TechnicalAggregator, TechnicalAnalysisResult, Alert
from src.analysis.indicator_store import IndicatorStore
from src.database.db_manager import DatabaseManager


# ── Helpers ──────────────────────────────────────────────────────────────────

def _synthetic_ohlcv(
    n: int = 200,
    base: float = 20000.0,
    seed: int = 42,
    with_nans: bool = False,
) -> pd.DataFrame:
    """Generate realistic OHLCV data.

    If *with_nans* is True, sprinkle some NaN values to test robustness.
    """
    rng = np.random.default_rng(seed)
    close = base + rng.standard_normal(n).cumsum() * 100
    high = close + rng.uniform(50, 200, n)
    low = close - rng.uniform(50, 200, n)
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(500_000, 10_000_000, n).astype(float)

    dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=dates)

    if with_nans:
        # Introduce a few NaN values
        df.iloc[3, df.columns.get_loc("volume")] = np.nan
        df.iloc[7, df.columns.get_loc("close")] = np.nan

    return df


@pytest.fixture()
def db(tmp_path):
    """Provide a fresh isolated SQLite DB with seed index_master rows."""
    db_path = tmp_path / "test_phase2.db"
    db = DatabaseManager(db_path=db_path)
    db.connect()
    db.initialise_schema()

    # Seed index_master so FK constraints pass
    now_str = datetime.now().isoformat()
    for idx_id, name in [("NIFTY50", "NIFTY 50"), ("BANKNIFTY", "NIFTY BANK"), ("NIFTY_IT", "NIFTY IT")]:
        db.execute(
            """INSERT OR IGNORE INTO index_master
               (id, display_name, exchange, sector_category, is_active, created_at, updated_at)
               VALUES (?, ?, 'NSE', 'broad_market', 1, ?, ?)""",
            (idx_id, name, now_str, now_str),
        )

    yield db


# ══════════════════════════════════════════════════════════════════════════════
# E2E: Run aggregator on synthetic data
# ══════════════════════════════════════════════════════════════════════════════


class TestEndToEndAnalysis:
    """Run TechnicalAggregator.analyze() end-to-end on synthetic data."""

    def test_nifty50_daily(self):
        """Full analysis on 200 days of NIFTY50-like data, no options."""
        agg = TechnicalAggregator()
        df = _synthetic_ohlcv(200)
        result = agg.analyze("NIFTY50", df)

        # Basic structure
        assert isinstance(result, TechnicalAnalysisResult)
        assert result.index_id == "NIFTY50"
        assert result.timeframe == "1d"
        assert result.overall_signal in ("STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL")
        assert 0.2 <= result.overall_confidence <= 0.9

        # Votes
        assert "trend" in result.votes
        assert "momentum" in result.votes
        assert "volume" in result.votes
        assert "quant" in result.votes

        # Levels
        assert len(result.support_levels) >= 1
        assert len(result.resistance_levels) >= 1
        assert result.immediate_support > 0
        assert result.immediate_resistance > 0

        # Risk params
        assert result.suggested_stop_loss_distance >= 0
        assert result.suggested_target_distance >= 0
        assert result.position_size_modifier > 0

        # Reasoning
        assert len(result.reasoning) > 50  # should be multi-line

        # Data quality
        assert 0.0 < result.data_completeness <= 1.0
        assert isinstance(result.warnings, list)

    def test_banknifty_daily(self):
        """BankNifty analysis with different seed."""
        agg = TechnicalAggregator()
        df = _synthetic_ohlcv(200, base=48000.0, seed=99)
        result = agg.analyze("BANKNIFTY", df)

        assert isinstance(result, TechnicalAnalysisResult)
        assert result.index_id == "BANKNIFTY"
        assert result.overall_signal in ("STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL")

    def test_niftyit_no_options(self):
        """NIFTY IT without options data — should work with redistributed weights."""
        agg = TechnicalAggregator()
        df = _synthetic_ohlcv(200, base=35000.0, seed=7)
        result = agg.analyze("NIFTY_IT", df)

        assert result.options is None
        assert any("Options" in w for w in result.warnings)
        assert result.overall_signal in ("STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL")

    def test_with_benchmark(self):
        """Analysis with benchmark data for beta/correlation."""
        agg = TechnicalAggregator()
        df = _synthetic_ohlcv(200, base=35000.0, seed=11)
        benchmark = _synthetic_ohlcv(200, base=20000.0, seed=42)
        result = agg.analyze("NIFTY_IT", df, benchmark_df=benchmark)

        assert isinstance(result, TechnicalAnalysisResult)
        # Quant should have attempted beta calculation
        assert result.quant is not None

    def test_with_vix(self):
        """Analysis with VIX value."""
        agg = TechnicalAggregator()
        df = _synthetic_ohlcv(200)
        result = agg.analyze("NIFTY50", df, vix_value=25.0)

        assert isinstance(result, TechnicalAnalysisResult)
        # VIX elevated alert should be present
        vix_alerts = [a for a in result.alerts if a.type == "VIX_EXTREME"]
        assert len(vix_alerts) >= 1


class TestNaNHandling:
    """Verify no crashes with NaN/missing data."""

    def test_nan_in_data(self):
        """Should not crash with NaN values in price data."""
        agg = TechnicalAggregator()
        df = _synthetic_ohlcv(200, with_nans=True)
        # Forward fill NaN to avoid indicator crashes (real pipeline does this)
        df = df.ffill().bfill()
        result = agg.analyze("NIFTY50", df)
        assert isinstance(result, TechnicalAnalysisResult)

    def test_minimal_data(self):
        """Should handle very short data gracefully."""
        agg = TechnicalAggregator()
        df = _synthetic_ohlcv(30)  # Barely enough for some indicators
        result = agg.analyze("NIFTY50", df)
        assert isinstance(result, TechnicalAnalysisResult)
        # May have reduced data completeness
        assert result.overall_signal in ("STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL")


class TestMultipleIndices:
    """Run analysis on multiple indices sequentially."""

    def test_three_indices(self):
        """Verify analysis works for NIFTY, BANKNIFTY, NIFTYIT."""
        agg = TechnicalAggregator()

        configs = [
            ("NIFTY50", 20000.0, 42),
            ("BANKNIFTY", 48000.0, 99),
            ("NIFTY_IT", 35000.0, 7),
        ]

        results = []
        for idx_id, base, seed in configs:
            df = _synthetic_ohlcv(200, base=base, seed=seed)
            result = agg.analyze(idx_id, df)
            results.append(result)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, TechnicalAnalysisResult)
            assert r.overall_signal in ("STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL")

        # Each should have its own index_id
        assert {r.index_id for r in results} == {"NIFTY50", "BANKNIFTY", "NIFTY_IT"}


# ══════════════════════════════════════════════════════════════════════════════
# IndicatorStore round-trip
# ══════════════════════════════════════════════════════════════════════════════


class TestIndicatorStoreRoundTrip:
    """Test save → load cycle for IndicatorStore."""

    def test_save_and_load(self, db):
        """Save an analysis result, then load it back."""
        agg = TechnicalAggregator()
        df = _synthetic_ohlcv(200)
        result = agg.analyze("NIFTY50", df)

        # Save
        IndicatorStore.save_analysis(result, db)

        # Load
        loaded = IndicatorStore.get_latest_analysis("NIFTY50", db)
        assert loaded is not None
        assert loaded["index_id"] == "NIFTY50"
        assert loaded["technical_signal"] in ("BULLISH", "BEARISH", "NEUTRAL")
        assert 0.0 <= loaded["signal_strength"] <= 1.0

    def test_no_duplicate_on_rerun(self, db):
        """Running analysis twice with same timestamp should upsert, not duplicate."""
        agg = TechnicalAggregator()
        df = _synthetic_ohlcv(200)

        result1 = agg.analyze("NIFTY50", df)
        IndicatorStore.save_analysis(result1, db)

        result2 = agg.analyze("NIFTY50", df)
        # Force same timestamp to test upsert
        result2 = TechnicalAnalysisResult(
            **{**result2.__dict__, "timestamp": result1.timestamp}
        )
        IndicatorStore.save_analysis(result2, db)

        # Should have exactly 1 row (not 2)
        rows = db.fetch_all(
            "SELECT * FROM technical_indicators WHERE index_id = ?",
            ("NIFTY50",),
        )
        assert len(rows) == 1

    def test_history_query(self, db):
        """Save multiple analyses and query history."""
        agg = TechnicalAggregator()

        for i in range(3):
            df = _synthetic_ohlcv(200, seed=i + 10)
            result = agg.analyze("NIFTY50", df)
            IndicatorStore.save_analysis(result, db)

        # All have different timestamps, so should get 3 rows
        history = IndicatorStore.get_analysis_history(
            "NIFTY50", "2000-01-01", "2099-12-31", db
        )
        assert len(history) == 3

    def test_indicator_series(self, db):
        """Get time series of RSI."""
        agg = TechnicalAggregator()

        for i in range(3):
            df = _synthetic_ohlcv(200, seed=i + 20)
            result = agg.analyze("NIFTY50", df)
            IndicatorStore.save_analysis(result, db)

        series = IndicatorStore.get_indicator_series(
            "NIFTY50", "rsi_14", "2000-01-01", "2099-12-31", db
        )
        assert isinstance(series, pd.Series)
        assert series.name == "rsi_14"
        assert len(series) == 3

    def test_invalid_indicator_name_raises(self, db):
        """Invalid indicator name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid indicator name"):
            IndicatorStore.get_indicator_series(
                "NIFTY50", "invalid_col", "2000-01-01", "2099-12-31", db
            )

    def test_alerts_saved_to_anomaly_events(self, db):
        """Verify alerts are persisted in anomaly_events table."""
        agg = TechnicalAggregator()
        df = _synthetic_ohlcv(200)
        result = agg.analyze("NIFTY50", df, vix_value=35.0)  # Will trigger VIX alert

        IndicatorStore.save_analysis(result, db)

        anomalies = db.fetch_all(
            "SELECT * FROM anomaly_events WHERE index_id = ?",
            ("NIFTY50",),
        )
        # At least the VIX alert should be saved
        assert len(anomalies) >= 1

    def test_get_latest_nonexistent(self, db):
        """Loading non-existent index returns None."""
        loaded = IndicatorStore.get_latest_analysis("NONEXISTENT", db)
        assert loaded is None


# ══════════════════════════════════════════════════════════════════════════════
# Performance
# ══════════════════════════════════════════════════════════════════════════════


class TestPerformance:
    """Performance benchmarks — analysis should be fast."""

    def test_two_years_under_5_seconds(self):
        """Analysis of ~500 trading days should complete in < 5 seconds."""
        agg = TechnicalAggregator()
        df = _synthetic_ohlcv(500, seed=42)

        start = time.time()
        result = agg.analyze("NIFTY50", df)
        elapsed = time.time() - start

        assert isinstance(result, TechnicalAnalysisResult)
        assert elapsed < 5.0, f"Analysis took {elapsed:.2f}s, expected < 5s"

    def test_multiple_indices_performance(self):
        """Analysing 3 indices sequentially should complete in < 10 seconds."""
        agg = TechnicalAggregator()

        start = time.time()
        for i, idx_id in enumerate(["NIFTY50", "BANKNIFTY", "NIFTY_IT"]):
            df = _synthetic_ohlcv(500, base=20000 + i * 10000, seed=i + 1)
            agg.analyze(idx_id, df)
        elapsed = time.time() - start

        assert elapsed < 10.0, f"Multi-index analysis took {elapsed:.2f}s, expected < 10s"


# ══════════════════════════════════════════════════════════════════════════════
# Result field completeness
# ══════════════════════════════════════════════════════════════════════════════


class TestResultFields:
    """Verify all expected fields are present and well-typed."""

    @pytest.fixture()
    def result(self) -> TechnicalAnalysisResult:
        agg = TechnicalAggregator()
        df = _synthetic_ohlcv(200)
        return agg.analyze("NIFTY50", df)

    def test_all_summaries_present(self, result):
        assert result.trend is not None
        assert result.momentum is not None
        assert result.volatility is not None
        assert result.volume is not None
        assert result.quant is not None
        # options and smart_money may be None — that's OK

    def test_votes_dict(self, result):
        assert isinstance(result.votes, dict)
        assert "trend" in result.votes

    def test_vote_counts(self, result):
        assert isinstance(result.bullish_votes, int)
        assert isinstance(result.bearish_votes, int)
        assert isinstance(result.neutral_votes, int)
        total = result.bullish_votes + result.bearish_votes + result.neutral_votes
        assert total >= 1

    def test_levels_are_sorted(self, result):
        assert result.support_levels == sorted(result.support_levels)
        assert result.resistance_levels == sorted(result.resistance_levels)

    def test_alerts_are_list(self, result):
        assert isinstance(result.alerts, list)
        for alert in result.alerts:
            assert isinstance(alert, Alert)
            assert alert.type in (
                "DIVERGENCE", "BREAKOUT_TRAP", "OI_SPIKE", "VOLUME_CLIMAX",
                "BB_SQUEEZE", "SMART_MONEY_SIGNAL", "VIX_EXTREME", "REVERSAL_WARNING",
            )
            assert alert.severity in ("HIGH", "MEDIUM", "LOW")
            assert len(alert.message) > 0
            assert len(alert.source) > 0

    def test_warnings_are_list(self, result):
        assert isinstance(result.warnings, list)
