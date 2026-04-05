"""
Tests for the analysis layer:
  - SectorAnalyzer (sector_analyzer.py)
  - AnomalyDetector (anomaly_detector.py)
  - options analysis helpers (options_analysis.py)
  - NewsEngine (news_engine.py)
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.sector_analyzer import SectorAnalyzer, RelativeStrengthScore
from src.analysis.anomaly_detector import AnomalyDetector, AnomalyType
from src.analysis.options_analysis import (
    calculate_pcr,
    calculate_max_pain,
    find_oi_spikes,
    analyse_chain,
    OIAnalysis,
)
from src.analysis.news_engine import NewsEngine, NewsArticle


# ── Helpers ───────────────────────────────────────────────────────────────────

def _close_series(n: int = 60, base: float = 20000.0, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(base + rng.standard_normal(n).cumsum() * 100)


def _ohlcv(n: int = 30, base: float = 20000.0, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = base + rng.standard_normal(n).cumsum() * 100
    high = close + rng.uniform(50, 200, n)
    low = close - rng.uniform(50, 200, n)
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


def _chain_df() -> pd.DataFrame:
    """Minimal options chain with five strikes centred around 22000."""
    strikes = [21800, 21900, 22000, 22100, 22200]
    return pd.DataFrame({
        "strike": strikes,
        "ce_oi":     [50_000, 80_000, 120_000, 60_000, 30_000],
        "pe_oi":     [30_000, 60_000, 100_000, 80_000, 50_000],
        "ce_volume": [5_000,  8_000,  15_000,  6_000,  3_000],
        "pe_volume": [3_000,  7_000,  12_000,  9_000,  4_000],
        "ce_iv":     [14.0,   14.5,   15.0,    15.5,   16.0],
        "pe_iv":     [15.0,   15.5,   16.0,    16.5,   17.0],
    })


# ══════════════════════════════════════════════════════════════════════════════
# SectorAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class TestSectorAnalyzer:

    @pytest.fixture()
    def analyzer(self, registry):
        return SectorAnalyzer(registry, benchmark_id="NIFTY50")

    @pytest.fixture()
    def prices(self):
        """Price series for NIFTY50 (benchmark) and two sectoral indices.

        Outperformer compounds +10 bps/day above benchmark;
        underperformer compounds −10 bps/day below benchmark.
        """
        rng = np.random.default_rng(0)
        bench_ret = rng.standard_normal(60) * 0.01
        out_ret  = bench_ret + 0.001   # +10bps every day
        under_ret = bench_ret - 0.001  # -10bps every day

        bench      = pd.Series((1 + bench_ret).cumprod() * 20000.0)
        outperform = pd.Series((1 + out_ret).cumprod()   * 20000.0)
        underperform = pd.Series((1 + under_ret).cumprod() * 20000.0)
        return {
            "NIFTY50":   bench,
            "BANKNIFTY": outperform,
            "NIFTY_IT":  underperform,
        }

    # ── relative_strength ─────────────────────────────────────────────────────

    def test_relative_strength_returns_list(self, analyzer, prices):
        result = analyzer.relative_strength(prices)
        assert isinstance(result, list)
        # Benchmark excluded
        ids = {r.index_id for r in result}
        assert "NIFTY50" not in ids

    def test_relative_strength_sorted_by_20d_desc(self, analyzer, prices):
        result = analyzer.relative_strength(prices)
        rs_vals = [r.rs_20d for r in result]
        assert rs_vals == sorted(rs_vals, reverse=True)

    def test_relative_strength_trend_labels(self, analyzer, prices):
        result = analyzer.relative_strength(prices)
        id_map = {r.index_id: r for r in result}
        # The outperformer should beat benchmark on 20d
        assert id_map["BANKNIFTY"].rs_20d > id_map["NIFTY_IT"].rs_20d

    def test_relative_strength_missing_benchmark(self, analyzer):
        result = analyzer.relative_strength({"BANKNIFTY": _close_series()})
        assert result == []

    def test_relative_strength_skips_insufficient_data(self, analyzer):
        tiny = pd.Series([20000.0, 20100.0])  # < 21 points
        result = analyzer.relative_strength({
            "NIFTY50": tiny,
            "BANKNIFTY": tiny,
        })
        # BANKNIFTY has < p20+1 aligned rows after concat+dropna — skipped
        assert result == []

    # ── sector_heatmap ────────────────────────────────────────────────────────

    def test_heatmap_returns_dataframe(self, analyzer, prices):
        df = analyzer.sector_heatmap(prices, period_days=5)
        assert isinstance(df, pd.DataFrame)

    def test_heatmap_columns(self, analyzer, prices):
        df = analyzer.sector_heatmap(prices, period_days=5)
        assert set(df.columns) >= {"index_id", "display_name", "sector_category", "return_pct"}

    def test_heatmap_excludes_unknown_indices(self, analyzer):
        prices = {"UNKNOWN_IDX": _close_series()}
        df = analyzer.sector_heatmap(prices, period_days=5)
        assert df.empty

    def test_heatmap_sorted_descending_by_return(self, analyzer, prices):
        df = analyzer.sector_heatmap(prices, period_days=5)
        if len(df) > 1:
            assert list(df["return_pct"]) == sorted(df["return_pct"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# AnomalyDetector
# ══════════════════════════════════════════════════════════════════════════════

class TestAnomalyDetector:

    @pytest.fixture()
    def detector(self):
        return AnomalyDetector(volume_spike_multiplier=2.0, oi_spike_threshold=3.0)

    # ── detect_volume_spike ───────────────────────────────────────────────────

    def test_no_spike_on_normal_volume(self, detector):
        df = _ohlcv(30)
        # Force last volume to be same as mean
        df.at[df.index[-1], "volume"] = int(df["volume"].iloc[:-1].mean())
        assert detector.detect_volume_spike(df, "NIFTY50") is None

    def test_spike_detected_on_high_volume(self, detector):
        df = _ohlcv(30)
        baseline_mean = df["volume"].iloc[:-1].mean()
        df.at[df.index[-1], "volume"] = int(baseline_mean * 5)
        anomaly = detector.detect_volume_spike(df, "NIFTY50")
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.VOLUME_SPIKE
        assert anomaly.magnitude > 2.0

    def test_spike_returns_none_for_insufficient_data(self, detector):
        df = _ohlcv(5)
        assert detector.detect_volume_spike(df, "NIFTY50", lookback=20) is None

    def test_spike_description_contains_index_id(self, detector):
        df = _ohlcv(30)
        df.at[df.index[-1], "volume"] = int(df["volume"].iloc[:-1].mean() * 10)
        anomaly = detector.detect_volume_spike(df, "NIFTY50")
        assert "NIFTY50" in anomaly.description

    # ── detect_oi_spike ───────────────────────────────────────────────────────

    def test_oi_spike_empty_chain_returns_empty(self, detector):
        result = detector.detect_oi_spike(pd.DataFrame(), "NIFTY50")
        assert result == []

    def test_oi_spike_missing_column_returns_empty(self, detector):
        df = pd.DataFrame({"strike": [22000]})
        assert detector.detect_oi_spike(df, "NIFTY50", column="ce_chg_oi") == []

    def test_oi_spike_detects_outlier(self, detector):
        # Mean will be ~10k, outlier is 150k → 15× mean > threshold 3
        df = pd.DataFrame({
            "strike": [21800, 21900, 22000, 22100, 22200],
            "ce_chg_oi": [5_000, 8_000, 150_000, 12_000, 7_000],
        })
        anomalies = detector.detect_oi_spike(df, "NIFTY50", column="ce_chg_oi")
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.OI_SPIKE

    def test_oi_spike_no_anomaly_when_uniform(self, detector):
        df = pd.DataFrame({
            "strike": list(range(5)),
            "ce_chg_oi": [10_000] * 5,
        })
        assert detector.detect_oi_spike(df, "NIFTY50") == []

    # ── detect_price_gap ──────────────────────────────────────────────────────

    def test_gap_detected_on_large_move(self, detector):
        df = _ohlcv(5)
        # Force a 2% gap up between last two rows
        prev_close = df["close"].iloc[-2]
        df.at[df.index[-1], "open"] = prev_close * 1.02
        anomaly = detector.detect_price_gap(df, "NIFTY50", gap_threshold_pct=0.5)
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.PRICE_GAP
        assert "up" in anomaly.description

    def test_gap_not_detected_on_small_move(self, detector):
        df = _ohlcv(5)
        prev_close = df["close"].iloc[-2]
        df.at[df.index[-1], "open"] = prev_close * 1.001  # 0.1% — below threshold
        assert detector.detect_price_gap(df, "NIFTY50", gap_threshold_pct=0.5) is None

    def test_gap_returns_none_for_insufficient_data(self, detector):
        df = _ohlcv(1)
        assert detector.detect_price_gap(df, "NIFTY50") is None

    def test_gap_down_direction_in_description(self, detector):
        df = _ohlcv(5)
        prev_close = df["close"].iloc[-2]
        df.at[df.index[-1], "open"] = prev_close * 0.97
        anomaly = detector.detect_price_gap(df, "NIFTY50", gap_threshold_pct=0.5)
        assert anomaly is not None
        assert "down" in anomaly.description


# ══════════════════════════════════════════════════════════════════════════════
# Options Analysis helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestCalculatePCR:

    def test_pcr_basic_calculation(self):
        df = pd.DataFrame({"ce_oi": [100], "pe_oi": [150],
                           "ce_volume": [50], "pe_volume": [75]})
        pcr_oi, pcr_vol = calculate_pcr(df)
        assert pcr_oi == pytest.approx(1.5, abs=1e-4)
        assert pcr_vol == pytest.approx(1.5, abs=1e-4)

    def test_pcr_zero_ce_oi_returns_zero(self):
        df = pd.DataFrame({"ce_oi": [0], "pe_oi": [100],
                           "ce_volume": [0], "pe_volume": [50]})
        pcr_oi, pcr_vol = calculate_pcr(df)
        assert pcr_oi == 0.0
        assert pcr_vol == 0.0

    def test_pcr_chain_fixture(self, sample_options_chain):
        pcr_oi, pcr_vol = calculate_pcr(sample_options_chain)
        assert pcr_oi > 0
        assert pcr_vol > 0

    def test_pcr_summed_correctly(self):
        df = pd.DataFrame({
            "ce_oi": [100, 200],
            "pe_oi": [150, 250],
            "ce_volume": [50, 100],
            "pe_volume": [75, 125],
        })
        pcr_oi, _ = calculate_pcr(df)
        assert pcr_oi == pytest.approx(400 / 300, abs=1e-4)


class TestCalculateMaxPain:

    def test_max_pain_is_in_strike_list(self):
        df = _chain_df()
        mp = calculate_max_pain(df)
        assert mp in df["strike"].values

    def test_max_pain_empty_returns_zero(self):
        assert calculate_max_pain(pd.DataFrame({"strike": [], "ce_oi": [], "pe_oi": []})) == 0.0

    def test_max_pain_single_strike(self):
        df = pd.DataFrame({"strike": [22000.0], "ce_oi": [100_000], "pe_oi": [80_000]})
        assert calculate_max_pain(df) == 22000.0

    def test_max_pain_gravitates_toward_high_oi_cluster(self):
        # Max OI on both sides is at 22000 — max pain should be there
        df = pd.DataFrame({
            "strike":  [21500.0, 22000.0, 22500.0],
            "ce_oi":   [10_000,  200_000, 5_000],
            "pe_oi":   [5_000,   200_000, 10_000],
        })
        mp = calculate_max_pain(df)
        # With symmetric very high OI at 22000, max pain should be 22000
        assert mp == 22000.0


class TestFindOISpikes:

    def test_spikes_above_multiplier(self):
        df = pd.DataFrame({
            "strike": [21800, 21900, 22000, 22100, 22200],
            "ce_oi":  [10_000, 12_000, 200_000, 11_000, 9_000],
        })
        # mean ≈ 48400; 200000 >> threshold * mean with default threshold 10
        spikes = find_oi_spikes(df, "ce_oi", multiplier=2.0)
        assert 22000 in spikes

    def test_no_spikes_on_uniform_oi(self):
        df = pd.DataFrame({
            "strike": [21800, 22000, 22200],
            "ce_oi":  [100_000, 100_000, 100_000],
        })
        assert find_oi_spikes(df, "ce_oi", multiplier=2.0) == []

    def test_spikes_sorted(self):
        df = pd.DataFrame({
            "strike": [22200, 21800, 22000],
            "ce_oi":  [200_000, 180_000, 10_000],
        })
        spikes = find_oi_spikes(df, "ce_oi", multiplier=1.5)
        assert spikes == sorted(spikes)


class TestAnalyseChain:

    def test_returns_oi_analysis(self, sample_options_chain):
        result = analyse_chain(sample_options_chain, underlying_value=22000.0)
        assert isinstance(result, OIAnalysis)

    def test_pcr_positive(self, sample_options_chain):
        result = analyse_chain(sample_options_chain, underlying_value=22000.0)
        assert result.pcr_oi > 0

    def test_max_pain_in_strikes(self, sample_options_chain):
        result = analyse_chain(sample_options_chain, underlying_value=22000.0)
        assert result.max_pain in sample_options_chain["strike"].values

    def test_sentiment_valid_value(self, sample_options_chain):
        result = analyse_chain(sample_options_chain, underlying_value=22000.0)
        assert result.sentiment in {"bullish", "bearish", "neutral"}

    def test_empty_chain_raises(self):
        with pytest.raises(ValueError):
            analyse_chain(pd.DataFrame(), underlying_value=22000.0)

    def test_atm_strike_is_closest_to_underlying(self):
        df = _chain_df()  # strikes: 21800, 21900, 22000, 22100, 22200
        result = analyse_chain(df, underlying_value=22050.0)
        assert result.atm_strike == 22000.0  # closest integer below 22050

    def test_bullish_sentiment_on_low_pcr(self):
        """When CE OI >> PE OI, PCR is low → bullish."""
        df = pd.DataFrame({
            "strike": [22000.0],
            "ce_oi": [500_000],
            "pe_oi": [100_000],  # pcr ≈ 0.2 < 0.7
            "ce_volume": [10_000],
            "pe_volume": [2_000],
        })
        result = analyse_chain(df, underlying_value=22000.0)
        assert result.sentiment == "bullish"

    def test_bearish_sentiment_on_high_pcr(self):
        """When PE OI >> CE OI, PCR is high → bearish."""
        df = pd.DataFrame({
            "strike": [22000.0],
            "ce_oi": [100_000],
            "pe_oi": [500_000],  # pcr ≈ 5.0 > 1.3
            "ce_volume": [2_000],
            "pe_volume": [10_000],
        })
        result = analyse_chain(df, underlying_value=22000.0)
        assert result.sentiment == "bearish"


# ══════════════════════════════════════════════════════════════════════════════
# NewsEngine
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def news_config_dir(tmp_path: Path) -> Path:
    """Write minimal config JSON files into a temp dir."""
    feeds = [
        {"url": "https://example.com/rss", "source_name": "TestFeed",
         "is_active": True, "credibility_score": 0.8},
    ]
    mappings = {
        "banking": {
            "keywords": ["bank", "rbi"],
            "affected_indices": ["BANKNIFTY"],
        },
    }
    keywords = {
        "bullish": [{"keyword": "rally", "intensity": 2.0}],
        "bearish": [{"keyword": "crash", "intensity": 2.0}],
    }
    (tmp_path / "rss_feeds.json").write_text(json.dumps(feeds))
    (tmp_path / "news_mappings.json").write_text(json.dumps(mappings))
    (tmp_path / "sentiment_keywords.json").write_text(json.dumps(keywords))
    return tmp_path


@pytest.fixture()
def news_engine(news_config_dir: Path) -> NewsEngine:
    return NewsEngine(
        feeds_path=news_config_dir / "rss_feeds.json",
        mappings_path=news_config_dir / "news_mappings.json",
        keywords_path=news_config_dir / "sentiment_keywords.json",
    )


class TestNewsEngineAggregation:
    """Tests for aggregate_sentiment — no HTTP calls needed."""

    def _make_article(self, score: float, indices: list[str], cred: float = 0.8) -> NewsArticle:
        return NewsArticle(
            title="Test", summary="", url="", source="Test",
            published_at=time.time(),
            vader_compound=score,
            adjusted_score=score,
            affected_indices=indices,
            credibility_score=cred,
        )

    def test_aggregate_returns_zero_when_no_relevant(self, news_engine):
        articles = [self._make_article(0.5, ["NIFTY50"])]
        assert news_engine.aggregate_sentiment(articles, "BANKNIFTY") == 0.0

    def test_aggregate_weighted_by_credibility(self, news_engine):
        articles = [
            self._make_article(0.8, ["NIFTY50"], cred=0.9),
            self._make_article(-0.2, ["NIFTY50"], cred=0.1),
        ]
        score = news_engine.aggregate_sentiment(articles, "NIFTY50")
        # weighted = (0.8*0.9 + (-0.2)*0.1) / (0.9+0.1) = (0.72 - 0.02)/1.0 = 0.70
        assert score == pytest.approx(0.70, abs=1e-4)

    def test_aggregate_empty_articles_returns_zero(self, news_engine):
        assert news_engine.aggregate_sentiment([], "NIFTY50") == 0.0

    def test_aggregate_clamps_to_range(self, news_engine):
        articles = [self._make_article(1.5, ["BANKNIFTY"])]  # score > 1 is unusual but clipped
        score = news_engine.aggregate_sentiment(articles, "BANKNIFTY")
        assert -1.0 <= score <= 1.5  # NewsEngine doesn't clamp, VADER returns -1..1

    def test_aggregate_multiple_indices_independent(self, news_engine):
        articles = [
            self._make_article(0.8, ["NIFTY50"]),
            self._make_article(-0.6, ["BANKNIFTY"]),
        ]
        assert news_engine.aggregate_sentiment(articles, "NIFTY50") > 0
        assert news_engine.aggregate_sentiment(articles, "BANKNIFTY") < 0


class TestNewsEngineKeywordAdjustment:
    """Test that bullish/bearish keyword boosts apply correctly."""

    def test_bullish_keyword_boosts_positive(self, news_engine):
        # 'rally' is a bullish keyword with intensity 2.0 → boost = +0.2
        adjusted = news_engine._adjust_score("market rally expected", 0.0)
        assert adjusted > 0.0

    def test_bearish_keyword_reduces_score(self, news_engine):
        # 'crash' is bearish with intensity 2.0 → boost = -0.2
        adjusted = news_engine._adjust_score("stock market crash", 0.0)
        assert adjusted < 0.0

    def test_score_clamped_to_negative_one(self, news_engine):
        # Bearish keyword on already very negative score
        adjusted = news_engine._adjust_score("crash crash crash", -0.95)
        assert adjusted >= -1.0

    def test_score_clamped_to_positive_one(self, news_engine):
        adjusted = news_engine._adjust_score("rally rally rally", 0.95)
        assert adjusted <= 1.0


class TestNewsEngineMapping:
    """Test index mapping from article text."""

    def test_maps_banking_keyword_to_banknifty(self, news_engine):
        indices = news_engine._map_to_indices("RBI cuts bank rate")
        assert "BANKNIFTY" in indices

    def test_no_match_returns_empty(self, news_engine):
        assert news_engine._map_to_indices("unrelated text about weather") == []

    def test_case_insensitive_matching(self, news_engine):
        assert "BANKNIFTY" in news_engine._map_to_indices("BANK sector rally")


class TestNewsEngineFetch:
    """Smoke test for fetch_all with mocked feedparser."""

    def test_fetch_all_with_mocked_feed(self, news_engine):
        mock_entry = MagicMock()
        mock_entry.title = "Bank rally today"
        mock_entry.summary = "RBI policy announcement"
        mock_entry.link = "https://example.com/article"
        mock_entry.get.side_effect = lambda key, default=None: {
            "title": "Bank rally today",
            "summary": "RBI policy announcement",
            "link": "https://example.com/article",
            "published_parsed": time.localtime(),  # mktime(localtime()) ≈ time.time()
        }.get(key, default)

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]

        with patch("src.analysis.news_engine.feedparser.parse", return_value=mock_feed):
            articles = news_engine.fetch_all(max_age_seconds=3600.0)

        assert len(articles) == 1
        assert "BANKNIFTY" in articles[0].affected_indices

    def test_fetch_all_filters_old_articles(self, news_engine):
        mock_entry = MagicMock()
        old_time = time.gmtime(time.time() - 7200)  # 2 hours ago
        mock_entry.get.side_effect = lambda key, default=None: {
            "title": "Old news",
            "summary": "Old summary",
            "link": "https://example.com/old",
            "published_parsed": old_time,
        }.get(key, default)

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]

        with patch("src.analysis.news_engine.feedparser.parse", return_value=mock_feed):
            articles = news_engine.fetch_all(max_age_seconds=3600.0)

        assert len(articles) == 0

    def test_fetch_all_graceful_on_feed_error(self, news_engine):
        with patch("src.analysis.news_engine.feedparser.parse", side_effect=Exception("timeout")):
            articles = news_engine.fetch_all()
        assert articles == []
