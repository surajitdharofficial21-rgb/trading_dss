"""
Tests for TechnicalAggregator — the master analysis aggregator.

Covers:
  - All-bullish / all-bearish / mixed inputs
  - Missing options data (weight redistribution)
  - Support/resistance clustering
  - Reasoning generation
  - Alert generation
  - Confidence modifiers
  - Data completeness
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.technical_aggregator import (
    Alert,
    TechnicalAggregator,
    TechnicalAnalysisResult,
    _cluster_levels,
    _score_to_signal,
    _vote_to_numeric,
)
from src.analysis.indicators.trend import TrendSummary
from src.analysis.indicators.momentum import MomentumSummary
from src.analysis.indicators.volatility import VolatilitySummary
from src.analysis.indicators.volume import VolumeSummary
from src.analysis.indicators.options_indicators import OptionsSummary
from src.analysis.indicators.quant import QuantSummary
from src.analysis.indicators.smart_money import SmartMoneyScore


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ohlcv(n: int = 100, base: float = 20000.0, seed: int = 42, trend: str = "flat") -> pd.DataFrame:
    """Generate synthetic OHLCV data with configurable trend."""
    rng = np.random.default_rng(seed)

    if trend == "up":
        close = base + np.arange(n) * 50.0 + rng.standard_normal(n).cumsum() * 20
    elif trend == "down":
        close = base - np.arange(n) * 50.0 + rng.standard_normal(n).cumsum() * 20
    else:
        close = base + rng.standard_normal(n).cumsum() * 100

    high = close + rng.uniform(50, 200, n)
    low = close - rng.uniform(50, 200, n)
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, n)

    dates = pd.date_range(start="2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=dates)


def _bullish_trend(now: datetime) -> TrendSummary:
    return TrendSummary(
        index_id="TEST", timeframe="1d", timestamp=now,
        price_vs_ema20="ABOVE", price_vs_ema50="ABOVE", price_vs_ema200="ABOVE",
        ema_alignment="BULLISH", golden_cross=False, death_cross=False,
        macd_signal="BULLISH", macd_crossover=None, macd_histogram_trend="INCREASING",
        trend_strength="STRONG", trend_direction="BULLISH",
        trend_vote="STRONG_BULLISH", trend_confidence=0.9,
    )


def _bearish_trend(now: datetime) -> TrendSummary:
    return TrendSummary(
        index_id="TEST", timeframe="1d", timestamp=now,
        price_vs_ema20="BELOW", price_vs_ema50="BELOW", price_vs_ema200="BELOW",
        ema_alignment="BEARISH", golden_cross=False, death_cross=True,
        macd_signal="BEARISH", macd_crossover=None, macd_histogram_trend="DECREASING",
        trend_strength="STRONG", trend_direction="BEARISH",
        trend_vote="STRONG_BEARISH", trend_confidence=0.9,
    )


def _neutral_trend(now: datetime) -> TrendSummary:
    return TrendSummary(
        index_id="TEST", timeframe="1d", timestamp=now,
        price_vs_ema20="ABOVE", price_vs_ema50="BELOW", price_vs_ema200="ABOVE",
        ema_alignment="MIXED", golden_cross=False, death_cross=False,
        macd_signal="NEUTRAL", macd_crossover=None, macd_histogram_trend="DECREASING",
        trend_strength="WEAK", trend_direction="BULLISH",
        trend_vote="NEUTRAL", trend_confidence=0.5,
    )


def _bullish_momentum(now: datetime) -> MomentumSummary:
    return MomentumSummary(
        timestamp=now, rsi_value=35.0, rsi_zone="NEUTRAL", rsi_divergence=None,
        stochastic_k=25.0, stochastic_zone="OVERSOLD", stochastic_crossover="BULLISH_CROSS",
        cci_value=-150.0, cci_zone="OVERSOLD",
        momentum_vote="BULLISH", momentum_confidence=0.75,
        overbought_consensus=False, oversold_consensus=True,
        divergence_detected=False, reversal_warning=None,
    )


def _bearish_momentum(now: datetime) -> MomentumSummary:
    return MomentumSummary(
        timestamp=now, rsi_value=75.0, rsi_zone="OVERBOUGHT", rsi_divergence="BEARISH_DIVERGENCE",
        stochastic_k=85.0, stochastic_zone="OVERBOUGHT", stochastic_crossover="BEARISH_CROSS",
        cci_value=200.0, cci_zone="OVERBOUGHT",
        momentum_vote="BEARISH", momentum_confidence=0.8,
        overbought_consensus=True, oversold_consensus=False,
        divergence_detected=True, reversal_warning="POTENTIAL_TOP",
    )


def _neutral_momentum(now: datetime) -> MomentumSummary:
    return MomentumSummary(
        timestamp=now, rsi_value=50.0, rsi_zone="NEUTRAL", rsi_divergence=None,
        stochastic_k=50.0, stochastic_zone="NEUTRAL", stochastic_crossover=None,
        cci_value=0.0, cci_zone="NEUTRAL",
        momentum_vote="NEUTRAL", momentum_confidence=0.5,
        overbought_consensus=False, oversold_consensus=False,
        divergence_detected=False, reversal_warning=None,
    )


def _normal_volatility(now: datetime) -> VolatilitySummary:
    return VolatilitySummary(
        timestamp=now,
        bb_position="UPPER_ZONE", bb_squeeze=False, bb_bandwidth_percentile=50.0,
        atr_value=200.0, atr_pct=1.0, volatility_level="NORMAL",
        suggested_sl=300.0, suggested_target=600.0,
        hv_current=15.0, hv_regime="NORMAL", vix_regime=None,
        volatility_vote="NORMAL", volatility_confidence=0.6,
        position_size_modifier=1.0, breakout_alert=False, mean_reversion_setup=False,
    )


def _extreme_volatility(now: datetime) -> VolatilitySummary:
    return VolatilitySummary(
        timestamp=now,
        bb_position="BELOW_LOWER", bb_squeeze=True, bb_bandwidth_percentile=5.0,
        atr_value=500.0, atr_pct=2.5, volatility_level="EXTREME",
        suggested_sl=750.0, suggested_target=1500.0,
        hv_current=40.0, hv_regime="EXTREME", vix_regime="EXTREME",
        volatility_vote="EXTREME", volatility_confidence=0.9,
        position_size_modifier=0.5, breakout_alert=True, mean_reversion_setup=True,
    )


def _bullish_volume(now: datetime) -> VolumeSummary:
    return VolumeSummary(
        timestamp=now,
        price_vs_vwap="ABOVE", vwap_zone="ABOVE_1SD",
        institutional_bias="BUYING",
        obv_trend="RISING", obv_divergence=None,
        accumulation_distribution="ACCUMULATION",
        poc=20000.0, value_area_high=20200.0, value_area_low=19800.0,
        in_value_area=True, volume_ratio=1.5,
        volume_confirms_price=True,
        volume_vote="BULLISH", volume_confidence=0.7,
    )


def _bearish_volume(now: datetime) -> VolumeSummary:
    return VolumeSummary(
        timestamp=now,
        price_vs_vwap="BELOW", vwap_zone="BELOW_2SD",
        institutional_bias="SELLING",
        obv_trend="FALLING", obv_divergence="BEARISH_DIVERGENCE",
        accumulation_distribution="DISTRIBUTION",
        poc=20000.0, value_area_high=20200.0, value_area_low=19800.0,
        in_value_area=False, volume_ratio=3.5,
        volume_confirms_price=True,
        volume_vote="STRONG_BEARISH", volume_confidence=0.85,
    )


def _bullish_options(now: datetime) -> OptionsSummary:
    from datetime import date
    return OptionsSummary(
        timestamp=now, index_id="TEST", expiry_date=date(2024, 6, 27),
        days_to_expiry=5, pcr=1.3, pcr_signal="BULLISH",
        oi_support=19800.0, oi_resistance=20500.0,
        expected_range=(19700.0, 20600.0),
        max_pain=20100.0, max_pain_pull="MODERATE",
        oi_change_signal="BULLISH", dominant_buildup="LONG_BUILDUP",
        atm_iv=15.0, iv_regime="NORMAL", iv_skew=0.05,
        options_vote="STRONG_BULLISH", options_confidence=0.85,
    )


def _bearish_options(now: datetime) -> OptionsSummary:
    from datetime import date
    return OptionsSummary(
        timestamp=now, index_id="TEST", expiry_date=date(2024, 6, 27),
        days_to_expiry=5, pcr=0.6, pcr_signal="BEARISH",
        oi_support=19500.0, oi_resistance=20000.0,
        expected_range=(19400.0, 20100.0),
        max_pain=19800.0, max_pain_pull="STRONG",
        oi_change_signal="BEARISH", dominant_buildup="SHORT_BUILDUP",
        atm_iv=22.0, iv_regime="HIGH", iv_skew=-0.08,
        options_vote="STRONG_BEARISH", options_confidence=0.8,
    )


def _bullish_quant(now: datetime) -> QuantSummary:
    return QuantSummary(
        timestamp=now,
        zscore=-2.5, zscore_zone="OVERSOLD",
        mean_reversion_signal="BUY",
        beta=1.1, alpha=0.02, beta_interpretation="MODERATE_HIGH",
        statistical_regime="MEAN_REVERTING",
        quant_vote="BULLISH", quant_confidence=0.5,
    )


def _bearish_quant(now: datetime) -> QuantSummary:
    return QuantSummary(
        timestamp=now,
        zscore=2.5, zscore_zone="OVERBOUGHT",
        mean_reversion_signal="SELL",
        beta=1.1, alpha=-0.01, beta_interpretation="MODERATE_HIGH",
        statistical_regime="MEAN_REVERTING",
        quant_vote="BEARISH", quant_confidence=0.5,
    )


def _neutral_quant(now: datetime) -> QuantSummary:
    return QuantSummary(
        timestamp=now,
        zscore=0.2, zscore_zone="FAIR_VALUE", mean_reversion_signal=None,
        beta=None, alpha=None, beta_interpretation=None,
        statistical_regime="NORMAL",
        quant_vote="NEUTRAL", quant_confidence=0.3,
    )


def _bullish_smart_money() -> SmartMoneyScore:
    return SmartMoneyScore(
        score=75.0, grade="A",
        smfi_component=80.0, vsd_component=60.0, btd_component=70.0,
        oimi_component=80.0, lai_component=65.0,
        smart_money_bias="STRONGLY_BULLISH",
        key_finding="Strong institutional accumulation",
        actionable_insight="Institutions are building long positions",
        data_completeness=0.8, confidence=0.75,
    )


def _bearish_smart_money() -> SmartMoneyScore:
    return SmartMoneyScore(
        score=-70.0, grade="F",
        smfi_component=20.0, vsd_component=40.0, btd_component=30.0,
        oimi_component=25.0, lai_component=35.0,
        smart_money_bias="STRONGLY_BEARISH",
        key_finding="Distribution detected across indicators",
        actionable_insight="Institutional selling pressure building",
        data_completeness=0.8, confidence=0.7,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests for helper functions
# ══════════════════════════════════════════════════════════════════════════════


class TestVoteConversion:
    def test_all_vote_strings(self):
        assert _vote_to_numeric("STRONG_BULLISH") == 2.0
        assert _vote_to_numeric("BULLISH") == 1.0
        assert _vote_to_numeric("NEUTRAL") == 0.0
        assert _vote_to_numeric("BEARISH") == -1.0
        assert _vote_to_numeric("STRONG_BEARISH") == -2.0
        assert _vote_to_numeric("STRONGLY_BULLISH") == 2.0
        assert _vote_to_numeric("STRONGLY_BEARISH") == -2.0

    def test_unknown_vote_defaults_to_zero(self):
        assert _vote_to_numeric("UNKNOWN") == 0.0


class TestScoreToSignal:
    def test_strong_buy(self):
        assert _score_to_signal(1.5) == "STRONG_BUY"

    def test_buy(self):
        assert _score_to_signal(0.8) == "BUY"

    def test_neutral(self):
        assert _score_to_signal(0.0) == "NEUTRAL"

    def test_sell(self):
        assert _score_to_signal(-0.8) == "SELL"

    def test_strong_sell(self):
        assert _score_to_signal(-1.5) == "STRONG_SELL"

    def test_boundary_buy(self):
        assert _score_to_signal(0.5) == "BUY"

    def test_boundary_sell(self):
        assert _score_to_signal(-0.5) == "SELL"


class TestClusterLevels:
    def test_empty_list(self):
        assert _cluster_levels([]) == []

    def test_single_level(self):
        assert _cluster_levels([20000.0]) == [20000.0]

    def test_distant_levels_not_clustered(self):
        result = _cluster_levels([20000.0, 21000.0, 22000.0])
        assert len(result) == 3

    def test_close_levels_clustered(self):
        # 0.3% of 20000 = 60, so within 60 should cluster
        result = _cluster_levels([20000.0, 20030.0, 20050.0])
        assert len(result) == 1
        assert abs(result[0] - 20026.67) < 1.0

    def test_mixed_clustering(self):
        result = _cluster_levels([20000.0, 20010.0, 21000.0, 21005.0])
        assert len(result) == 2


# ══════════════════════════════════════════════════════════════════════════════
# TechnicalAggregator full integration tests (with mocked indicators)
# ══════════════════════════════════════════════════════════════════════════════


class TestAggregatorAllBullish:
    """All indicators bullish → expect STRONG_BUY with high confidence."""

    @pytest.fixture()
    def result(self) -> TechnicalAnalysisResult:
        now = datetime.now()
        df = _ohlcv(100, trend="up")

        with (
            patch("src.analysis.technical_aggregator.TrendIndicators") as MockTrend,
            patch("src.analysis.technical_aggregator.MomentumIndicators") as MockMom,
            patch("src.analysis.technical_aggregator.VolatilityIndicators") as MockVol,
            patch("src.analysis.technical_aggregator.VolumeIndicators") as MockVolu,
            patch("src.analysis.technical_aggregator.QuantIndicators") as MockQuant,
            patch("src.analysis.technical_aggregator.SmartMoneyIndicators") as MockSM,
        ):
            MockTrend.return_value.get_trend_summary.return_value = _bullish_trend(now)
            MockTrend.return_value.calculate_ema.return_value = pd.Series([20000.0] * 100)
            MockMom.return_value.get_momentum_summary.return_value = _bullish_momentum(now)
            MockVol.return_value.get_volatility_summary.return_value = _normal_volatility(now)
            MockVol.return_value.calculate_bollinger_bands.return_value = MagicMock(
                upper=pd.Series([20400.0]), lower=pd.Series([19600.0])
            )
            MockVolu.return_value.get_volume_summary.return_value = _bullish_volume(now)
            MockQuant.return_value.get_quant_summary.return_value = _bullish_quant(now)
            MockSM.return_value.calculate_smart_money_score.return_value = _bullish_smart_money()

            agg = TechnicalAggregator()
            return agg.analyze("NIFTY50", df, options_chain=MagicMock())

    def test_signal_is_strong_buy_or_buy(self, result):
        assert result.overall_signal in ("STRONG_BUY", "BUY")

    def test_high_confidence(self, result):
        assert result.overall_confidence >= 0.6

    def test_bullish_votes_dominate(self, result):
        assert result.bullish_votes > result.bearish_votes

    def test_reasoning_not_empty(self, result):
        assert len(result.reasoning) > 0
        assert result.overall_signal in result.reasoning


class TestAggregatorAllBearish:
    """All indicators bearish → expect STRONG_SELL."""

    @pytest.fixture()
    def result(self) -> TechnicalAnalysisResult:
        now = datetime.now()
        df = _ohlcv(100, trend="down")

        with (
            patch("src.analysis.technical_aggregator.TrendIndicators") as MockTrend,
            patch("src.analysis.technical_aggregator.MomentumIndicators") as MockMom,
            patch("src.analysis.technical_aggregator.VolatilityIndicators") as MockVol,
            patch("src.analysis.technical_aggregator.VolumeIndicators") as MockVolu,
            patch("src.analysis.technical_aggregator.OptionsIndicators") as MockOpts,
            patch("src.analysis.technical_aggregator.QuantIndicators") as MockQuant,
            patch("src.analysis.technical_aggregator.SmartMoneyIndicators") as MockSM,
        ):
            MockTrend.return_value.get_trend_summary.return_value = _bearish_trend(now)
            MockTrend.return_value.calculate_ema.return_value = pd.Series([20000.0] * 100)
            MockMom.return_value.get_momentum_summary.return_value = _bearish_momentum(now)
            MockVol.return_value.get_volatility_summary.return_value = _extreme_volatility(now)
            MockVol.return_value.calculate_bollinger_bands.return_value = MagicMock(
                upper=pd.Series([20400.0]), lower=pd.Series([19600.0])
            )
            MockVolu.return_value.get_volume_summary.return_value = _bearish_volume(now)
            MockOpts.return_value.get_options_summary.return_value = _bearish_options(now)
            MockQuant.return_value.get_quant_summary.return_value = _bearish_quant(now)
            MockSM.return_value.calculate_smart_money_score.return_value = _bearish_smart_money()

            agg = TechnicalAggregator()
            return agg.analyze("NIFTY50", df, options_chain=MagicMock())

    def test_signal_is_sell(self, result):
        assert result.overall_signal in ("STRONG_SELL", "SELL")

    def test_bearish_votes_dominate(self, result):
        assert result.bearish_votes > result.bullish_votes

    def test_alerts_include_divergence(self, result):
        alert_types = [a.type for a in result.alerts]
        assert "DIVERGENCE" in alert_types

    def test_alerts_include_bb_squeeze(self, result):
        alert_types = [a.type for a in result.alerts]
        assert "BB_SQUEEZE" in alert_types

    def test_death_cross_alert(self, result):
        msgs = [a.message for a in result.alerts]
        assert any("Death cross" in m for m in msgs)

    def test_volume_climax_alert(self, result):
        alert_types = [a.type for a in result.alerts]
        assert "VOLUME_CLIMAX" in alert_types


class TestAggregatorMixed:
    """Mixed inputs → expect NEUTRAL with lower confidence."""

    @pytest.fixture()
    def result(self) -> TechnicalAnalysisResult:
        now = datetime.now()
        df = _ohlcv(100, trend="flat")

        with (
            patch("src.analysis.technical_aggregator.TrendIndicators") as MockTrend,
            patch("src.analysis.technical_aggregator.MomentumIndicators") as MockMom,
            patch("src.analysis.technical_aggregator.VolatilityIndicators") as MockVol,
            patch("src.analysis.technical_aggregator.VolumeIndicators") as MockVolu,
            patch("src.analysis.technical_aggregator.QuantIndicators") as MockQuant,
            patch("src.analysis.technical_aggregator.SmartMoneyIndicators") as MockSM,
        ):
            MockTrend.return_value.get_trend_summary.return_value = _bullish_trend(now)
            MockTrend.return_value.calculate_ema.return_value = pd.Series([20000.0] * 100)
            MockMom.return_value.get_momentum_summary.return_value = _bearish_momentum(now)
            MockVol.return_value.get_volatility_summary.return_value = _normal_volatility(now)
            MockVol.return_value.calculate_bollinger_bands.return_value = MagicMock(
                upper=pd.Series([20400.0]), lower=pd.Series([19600.0])
            )
            MockVolu.return_value.get_volume_summary.return_value = _bullish_volume(now)
            MockQuant.return_value.get_quant_summary.return_value = _neutral_quant(now)
            # No smart money
            MockSM.return_value.calculate_smart_money_score.return_value = None

            agg = TechnicalAggregator()
            return agg.analyze("NIFTY50", df)  # No options

    def test_signal_neutral_or_moderate(self, result):
        assert result.overall_signal in ("BUY", "NEUTRAL", "SELL")

    def test_lower_confidence(self, result):
        # Mixed signals → confidence shouldn't be very high
        assert result.overall_confidence <= 0.85

    def test_neutral_votes_present(self, result):
        # At least some neutral or split
        total = result.bullish_votes + result.bearish_votes + result.neutral_votes
        assert total > 0


class TestMissingOptionsData:
    """Without options data, weights should redistribute."""

    @pytest.fixture()
    def result(self) -> TechnicalAnalysisResult:
        now = datetime.now()
        df = _ohlcv(100)

        with (
            patch("src.analysis.technical_aggregator.TrendIndicators") as MockTrend,
            patch("src.analysis.technical_aggregator.MomentumIndicators") as MockMom,
            patch("src.analysis.technical_aggregator.VolatilityIndicators") as MockVol,
            patch("src.analysis.technical_aggregator.VolumeIndicators") as MockVolu,
            patch("src.analysis.technical_aggregator.QuantIndicators") as MockQuant,
            patch("src.analysis.technical_aggregator.SmartMoneyIndicators") as MockSM,
        ):
            MockTrend.return_value.get_trend_summary.return_value = _bullish_trend(now)
            MockTrend.return_value.calculate_ema.return_value = pd.Series([20000.0] * 100)
            MockMom.return_value.get_momentum_summary.return_value = _bullish_momentum(now)
            MockVol.return_value.get_volatility_summary.return_value = _normal_volatility(now)
            MockVol.return_value.calculate_bollinger_bands.return_value = MagicMock(
                upper=pd.Series([20400.0]), lower=pd.Series([19600.0])
            )
            MockVolu.return_value.get_volume_summary.return_value = _bullish_volume(now)
            MockQuant.return_value.get_quant_summary.return_value = _bullish_quant(now)
            MockSM.return_value.calculate_smart_money_score.return_value = _bullish_smart_money()

            agg = TechnicalAggregator()
            return agg.analyze("NIFTY_IT", df)  # No options_chain

    def test_options_is_none(self, result):
        assert result.options is None

    def test_still_produces_signal(self, result):
        assert result.overall_signal in ("STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL")

    def test_warnings_mention_options(self, result):
        assert any("Options" in w for w in result.warnings)

    def test_options_not_in_votes(self, result):
        assert "options" not in result.votes


class TestSupportResistanceClustering:
    """Test that nearby levels get clustered properly."""

    def test_cluster_near_levels(self):
        levels = [20000.0, 20010.0, 20040.0, 21000.0]
        result = _cluster_levels(levels, threshold_pct=0.003)
        # 20000, 20010, 20040 should cluster (within 0.3% of each other)
        # 21000 is separate
        assert len(result) == 2

    def test_no_clustering_far_apart(self):
        levels = [19000.0, 20000.0, 21000.0]
        result = _cluster_levels(levels)
        assert len(result) == 3

    def test_result_is_sorted(self):
        levels = [21000.0, 19000.0, 20000.0]
        result = _cluster_levels(levels)
        assert result == sorted(result)


class TestReasoningGeneration:
    """Verify reasoning covers all categories."""

    def test_reasoning_includes_all_sections(self):
        now = datetime.now()
        result = TechnicalAnalysisResult(
            index_id="NIFTY50", timestamp=now, timeframe="1d",
            trend=_bullish_trend(now), momentum=_bullish_momentum(now),
            volatility=_normal_volatility(now), volume=_bullish_volume(now),
            options=_bullish_options(now), quant=_bullish_quant(now),
            smart_money=_bullish_smart_money(),
            votes={"trend": "STRONG_BULLISH", "momentum": "BULLISH"},
            bullish_votes=5, bearish_votes=0, neutral_votes=0,
            overall_signal="STRONG_BUY", overall_confidence=0.85,
            support_levels=[19800.0], resistance_levels=[20500.0],
            immediate_support=19800.0, immediate_resistance=20500.0,
            suggested_stop_loss_distance=300.0, suggested_target_distance=600.0,
            position_size_modifier=1.0,
            alerts=[Alert(type="BB_SQUEEZE", severity="HIGH", message="Squeeze!", source="vol")],
        )

        reasoning = TechnicalAggregator.generate_reasoning(result)
        assert "Trend:" in reasoning
        assert "Momentum:" in reasoning
        assert "Volume:" in reasoning
        assert "Options:" in reasoning
        assert "Quant:" in reasoning
        assert "Smart Money:" in reasoning
        assert "Volatility:" in reasoning
        assert "Key levels:" in reasoning
        assert "Risk:" in reasoning

    def test_reasoning_with_no_options(self):
        now = datetime.now()
        result = TechnicalAnalysisResult(
            index_id="NIFTY_IT", timestamp=now, timeframe="1d",
            trend=_neutral_trend(now), momentum=_neutral_momentum(now),
            volatility=_normal_volatility(now), volume=_bullish_volume(now),
            options=None, quant=_neutral_quant(now), smart_money=None,
            votes={"trend": "NEUTRAL"},
            bullish_votes=1, bearish_votes=0, neutral_votes=3,
            overall_signal="NEUTRAL", overall_confidence=0.5,
            support_levels=[19800.0], resistance_levels=[20200.0],
            immediate_support=19800.0, immediate_resistance=20200.0,
            suggested_stop_loss_distance=300.0, suggested_target_distance=600.0,
            position_size_modifier=1.0,
        )

        reasoning = TechnicalAggregator.generate_reasoning(result)
        assert "Options: N/A" in reasoning
        assert "Smart Money: N/A" in reasoning


class TestAlertGeneration:
    """Test specific alert conditions."""

    def test_divergence_alert(self):
        now = datetime.now()
        alerts = TechnicalAggregator._generate_alerts(
            trend=_neutral_trend(now),
            momentum=_bearish_momentum(now),  # has divergence
            volatility=_normal_volatility(now),
            volume=_bullish_volume(now),
            options=None, smart_money=None, vix_value=None,
        )
        assert any(a.type == "DIVERGENCE" for a in alerts)

    def test_bb_squeeze_alert(self):
        now = datetime.now()
        alerts = TechnicalAggregator._generate_alerts(
            trend=_neutral_trend(now),
            momentum=_neutral_momentum(now),
            volatility=_extreme_volatility(now),  # has bb_squeeze
            volume=_bullish_volume(now),
            options=None, smart_money=None, vix_value=None,
        )
        assert any(a.type == "BB_SQUEEZE" for a in alerts)

    def test_vix_extreme_alert(self):
        now = datetime.now()
        alerts = TechnicalAggregator._generate_alerts(
            trend=_neutral_trend(now),
            momentum=_neutral_momentum(now),
            volatility=_normal_volatility(now),
            volume=_bullish_volume(now),
            options=None, smart_money=None, vix_value=35.0,
        )
        vix_alerts = [a for a in alerts if a.type == "VIX_EXTREME"]
        assert len(vix_alerts) == 1
        assert vix_alerts[0].severity == "HIGH"

    def test_vix_elevated_alert(self):
        now = datetime.now()
        alerts = TechnicalAggregator._generate_alerts(
            trend=_neutral_trend(now),
            momentum=_neutral_momentum(now),
            volatility=_normal_volatility(now),
            volume=_bullish_volume(now),
            options=None, smart_money=None, vix_value=22.0,
        )
        vix_alerts = [a for a in alerts if a.type == "VIX_EXTREME"]
        assert len(vix_alerts) == 1
        assert vix_alerts[0].severity == "MEDIUM"

    def test_smart_money_alert(self):
        now = datetime.now()
        alerts = TechnicalAggregator._generate_alerts(
            trend=_neutral_trend(now),
            momentum=_neutral_momentum(now),
            volatility=_normal_volatility(now),
            volume=_bullish_volume(now),
            options=None,
            smart_money=_bullish_smart_money(),  # score 75 > 60
            vix_value=None,
        )
        assert any(a.type == "SMART_MONEY_SIGNAL" for a in alerts)

    def test_volume_climax_alert(self):
        now = datetime.now()
        alerts = TechnicalAggregator._generate_alerts(
            trend=_neutral_trend(now),
            momentum=_neutral_momentum(now),
            volatility=_normal_volatility(now),
            volume=_bearish_volume(now),  # volume_ratio = 3.5
            options=None, smart_money=None, vix_value=None,
        )
        assert any(a.type == "VOLUME_CLIMAX" for a in alerts)

    def test_reversal_warning_from_momentum(self):
        now = datetime.now()
        alerts = TechnicalAggregator._generate_alerts(
            trend=_neutral_trend(now),
            momentum=_bearish_momentum(now),  # has reversal_warning
            volatility=_normal_volatility(now),
            volume=_bullish_volume(now),
            options=None, smart_money=None, vix_value=None,
        )
        assert any(a.type == "REVERSAL_WARNING" for a in alerts)

    def test_golden_cross_alert(self):
        now = datetime.now()
        trend = _bullish_trend(now)
        # Manually set golden cross
        object.__setattr__(trend, "golden_cross", True) if hasattr(trend, "__dataclass_fields__") else None
        trend = TrendSummary(
            **{**trend.__dict__, "golden_cross": True, "death_cross": False}
        )
        alerts = TechnicalAggregator._generate_alerts(
            trend=trend,
            momentum=_neutral_momentum(now),
            volatility=_normal_volatility(now),
            volume=_bullish_volume(now),
            options=None, smart_money=None, vix_value=None,
        )
        assert any("Golden cross" in a.message for a in alerts)


class TestConfidenceModifiers:
    """Test that confidence responds to volatility, ADX, and data completeness."""

    def test_high_volatility_reduces_confidence(self):
        now = datetime.now()
        votes = {"trend": "BULLISH", "momentum": "BULLISH", "volume": "BULLISH"}

        conf_normal = TechnicalAggregator._calculate_confidence(
            votes=votes,
            volatility=_normal_volatility(now),
            trend=_bullish_trend(now),
            data_completeness=1.0,
        )
        conf_extreme = TechnicalAggregator._calculate_confidence(
            votes=votes,
            volatility=_extreme_volatility(now),
            trend=_bullish_trend(now),
            data_completeness=1.0,
        )
        assert conf_normal > conf_extreme

    def test_weak_trend_reduces_confidence(self):
        now = datetime.now()
        votes = {"trend": "BULLISH", "momentum": "BULLISH"}

        conf_strong = TechnicalAggregator._calculate_confidence(
            votes=votes,
            volatility=_normal_volatility(now),
            trend=_bullish_trend(now),  # STRONG
            data_completeness=1.0,
        )
        conf_weak = TechnicalAggregator._calculate_confidence(
            votes=votes,
            volatility=_normal_volatility(now),
            trend=_neutral_trend(now),  # WEAK
            data_completeness=1.0,
        )
        assert conf_strong > conf_weak

    def test_missing_data_reduces_confidence(self):
        now = datetime.now()
        votes = {"trend": "BULLISH", "momentum": "BULLISH"}

        conf_full = TechnicalAggregator._calculate_confidence(
            votes=votes,
            volatility=_normal_volatility(now),
            trend=_bullish_trend(now),
            data_completeness=1.0,
        )
        conf_partial = TechnicalAggregator._calculate_confidence(
            votes=votes,
            volatility=_normal_volatility(now),
            trend=_bullish_trend(now),
            data_completeness=0.5,
        )
        assert conf_full > conf_partial

    def test_confidence_floor(self):
        now = datetime.now()
        votes = {"a": "BULLISH", "b": "BEARISH", "c": "NEUTRAL"}
        conf = TechnicalAggregator._calculate_confidence(
            votes=votes,
            volatility=_extreme_volatility(now),
            trend=_neutral_trend(now),  # WEAK
            data_completeness=0.3,
        )
        assert conf >= 0.2

    def test_confidence_ceiling(self):
        now = datetime.now()
        votes = {"a": "BULLISH", "b": "BULLISH", "c": "BULLISH", "d": "BULLISH"}
        conf = TechnicalAggregator._calculate_confidence(
            votes=votes,
            volatility=_normal_volatility(now),
            trend=_bullish_trend(now),
            data_completeness=1.0,
        )
        assert conf <= 0.9


class TestDataCompleteness:
    """Test data_completeness calculation."""

    def test_all_categories_available(self):
        now = datetime.now()
        df = _ohlcv(100)

        with (
            patch("src.analysis.technical_aggregator.TrendIndicators") as MockTrend,
            patch("src.analysis.technical_aggregator.MomentumIndicators") as MockMom,
            patch("src.analysis.technical_aggregator.VolatilityIndicators") as MockVol,
            patch("src.analysis.technical_aggregator.VolumeIndicators") as MockVolu,
            patch("src.analysis.technical_aggregator.OptionsIndicators") as MockOpts,
            patch("src.analysis.technical_aggregator.QuantIndicators") as MockQuant,
            patch("src.analysis.technical_aggregator.SmartMoneyIndicators") as MockSM,
        ):
            MockTrend.return_value.get_trend_summary.return_value = _bullish_trend(now)
            MockTrend.return_value.calculate_ema.return_value = pd.Series([20000.0] * 100)
            MockMom.return_value.get_momentum_summary.return_value = _bullish_momentum(now)
            MockVol.return_value.get_volatility_summary.return_value = _normal_volatility(now)
            MockVol.return_value.calculate_bollinger_bands.return_value = MagicMock(
                upper=pd.Series([20400.0]), lower=pd.Series([19600.0])
            )
            MockVolu.return_value.get_volume_summary.return_value = _bullish_volume(now)
            MockOpts.return_value.get_options_summary.return_value = _bullish_options(now)
            MockQuant.return_value.get_quant_summary.return_value = _bullish_quant(now)
            MockSM.return_value.calculate_smart_money_score.return_value = _bullish_smart_money()

            agg = TechnicalAggregator()
            result = agg.analyze("NIFTY50", df, options_chain=MagicMock())

        assert result.data_completeness == 1.0

    def test_missing_options_and_smart_money(self):
        now = datetime.now()
        df = _ohlcv(100)

        with (
            patch("src.analysis.technical_aggregator.TrendIndicators") as MockTrend,
            patch("src.analysis.technical_aggregator.MomentumIndicators") as MockMom,
            patch("src.analysis.technical_aggregator.VolatilityIndicators") as MockVol,
            patch("src.analysis.technical_aggregator.VolumeIndicators") as MockVolu,
            patch("src.analysis.technical_aggregator.QuantIndicators") as MockQuant,
            patch("src.analysis.technical_aggregator.SmartMoneyIndicators") as MockSM,
        ):
            MockTrend.return_value.get_trend_summary.return_value = _bullish_trend(now)
            MockTrend.return_value.calculate_ema.return_value = pd.Series([20000.0] * 100)
            MockMom.return_value.get_momentum_summary.return_value = _bullish_momentum(now)
            MockVol.return_value.get_volatility_summary.return_value = _normal_volatility(now)
            MockVol.return_value.calculate_bollinger_bands.return_value = MagicMock(
                upper=pd.Series([20400.0]), lower=pd.Series([19600.0])
            )
            MockVolu.return_value.get_volume_summary.return_value = _bullish_volume(now)
            MockQuant.return_value.get_quant_summary.return_value = _bullish_quant(now)
            # Smart money returns None (low data completeness)
            sm_low = SmartMoneyScore(
                score=10.0, grade="C",
                smfi_component=50.0, vsd_component=50.0, btd_component=50.0,
                oimi_component=50.0, lai_component=50.0,
                smart_money_bias="NEUTRAL", key_finding="Low data",
                actionable_insight="Not enough data",
                data_completeness=0.2, confidence=0.2,
            )
            MockSM.return_value.calculate_smart_money_score.return_value = sm_low

            agg = TechnicalAggregator()
            result = agg.analyze("NIFTY_IT", df)  # No options

        # Options + smart money missing = 5/7
        assert result.data_completeness == pytest.approx(5 / 7, abs=0.01)
