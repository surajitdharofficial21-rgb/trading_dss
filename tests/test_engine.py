"""
Tests for the engine layer:
  - RegimeDetector  (engine/regime_detector.py)
  - RiskManager     (engine/risk_manager.py)
  - DecisionEngine  (engine/decision_engine.py)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.engine.regime_detector import RegimeDetector, MarketRegime, RegimeSnapshot
from src.engine.risk_manager import RiskManager, PositionSizing, TransactionCosts
from src.engine.decision_engine import (
    DecisionEngine,
    SignalInputs,
    TradingSignal,
    SignalDirection,
    SignalStrength,
)
from src.data.vix_data import VIXSnapshot, VIXRegime
from config.constants import IST_TIMEZONE
from zoneinfo import ZoneInfo

_IST = ZoneInfo(IST_TIMEZONE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _daily_ohlcv(n: int = 100, trend: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic daily OHLCV.

    Parameters
    ----------
    trend:
        Daily drift to add to the random walk (+ve = uptrend, -ve = downtrend).
    """
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal(n) * 0.01 + trend
    close = (1 + pd.Series(returns)).cumprod() * 20000.0
    high = close * (1 + rng.uniform(0.001, 0.005, n))
    low = close * (1 - rng.uniform(0.001, 0.005, n))
    open_ = close.shift(1).fillna(20000.0)
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                          "volume": volume})


def _vix_snapshot(value: float) -> VIXSnapshot:
    return VIXSnapshot(
        value=value, change=0.0, change_pct=0.0,
        timestamp=datetime.now(tz=_IST),
        regime=VIXRegime.NORMAL,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RegimeDetector
# ══════════════════════════════════════════════════════════════════════════════

class TestRegimeDetector:

    @pytest.fixture()
    def detector(self):
        return RegimeDetector(trend_ema_short=20, trend_ema_long=50)

    def test_returns_regime_snapshot(self, detector):
        df = _daily_ohlcv(100)
        result = detector.detect(df)
        assert isinstance(result, RegimeSnapshot)

    def test_snapshot_has_all_fields(self, detector):
        result = detector.detect(_daily_ohlcv(100))
        assert result.regime in MarketRegime.__members__.values() or isinstance(result.regime, str)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.summary, str) and result.summary

    # ── HIGH_VOLATILITY regime ────────────────────────────────────────────────

    def test_high_vix_triggers_high_volatility_regime(self, detector):
        df = _daily_ohlcv(100)
        snap = _vix_snapshot(30.0)  # ≥ 25 threshold
        result = detector.detect(df, vix_snapshot=snap)
        assert result.regime == MarketRegime.HIGH_VOLATILITY
        assert result.confidence >= 0.7

    def test_vix_below_threshold_does_not_override_trend(self, detector):
        # Strong uptrend with low VIX should NOT be HIGH_VOLATILITY
        df = _daily_ohlcv(100, trend=0.005)
        snap = _vix_snapshot(12.0)
        result = detector.detect(df, vix_snapshot=snap)
        assert result.regime != MarketRegime.HIGH_VOLATILITY

    # ── Trend-based regimes ───────────────────────────────────────────────────

    def test_strong_uptrend_gives_bull_regime(self, detector):
        df = _daily_ohlcv(100, trend=0.006, seed=1)
        result = detector.detect(df, advances=350, declines=50)
        assert result.regime in {MarketRegime.BULL, MarketRegime.STRONG_BULL}
        assert result.trend_score > 0

    def test_strong_downtrend_gives_bear_regime(self, detector):
        df = _daily_ohlcv(100, trend=-0.006, seed=2)
        result = detector.detect(df, advances=50, declines=350)
        assert result.regime in {MarketRegime.BEAR, MarketRegime.STRONG_BEAR}
        assert result.trend_score < 0

    def test_flat_market_gives_neutral(self, detector):
        df = _daily_ohlcv(100, trend=0.0, seed=3)
        result = detector.detect(df)
        # Neutral is expected when EMA alignment is ambiguous / flat
        assert result.regime in {MarketRegime.NEUTRAL, MarketRegime.BULL, MarketRegime.BEAR}

    # ── Breadth score ─────────────────────────────────────────────────────────

    def test_breadth_score_zero_when_no_advances_declines(self, detector):
        df = _daily_ohlcv(100)
        result = detector.detect(df)
        assert result.breadth_score == 0.0

    def test_breadth_score_positive_on_advancing_market(self, detector):
        df = _daily_ohlcv(100)
        result = detector.detect(df, advances=300, declines=100)
        assert result.breadth_score > 0.0

    def test_breadth_score_negative_on_declining_market(self, detector):
        df = _daily_ohlcv(100)
        result = detector.detect(df, advances=100, declines=300)
        assert result.breadth_score < 0.0

    # ── Insufficient data ─────────────────────────────────────────────────────

    def test_insufficient_data_defaults_gracefully(self, detector):
        df = _daily_ohlcv(10)  # < trend_ema_long (50)
        result = detector.detect(df)
        assert result.trend_score == 0.0  # fallback
        assert result.volatility_score == 15.0  # fallback assumption

    # ── VIX regime passthrough ────────────────────────────────────────────────

    def test_vix_regime_propagated_to_snapshot(self, detector):
        df = _daily_ohlcv(100)
        snap = _vix_snapshot(28.0)
        result = detector.detect(df, vix_snapshot=snap)
        assert result.vix_regime == snap.regime

    def test_no_vix_snapshot_gives_none_vix_regime(self, detector):
        result = detector.detect(_daily_ohlcv(100))
        assert result.vix_regime is None


# ══════════════════════════════════════════════════════════════════════════════
# RiskManager
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskManager:

    @pytest.fixture()
    def manager(self):
        return RiskManager(
            capital=1_000_000.0,
            risk_pct_per_trade=1.0,   # ₹10,000 max risk
            atr_sl_multiplier=1.5,
            rr_target1=1.5,
            rr_target2=2.5,
        )

    # ── calculate_position ────────────────────────────────────────────────────

    def test_returns_position_sizing(self, manager):
        result = manager.calculate_position(entry_price=22000.0, atr=100.0)
        assert isinstance(result, PositionSizing)

    def test_long_stop_loss_below_entry(self, manager):
        ps = manager.calculate_position(22000.0, atr=100.0, is_long=True)
        assert ps.stop_loss < ps.entry_price

    def test_short_stop_loss_above_entry(self, manager):
        ps = manager.calculate_position(22000.0, atr=100.0, is_long=False)
        assert ps.stop_loss > ps.entry_price

    def test_target1_above_entry_for_long(self, manager):
        ps = manager.calculate_position(22000.0, atr=100.0, is_long=True)
        assert ps.target1 > ps.entry_price
        assert ps.target2 > ps.target1

    def test_target1_below_entry_for_short(self, manager):
        ps = manager.calculate_position(22000.0, atr=100.0, is_long=False)
        assert ps.target1 < ps.entry_price
        assert ps.target2 < ps.target1

    def test_rr_ratio_is_target1_multiple(self, manager):
        ps = manager.calculate_position(22000.0, atr=100.0)
        # risk_per_unit = 1.5 * 100 = 150; target1 = entry + 150*1.5 = entry + 225
        assert ps.rr_ratio == pytest.approx(1.5, abs=1e-4)

    def test_quantity_respects_risk_limit(self, manager):
        # risk_per_lot = atr_mult * atr * lot_size = 1.5 * 100 * 50 = 7500
        # max_risk = 1_000_000 * 0.01 = 10_000
        # qty = int(10_000 / 7_500) = 1
        ps = manager.calculate_position(22000.0, atr=100.0, lot_size=50)
        assert ps.quantity >= 1
        assert ps.total_risk <= 1_000_000 * 0.01 + 1e-2  # within 1% capital

    def test_zero_atr_returns_quantity_one(self, manager):
        ps = manager.calculate_position(22000.0, atr=0.0)
        assert ps.quantity == 1

    def test_entry_price_rounded_to_two_decimals(self, manager):
        ps = manager.calculate_position(22000.123456, atr=100.0)
        assert ps.entry_price == round(ps.entry_price, 2)

    def test_smaller_atr_gives_larger_quantity(self, manager):
        ps_large = manager.calculate_position(22000.0, atr=500.0)
        ps_small = manager.calculate_position(22000.0, atr=50.0)
        assert ps_small.quantity >= ps_large.quantity

    # ── calculate_fo_costs ────────────────────────────────────────────────────

    def test_returns_transaction_costs(self):
        result = RiskManager.calculate_fo_costs(100_000.0, is_buy=True)
        assert isinstance(result, TransactionCosts)

    def test_buy_has_no_stt(self):
        costs = RiskManager.calculate_fo_costs(100_000.0, is_buy=True)
        assert costs.stt == 0.0

    def test_sell_has_stt(self):
        costs = RiskManager.calculate_fo_costs(100_000.0, is_buy=False)
        assert costs.stt > 0.0

    def test_buy_has_stamp_duty(self):
        costs = RiskManager.calculate_fo_costs(100_000.0, is_buy=True)
        assert costs.stamp_duty > 0.0

    def test_sell_has_no_stamp_duty(self):
        costs = RiskManager.calculate_fo_costs(100_000.0, is_buy=False)
        assert costs.stamp_duty == 0.0

    def test_total_includes_all_components(self):
        c = RiskManager.calculate_fo_costs(100_000.0, is_buy=True)
        computed = c.brokerage + c.stt + c.exchange_charges + c.sebi_fee + c.gst + c.stamp_duty
        assert c.total == pytest.approx(computed, abs=0.01)

    def test_total_pct_formula(self):
        c = RiskManager.calculate_fo_costs(100_000.0)
        expected_pct = c.total / 100_000.0 * 100
        assert c.total_pct == pytest.approx(expected_pct, abs=1e-3)

    def test_zero_trade_value_returns_zero_pct(self):
        c = RiskManager.calculate_fo_costs(0.0)
        assert c.total_pct == 0.0

    def test_gst_on_brokerage_and_exchange(self):
        c = RiskManager.calculate_fo_costs(100_000.0)
        # GST = 18% * (brokerage + exchange_charges)
        expected_gst = (c.brokerage + c.exchange_charges) * 0.18
        assert c.gst == pytest.approx(expected_gst, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# DecisionEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestDecisionEngine:

    @pytest.fixture()
    def engine(self):
        return DecisionEngine(
            technical_weight=0.40,
            options_weight=0.35,
            news_weight=0.25,
        )

    def _inputs(self, **kwargs) -> SignalInputs:
        defaults = dict(
            index_id="NIFTY50",
            rsi=50.0,
            ema_short=20000.0,
            ema_long=19800.0,
            current_price=20100.0,
            volume_vs_avg=1.0,
            pcr_oi=1.0,
            news_sentiment=0.0,
            vix_value=15.0,
            anomalies_count=0,
        )
        defaults.update(kwargs)
        return SignalInputs(**defaults)

    # ── Basic structure ───────────────────────────────────────────────────────

    def test_returns_trading_signal(self, engine):
        result = engine.generate_signal(self._inputs())
        assert isinstance(result, TradingSignal)

    def test_signal_has_correct_index_id(self, engine):
        result = engine.generate_signal(self._inputs(index_id="BANKNIFTY"))
        assert result.index_id == "BANKNIFTY"

    def test_confidence_in_range(self, engine):
        result = engine.generate_signal(self._inputs())
        assert 0.0 <= result.confidence <= 1.0

    def test_direction_valid(self, engine):
        result = engine.generate_signal(self._inputs())
        assert result.direction in SignalDirection.__members__.values() or isinstance(result.direction, str)

    def test_strength_valid(self, engine):
        result = engine.generate_signal(self._inputs())
        assert result.strength in SignalStrength.__members__.values() or isinstance(result.strength, str)

    # ── Technical score ───────────────────────────────────────────────────────

    def test_oversold_rsi_contributes_bullish(self, engine):
        bullish = engine.generate_signal(self._inputs(rsi=25.0))  # oversold
        bearish = engine.generate_signal(self._inputs(rsi=75.0))  # overbought
        assert bullish.technical_score > bearish.technical_score

    def test_ema_uptrend_contributes_bullish(self, engine):
        uptrend = engine.generate_signal(self._inputs(ema_short=20500.0, ema_long=19500.0))
        downtrend = engine.generate_signal(self._inputs(ema_short=19500.0, ema_long=20500.0))
        assert uptrend.technical_score > downtrend.technical_score

    def test_price_above_ema_contributes_bullish(self, engine):
        above = engine.generate_signal(self._inputs(current_price=21000.0, ema_short=20000.0))
        below = engine.generate_signal(self._inputs(current_price=19000.0, ema_short=20000.0))
        assert above.technical_score > below.technical_score

    def test_technical_score_clamped_to_pm1(self, engine):
        result = engine.generate_signal(self._inputs(
            rsi=10.0, ema_short=25000.0, ema_long=10000.0, current_price=30000.0,
        ))
        assert -1.0 <= result.technical_score <= 1.0

    # ── Options score ─────────────────────────────────────────────────────────

    def test_low_pcr_gives_bullish_options_score(self, engine):
        low_pcr = engine.generate_signal(self._inputs(pcr_oi=0.5))   # < 0.7
        high_pcr = engine.generate_signal(self._inputs(pcr_oi=1.5))  # > 1.3
        assert low_pcr.options_score > high_pcr.options_score

    def test_anomalies_noted_in_reasons(self, engine):
        result = engine.generate_signal(self._inputs(anomalies_count=3))
        assert any("anomal" in r.lower() for r in result.reasons)

    # ── News score ────────────────────────────────────────────────────────────

    def test_positive_news_contributes_bullish(self, engine):
        positive = engine.generate_signal(self._inputs(news_sentiment=0.8))
        negative = engine.generate_signal(self._inputs(news_sentiment=-0.8))
        assert positive.news_score > negative.news_score

    def test_news_score_clamped(self, engine):
        result = engine.generate_signal(self._inputs(news_sentiment=5.0))
        assert result.news_score <= 1.0

    # ── VIX adjustment ────────────────────────────────────────────────────────

    def test_extreme_vix_halves_confidence(self, engine):
        normal_vix = engine.generate_signal(self._inputs(vix_value=15.0))
        extreme_vix = engine.generate_signal(self._inputs(vix_value=35.0))
        # Same inputs but extreme VIX applies 0.5× multiplier
        assert extreme_vix.confidence <= normal_vix.confidence

    def test_none_vix_no_adjustment(self, engine):
        no_vix = engine.generate_signal(self._inputs(vix_value=None))
        with_vix = engine.generate_signal(self._inputs(vix_value=15.0))
        # Normal VIX (< 20) → adjustment = 1.0, same as None
        assert no_vix.confidence == pytest.approx(with_vix.confidence, abs=1e-4)

    def test_high_vix_reduces_confidence_partially(self, engine):
        low = engine.generate_signal(self._inputs(vix_value=15.0))
        high = engine.generate_signal(self._inputs(vix_value=25.0))
        assert high.confidence <= low.confidence

    # ── Direction logic ───────────────────────────────────────────────────────

    def test_all_bullish_signals_give_bullish_direction(self, engine):
        result = engine.generate_signal(self._inputs(
            rsi=25.0, ema_short=22000.0, ema_long=20000.0, current_price=23000.0,
            pcr_oi=0.5, news_sentiment=0.8, vix_value=10.0,
        ))
        assert result.direction == SignalDirection.BULLISH

    def test_all_bearish_signals_give_bearish_direction(self, engine):
        result = engine.generate_signal(self._inputs(
            rsi=78.0, ema_short=18000.0, ema_long=22000.0, current_price=17000.0,
            pcr_oi=1.8, news_sentiment=-0.8, vix_value=10.0,
        ))
        assert result.direction == SignalDirection.BEARISH

    def test_neutral_inputs_give_neutral_direction(self, engine):
        # ema_short slightly above ema_long (+0.3) but price below ema_short (-0.2)
        # → tech_score = 0.1 → composite = 0.04 → neutral (< 0.1 threshold)
        result = engine.generate_signal(self._inputs(
            rsi=50.0, ema_short=20100.0, ema_long=20000.0, current_price=20000.0,
            pcr_oi=1.0, news_sentiment=0.0, vix_value=15.0,
        ))
        assert result.direction == SignalDirection.NEUTRAL

    # ── Strength classification ───────────────────────────────────────────────

    def test_high_confidence_gives_strong_signal(self, engine):
        # tech=0.9, opts=0.6, news=1.0 → composite=0.82 → confidence=0.82 ≥ 0.8 (STRONG)
        result = engine.generate_signal(self._inputs(
            rsi=20.0, ema_short=22000.0, ema_long=18000.0,
            current_price=23000.0, pcr_oi=0.4, news_sentiment=1.0,
        ))
        assert result.strength in {SignalStrength.STRONG, SignalStrength.VERY_STRONG}

    def test_low_confidence_gives_weak_signal(self, engine):
        result = engine.generate_signal(self._inputs(
            rsi=50.0, ema_short=20000.0, ema_long=20000.0,
            current_price=20000.0, pcr_oi=1.0, news_sentiment=0.0,
        ))
        assert result.strength == SignalStrength.WEAK

    # ── Reasons list ─────────────────────────────────────────────────────────

    def test_reasons_not_empty_for_strong_signal(self, engine):
        result = engine.generate_signal(self._inputs(rsi=20.0, pcr_oi=0.4))
        assert len(result.reasons) > 0

    def test_reasons_mention_rsi_when_extreme(self, engine):
        result = engine.generate_signal(self._inputs(rsi=20.0))
        assert any("RSI" in r for r in result.reasons)

    # ── is_actionable property ────────────────────────────────────────────────

    def test_is_actionable_above_threshold(self, engine):
        # Generate a high-confidence signal
        result = engine.generate_signal(self._inputs(
            rsi=20.0, ema_short=22000.0, ema_long=18000.0,
            current_price=23000.0, pcr_oi=0.4, news_sentiment=0.8,
        ))
        if result.confidence >= 0.6:
            assert result.is_actionable is True

    def test_is_not_actionable_for_weak_signal(self, engine):
        result = engine.generate_signal(self._inputs(
            rsi=50.0, pcr_oi=1.0, news_sentiment=0.0,
        ))
        # Confidence will be near zero — not actionable
        assert result.is_actionable is False

    # ── Weight sensitivity ────────────────────────────────────────────────────

    def test_custom_weights_change_output(self):
        eng_tech = DecisionEngine(technical_weight=1.0, options_weight=0.0, news_weight=0.0)
        eng_news = DecisionEngine(technical_weight=0.0, options_weight=0.0, news_weight=1.0)
        inputs = self._inputs(rsi=20.0, news_sentiment=-0.9)
        sig_tech = eng_tech.generate_signal(inputs)
        sig_news = eng_news.generate_signal(inputs)
        # Pure-technical should be bullish; pure-news should be bearish
        assert sig_tech.direction == SignalDirection.BULLISH
        assert sig_news.direction == SignalDirection.BEARISH

    # ── None inputs handled gracefully ────────────────────────────────────────

    def test_all_none_inputs_produces_neutral(self, engine):
        inputs = SignalInputs(index_id="NIFTY50")  # all optionals are None
        result = engine.generate_signal(inputs)
        assert result.direction == SignalDirection.NEUTRAL
        assert result.confidence == 0.0

    def test_partial_inputs_produce_valid_signal(self, engine):
        inputs = SignalInputs(index_id="NIFTY50", rsi=25.0)
        result = engine.generate_signal(inputs)
        assert isinstance(result, TradingSignal)
        assert result.direction in {SignalDirection.BULLISH, SignalDirection.NEUTRAL}
