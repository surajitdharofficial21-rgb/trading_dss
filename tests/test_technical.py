"""Tests for technical analysis indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.technical import (
    ema, rsi, bollinger_bands, atr, pivot_points,
    support_resistance_levels, TechnicalAnalysisError,
)


@pytest.fixture()
def close_series() -> pd.Series:
    rng = np.random.default_rng(99)
    return pd.Series(20000.0 + rng.standard_normal(100).cumsum() * 50)


class TestEMA:
    def test_length(self, close_series: pd.Series) -> None:
        result = ema(close_series, 20)
        assert len(result) == len(close_series)

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(TechnicalAnalysisError):
            ema(pd.Series([1.0, 2.0]), 20)

    def test_values_reasonable(self, close_series: pd.Series) -> None:
        result = ema(close_series, 10)
        assert not result.isna().all()
        assert result.min() > 0


class TestRSI:
    def test_range(self, close_series: pd.Series) -> None:
        result = rsi(close_series)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_period_default(self, close_series: pd.Series) -> None:
        result = rsi(close_series, 14)
        assert len(result) == len(close_series)


class TestBollingerBands:
    def test_columns(self, close_series: pd.Series) -> None:
        result = bollinger_bands(close_series)
        assert set(result.columns) == {"bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_pct_b"}

    def test_upper_above_lower(self, close_series: pd.Series) -> None:
        result = bollinger_bands(close_series).dropna()
        assert (result["bb_upper"] >= result["bb_lower"]).all()

    def test_mid_between_bands(self, close_series: pd.Series) -> None:
        result = bollinger_bands(close_series).dropna()
        assert (result["bb_mid"] >= result["bb_lower"]).all()
        assert (result["bb_mid"] <= result["bb_upper"]).all()


class TestATR:
    def test_length(self, sample_ohlcv: pd.DataFrame) -> None:
        result = atr(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)

    def test_non_negative(self, sample_ohlcv: pd.DataFrame) -> None:
        result = atr(sample_ohlcv).dropna()
        assert (result >= 0).all()


class TestPivotPoints:
    def test_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        result = pivot_points(sample_ohlcv)
        assert set(result.columns) == {"P", "R1", "R2", "R3", "S1", "S2", "S3"}

    def test_resistances_above_pivot(self, sample_ohlcv: pd.DataFrame) -> None:
        result = pivot_points(sample_ohlcv).iloc[0]
        assert result["R1"] > result["P"]
        assert result["R2"] > result["R1"]
        assert result["R3"] > result["R2"]

    def test_supports_below_pivot(self, sample_ohlcv: pd.DataFrame) -> None:
        result = pivot_points(sample_ohlcv).iloc[0]
        assert result["S1"] < result["P"]
        assert result["S2"] < result["S1"]
        assert result["S3"] < result["S2"]
