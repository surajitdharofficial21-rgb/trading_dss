"""Tests for data validation and sanitization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.data_validator import (
    validate_ohlcv,
    sanitize_ohlcv,
    validate_price_tick,
    DataValidationError,
)


class TestValidateOHLCV:
    def test_valid_data_passes(self, sample_ohlcv: pd.DataFrame) -> None:
        result = validate_ohlcv(sample_ohlcv, "NIFTY50")
        assert result.is_valid

    def test_missing_columns(self) -> None:
        df = pd.DataFrame({"open": [100.0], "close": [101.0]})
        result = validate_ohlcv(df)
        assert not result.is_valid
        assert any("Missing" in e for e in result.errors)

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = validate_ohlcv(df)
        assert not result.is_valid

    def test_high_below_low_detected(self) -> None:
        df = pd.DataFrame({
            "open": [100.0],
            "high": [99.0],   # high < low — invalid
            "low": [100.0],
            "close": [100.0],
            "volume": [1000],
        }, index=pd.to_datetime(["2024-01-01"]))
        result = validate_ohlcv(df)
        assert not result.is_valid
        assert any("high < low" in e for e in result.errors)

    def test_negative_price_detected(self) -> None:
        df = pd.DataFrame({
            "open": [-100.0],
            "high": [100.0],
            "low": [-100.0],
            "close": [100.0],
            "volume": [1000],
        }, index=pd.to_datetime(["2024-01-01"]))
        result = validate_ohlcv(df)
        assert not result.is_valid


class TestSanitizeOHLCV:
    def test_negative_prices_become_nan_then_ffill(self) -> None:
        df = pd.DataFrame({
            "open": [100.0, -1.0, 102.0],
            "high": [101.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 102.0, 103.0],
            "volume": [1000, 2000, 3000],
        })
        result = sanitize_ohlcv(df)
        assert result["open"].iloc[1] == 100.0  # forward-filled

    def test_negative_volume_clipped(self) -> None:
        df = pd.DataFrame({
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [-500],
        })
        result = sanitize_ohlcv(df)
        assert result["volume"].iloc[0] == 0


class TestValidatePriceTick:
    def test_valid_price(self) -> None:
        assert validate_price_tick(22000.0, "NIFTY50") == 22000.0

    def test_negative_price_raises(self) -> None:
        with pytest.raises(DataValidationError):
            validate_price_tick(-1.0, "NIFTY50")

    def test_zero_price_raises(self) -> None:
        with pytest.raises(DataValidationError):
            validate_price_tick(0.0, "NIFTY50")

    def test_infinite_price_raises(self) -> None:
        with pytest.raises(DataValidationError):
            validate_price_tick(float("inf"), "NIFTY50")
