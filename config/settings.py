"""
Application settings loaded from environment variables.

All configurable parameters are defined here using pydantic BaseSettings.
Values are read from .env file with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Project root — two levels up from this file (config/settings.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MarketHoursSettings(BaseSettings):
    """Indian market session timings (all times in IST / Asia/Kolkata)."""

    pre_market_open: str = Field(default="09:00", description="Pre-market session start HH:MM")
    market_open: str = Field(default="09:15", description="Regular session open HH:MM")
    market_close: str = Field(default="15:30", description="Regular session close HH:MM")
    post_market_close: str = Field(default="16:00", description="Post-market session end HH:MM")

    model_config = {"env_prefix": "MARKET_HOURS_", "extra": "ignore"}


class DatabaseSettings(BaseSettings):
    """SQLite database configuration."""

    path: Path = Field(
        default=PROJECT_ROOT / "data" / "db" / "trading_dss.db",
        description="Absolute path to SQLite database file",
    )
    wal_mode: bool = Field(default=True, description="Enable WAL journal mode for concurrency")
    connection_pool_size: int = Field(default=5, ge=1, le=20)
    timeout_seconds: int = Field(default=30, ge=5, le=120)

    model_config = {"env_prefix": "DB_", "extra": "ignore"}


class RateLimitSettings(BaseSettings):
    """Per-source HTTP rate limit configuration."""

    nse_requests_per_minute: int = Field(default=30, ge=1, le=120)
    bse_requests_per_minute: int = Field(default=20, ge=1, le=120)
    yfinance_requests_per_minute: int = Field(default=60, ge=1, le=120)
    news_requests_per_minute: int = Field(default=30, ge=1, le=120)

    model_config = {"env_prefix": "RATE_LIMIT_", "extra": "ignore"}


class PollingSettings(BaseSettings):
    """Data refresh intervals in seconds."""

    price_interval: int = Field(default=60, ge=15, description="Spot price refresh interval (s)")
    options_interval: int = Field(default=180, ge=60, description="Options chain refresh interval (s)")
    news_interval: int = Field(default=120, ge=60, description="News/RSS refresh interval (s)")
    fii_dii_interval: int = Field(default=900, ge=300, description="FII/DII data refresh interval (s)")
    vix_interval: int = Field(default=60, ge=15, description="VIX refresh interval (s)")

    model_config = {"env_prefix": "POLL_", "extra": "ignore"}


class SignalThresholds(BaseSettings):
    """Thresholds that trigger trading signal conditions."""

    oi_spike_threshold: float = Field(
        default=10.0, ge=1.0,
        description="OI change % to flag as spike",
    )
    volume_spike_multiplier: float = Field(
        default=1.5, ge=1.0,
        description="Multiple of 20-day avg volume to flag spike",
    )
    pcr_extreme_low: float = Field(
        default=0.7, ge=0.1,
        description="PCR below this is considered extreme bullish",
    )
    pcr_extreme_high: float = Field(
        default=1.3, le=5.0,
        description="PCR above this is considered extreme bearish",
    )
    vix_low_threshold: float = Field(default=12.0, description="VIX below this → complacency")
    vix_high_threshold: float = Field(default=20.0, description="VIX above this → fear zone")
    vix_extreme_threshold: float = Field(default=30.0, description="VIX above this → extreme fear")
    # Regime classification thresholds (used by VIXTracker.get_vix_regime)
    vix_normal_threshold: float = Field(default=13.0, description="VIX below this → LOW_VOL regime")
    vix_elevated_threshold: float = Field(default=18.0, description="VIX above this → ELEVATED regime")
    vix_panic_threshold: float = Field(default=25.0, description="VIX above this → HIGH_VOL regime")
    sentiment_bullish_threshold: float = Field(default=0.3, description="Sentiment score for bullish")
    sentiment_bearish_threshold: float = Field(default=-0.3, description="Sentiment score for bearish")
    rsi_overbought: float = Field(default=70.0, ge=50.0, le=90.0)
    rsi_oversold: float = Field(default=30.0, ge=10.0, le=50.0)

    model_config = {"env_prefix": "THRESHOLD_", "extra": "ignore"}


class TelegramSettings(BaseSettings):
    """Telegram bot configuration."""

    bot_token: Optional[str] = Field(default=None, description="Telegram bot API token")
    chat_id: Optional[str] = Field(default=None, description="Target chat/channel ID")
    alert_min_confidence: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Minimum signal confidence to send alert",
    )

    model_config = {"env_prefix": "TELEGRAM_", "extra": "ignore"}


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Root log level")
    log_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "logs",
        description="Directory where log files are written",
    )
    max_bytes: int = Field(default=10 * 1024 * 1024, description="Max log file size (10 MB)")
    backup_count: int = Field(default=5, description="Number of rotated log backups")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Ensure log level is a valid Python logging level."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log level must be one of {valid}")
        return v.upper()

    model_config = {"env_prefix": "LOG_", "extra": "ignore"}


class Settings(BaseSettings):
    """
    Root application settings.

    Loads from .env file in project root. All nested settings groups
    delegate to their own env prefixes.
    """

    # ── Application identity ────────────────────────────────────────────────
    app_name: str = Field(default="trading_dss", description="Application name")
    environment: str = Field(default="development", description="development | staging | production")
    debug: bool = Field(default=False, description="Enable debug mode")

    # ── File paths ──────────────────────────────────────────────────────────
    config_dir: Path = Field(default=PROJECT_ROOT / "config")
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    cache_dir: Path = Field(default=PROJECT_ROOT / "data" / "cache")

    # ── Derived config file paths ───────────────────────────────────────────
    @property
    def indices_config_path(self) -> Path:
        return self.config_dir / "indices.json"

    @property
    def news_mappings_path(self) -> Path:
        return self.config_dir / "news_mappings.json"

    @property
    def rss_feeds_path(self) -> Path:
        return self.config_dir / "rss_feeds.json"

    @property
    def sentiment_keywords_path(self) -> Path:
        return self.config_dir / "sentiment_keywords.json"

    # ── Nested settings ─────────────────────────────────────────────────────
    market_hours: MarketHoursSettings = Field(default_factory=MarketHoursSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    rate_limits: RateLimitSettings = Field(default_factory=RateLimitSettings)
    polling: PollingSettings = Field(default_factory=PollingSettings)
    thresholds: SignalThresholds = Field(default_factory=SignalThresholds)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "env_nested_delimiter": "__",
    }


# Module-level singleton — import this everywhere instead of constructing your own
settings = Settings()
