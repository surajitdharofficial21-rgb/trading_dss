"""
Structured logging configuration for trading_dss.

Sets up:
- Console handler (INFO+) with ANSI colour output
- Rotating file handler (DEBUG+) — 10 MB max, 5 backups
- Separate error-only rotating file
- All timestamps in IST (Asia/Kolkata)
"""

from __future__ import annotations

import logging
import logging.handlers
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# IST offset: UTC+5:30
_IST = timezone(timedelta(hours=5, minutes=30))

LOG_FORMAT: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


class _ISTFormatter(logging.Formatter):
    """Formatter that forces timestamps to IST regardless of server timezone."""

    def converter(self, timestamp: float) -> datetime:  # type: ignore[override]
        return datetime.fromtimestamp(timestamp, tz=_IST)

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        dt = self.converter(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime(DATE_FORMAT)


# ANSI colour codes for console output
_COLOURS: dict[str, str] = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",
}


class _ColourFormatter(_ISTFormatter):
    """Console formatter that colours the level name."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, _COLOURS["RESET"])
        reset = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname:<8}{reset}"
        return super().format(record)


def setup_logging(
    log_dir: Path,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configure the root logger with console and rotating file handlers.

    Parameters
    ----------
    log_dir:
        Directory where log files are written. Created if it does not exist.
    console_level:
        Minimum level written to stdout.
    file_level:
        Minimum level written to the rotating log file.
    max_bytes:
        Maximum size of each log file before rotation (default 10 MB).
    backup_count:
        Number of rotated backup files to keep.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers filter individually

    # ── Console handler ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(_ColourFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))

    # ── Rotating main log ────────────────────────────────────────────────────
    main_log_path = log_dir / "trading_dss.log"
    file_handler = logging.handlers.RotatingFileHandler(
        filename=main_log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(_ISTFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))

    # ── Error-only log ───────────────────────────────────────────────────────
    error_log_path = log_dir / "errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        filename=error_log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(_ISTFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))

    # Avoid duplicate handlers if setup_logging is called more than once
    if not root.handlers:
        root.addHandler(console_handler)
        root.addHandler(file_handler)
        root.addHandler(error_handler)
    else:
        root.handlers.clear()
        root.addHandler(console_handler)
        root.addHandler(file_handler)
        root.addHandler(error_handler)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "apscheduler", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
