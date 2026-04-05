"""
Data quality checks and sanitization for incoming market data.

Every public validator returns a :class:`ValidationResult` containing
``is_valid``, ``cleaned_data``, ``errors``, and ``warnings`` — callers
never need to catch exceptions for validation failures.

Backwards-compatible note
--------------------------
The original :func:`validate_ohlcv`, :func:`sanitize_ohlcv`, and
:func:`validate_price_tick` functions are preserved unchanged.
"""

from __future__ import annotations

import html
import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from config.constants import IST_TIMEZONE

logger = logging.getLogger(__name__)

_IST = ZoneInfo(IST_TIMEZONE)

# ---------------------------------------------------------------------------
# Validation bounds (sane defaults for Indian equity markets)
# ---------------------------------------------------------------------------

_MIN_PRICE: float = 0.05
_MAX_PRICE: float = 1_500_000.0        # Nifty 50 * lot-size upper bound
_MAX_OI_CHANGE_PCT: float = 500.0      # Flag if OI jumps > 500%
_MAX_IV: float = 200.0                 # Implied volatility cap
_MIN_IV: float = 0.0
_MAX_NEWS_TITLE_LEN: int = 1_000
_MIN_NEWS_CONTENT_LEN: int = 10
_MAX_OHLC_SPREAD_PCT: float = 20.0    # Intraday H-L as % of close → warn above this


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class DataValidationError(Exception):
    """Raised when incoming data fails a hard validation check."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """
    Result of a single validation pass.

    Attributes
    ----------
    is_valid:
        ``False`` if any hard error was found.
    errors:
        List of error messages (hard failures — data should not be saved).
    warnings:
        List of warning messages (soft issues — data may still be usable).
    cleaned_data:
        A sanitized copy of the input dict, or ``None`` if validation failed.
        Only populated by dict-based validators (not the DataFrame-based ones).
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cleaned_data: Optional[dict] = field(default=None)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> Optional[float]:
    """Return ``float(v)`` or ``None`` if conversion fails or result is NaN/Inf."""
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    """Return ``int(float(v))`` or ``None`` on failure."""
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _require_float(
    data: dict,
    key: str,
    errors: list[str],
    *,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_zero: bool = False,
) -> Optional[float]:
    """
    Parse *key* from *data* as float, appending to *errors* on failure.

    Returns the parsed float, or ``None`` if it's missing / invalid.
    """
    raw = data.get(key)
    if raw is None:
        errors.append(f"Missing required field: {key!r}")
        return None
    v = _safe_float(raw)
    if v is None:
        errors.append(f"{key!r}: non-numeric value {raw!r}")
        return None
    if not allow_zero and v == 0.0:
        errors.append(f"{key!r}: value is exactly 0 (possible missing data)")
        return None
    if min_val is not None and v < min_val:
        errors.append(f"{key!r}: {v} is below minimum {min_val}")
        return None
    if max_val is not None and v > max_val:
        errors.append(f"{key!r}: {v} exceeds maximum {max_val}")
        return None
    return v


# ---------------------------------------------------------------------------
# Dict-based validators (new API)
# ---------------------------------------------------------------------------


def validate_price_data(data: dict[str, Any]) -> ValidationResult:
    """
    Validate a live price tick dict (from scraper output).

    Required keys: ``ltp``, ``open``, ``high``, ``low``, ``close``.
    Optional keys: ``volume``, ``timestamp``.

    Parameters
    ----------
    data:
        Raw price dict (e.g. from :meth:`~src.data.nse_scraper.NSEScraper.get_index_quote`).

    Returns
    -------
    ValidationResult:
        ``cleaned_data`` is populated only when ``is_valid=True``.
    """
    errors: list[str] = []
    warnings: list[str] = []
    cleaned: dict[str, Any] = {}

    ltp = _require_float(data, "ltp", errors, min_val=_MIN_PRICE, max_val=_MAX_PRICE)
    if ltp is not None:
        cleaned["ltp"] = ltp

    # Parse OHLC
    ohlc_ok = True
    for key in ("open", "high", "low", "close"):
        v = _require_float(data, key, errors, min_val=_MIN_PRICE, max_val=_MAX_PRICE)
        if v is None:
            ohlc_ok = False
        else:
            cleaned[key] = v

    if ohlc_ok:
        high, low, close = cleaned["high"], cleaned["low"], cleaned["close"]

        if high < low:
            errors.append(f"high ({high}) < low ({low})")

        if close < low or close > high:
            errors.append(f"close ({close}) is outside [low ({low}), high ({high})]")

        spread_pct = (high - low) / max(close, 1) * 100
        if spread_pct > _MAX_OHLC_SPREAD_PCT:
            warnings.append(
                f"Intraday H-L spread is {spread_pct:.1f}% of close — "
                "possible garbage data"
            )

    # Volume (optional, but must be non-negative integer if present)
    raw_vol = data.get("volume")
    if raw_vol is not None:
        vol = _safe_int(raw_vol)
        if vol is None:
            warnings.append(f"volume {raw_vol!r} is not numeric — skipped")
        elif vol < 0:
            errors.append(f"volume is negative: {vol}")
        else:
            cleaned["volume"] = vol

    # Timestamp (optional)
    raw_ts = data.get("timestamp")
    if raw_ts is not None:
        try:
            if isinstance(raw_ts, datetime):
                cleaned["timestamp"] = raw_ts.isoformat()
            else:
                cleaned["timestamp"] = str(raw_ts)
        except Exception:
            warnings.append("timestamp is unparseable — skipped")

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        cleaned_data=cleaned if is_valid else None,
    )


def validate_options_data(data: dict[str, Any]) -> ValidationResult:
    """
    Validate a single options strike-level dict.

    Required keys: ``strike``, ``option_type``, ``expiry``.
    Optional keys: ``oi``, ``ltp``, ``iv``, ``volume``.

    Parameters
    ----------
    data:
        Raw options data dict from scraper output.

    Returns
    -------
    ValidationResult:
        ``cleaned_data`` is populated only when ``is_valid=True``.
    """
    errors: list[str] = []
    warnings: list[str] = []
    cleaned: dict[str, Any] = {}

    # Strike price
    strike = _require_float(data, "strike", errors, min_val=_MIN_PRICE, max_val=_MAX_PRICE)
    if strike is not None:
        cleaned["strike"] = strike

    # Option type
    opt_type = data.get("option_type", "")
    if str(opt_type).upper() not in ("CE", "PE"):
        errors.append(f"option_type must be 'CE' or 'PE', got {opt_type!r}")
    else:
        cleaned["option_type"] = str(opt_type).upper()

    # Expiry date
    raw_expiry = data.get("expiry")
    if raw_expiry is None:
        errors.append("Missing required field: 'expiry'")
    else:
        try:
            if isinstance(raw_expiry, date):
                expiry_date = raw_expiry
            else:
                # Accept "DD-Mon-YYYY" (NSE format) or ISO
                raw_s = str(raw_expiry)
                try:
                    expiry_date = datetime.strptime(raw_s, "%d-%b-%Y").date()
                except ValueError:
                    expiry_date = datetime.fromisoformat(raw_s).date()

            today = datetime.now(tz=_IST).date()
            if expiry_date < today:
                warnings.append(f"Expiry {expiry_date} is in the past")
            cleaned["expiry"] = expiry_date.isoformat()
        except Exception:
            errors.append(f"expiry {raw_expiry!r} is not a valid date")

    # Open interest (non-negative)
    raw_oi = data.get("oi")
    if raw_oi is not None:
        oi = _safe_int(raw_oi)
        if oi is None:
            warnings.append(f"oi {raw_oi!r} is not numeric — skipped")
        elif oi < 0:
            errors.append(f"oi is negative: {oi}")
        else:
            cleaned["oi"] = oi

    # LTP (non-negative; deep OTM options may have 0.05 LTP)
    raw_ltp = data.get("ltp")
    if raw_ltp is not None:
        ltp = _safe_float(raw_ltp)
        if ltp is None:
            warnings.append(f"ltp {raw_ltp!r} is not numeric — skipped")
        elif ltp < 0:
            errors.append(f"ltp is negative: {ltp}")
        else:
            cleaned["ltp"] = ltp

    # Implied volatility
    raw_iv = data.get("iv")
    if raw_iv is not None:
        iv = _safe_float(raw_iv)
        if iv is None:
            warnings.append(f"iv {raw_iv!r} is not numeric — skipped")
        elif iv < _MIN_IV or iv > _MAX_IV:
            errors.append(f"iv {iv} is outside valid range [0, {_MAX_IV}]")
        else:
            cleaned["iv"] = iv

    # Volume
    raw_vol = data.get("volume")
    if raw_vol is not None:
        vol = _safe_int(raw_vol)
        if vol is not None and vol >= 0:
            cleaned["volume"] = vol
        elif vol is not None:
            errors.append(f"volume is negative: {vol}")

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        cleaned_data=cleaned if is_valid else None,
    )


def validate_news_data(data: dict[str, Any]) -> ValidationResult:
    """
    Validate a news article dict before database insertion.

    Required keys: ``title``, ``source``, ``published_at``.
    Optional keys: ``content``, ``summary``, ``url``.

    Parameters
    ----------
    data:
        Raw news dict (e.g. from RSS feed parser).

    Returns
    -------
    ValidationResult:
        ``cleaned_data`` has sanitized strings and a normalized timestamp.
    """
    errors: list[str] = []
    warnings: list[str] = []
    cleaned: dict[str, Any] = {}

    # Title
    raw_title = data.get("title")
    if not raw_title or not str(raw_title).strip():
        errors.append("title is missing or empty")
    else:
        title = sanitize_string(str(raw_title))
        if not title:
            errors.append("title is empty after sanitization")
        elif len(title) > _MAX_NEWS_TITLE_LEN:
            warnings.append(f"title truncated from {len(title)} to {_MAX_NEWS_TITLE_LEN} chars")
            title = title[:_MAX_NEWS_TITLE_LEN]
        cleaned["title"] = title

    # Source
    raw_source = data.get("source")
    if not raw_source or not str(raw_source).strip():
        errors.append("source is missing or empty")
    else:
        cleaned["source"] = sanitize_string(str(raw_source))

    # Published at
    raw_ts = data.get("published_at")
    if raw_ts is None:
        errors.append("published_at is missing")
    else:
        try:
            if isinstance(raw_ts, datetime):
                ts = raw_ts
            else:
                ts = datetime.fromisoformat(str(raw_ts))
            cleaned["published_at"] = ts.isoformat()
        except Exception:
            errors.append(f"published_at {raw_ts!r} is not a valid datetime")

    # Content / summary (at least one should be present and non-trivial)
    content = sanitize_string(str(data.get("content") or data.get("summary") or ""))
    if len(content) < _MIN_NEWS_CONTENT_LEN:
        warnings.append(
            f"content/summary is very short ({len(content)} chars) — may be incomplete"
        )
    if content:
        cleaned["content"] = content

    # URL (optional but store if present)
    raw_url = data.get("url")
    if raw_url:
        cleaned["url"] = str(raw_url).strip()

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        cleaned_data=cleaned if is_valid else None,
    )


# ---------------------------------------------------------------------------
# String sanitization
# ---------------------------------------------------------------------------

_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_string(text: str) -> str:
    """
    Clean an arbitrary string for safe storage.

    Steps:
    1. HTML-unescape entities (``&amp;`` → ``&``)
    2. Strip HTML tags
    3. Remove ASCII control characters (keep newlines/tabs)
    4. Normalize Unicode to NFC form
    5. Collapse consecutive whitespace to a single space
    6. Strip leading/trailing whitespace

    Parameters
    ----------
    text:
        Raw input string (may contain HTML markup or escaped entities).

    Returns
    -------
    str:
        Cleaned string.
    """
    if not text:
        return ""
    # 1. Remove HTML tags first (before unescaping, so &lt;tag&gt; stays as-is)
    text = _RE_HTML_TAG.sub(" ", text)
    # 2. Unescape HTML entities (&amp; → &, &lt; → <, etc.)
    text = html.unescape(text)
    # 3. Remove control characters (but keep \n \r \t)
    text = _RE_CONTROL_CHARS.sub("", text)
    # 4. Normalize Unicode
    text = unicodedata.normalize("NFC", text)
    # 5. Collapse whitespace
    text = _RE_MULTI_SPACE.sub(" ", text)
    # 6. Strip
    return text.strip()


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------


def detect_stale_data(
    latest_timestamp: datetime,
    max_age_seconds: float,
) -> bool:
    """
    Return ``True`` if *latest_timestamp* is older than *max_age_seconds*.

    Parameters
    ----------
    latest_timestamp:
        The most recent data timestamp (timezone-aware or naive IST).
    max_age_seconds:
        Maximum acceptable age in seconds.

    Returns
    -------
    bool:
        ``True`` when the data is stale.
    """
    now = datetime.now(tz=_IST)
    if latest_timestamp.tzinfo is None:
        latest_timestamp = latest_timestamp.replace(tzinfo=_IST)
    else:
        latest_timestamp = latest_timestamp.astimezone(_IST)

    age = (now - latest_timestamp).total_seconds()
    if age > max_age_seconds:
        logger.warning(
            "Stale data detected: age=%.0fs, max_age=%.0fs", age, max_age_seconds
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Backward-compatible DataFrame validators (original API, unchanged)
# ---------------------------------------------------------------------------


def validate_ohlcv(df: pd.DataFrame, symbol: str = "") -> ValidationResult:
    """
    Validate an OHLCV DataFrame for structural and numeric correctness.

    Expected columns: ``open``, ``high``, ``low``, ``close``, ``volume``.
    Index must be ``DatetimeIndex``.

    Parameters
    ----------
    df:
        DataFrame to validate.
    symbol:
        Label used in error messages.

    Returns
    -------
    ValidationResult:
        ``is_valid=False`` if any hard errors are found.
        ``cleaned_data`` is ``None`` (use :func:`sanitize_ohlcv` separately).
    """
    errors: list[str] = []
    warnings: list[str] = []
    tag = f"[{symbol}] " if symbol else ""

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns.str.lower())
    if missing:
        errors.append(f"{tag}Missing columns: {missing}")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    df = df.copy()
    df.columns = df.columns.str.lower()

    if df.empty:
        errors.append(f"{tag}DataFrame is empty")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"{tag}Index must be DatetimeIndex, got {type(df.index).__name__}")

    null_counts = df[list(required)].isnull().sum()
    if null_counts.any():
        warnings.append(f"{tag}Null values detected: {null_counts[null_counts > 0].to_dict()}")

    invalid_hl = df[df["high"] < df["low"]]
    if not invalid_hl.empty:
        errors.append(f"{tag}{len(invalid_hl)} rows where high < low")

    invalid_close = df[(df["close"] < df["low"]) | (df["close"] > df["high"])]
    if not invalid_close.empty:
        errors.append(f"{tag}{len(invalid_close)} rows where close outside [low, high]")

    for col in ("open", "high", "low", "close"):
        neg = df[df[col] <= 0]
        if not neg.empty:
            errors.append(f"{tag}{len(neg)} rows with non-positive {col}")

    neg_vol = df[df["volume"] < 0]
    if not neg_vol.empty:
        errors.append(f"{tag}{len(neg_vol)} rows with negative volume")

    daily_change = df["close"].pct_change().abs()
    extremes = daily_change[daily_change > 0.20]
    if not extremes.empty:
        warnings.append(
            f"{tag}{len(extremes)} bars with >20% price change — possible data error"
        )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard fixes to an OHLCV DataFrame.

    - Forward-fill small gaps (up to 2 consecutive NaNs)
    - Clip negative prices to NaN then forward-fill
    - Ensure volume is non-negative integer

    Parameters
    ----------
    df:
        Input DataFrame with lowercase OHLCV columns.

    Returns
    -------
    pd.DataFrame:
        Cleaned copy.
    """
    df = df.copy()
    df.columns = df.columns.str.lower()

    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = df[col].where(df[col] > 0)
            df[col] = df[col].ffill(limit=2)

    if "volume" in df.columns:
        df["volume"] = df["volume"].clip(lower=0).fillna(0).astype("int64")

    return df


def validate_price_tick(
    price: float,
    symbol: str,
    *,
    min_price: float = 0.05,
    max_price: float = 1_000_000.0,
) -> float:
    """
    Validate a single price tick value.

    Parameters
    ----------
    price:
        Price to validate.
    symbol:
        Label used in error message.
    min_price:
        Minimum acceptable price.
    max_price:
        Maximum acceptable price.

    Returns
    -------
    float:
        The validated price.

    Raises
    ------
    DataValidationError:
        If the price is outside acceptable bounds or not finite.
    """
    if not math.isfinite(price):
        raise DataValidationError(f"{symbol}: price {price!r} is not finite")
    if price < min_price:
        raise DataValidationError(f"{symbol}: price {price} below minimum {min_price}")
    if price > max_price:
        raise DataValidationError(f"{symbol}: price {price} above maximum {max_price}")
    return price


def validate_options_chain(data: dict[str, Any], symbol: str = "") -> ValidationResult:
    """
    Validate the structure of a raw NSE options chain API response.

    The NSE response shape is::

        {
            "records": {
                "underlyingValue": float,
                "expiryDates": [...],
                "data": [{"strikePrice": ..., "CE": {...}, "PE": {...}}, ...]
            }
        }

    Parameters
    ----------
    data:
        Raw NSE options chain dict.
    symbol:
        Label used in error messages.
    """
    errors: list[str] = []
    warnings: list[str] = []
    tag = f"[{symbol}] " if symbol else ""

    if not isinstance(data, dict):
        errors.append(f"{tag}Expected a dict, got {type(data).__name__}")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    records = data.get("records")
    if not records or not isinstance(records, dict):
        errors.append(f"{tag}Options chain missing 'records' dict")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    if "underlyingValue" not in records:
        warnings.append(f"{tag}Missing 'underlyingValue' in records")

    expiry_dates = records.get("expiryDates")
    if not expiry_dates:
        errors.append(f"{tag}Options chain has no expiry dates")

    strike_data = records.get("data")
    if not strike_data or not isinstance(strike_data, list):
        errors.append(f"{tag}Options chain has no strike data")
    elif len(strike_data) == 0:
        warnings.append(f"{tag}Options chain strike data is empty")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
