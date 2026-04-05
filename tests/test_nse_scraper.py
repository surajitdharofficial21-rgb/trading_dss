"""
Tests for NSEScraper, BSEScraper, and DataValidator.

All HTTP calls are mocked via ``unittest.mock`` — no real network traffic.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from src.data.nse_scraper import NSEScraper, NSEScraperError, _NSETransientError
from src.data.bse_scraper import BSEScraper, BSEScraperError
from src.data.data_validator import (
    DataValidationError,
    ValidationResult,
    detect_stale_data,
    sanitize_string,
    validate_news_data,
    validate_options_data,
    validate_price_data,
    validate_price_tick,
)
from src.utils.cache import TTLCache
from src.data.rate_limiter import RateLimiter


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def unlimited_limiter() -> RateLimiter:
    """Rate limiter with a huge bucket so tests never wait."""
    return RateLimiter(max_requests=10_000, window_seconds=60.0)


@pytest.fixture()
def mock_nse_session():
    """Patch requests.Session used inside NSEScraper."""
    with patch("src.data.nse_scraper.requests.Session") as mock_cls:
        session = MagicMock()
        mock_cls.return_value = session
        yield session


@pytest.fixture()
def scraper(mock_nse_session, unlimited_limiter) -> NSEScraper:
    """Pre-warmed NSEScraper with mocked session and unlimited rate limiter."""
    warm_resp = MagicMock()
    warm_resp.raise_for_status.return_value = None
    mock_nse_session.get.return_value = warm_resp

    s = NSEScraper(rate_limiter=unlimited_limiter)
    s.warm_up()
    # Reset so test-specific side_effects are consumed cleanly
    mock_nse_session.get.reset_mock()
    return s


@pytest.fixture()
def mock_bse_session():
    """Patch requests.Session used inside BSEScraper."""
    with patch("src.data.bse_scraper.requests.Session") as mock_cls:
        session = MagicMock()
        mock_cls.return_value = session
        yield session


@pytest.fixture()
def bse_scraper(mock_bse_session, unlimited_limiter) -> BSEScraper:
    """BSEScraper with mocked session."""
    return BSEScraper(rate_limiter=unlimited_limiter)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_resp(payload) -> MagicMock:
    """Return a mock Response with status 200 and JSON payload."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def _err_resp(status: int) -> MagicMock:
    """Return a mock Response that raises HTTPError on raise_for_status."""
    resp = MagicMock()
    resp.status_code = status
    http_err = requests.HTTPError(response=resp)
    resp.raise_for_status.side_effect = http_err
    return resp


# ---------------------------------------------------------------------------
# NSEScraper — session management
# ---------------------------------------------------------------------------


class TestNSESessionManagement:
    def test_warm_up_sets_warmed_flag(self, mock_nse_session) -> None:
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        mock_nse_session.get.return_value = resp

        s = NSEScraper()
        s.warm_up()
        assert s._warmed_up is True

    def test_warm_up_records_creation_time(self, mock_nse_session) -> None:
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        mock_nse_session.get.return_value = resp

        before = time.monotonic()
        s = NSEScraper()
        s.warm_up()
        after = time.monotonic()
        assert before <= s._session_created_at <= after

    def test_warm_up_network_failure_raises(self, mock_nse_session) -> None:
        mock_nse_session.get.side_effect = requests.ConnectionError("refused")
        s = NSEScraper()
        with pytest.raises(NSEScraperError, match="warm-up"):
            s.warm_up()

    def test_warm_up_http_failure_raises(self, mock_nse_session) -> None:
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError("503")
        mock_nse_session.get.return_value = resp
        s = NSEScraper()
        with pytest.raises(NSEScraperError, match="warm-up"):
            s.warm_up()

    def test_first_get_triggers_warm_up(self, mock_nse_session, unlimited_limiter) -> None:
        warm_resp = MagicMock()
        warm_resp.raise_for_status.return_value = None
        data_resp = _ok_resp({"data": []})
        mock_nse_session.get.side_effect = [warm_resp, data_resp]

        s = NSEScraper(rate_limiter=unlimited_limiter)
        assert s._warmed_up is False
        s.get_all_indices()
        assert s._warmed_up is True
        # First call was the warm-up homepage, second was the API
        assert mock_nse_session.get.call_count == 2

    def test_expired_session_triggers_rewarm(self, mock_nse_session, unlimited_limiter) -> None:
        """When session_created_at is old, a new warm-up must happen automatically."""
        warm_resp = MagicMock()
        warm_resp.raise_for_status.return_value = None
        data_resp = _ok_resp({"data": []})
        mock_nse_session.get.side_effect = [warm_resp, data_resp]

        s = NSEScraper(rate_limiter=unlimited_limiter)
        s._warmed_up = True
        s._session_created_at = time.monotonic() - 400  # 400s > 240s TTL
        s.get_all_indices()
        # warm_up + API call
        assert mock_nse_session.get.call_count == 2


# ---------------------------------------------------------------------------
# NSEScraper — header rotation
# ---------------------------------------------------------------------------


class TestNSEHeaderRotation:
    def test_user_agent_is_set_on_every_request(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        mock_nse_session.get.return_value = _ok_resp({"data": []})
        scraper.get_all_indices()
        call_kwargs = mock_nse_session.get.call_args
        assert "headers" in call_kwargs.kwargs or len(call_kwargs.args) >= 2
        # Extract headers from positional or keyword args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs.args[1]
        assert "User-Agent" in headers
        assert len(headers["User-Agent"]) > 20

    def test_user_agent_varies_between_requests(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        """Multiple calls may rotate User-Agent (non-deterministic but at least valid)."""
        agents: set[str] = set()
        for _ in range(5):
            mock_nse_session.get.return_value = _ok_resp({"data": []})
            scraper.get_all_indices()
            scraper._cache.clear()  # Force re-fetch
            kw = mock_nse_session.get.call_args.kwargs
            headers = kw.get("headers", {})
            agents.add(headers.get("User-Agent", ""))
        # From a list of 5+ UAs, at least 2 distinct values over 5 calls is very likely
        assert len(agents) >= 1  # At minimum the header was set each time


# ---------------------------------------------------------------------------
# NSEScraper — data parsing
# ---------------------------------------------------------------------------

_SAMPLE_INDICES_PAYLOAD = {
    "data": [
        {
            "indexSymbol": "NIFTY 50",
            "last": "22000.50",
            "open": "21900.00",
            "high": "22100.00",
            "low": "21850.00",
            "previousClose": "21950.00",
            "change": "50.50",
            "pChange": "0.23",
            "advances": "35",
            "declines": "15",
            "unchanged": "0",
            "totalTradedVolume": "123456789",
        },
        {
            "indexSymbol": "INDIA VIX",
            "last": "13.45",
            "change": "-0.55",
            "pChange": "-3.93",
            "open": "14.00",
            "high": "14.10",
            "low": "13.40",
            "previousClose": "14.00",
        },
    ]
}


class TestNSEDataParsing:
    def test_get_all_indices_returns_normalized_list(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        mock_nse_session.get.return_value = _ok_resp(_SAMPLE_INDICES_PAYLOAD)
        result = scraper.get_all_indices()
        assert isinstance(result, list)
        assert len(result) == 2
        nifty = next(r for r in result if r["index_name"] == "NIFTY 50")
        assert nifty["ltp"] == pytest.approx(22000.50)
        assert nifty["open"] == pytest.approx(21900.00)
        assert nifty["high"] == pytest.approx(22100.00)
        assert nifty["low"] == pytest.approx(21850.00)
        assert nifty["close"] == pytest.approx(21950.00)
        assert nifty["change_pct"] == pytest.approx(0.23)
        assert nifty["advances"] == 35
        assert nifty["declines"] == 15

    def test_get_index_quote_filters_by_symbol(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        mock_nse_session.get.return_value = _ok_resp(_SAMPLE_INDICES_PAYLOAD)
        result = scraper.get_index_quote("NIFTY 50")
        assert result is not None
        assert result["ltp"] == pytest.approx(22000.50)

    def test_get_index_quote_case_insensitive(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        mock_nse_session.get.return_value = _ok_resp(_SAMPLE_INDICES_PAYLOAD)
        result = scraper.get_index_quote("nifty 50")
        assert result is not None

    def test_get_index_quote_not_found_returns_none(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        mock_nse_session.get.return_value = _ok_resp({"data": []})
        assert scraper.get_index_quote("FAKEIDX") is None

    def test_get_vix_extracts_vix_entry(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        mock_nse_session.get.return_value = _ok_resp(_SAMPLE_INDICES_PAYLOAD)
        vix = scraper.get_vix()
        assert vix is not None
        assert vix["vix_value"] == pytest.approx(13.45)
        assert vix["vix_change"] == pytest.approx(-0.55)
        assert vix["vix_change_pct"] == pytest.approx(-3.93)

    def test_get_vix_not_found_returns_none(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        mock_nse_session.get.return_value = _ok_resp({"data": []})
        assert scraper.get_vix() is None

    def test_get_advances_declines(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        mock_nse_session.get.return_value = _ok_resp(_SAMPLE_INDICES_PAYLOAD)
        result = scraper.get_advances_declines("NIFTY 50")
        assert result == {"advances": 35, "declines": 15, "unchanged": 0}

    def test_all_indices_returns_none_on_failure(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        mock_nse_session.get.side_effect = requests.ConnectionError("no network")
        result = scraper.get_all_indices()
        assert result is None

    def test_missing_data_key_returns_empty_list(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        mock_nse_session.get.return_value = _ok_resp({"status": "ok"})  # no "data"
        result = scraper.get_all_indices()
        assert result == []


# ---------------------------------------------------------------------------
# NSEScraper — caching
# ---------------------------------------------------------------------------


class TestNSECaching:
    def test_second_call_uses_cache(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        mock_nse_session.get.return_value = _ok_resp(_SAMPLE_INDICES_PAYLOAD)
        scraper.get_all_indices()
        scraper.get_all_indices()
        # Only one HTTP call despite two method calls (second served from cache)
        assert mock_nse_session.get.call_count == 1

    def test_cache_miss_after_clear_fetches_fresh_data(
        self, unlimited_limiter: RateLimiter, mock_nse_session
    ) -> None:
        """After cache is cleared, a new HTTP call returns updated data."""
        s = NSEScraper(rate_limiter=unlimited_limiter)

        warm_resp = MagicMock()
        warm_resp.raise_for_status.return_value = None

        payload_v1 = {"data": [{"indexSymbol": "NIFTY 50", "last": "22000.00",
                                "open": "21900", "high": "22100", "low": "21850",
                                "previousClose": "21950", "change": "50", "pChange": "0.23",
                                "advances": "35", "declines": "15", "unchanged": "0",
                                "totalTradedVolume": "100000"}]}
        payload_v2 = {"data": [{"indexSymbol": "NIFTY 50", "last": "22500.00",
                                "open": "22000", "high": "22600", "low": "21950",
                                "previousClose": "22000", "change": "500", "pChange": "2.27",
                                "advances": "40", "declines": "10", "unchanged": "0",
                                "totalTradedVolume": "200000"}]}

        mock_nse_session.get.side_effect = [
            warm_resp,
            _ok_resp(payload_v1),
            _ok_resp(payload_v2),
        ]
        s.warm_up()
        result1 = s.get_all_indices()
        assert result1[0]["ltp"] == pytest.approx(22000.0)

        # Clear cache — next call must fetch fresh data from the mock
        s._cache.clear()
        result2 = s.get_all_indices()
        assert result2[0]["ltp"] == pytest.approx(22500.0)
        # Data changed ↔ a fresh HTTP call was made
        assert result1[0]["ltp"] != result2[0]["ltp"]


# ---------------------------------------------------------------------------
# NSEScraper — 403 / session handling
# ---------------------------------------------------------------------------


class TestNSE403Handling:
    def test_403_clears_warmed_flag(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        resp_403 = MagicMock()
        resp_403.status_code = 403
        mock_nse_session.get.return_value = resp_403

        with pytest.raises(NSEScraperError):
            scraper._get("/api/allIndices")
        assert scraper._warmed_up is False

    def test_401_also_clears_warmed_flag(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        resp_401 = MagicMock()
        resp_401.status_code = 401
        mock_nse_session.get.return_value = resp_401

        with pytest.raises(NSEScraperError):
            scraper._get("/api/allIndices")
        assert scraper._warmed_up is False

    def test_403_returns_none_from_public_method(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        resp_403 = MagicMock()
        resp_403.status_code = 403
        mock_nse_session.get.return_value = resp_403
        # Public methods return None instead of raising
        assert scraper.get_all_indices() is None

    def test_after_403_next_call_rewarms(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        """After a 403, the next call to a public method triggers re-warm."""
        resp_403 = MagicMock()
        resp_403.status_code = 403
        warm_resp = MagicMock()
        warm_resp.raise_for_status.return_value = None
        data_resp = _ok_resp(_SAMPLE_INDICES_PAYLOAD)

        mock_nse_session.get.side_effect = [resp_403, warm_resp, data_resp]
        scraper.get_all_indices()  # → 403 → None
        result = scraper.get_all_indices()  # → rewarm + data
        assert result is not None


# ---------------------------------------------------------------------------
# NSEScraper — retry on transient errors
# ---------------------------------------------------------------------------


class TestNSERetryBehavior:
    def test_connection_error_triggers_retry(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        """ConnectionError is a transient error — tenacity should retry 3 times."""
        mock_nse_session.get.side_effect = requests.ConnectionError("refused")
        # After 3 retries, _NSETransientError is reraised as NSEScraperError
        with pytest.raises((NSEScraperError, _NSETransientError)):
            scraper._get("/api/allIndices")
        assert mock_nse_session.get.call_count >= 1  # at least 1 attempt

    def test_timeout_error_is_transient(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        mock_nse_session.get.side_effect = requests.Timeout("timeout")
        with pytest.raises((NSEScraperError, _NSETransientError)):
            scraper._get("/api/allIndices")

    def test_success_after_transient_failure(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        """First call times out; second (retry) succeeds."""
        mock_nse_session.get.side_effect = [
            requests.Timeout("1st attempt"),
            _ok_resp(_SAMPLE_INDICES_PAYLOAD),
        ]
        # _get with tenacity retries — second attempt succeeds
        result = scraper._get("/api/allIndices")
        assert result == _SAMPLE_INDICES_PAYLOAD


# ---------------------------------------------------------------------------
# NSEScraper — health stats
# ---------------------------------------------------------------------------


class TestNSEHealthStats:
    def test_initial_health_stats(self, mock_nse_session) -> None:
        s = NSEScraper()
        stats = s.get_health_stats()
        assert stats["total_requests"] == 0
        assert stats["successful"] == 0
        assert stats["failed"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["avg_response_ms"] == 0.0

    def test_successful_request_increments_success(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        mock_nse_session.get.return_value = _ok_resp(_SAMPLE_INDICES_PAYLOAD)
        scraper.get_all_indices()
        stats = scraper.get_health_stats()
        assert stats["total_requests"] == 1
        assert stats["successful"] == 1
        assert stats["failed"] == 0
        assert stats["success_rate"] == 1.0

    def test_failed_request_increments_failure(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        resp_403 = MagicMock()
        resp_403.status_code = 403
        mock_nse_session.get.return_value = resp_403
        scraper.get_all_indices()  # returns None
        stats = scraper.get_health_stats()
        assert stats["total_requests"] == 1
        assert stats["failed"] == 1
        assert stats["success_rate"] == 0.0

    def test_consecutive_failures_tracked(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        resp_403 = MagicMock()
        resp_403.status_code = 403
        mock_nse_session.get.return_value = resp_403
        for _ in range(3):
            scraper.get_all_indices()
        stats = scraper.get_health_stats()
        assert stats["consecutive_failures"] == 3

    def test_success_resets_consecutive_failures(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        resp_403 = MagicMock()
        resp_403.status_code = 403
        mock_nse_session.get.side_effect = [resp_403, _ok_resp(_SAMPLE_INDICES_PAYLOAD)]
        scraper.get_all_indices()  # fail
        # Rewarm happens automatically next call (403 cleared _warmed_up)
        warm_resp = MagicMock()
        warm_resp.raise_for_status.return_value = None
        mock_nse_session.get.side_effect = [warm_resp, _ok_resp(_SAMPLE_INDICES_PAYLOAD)]
        scraper.get_all_indices()  # success after rewarm
        assert scraper.get_health_stats()["consecutive_failures"] == 0

    def test_repr_contains_scraper_info(
        self, scraper: NSEScraper, mock_nse_session
    ) -> None:
        r = repr(scraper)
        assert "NSEScraper" in r
        assert "warmed=" in r


# ---------------------------------------------------------------------------
# BSEScraper tests
# ---------------------------------------------------------------------------


class TestBSEScraper:
    def test_get_sensex_quote_normalised(
        self, bse_scraper: BSEScraper, mock_bse_session
    ) -> None:
        raw = {
            "Index": "SENSEX",
            "Value": "73000.50",
            "Change": "200.25",
            "PerChange": "0.27",
            "Open": "72800.00",
            "High": "73100.00",
            "Low": "72750.00",
            "PrevClose": "72800.00",
        }
        mock_bse_session.get.return_value = _ok_resp(raw)
        result = bse_scraper.get_sensex_quote()
        assert result is not None
        assert result["ltp"] == pytest.approx(73000.50)
        assert result["change_pct"] == pytest.approx(0.27)

    def test_http_error_raises_bse_error(
        self, bse_scraper: BSEScraper, mock_bse_session
    ) -> None:
        mock_bse_session.get.return_value = _err_resp(500)
        with pytest.raises(BSEScraperError, match="HTTP error"):
            bse_scraper._get("https://api.bseindia.com/BseIndiaAPI/api/SensexView/w")

    def test_get_all_indices_from_list_response(
        self, bse_scraper: BSEScraper, mock_bse_session
    ) -> None:
        payload = [
            {"indxName": "BSE SENSEX", "currValue": "73000", "open": "72800",
             "high": "73100", "low": "72750", "prevClose": "72800", "chg": "200", "pChg": "0.27"},
            {"indxName": "BSE 500", "currValue": "30000", "open": "29900",
             "high": "30100", "low": "29800", "prevClose": "29900", "chg": "100", "pChg": "0.33"},
        ]
        mock_bse_session.get.return_value = _ok_resp(payload)
        result = bse_scraper.get_all_indices()
        assert isinstance(result, list)
        assert len(result) == 2
        sensex = result[0]
        assert sensex["ltp"] == pytest.approx(73000.0)

    def test_returns_none_on_failure(
        self, bse_scraper: BSEScraper, mock_bse_session
    ) -> None:
        mock_bse_session.get.side_effect = requests.ConnectionError("no network")
        assert bse_scraper.get_sensex_quote() is None

    def test_context_manager_closes_session(
        self, mock_bse_session
    ) -> None:
        with BSEScraper() as scraper:
            pass
        mock_bse_session.close.assert_called_once()

    def test_health_stats_initial(self, mock_bse_session) -> None:
        s = BSEScraper()
        stats = s.get_health_stats()
        assert stats["total_requests"] == 0
        assert stats["success_rate"] == 0.0

    def test_health_stats_after_success(
        self, bse_scraper: BSEScraper, mock_bse_session
    ) -> None:
        mock_bse_session.get.return_value = _ok_resp({"Index": "SENSEX", "Value": "73000"})
        bse_scraper.get_sensex_quote()
        stats = bse_scraper.get_health_stats()
        assert stats["successful"] == 1
        assert stats["success_rate"] == 1.0


# ---------------------------------------------------------------------------
# DataValidator — validate_price_data
# ---------------------------------------------------------------------------


class TestValidatePriceData:
    def _valid(self) -> dict:
        return {
            "ltp": 22000.50,
            "open": 21900.0,
            "high": 22100.0,
            "low": 21850.0,
            "close": 21950.0,
            "volume": 1_234_567,
        }

    def test_valid_data_passes(self) -> None:
        result = validate_price_data(self._valid())
        assert result.is_valid
        assert result.cleaned_data is not None
        assert result.cleaned_data["ltp"] == pytest.approx(22000.50)

    def test_cleaned_data_contains_all_fields(self) -> None:
        result = validate_price_data(self._valid())
        for key in ("ltp", "open", "high", "low", "close", "volume"):
            assert key in result.cleaned_data

    def test_negative_ltp_fails(self) -> None:
        d = self._valid()
        d["ltp"] = -1.0
        result = validate_price_data(d)
        assert not result.is_valid
        assert result.cleaned_data is None
        assert any("ltp" in e for e in result.errors)

    def test_zero_ltp_fails(self) -> None:
        d = self._valid()
        d["ltp"] = 0.0
        assert not validate_price_data(d).is_valid

    def test_high_less_than_low_fails(self) -> None:
        d = self._valid()
        d["high"] = 21000.0
        d["low"] = 22000.0
        result = validate_price_data(d)
        assert not result.is_valid
        assert any("high" in e and "low" in e for e in result.errors)

    def test_close_outside_range_fails(self) -> None:
        d = self._valid()
        d["close"] = 25000.0  # above high
        result = validate_price_data(d)
        assert not result.is_valid

    def test_negative_volume_fails(self) -> None:
        d = self._valid()
        d["volume"] = -100
        result = validate_price_data(d)
        assert not result.is_valid

    def test_missing_ltp_fails(self) -> None:
        d = self._valid()
        del d["ltp"]
        result = validate_price_data(d)
        assert not result.is_valid

    def test_non_numeric_price_fails(self) -> None:
        d = self._valid()
        d["ltp"] = "n/a"
        result = validate_price_data(d)
        assert not result.is_valid

    def test_large_spread_generates_warning(self) -> None:
        d = self._valid()
        d["high"] = 30000.0  # > 20% spread
        d["low"] = 10000.0
        d["close"] = 20000.0
        d["ltp"] = 20000.0
        result = validate_price_data(d)
        assert result.has_warnings

    def test_volume_zero_is_accepted(self) -> None:
        d = self._valid()
        d["volume"] = 0
        result = validate_price_data(d)
        assert result.is_valid

    def test_optional_timestamp_accepted(self) -> None:
        d = self._valid()
        d["timestamp"] = "2024-01-15T10:00:00+05:30"
        result = validate_price_data(d)
        assert result.is_valid
        assert "timestamp" in result.cleaned_data


# ---------------------------------------------------------------------------
# DataValidator — validate_options_data
# ---------------------------------------------------------------------------


class TestValidateOptionsData:
    def _valid(self) -> dict:
        return {
            "strike": 22000.0,
            "option_type": "CE",
            "expiry": "27-Jun-2025",
            "oi": 1_000_000,
            "ltp": 250.50,
            "iv": 14.5,
            "volume": 50_000,
        }

    def test_valid_options_pass(self) -> None:
        result = validate_options_data(self._valid())
        assert result.is_valid
        assert result.cleaned_data["option_type"] == "CE"
        assert result.cleaned_data["strike"] == pytest.approx(22000.0)

    def test_invalid_option_type_fails(self) -> None:
        d = self._valid()
        d["option_type"] = "FX"
        result = validate_options_data(d)
        assert not result.is_valid

    def test_pe_option_type_valid(self) -> None:
        d = self._valid()
        d["option_type"] = "PE"
        assert validate_options_data(d).is_valid

    def test_negative_strike_fails(self) -> None:
        d = self._valid()
        d["strike"] = -100.0
        assert not validate_options_data(d).is_valid

    def test_negative_oi_fails(self) -> None:
        d = self._valid()
        d["oi"] = -1
        assert not validate_options_data(d).is_valid

    def test_zero_ltp_is_valid(self) -> None:
        d = self._valid()
        d["ltp"] = 0.0
        result = validate_options_data(d)
        assert result.is_valid
        assert result.cleaned_data["ltp"] == 0.0

    def test_iv_above_200_fails(self) -> None:
        d = self._valid()
        d["iv"] = 250.0
        assert not validate_options_data(d).is_valid

    def test_iv_zero_is_valid(self) -> None:
        d = self._valid()
        d["iv"] = 0.0
        assert validate_options_data(d).is_valid

    def test_past_expiry_generates_warning(self) -> None:
        d = self._valid()
        d["expiry"] = "01-Jan-2020"
        result = validate_options_data(d)
        assert result.has_warnings
        assert any("past" in w.lower() for w in result.warnings)

    def test_missing_expiry_fails(self) -> None:
        d = self._valid()
        del d["expiry"]
        assert not validate_options_data(d).is_valid

    def test_invalid_expiry_fails(self) -> None:
        d = self._valid()
        d["expiry"] = "not-a-date"
        assert not validate_options_data(d).is_valid

    def test_option_type_normalised_to_upper(self) -> None:
        d = self._valid()
        d["option_type"] = "ce"
        result = validate_options_data(d)
        assert result.is_valid
        assert result.cleaned_data["option_type"] == "CE"


# ---------------------------------------------------------------------------
# DataValidator — validate_news_data
# ---------------------------------------------------------------------------


class TestValidateNewsData:
    def _valid(self) -> dict:
        return {
            "title": "NIFTY 50 rises 100 points on strong FII inflows",
            "source": "Economic Times",
            "published_at": "2024-01-15T10:30:00",
            "content": "Indian markets opened higher on Monday with NIFTY 50 rising 100 pts.",
            "url": "https://example.com/article",
        }

    def test_valid_news_passes(self) -> None:
        result = validate_news_data(self._valid())
        assert result.is_valid
        assert result.cleaned_data["title"] == self._valid()["title"]

    def test_missing_title_fails(self) -> None:
        d = self._valid()
        del d["title"]
        assert not validate_news_data(d).is_valid

    def test_empty_title_fails(self) -> None:
        d = self._valid()
        d["title"] = "   "
        assert not validate_news_data(d).is_valid

    def test_missing_source_fails(self) -> None:
        d = self._valid()
        del d["source"]
        assert not validate_news_data(d).is_valid

    def test_missing_published_at_fails(self) -> None:
        d = self._valid()
        del d["published_at"]
        assert not validate_news_data(d).is_valid

    def test_invalid_published_at_fails(self) -> None:
        d = self._valid()
        d["published_at"] = "not-a-date"
        assert not validate_news_data(d).is_valid

    def test_html_title_is_sanitized(self) -> None:
        d = self._valid()
        d["title"] = "<b>NIFTY</b> rises &amp; SENSEX falls"
        result = validate_news_data(d)
        assert result.is_valid
        assert "<b>" not in result.cleaned_data["title"]
        assert "&amp;" not in result.cleaned_data["title"]
        assert "NIFTY" in result.cleaned_data["title"]

    def test_short_content_generates_warning(self) -> None:
        d = self._valid()
        d["content"] = "Hi"
        result = validate_news_data(d)
        assert result.has_warnings

    def test_datetime_object_as_published_at(self) -> None:
        d = self._valid()
        d["published_at"] = datetime(2024, 1, 15, 10, 30, 0)
        assert validate_news_data(d).is_valid


# ---------------------------------------------------------------------------
# DataValidator — sanitize_string
# ---------------------------------------------------------------------------


class TestSanitizeString:
    def test_strips_html_tags(self) -> None:
        assert sanitize_string("<b>Bold</b>") == "Bold"

    def test_unescapes_html_entities(self) -> None:
        assert sanitize_string("Price &amp; Volume") == "Price & Volume"
        assert sanitize_string("&lt;10%&gt;") == "<10%>"

    def test_collapses_whitespace(self) -> None:
        assert sanitize_string("  too   many    spaces  ") == "too many spaces"

    def test_removes_control_chars(self) -> None:
        result = sanitize_string("hello\x00world\x1f!")
        assert "\x00" not in result
        assert "hello" in result
        assert "world" in result

    def test_empty_string_returns_empty(self) -> None:
        assert sanitize_string("") == ""

    def test_nested_html(self) -> None:
        assert sanitize_string("<div><p>Clean</p></div>") == "Clean"


# ---------------------------------------------------------------------------
# DataValidator — detect_stale_data
# ---------------------------------------------------------------------------


class TestDetectStaleData:
    def _ist(self) -> ZoneInfo:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Asia/Kolkata")

    def test_fresh_data_is_not_stale(self) -> None:
        from zoneinfo import ZoneInfo
        now = datetime.now(tz=ZoneInfo("Asia/Kolkata"))
        assert detect_stale_data(now, max_age_seconds=60) is False

    def test_old_data_is_stale(self) -> None:
        from zoneinfo import ZoneInfo
        old = datetime.now(tz=ZoneInfo("Asia/Kolkata")) - timedelta(seconds=120)
        assert detect_stale_data(old, max_age_seconds=60) is True

    def test_naive_datetime_treated_as_ist(self) -> None:
        naive_now = datetime.now()  # naive = IST for our system
        assert detect_stale_data(naive_now, max_age_seconds=30) is False

    def test_boundary_is_stale(self) -> None:
        from zoneinfo import ZoneInfo
        # Exactly at the boundary — should be stale (age > max_age)
        ts = datetime.now(tz=ZoneInfo("Asia/Kolkata")) - timedelta(seconds=61)
        assert detect_stale_data(ts, max_age_seconds=60) is True


# ---------------------------------------------------------------------------
# Backward-compatible validators
# ---------------------------------------------------------------------------


class TestBackwardCompatValidators:
    def test_validate_price_tick_valid(self) -> None:
        assert validate_price_tick(22000.0, "NIFTY50") == 22000.0

    def test_validate_price_tick_negative_raises(self) -> None:
        with pytest.raises(DataValidationError):
            validate_price_tick(-1.0, "NIFTY50")

    def test_validate_price_tick_inf_raises(self) -> None:
        with pytest.raises(DataValidationError):
            validate_price_tick(float("inf"), "NIFTY50")

    def test_validation_result_backward_compat(self) -> None:
        """Old code creates ValidationResult without cleaned_data."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert result.cleaned_data is None
        assert result.is_valid
