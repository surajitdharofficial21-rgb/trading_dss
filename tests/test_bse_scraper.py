"""Tests for BSE scraper (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data.bse_scraper import BSEScraper, BSEScraperError
from src.data.rate_limiter import RateLimiter


@pytest.fixture()
def unlimited_limiter() -> RateLimiter:
    return RateLimiter(max_requests=10_000, window_seconds=60.0)


@pytest.fixture()
def mock_bse_session():
    with patch("src.data.bse_scraper.requests.Session") as mock_cls:
        session = MagicMock()
        mock_cls.return_value = session
        yield session


def _ok_resp(payload) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


class TestBSEScraper:
    def test_get_sensex_quote_returns_normalised_dict(
        self, mock_bse_session, unlimited_limiter
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
        scraper = BSEScraper(rate_limiter=unlimited_limiter)
        result = scraper.get_sensex_quote()
        assert result is not None
        assert result["ltp"] == pytest.approx(73000.50)
        assert result["change_pct"] == pytest.approx(0.27)

    def test_http_error_returns_none(
        self, mock_bse_session, unlimited_limiter
    ) -> None:
        """Public methods return None instead of raising on HTTP errors."""
        resp = MagicMock()
        resp.status_code = 500
        resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_bse_session.get.return_value = resp

        scraper = BSEScraper(rate_limiter=unlimited_limiter)
        result = scraper.get_sensex_quote()
        assert result is None

    def test_get_raises_bse_error_when_called_directly(
        self, mock_bse_session, unlimited_limiter
    ) -> None:
        """_get() (the internal method) propagates BSEScraperError."""
        resp = MagicMock()
        resp.status_code = 500
        resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_bse_session.get.return_value = resp

        scraper = BSEScraper(rate_limiter=unlimited_limiter)
        with pytest.raises(BSEScraperError, match="HTTP error"):
            scraper._get("https://api.bseindia.com/BseIndiaAPI/api/SensexView/w")

    def test_context_manager_closes_session(self, mock_bse_session) -> None:
        with BSEScraper() as scraper:
            pass
        mock_bse_session.close.assert_called_once()

    def test_network_error_returns_none(
        self, mock_bse_session, unlimited_limiter
    ) -> None:
        mock_bse_session.get.side_effect = requests.ConnectionError("refused")
        scraper = BSEScraper(rate_limiter=unlimited_limiter)
        assert scraper.get_sensex_quote() is None

    def test_health_stats_after_failure(
        self, mock_bse_session, unlimited_limiter
    ) -> None:
        mock_bse_session.get.side_effect = requests.ConnectionError("refused")
        scraper = BSEScraper(rate_limiter=unlimited_limiter)
        scraper.get_sensex_quote()
        stats = scraper.get_health_stats()
        assert stats["failed"] >= 1
        assert stats["success_rate"] == 0.0
