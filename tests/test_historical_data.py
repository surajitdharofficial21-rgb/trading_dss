"""
Comprehensive tests for HistoricalDataManager, FIIDIIFetcher, and VIXTracker.

All network I/O is mocked — no actual HTTP calls or DB writes.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from src.data.fii_dii_data import FIIDIIData, FIIDIIFetcher
from src.data.historical_data import HistoricalDataError, HistoricalDataManager
from src.data.index_registry import Index, IndexRegistry
from src.data.vix_data import VIXData, VIXTracker

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_IDX_NIFTY = Index(
    id="NIFTY50",
    display_name="Nifty 50",
    exchange="NSE",
    yahoo_symbol="^NSEI",
    is_active=True,
)
_IDX_BANKNIFTY = Index(
    id="BANKNIFTY",
    display_name="Nifty Bank",
    exchange="NSE",
    yahoo_symbol="^NSEBANK",
    is_active=True,
    has_options=True,
    option_symbol="BANKNIFTY",
)
_IDX_NO_SYMBOL = Index(
    id="NODATA",
    display_name="No Data",
    exchange="NSE",
    yahoo_symbol=None,
    is_active=True,
)


@pytest.fixture()
def mock_registry() -> MagicMock:
    reg = MagicMock(spec=IndexRegistry)
    reg.get_index.return_value = _IDX_NIFTY
    reg.get_active_indices.return_value = [_IDX_NIFTY, _IDX_BANKNIFTY]
    reg.get_indices_with_options.return_value = [_IDX_BANKNIFTY]
    return reg


@pytest.fixture()
def mock_db() -> MagicMock:
    db = MagicMock()
    db.fetch_one.return_value = None
    db.fetch_all.return_value = []
    return db


@pytest.fixture()
def manager(mock_registry: MagicMock, mock_db: MagicMock) -> HistoricalDataManager:
    return HistoricalDataManager(registry=mock_registry, db=mock_db)


def _make_ohlcv_df(
    dates: list[str],
    *,
    tz: str = "UTC",
    include_dividends: bool = True,
) -> pd.DataFrame:
    """Return a minimal yfinance-style OHLCV DataFrame."""
    data: dict = {
        "Open":   [100.0 + i for i in range(len(dates))],
        "High":   [105.0 + i for i in range(len(dates))],
        "Low":    [98.0  + i for i in range(len(dates))],
        "Close":  [103.0 + i for i in range(len(dates))],
        "Volume": [10_000 * (i + 1) for i in range(len(dates))],
    }
    if include_dividends:
        data["Dividends"] = [0] * len(dates)
        data["Stock Splits"] = [0] * len(dates)

    return pd.DataFrame(data, index=pd.DatetimeIndex(dates, tz=tz))


# ===========================================================================
# TestHistoricalDataDownload
# ===========================================================================


class TestHistoricalDataDownload:
    @patch("src.data.historical_data.yf.Ticker")
    def test_columns_normalized_to_lowercase(
        self, mock_ticker_cls: MagicMock, manager: HistoricalDataManager
    ) -> None:
        mock_ticker_cls.return_value.history.return_value = _make_ohlcv_df(
            ["2024-04-01", "2024-04-02"]
        )
        df = manager.download_index_history("NIFTY50")
        assert "open"  in df.columns
        assert "high"  in df.columns
        assert "low"   in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        # Original-case columns must be gone
        assert "Open" not in df.columns

    @patch("src.data.historical_data.yf.Ticker")
    def test_dividends_and_splits_removed(
        self, mock_ticker_cls: MagicMock, manager: HistoricalDataManager
    ) -> None:
        mock_ticker_cls.return_value.history.return_value = _make_ohlcv_df(
            ["2024-04-01"], include_dividends=True
        )
        df = manager.download_index_history("NIFTY50")
        assert "dividends"    not in df.columns
        assert "stock splits" not in df.columns

    @patch("src.data.historical_data.yf.Ticker")
    def test_nan_close_rows_dropped(
        self, mock_ticker_cls: MagicMock, manager: HistoricalDataManager
    ) -> None:
        import numpy as np

        raw = _make_ohlcv_df(["2024-04-01", "2024-04-02", "2024-04-03"])
        raw.loc[raw.index[1], "Close"] = float("nan")
        mock_ticker_cls.return_value.history.return_value = raw
        df = manager.download_index_history("NIFTY50")
        assert len(df) == 2

    @patch("src.data.historical_data.yf.Ticker")
    def test_timestamp_column_present(
        self, mock_ticker_cls: MagicMock, manager: HistoricalDataManager
    ) -> None:
        mock_ticker_cls.return_value.history.return_value = _make_ohlcv_df(
            ["2024-04-01"]
        )
        df = manager.download_index_history("NIFTY50")
        assert "timestamp" in df.columns

    @patch("src.data.historical_data.yf.Ticker")
    def test_ohlcv_values_preserved(
        self, mock_ticker_cls: MagicMock, manager: HistoricalDataManager
    ) -> None:
        mock_ticker_cls.return_value.history.return_value = _make_ohlcv_df(
            ["2024-04-01", "2024-04-02"]
        )
        df = manager.download_index_history("NIFTY50")
        assert len(df) == 2
        assert df.iloc[0]["open"] == pytest.approx(100.0)
        assert df.iloc[1]["close"] == pytest.approx(104.0)

    def test_raises_on_missing_yahoo_symbol(
        self, mock_registry: MagicMock, mock_db: MagicMock
    ) -> None:
        mock_registry.get_index.return_value = _IDX_NO_SYMBOL
        mgr = HistoricalDataManager(registry=mock_registry, db=mock_db)
        with pytest.raises(HistoricalDataError, match="yahoo_symbol"):
            mgr.download_index_history("NODATA")

    def test_raises_when_registry_returns_none(
        self, mock_registry: MagicMock, mock_db: MagicMock
    ) -> None:
        mock_registry.get_index.return_value = None
        mgr = HistoricalDataManager(registry=mock_registry, db=mock_db)
        with pytest.raises(HistoricalDataError):
            mgr.download_index_history("GHOST")

    @patch("src.data.historical_data.yf.Ticker")
    def test_raises_on_empty_dataframe(
        self, mock_ticker_cls: MagicMock, manager: HistoricalDataManager
    ) -> None:
        mock_ticker_cls.return_value.history.return_value = pd.DataFrame()
        with pytest.raises(HistoricalDataError, match="No data"):
            manager.download_index_history("NIFTY50")


# ===========================================================================
# TestDownloadAllActive
# ===========================================================================


class TestDownloadAllActive:
    @patch("src.data.historical_data.yf.Ticker")
    @patch("src.data.historical_data.time.sleep")
    def test_returns_summary_with_correct_keys(
        self,
        _sleep: MagicMock,
        mock_ticker_cls: MagicMock,
        manager: HistoricalDataManager,
    ) -> None:
        mock_ticker_cls.return_value.history.return_value = _make_ohlcv_df(
            ["2024-04-01", "2024-04-02"]
        )
        result = manager.download_all_active_indices()
        assert "downloaded"    in result
        assert "failed"        in result
        assert "total_records" in result

    @patch("src.data.historical_data.yf.Ticker")
    @patch("src.data.historical_data.time.sleep")
    def test_counts_downloaded_indices(
        self,
        _sleep: MagicMock,
        mock_ticker_cls: MagicMock,
        manager: HistoricalDataManager,
    ) -> None:
        mock_ticker_cls.return_value.history.return_value = _make_ohlcv_df(
            ["2024-04-01"]
        )
        result = manager.download_all_active_indices()
        # Both NIFTY50 and BANKNIFTY have yahoo_symbol
        assert result["downloaded"] == 2
        assert result["failed"] == 0

    @patch("src.data.historical_data.yf.Ticker")
    @patch("src.data.historical_data.time.sleep")
    def test_counts_failed_indices(
        self,
        _sleep: MagicMock,
        mock_ticker_cls: MagicMock,
        mock_registry: MagicMock,
        mock_db: MagicMock,
    ) -> None:
        mock_ticker_cls.return_value.history.side_effect = Exception("network error")
        mgr = HistoricalDataManager(registry=mock_registry, db=mock_db)
        result = mgr.download_all_active_indices()
        assert result["failed"] == 2
        assert result["downloaded"] == 0

    @patch("src.data.historical_data.time.sleep")
    def test_skips_indices_without_yahoo_symbol(
        self,
        _sleep: MagicMock,
        mock_db: MagicMock,
    ) -> None:
        reg = MagicMock(spec=IndexRegistry)
        reg.get_active_indices.return_value = [_IDX_NO_SYMBOL]
        mgr = HistoricalDataManager(registry=reg, db=mock_db)
        result = mgr.download_all_active_indices()
        assert result["downloaded"] == 0
        assert result["failed"] == 0

    def test_requires_db(self, mock_registry: MagicMock) -> None:
        mgr = HistoricalDataManager(registry=mock_registry)
        with pytest.raises(ValueError, match="Database"):
            mgr.download_all_active_indices()


# ===========================================================================
# TestUpdateDailyData
# ===========================================================================


class TestUpdateDailyData:
    def test_requires_db(self, mock_registry: MagicMock) -> None:
        mgr = HistoricalDataManager(registry=mock_registry)
        with pytest.raises(ValueError, match="Database"):
            mgr.update_daily_data()

    @patch("src.data.historical_data.yf.Ticker")
    @patch("src.data.historical_data.time.sleep")
    def test_skips_up_to_date_index(
        self,
        _sleep: MagicMock,
        mock_ticker_cls: MagicMock,
        manager: HistoricalDataManager,
        mock_db: MagicMock,
    ) -> None:
        # Simulate last record was today → diff ≤ 1 → no download
        from datetime import datetime as dt
        import zoneinfo

        today_str = dt.now(tz=zoneinfo.ZoneInfo("Asia/Kolkata")).date().isoformat()
        mock_db.fetch_one.return_value = {"timestamp": f"{today_str}T15:30:00+05:30"}
        manager.update_daily_data()
        mock_ticker_cls.return_value.history.assert_not_called()

    @patch("src.data.historical_data.yf.Ticker")
    @patch("src.data.historical_data.time.sleep")
    def test_downloads_when_last_record_is_stale(
        self,
        _sleep: MagicMock,
        mock_ticker_cls: MagicMock,
        manager: HistoricalDataManager,
        mock_db: MagicMock,
    ) -> None:
        mock_ticker_cls.return_value.history.return_value = _make_ohlcv_df(
            ["2024-04-03"]
        )
        # Last stored date 5 days ago → stale
        mock_db.fetch_one.return_value = {"timestamp": "2020-01-01T15:30:00+05:30"}
        manager.update_daily_data()
        assert mock_ticker_cls.return_value.history.called
        assert mock_db.execute_many.called

    @patch("src.data.historical_data.yf.Ticker")
    @patch("src.data.historical_data.time.sleep")
    def test_backfills_when_no_stored_record(
        self,
        _sleep: MagicMock,
        mock_ticker_cls: MagicMock,
        manager: HistoricalDataManager,
        mock_db: MagicMock,
    ) -> None:
        mock_db.fetch_one.return_value = None  # no stored record at all
        mock_ticker_cls.return_value.history.return_value = _make_ohlcv_df(
            ["2024-04-01"]
        )
        manager.update_daily_data()
        assert mock_ticker_cls.return_value.history.called

    @patch("src.data.historical_data.yf.Ticker")
    @patch("src.data.historical_data.time.sleep")
    def test_skips_failed_index_and_continues(
        self,
        _sleep: MagicMock,
        mock_ticker_cls: MagicMock,
        mock_db: MagicMock,
    ) -> None:
        # NIFTY50: registry returns None → HistoricalDataError (non-retryable, 0 yf calls)
        # BANKNIFTY: registry returns valid index → yfinance succeeds (1 yf call)
        reg = MagicMock(spec=IndexRegistry)
        reg.get_active_indices.return_value = [_IDX_NIFTY, _IDX_BANKNIFTY]
        reg.get_index.side_effect = [None, _IDX_BANKNIFTY]
        mock_db.fetch_one.return_value = {"timestamp": "2020-01-01T15:30:00+05:30"}
        mock_ticker_cls.return_value.history.return_value = _make_ohlcv_df(["2024-04-01"])

        mgr = HistoricalDataManager(registry=reg, db=mock_db)
        mgr.update_daily_data()  # must not raise

        # BANKNIFTY download succeeded; data was persisted
        assert mock_db.execute_many.called


# ===========================================================================
# TestGetStoredHistory
# ===========================================================================


class TestGetStoredHistory:
    def test_requires_db(self, mock_registry: MagicMock) -> None:
        mgr = HistoricalDataManager(registry=mock_registry)
        with pytest.raises(ValueError, match="Database"):
            mgr.get_stored_history("NIFTY50", "2024-01-01", "2024-04-01")

    def test_returns_empty_df_when_no_rows(
        self, manager: HistoricalDataManager, mock_db: MagicMock
    ) -> None:
        mock_db.fetch_all.return_value = []
        df = manager.get_stored_history("NIFTY50", "2024-01-01", "2024-04-01")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_returns_dataframe_with_timestamp_index(
        self, manager: HistoricalDataManager, mock_db: MagicMock
    ) -> None:
        mock_db.fetch_all.return_value = [
            {
                "timestamp": "2024-04-01T09:15:00+05:30",
                "open": 22000.0,
                "high": 22100.0,
                "low": 21900.0,
                "close": 22050.0,
                "volume": 500000,
            }
        ]
        df = manager.get_stored_history("NIFTY50", "2024-04-01", "2024-04-01")
        assert not df.empty
        assert df.index.name == "timestamp"

    def test_passes_correct_params_to_db(
        self, manager: HistoricalDataManager, mock_db: MagicMock
    ) -> None:
        mock_db.fetch_all.return_value = []
        manager.get_stored_history("NIFTY50", "2024-01-01", "2024-03-31", timeframe="5m")
        args = mock_db.fetch_all.call_args[0][1]
        assert args[0] == "NIFTY50"
        assert args[1] == "5m"
        assert "2024-01-01" in args
        assert "2024-03-31" in args


# ===========================================================================
# TestSaveDataframeToDb (DataValidator integration)
# ===========================================================================


class TestSaveDataframeToDb:
    @patch("src.data.historical_data.yf.Ticker")
    def test_valid_rows_are_upserted(
        self, mock_ticker_cls: MagicMock, manager: HistoricalDataManager, mock_db: MagicMock
    ) -> None:
        mock_ticker_cls.return_value.history.return_value = _make_ohlcv_df(
            ["2024-04-01", "2024-04-02"]
        )
        df = manager.download_index_history("NIFTY50")
        manager._save_dataframe_to_db("NIFTY50", df, "1d")
        mock_db.execute_many.assert_called_once()
        rows = mock_db.execute_many.call_args[0][1]
        assert len(rows) == 2

    def test_invalid_rows_are_skipped(
        self, manager: HistoricalDataManager, mock_db: MagicMock
    ) -> None:
        # Build a df where close = 0 (invalid — fails validate_price_data positivity check)
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-04-01"], utc=True),
                "open":      [0.0],
                "high":      [0.0],
                "low":       [0.0],
                "close":     [0.0],  # zero is invalid
                "volume":    [0.0],
            }
        )
        manager._save_dataframe_to_db("NIFTY50", df, "1d")
        mock_db.execute_many.assert_not_called()

    def test_empty_df_does_not_call_db(
        self, manager: HistoricalDataManager, mock_db: MagicMock
    ) -> None:
        manager._save_dataframe_to_db("NIFTY50", pd.DataFrame(), "1d")
        mock_db.execute_many.assert_not_called()


# ===========================================================================
# TestFIIDIIFetcher
# ===========================================================================

_SAMPLE_FII_RAW = [
    {
        "date": "01-Apr-2024",
        "fiiBuyValue":  "50000.00",
        "fiiSellValue": "45000.00",
        "fiiNetValue":  "5000.00",
        "diiBuyValue":  "30000.00",
        "diiSellValue": "32000.00",
        "diiNetValue":  "-2000.00",
    },
    {
        "date": "29-Mar-2024",
        "fiiBuyValue":  "40000.00",
        "fiiSellValue": "42000.00",
        "fiiNetValue":  "-2000.00",
        "diiBuyValue":  "28000.00",
        "diiSellValue": "26000.00",
        "diiNetValue":  "2000.00",
    },
]


@pytest.fixture()
def mock_scraper_fii() -> MagicMock:
    scraper = MagicMock()
    scraper.get_fii_dii_activity.return_value = _SAMPLE_FII_RAW
    return scraper


@pytest.fixture()
def fii_fetcher(mock_scraper_fii: MagicMock) -> FIIDIIFetcher:
    return FIIDIIFetcher(scraper=mock_scraper_fii)


class TestFIIDIIFetcher:
    def test_fetch_today_returns_fii_dii_data(
        self, fii_fetcher: FIIDIIFetcher
    ) -> None:
        result = fii_fetcher.fetch_today_fii_dii()
        assert result is not None
        assert isinstance(result, FIIDIIData)

    def test_fetch_today_parses_values_correctly(
        self, fii_fetcher: FIIDIIFetcher
    ) -> None:
        result = fii_fetcher.fetch_today_fii_dii()
        assert result is not None
        assert result.fii_buy_value  == pytest.approx(50000.0)
        assert result.fii_sell_value == pytest.approx(45000.0)
        assert result.fii_net_value  == pytest.approx(5000.0)
        assert result.dii_buy_value  == pytest.approx(30000.0)
        assert result.dii_net_value  == pytest.approx(-2000.0)

    def test_fetch_today_returns_none_when_api_returns_none(self) -> None:
        scraper = MagicMock()
        scraper.get_fii_dii_activity.return_value = None
        fetcher = FIIDIIFetcher(scraper=scraper)
        assert fetcher.fetch_today_fii_dii() is None

    def test_fetch_today_returns_none_on_empty_list(self) -> None:
        scraper = MagicMock()
        scraper.get_fii_dii_activity.return_value = []
        fetcher = FIIDIIFetcher(scraper=scraper)
        assert fetcher.fetch_today_fii_dii() is None

    def test_fetch_historical_filters_by_date_range(
        self, fii_fetcher: FIIDIIFetcher
    ) -> None:
        # Only 01-Apr-2024 should be included (29-Mar-2024 is outside range)
        result = fii_fetcher.fetch_historical_fii_dii(
            start_date=date(2024, 4, 1),
            end_date=date(2024, 4, 30),
        )
        assert len(result) == 1
        assert result[0].date == date(2024, 4, 1)

    def test_fetch_historical_returns_all_in_range(
        self, fii_fetcher: FIIDIIFetcher
    ) -> None:
        result = fii_fetcher.fetch_historical_fii_dii(
            start_date=date(2024, 3, 1),
            end_date=date(2024, 4, 30),
        )
        assert len(result) == 2

    def test_get_fii_trend_buying(self, fii_fetcher: FIIDIIFetcher) -> None:
        # Both records: net = 5000 and -2000 → avg = 1500 > 500 → BUYING
        trend = fii_fetcher.get_fii_trend(days=2)
        assert trend["trend"] == "BUYING"
        assert trend["total_net"] == pytest.approx(3000.0)
        assert trend["avg_daily_net"] == pytest.approx(1500.0)

    def test_get_fii_trend_selling(self) -> None:
        scraper = MagicMock()
        scraper.get_fii_dii_activity.return_value = [
            {"date": "01-Apr-2024", "fiiNetValue": "-1000"},
            {"date": "29-Mar-2024", "fiiNetValue": "-2000"},
        ]
        fetcher = FIIDIIFetcher(scraper=scraper)
        trend = fetcher.get_fii_trend(days=2)
        assert trend["trend"] == "SELLING"

    def test_get_fii_trend_neutral(self) -> None:
        scraper = MagicMock()
        scraper.get_fii_dii_activity.return_value = [
            {"date": "01-Apr-2024", "fiiNetValue": "100"},
        ]
        fetcher = FIIDIIFetcher(scraper=scraper)
        trend = fetcher.get_fii_trend(days=1)
        assert trend["trend"] == "NEUTRAL"

    def test_get_fii_trend_empty_returns_neutral(self) -> None:
        scraper = MagicMock()
        scraper.get_fii_dii_activity.return_value = []
        fetcher = FIIDIIFetcher(scraper=scraper)
        trend = fetcher.get_fii_trend()
        assert trend["trend"] == "NEUTRAL"
        assert trend["total_net"] == 0.0

    def test_get_fii_trend_consecutive_buy_days(self) -> None:
        scraper = MagicMock()
        scraper.get_fii_dii_activity.return_value = [
            {"date": "03-Apr-2024", "fiiNetValue": "2000"},
            {"date": "02-Apr-2024", "fiiNetValue": "1500"},
            {"date": "01-Apr-2024", "fiiNetValue": "3000"},
        ]
        fetcher = FIIDIIFetcher(scraper=scraper)
        trend = fetcher.get_fii_trend(days=3)
        assert trend["consecutive_buy_days"] == 3
        assert trend["consecutive_sell_days"] == 0

    def test_get_fii_trend_consecutive_sell_days(self) -> None:
        scraper = MagicMock()
        scraper.get_fii_dii_activity.return_value = [
            {"date": "03-Apr-2024", "fiiNetValue": "-800"},
            {"date": "02-Apr-2024", "fiiNetValue": "-600"},
        ]
        fetcher = FIIDIIFetcher(scraper=scraper)
        trend = fetcher.get_fii_trend(days=2)
        assert trend["consecutive_sell_days"] == 2

    def test_save_to_db_inserts_fii_and_dii_rows(
        self, fii_fetcher: FIIDIIFetcher
    ) -> None:
        db = MagicMock()
        data = FIIDIIData(
            date=date(2024, 4, 1),
            fii_buy_value=50000.0, fii_sell_value=45000.0, fii_net_value=5000.0,
            dii_buy_value=30000.0, dii_sell_value=32000.0, dii_net_value=-2000.0,
        )
        fii_fetcher.save_to_db(data, db)
        # FII row + DII row = 2 execute calls (no F&O since fo fields are 0)
        assert db.execute.call_count == 2

    def test_save_to_db_inserts_fo_row_when_nonzero(
        self, fii_fetcher: FIIDIIFetcher
    ) -> None:
        db = MagicMock()
        data = FIIDIIData(
            date=date(2024, 4, 1),
            fii_buy_value=50000.0, fii_sell_value=45000.0, fii_net_value=5000.0,
            dii_buy_value=30000.0, dii_sell_value=32000.0, dii_net_value=-2000.0,
            fii_fo_buy=10000.0, fii_fo_sell=8000.0, fii_fo_net=2000.0,
        )
        fii_fetcher.save_to_db(data, db)
        # FII + DII + F&O = 3 execute calls
        assert db.execute.call_count == 3

    def test_save_to_db_logs_on_db_error(
        self, fii_fetcher: FIIDIIFetcher
    ) -> None:
        db = MagicMock()
        db.execute.side_effect = Exception("db locked")
        data = FIIDIIData(
            date=date(2024, 4, 1),
            fii_buy_value=1.0, fii_sell_value=1.0, fii_net_value=0.0,
            dii_buy_value=1.0, dii_sell_value=1.0, dii_net_value=0.0,
        )
        # Must not propagate the exception
        fii_fetcher.save_to_db(data, db)


# ===========================================================================
# TestVIXTracker
# ===========================================================================


@pytest.fixture()
def mock_scraper_vix() -> MagicMock:
    scraper = MagicMock()
    scraper.get_vix.return_value = {
        "vix_value": 15.5,
        "vix_change": 0.3,
        "vix_change_pct": 1.97,
    }
    return scraper


@pytest.fixture()
def vix_tracker(mock_scraper_vix: MagicMock) -> VIXTracker:
    return VIXTracker(scraper=mock_scraper_vix)


class TestVIXTracker:
    def test_get_current_vix_returns_vixdata(
        self, vix_tracker: VIXTracker
    ) -> None:
        result = vix_tracker.get_current_vix()
        assert result is not None
        assert isinstance(result, VIXData)
        assert result.value == pytest.approx(15.5)
        assert result.change == pytest.approx(0.3)
        assert result.change_pct == pytest.approx(1.97)

    def test_get_current_vix_timestamp_is_aware(
        self, vix_tracker: VIXTracker
    ) -> None:
        result = vix_tracker.get_current_vix()
        assert result is not None
        assert result.timestamp.tzinfo is not None

    def test_get_current_vix_returns_none_when_scraper_none(self) -> None:
        scraper = MagicMock()
        scraper.get_vix.return_value = None
        tracker = VIXTracker(scraper=scraper)
        assert tracker.get_current_vix() is None

    # ── VIX regime classification (uses configurable settings thresholds) ──

    def test_regime_low_vol(self, vix_tracker: VIXTracker) -> None:
        # Below vix_normal_threshold (default 13.0)
        vix_tracker.get_current_vix = MagicMock(
            return_value=VIXData(12.0, -1.0, -5.0, datetime.now())
        )
        assert vix_tracker.get_vix_regime() == "LOW_VOL"

    def test_regime_normal(self, vix_tracker: VIXTracker) -> None:
        # 13.0 ≤ val < 18.0
        vix_tracker.get_current_vix = MagicMock(
            return_value=VIXData(15.0, 0.0, 0.0, datetime.now())
        )
        assert vix_tracker.get_vix_regime() == "NORMAL"

    def test_regime_elevated(self, vix_tracker: VIXTracker) -> None:
        # 18.0 ≤ val < 25.0
        vix_tracker.get_current_vix = MagicMock(
            return_value=VIXData(20.0, 2.0, 11.0, datetime.now())
        )
        assert vix_tracker.get_vix_regime() == "ELEVATED"

    def test_regime_high_vol(self, vix_tracker: VIXTracker) -> None:
        # val ≥ 25.0
        vix_tracker.get_current_vix = MagicMock(
            return_value=VIXData(30.0, 5.0, 20.0, datetime.now())
        )
        assert vix_tracker.get_vix_regime() == "HIGH_VOL"

    def test_regime_boundary_at_normal_threshold(
        self, vix_tracker: VIXTracker
    ) -> None:
        # Exactly at the LOW_VOL ceiling → NORMAL (not LOW_VOL)
        vix_tracker.get_current_vix = MagicMock(
            return_value=VIXData(13.0, 0.0, 0.0, datetime.now())
        )
        assert vix_tracker.get_vix_regime() == "NORMAL"

    def test_regime_boundary_at_elevated_threshold(
        self, vix_tracker: VIXTracker
    ) -> None:
        vix_tracker.get_current_vix = MagicMock(
            return_value=VIXData(18.0, 0.0, 0.0, datetime.now())
        )
        assert vix_tracker.get_vix_regime() == "ELEVATED"

    def test_regime_boundary_at_panic_threshold(
        self, vix_tracker: VIXTracker
    ) -> None:
        vix_tracker.get_current_vix = MagicMock(
            return_value=VIXData(25.0, 0.0, 0.0, datetime.now())
        )
        assert vix_tracker.get_vix_regime() == "HIGH_VOL"

    def test_regime_fallback_to_normal_when_no_vix_data(self) -> None:
        scraper = MagicMock()
        scraper.get_vix.return_value = None
        tracker = VIXTracker(scraper=scraper)
        assert tracker.get_vix_regime() == "NORMAL"

    def test_save_to_db_calls_execute(self, vix_tracker: VIXTracker) -> None:
        db = MagicMock()
        data = VIXData(
            value=15.5,
            change=0.3,
            change_pct=1.97,
            timestamp=datetime.now(tz=__import__("zoneinfo").ZoneInfo("Asia/Kolkata")),
        )
        vix_tracker.save_to_db(data, db)
        db.execute.assert_called_once()

    def test_save_to_db_logs_on_error(self, vix_tracker: VIXTracker) -> None:
        db = MagicMock()
        db.execute.side_effect = Exception("db error")
        data = VIXData(
            value=15.5, change=0.3, change_pct=1.97,
            timestamp=datetime.now(tz=__import__("zoneinfo").ZoneInfo("Asia/Kolkata")),
        )
        # Must not propagate
        vix_tracker.save_to_db(data, db)
