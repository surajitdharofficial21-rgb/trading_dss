"""
Unit tests for the DataCollector orchestrator.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.data.data_collector import DataCollector


@pytest.fixture
def mock_db():
    db = MagicMock()
    return db


@pytest.fixture
def mock_registry(registry):
    return registry


@pytest.fixture
def data_collector(mock_db, mock_registry):
    """Return a DataCollector instance with mocked internal components."""
    with patch("src.data.data_collector.NSEScraper"), \
         patch("src.data.data_collector.BSEScraper"), \
         patch("src.data.data_collector.OptionsChainFetcher"), \
         patch("src.data.data_collector.HistoricalDataManager"), \
         patch("src.data.data_collector.FIIDIIFetcher"), \
         patch("src.data.data_collector.VIXTracker"), \
         patch("src.data.data_collector.MarketHoursManager"):
        
        collector = DataCollector(
            db=mock_db,
            registry=mock_registry,
            dry_run=True,
            force_start=False
        )
        yield collector


def test_scheduler_starts_stops_cleanly(data_collector):
    """Test that the APScheduler starts and stops gracefully without throwing errors."""
    assert not data_collector.scheduler.running
    
    data_collector.start()
    assert data_collector.scheduler.running
    
    data_collector.stop()
    assert not data_collector.scheduler.running


def test_jobs_registered_correctly(data_collector):
    """Test that all scheduled jobs are registered properly during initialization."""
    jobs = {job.id for job in data_collector.scheduler.get_jobs()}
    
    expected_jobs = {
        "collect_index_prices",
        "collect_options_chain",
        "collect_vix",
        "health_check",
        "warm_up",
        "collect_fii_dii",
        "update_historical",
        "cleanup"
    }
    
    # Assert all expected jobs are present
    for job in expected_jobs:
        assert job in jobs


def test_market_hours_filtering(data_collector):
    """Test jobs do not execute when the market is closed unless force_start=True."""
    # Ensure market is mocked as CLOSED
    data_collector.market_hours.get_market_status.return_value = {"status": "CLOSED"}
    data_collector.force_start = False
    
    # Try running the price collection
    data_collector.collect_index_prices()
    
    # Since it shouldn't run, nse_scraper should NOT be called
    data_collector.nse_scraper.get_all_indices.assert_not_called()

    # Now test with force_start = True
    data_collector.force_start = True
    data_collector.nse_scraper.get_all_indices.return_value = []
    
    data_collector.collect_index_prices()
    data_collector.nse_scraper.get_all_indices.assert_called_once()


def test_error_recovery_thresholds(data_collector):
    """Test that a component failure triggers auto-recovery logic at threshold but doesn't crash."""
    component = "nse"
    
    # Trigger 9 failures (below recovery threshold)
    for _ in range(9):
        data_collector._handle_failure(component, Exception("Test Error"))
        
    assert data_collector._failures[component] == 9
    
    # Capture the original object reference
    original_scraper = data_collector.nse_scraper
    
    with patch("src.data.data_collector.time.monotonic", return_value=100.0), \
         patch("src.data.data_collector.NSEScraper") as mock_nse, \
         patch("src.data.data_collector.OptionsChainFetcher"), \
         patch("src.data.data_collector.FIIDIIFetcher"), \
         patch("src.data.data_collector.VIXTracker"):
        
        # Trigger 10th failure (reaches recovery threshold)
        data_collector._handle_failure(component, Exception("Test Critical Error"))
        
        # Verify the failure count is 10
        assert data_collector._failures[component] == 10
        
        # Verify auto-recovery triggered object reassignment
        assert data_collector.nse_scraper is not original_scraper
        mock_nse.assert_called()


def test_graceful_shutdown_controls(data_collector):
    """Test start, pause, resume, and stop controls for the collector."""
    data_collector.start()
    
    status = data_collector.get_status()
    assert "health" in status
    assert "jobs" in status
    assert isinstance(status["jobs"], list)
    
    data_collector.pause()
    data_collector.resume()
    data_collector.stop()
    
    assert not data_collector.scheduler.running
