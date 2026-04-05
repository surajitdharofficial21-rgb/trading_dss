"""
Integration Tests for the Trading DSS Data Layer.
Verifies the end-to-end functionality of DataCollector orchestrated pipelines.
"""

from __future__ import annotations

import json
from datetime import datetime, date, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from apscheduler.job import Job

from config.settings import settings
from config.constants import IST_TIMEZONE
from zoneinfo import ZoneInfo
from src.data.data_collector import DataCollector
from src.database.db_manager import DatabaseManager
from src.data.index_registry import IndexRegistry
from src.data.options_chain import OptionsChainData, OptionStrike, OISummary


_IST = ZoneInfo(IST_TIMEZONE)


@pytest.fixture
def test_db(tmp_path):
    """Provides a fresh isolated SQLite file DB for the integration test."""
    db_path = tmp_path / "test_trading.db"
    db = DatabaseManager(db_path=db_path)
    db.connect()
    db.initialise_schema()
    yield db


def get_mock_nse_prices():
    return [
        {
            "index_name": "NIFTY 50",
            "indexSymbol": "NIFTY 50",
            "ltp": 22000.50,
            "open": 21900.00,
            "high": 22100.00,
            "low": 21850.00,
            "close": 21950.00,
            "volume": 100000,
            "timestamp": "2024-06-01T10:00:00"
        },
        {
            # Bad data simulating negative prices to be rejected by DataValidator
            "index_name": "NIFTY BANK",
            "indexSymbol": "NIFTY BANK",
            "ltp": -500.00,
            "open": 48000.00,
            "high": 48100.00,
            "low": 47900.00,
            "close": 47950.00,
            "volume": 200000,
            "timestamp": "2024-06-01T10:00:00"
        }
    ]


def test_full_price_collection_flow(test_db, registry):
    """
    Test E2E price collection simulating data fetching, validation, and insertion.
    Verifies valid entries are UPSERTED while negative boundaries are rejected.
    """
    registry.sync_to_db(test_db)
    collector = DataCollector(db=test_db, registry=registry, dry_run=False, force_start=True)
    
    with patch.object(collector.nse_scraper, "get_all_indices", return_value=get_mock_nse_prices()):
        collector.collect_index_prices()
        
    prices = test_db.execute("SELECT index_id, close, volume FROM price_data").fetchall()
    
    # NIFTY 50 is inserted, but NIFTY BANK with negative close gets rejected
    assert len(prices) == 1
    assert prices[0][0] == "NIFTY50"
    assert prices[0][1] == 21950.00  # Based on our mock
    assert prices[0][2] == 100000.0


def test_options_chain_flow(test_db, registry):
    """
    Test parsing an Options chain, calculating OISummary, inserting, and identifying spikes.
    """
    registry.sync_to_db(test_db)
    collector = DataCollector(db=test_db, registry=registry, dry_run=False, force_start=True)
    
    now = datetime.now(tz=_IST)
    
    # Snapshot 1
    chain_1 = OptionsChainData(
        index_id="NIFTY50",
        spot_price=22050.0,
        timestamp=now,
        expiry_date=date(2024, 6, 27),
        strikes=[
            OptionStrike(strike_price=22000.0, ce_oi=50000, ce_oi_change=0, ce_volume=1000, ce_iv=15.0, ce_ltp=100.0,
                                               pe_oi=60000, pe_oi_change=0, pe_volume=1200, pe_iv=16.0, pe_ltp=110.0),
            OptionStrike(strike_price=22100.0, ce_oi=70000, ce_oi_change=0, ce_volume=2000, ce_iv=15.5, ce_ltp=80.0,
                                               pe_oi=40000, pe_oi_change=0, pe_volume=800, pe_iv=16.5, pe_ltp=130.0)
        ],
        available_expiries=tuple([date(2024, 6, 27)])
    )
    
    # Snapshot 2 (Spike in 22000 CE OI by +100%, 22100 PE drops)
    chain_2 = OptionsChainData(
        index_id="NIFTY50",
        spot_price=22050.0,
        timestamp=now + timedelta(minutes=5),
        expiry_date=date(2024, 6, 27),
        strikes=[
            OptionStrike(strike_price=22000.0, ce_oi=100000, ce_oi_change=50000, ce_volume=5000, ce_iv=15.0, ce_ltp=100.0,
                                               pe_oi=60000, pe_oi_change=0, pe_volume=1200, pe_iv=16.0, pe_ltp=110.0),
            OptionStrike(strike_price=22100.0, ce_oi=70000, ce_oi_change=0, ce_volume=2000, ce_iv=15.5, ce_ltp=80.0,
                                               pe_oi=30000, pe_oi_change=-10000, pe_volume=800, pe_iv=16.5, pe_ltp=130.0)
        ],
        available_expiries=tuple([date(2024, 6, 27)])
    )
    
    with patch.object(collector.registry, "get_indices_with_options", return_value=[registry.get_index("NIFTY50")]), \
         patch.object(collector.options_fetcher, "get_options_chain", side_effect=[chain_1, chain_2]), \
         patch.object(collector.options_fetcher, "get_memory_snapshots", side_effect=[[], [chain_1, chain_2]]):
        # Run collection 1
        collector.collect_options_chain() # Populates in memory snapshots
        
        # Verify OI summarization and Max Pain
        db_summ = test_db.execute("SELECT max_pain_strike, pcr FROM oi_aggregated").fetchone()
        assert db_summ[0] == 22000.0 # 22000 has Max CE + PE OI combined
        assert round(db_summ[1], 2) > 0 # PCR valid
        
        # Run collection 2 to trigger Spike Anomaly event
        collector.collect_options_chain()
        
        anomalies = test_db.execute("SELECT severity, details FROM anomaly_events").fetchall()
        assert len(anomalies) > 0
        spike = json.loads(anomalies[0][1])
        assert spike["strike"] == 22000.0
        assert spike["option_type"] == "CE"
        assert spike["change_pct"] == 100.0
        assert anomalies[0][0] == "HIGH"


def test_market_hours_integration(test_db, registry):
    """
    Overrides `now()` checking different market regimes.
    """
    collector = DataCollector(db=test_db, registry=registry, dry_run=False, force_start=False)
    
    with patch.object(collector.market_hours, "get_market_status") as mock_status, \
         patch.object(collector.nse_scraper, "get_all_indices", return_value=[]) as mock_nse:

        # Simulate 8:00 AM Working Day
        mock_status.return_value = {"status": "CLOSED", "reason": "PRE_MARKET"}
        collector.collect_index_prices()
        mock_nse.assert_not_called()

        # Simulate 10:00 AM Working Day
        mock_status.return_value = {"status": "OPEN", "reason": "REGULAR_TRADING"}
        collector.collect_index_prices()
        assert mock_nse.call_count == 1
        
        # Simulate 16:00 PM Working Day
        mock_status.return_value = {"status": "CLOSED", "reason": "POST_MARKET"}
        collector.collect_index_prices()
        assert mock_nse.call_count == 1 # unchanged because closed
        
        # Simulate Saturday
        collector.market_hours.is_weekend = MagicMock(return_value=True)
        collector.collect_index_prices()
        assert mock_nse.call_count == 1 # unchanged because weekend


def test_error_recovery(test_db, registry):
    """
    Verify error handling doesn't hard-crash the application but logs status 
    and keeps executing other independent pipelines.
    """
    registry.sync_to_db(test_db)
    collector = DataCollector(db=test_db, registry=registry, dry_run=False, force_start=True)
    
    with patch.object(collector.nse_scraper, "get_all_indices", side_effect=Exception("NSE Blocked")) as mse, \
         patch.object(collector.bse_scraper, "get_all_indices", return_value=[]) as mbe:
        
        # Call 3 times
        for _ in range(3):
            collector.collect_index_prices()
            
        assert mse.call_count == 3
        # Should execute fall back
        assert mbe.call_count == 3
        
        # Verify it wrote NO records, but didn't crash
        ct = test_db.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
        assert ct == 0
        
        collector.health_check()
        
        # Verify internal failure mechanism tracks
        assert collector._failures["nse"] == 3


def test_database_integrity_and_cleanup(test_db, registry):
    """
    Check composite integrity keys and VACUUM cleanups without blowing out limits.
    """
    registry.sync_to_db(test_db)
    collector = DataCollector(db=test_db, registry=registry, dry_run=False, force_start=True)
    
    # Intentionally insert duplicate options chain
    now = datetime.now(tz=_IST)
    chain = OptionsChainData(
        index_id="NIFTY50",
        spot_price=22050.0,
        timestamp=now,
        expiry_date=date(2024, 6, 27),
        strikes=[
            OptionStrike(strike_price=22000.0, ce_oi=50000, ce_oi_change=0, ce_volume=1000, ce_iv=15.0, ce_ltp=100.0,
                                               pe_oi=60000, pe_oi_change=0, pe_volume=1200, pe_iv=16.0, pe_ltp=110.0)
        ],
        available_expiries=tuple([date(2024, 6, 27)])
    )
    
    # 1. Insert Chain
    with patch.object(collector.options_fetcher, "get_options_chain", return_value=chain):
        collector.collect_options_chain() # OK
        
    # Verify records present
    assert test_db.execute("SELECT COUNT(*) FROM options_chain_snapshot").fetchone()[0] == 2
    
    # 2. Mock time forward 35 days and run cleanup
    old = (now - timedelta(days=35)).isoformat()
    test_db.execute("UPDATE options_chain_snapshot SET timestamp = ?", (old,))
    
    collector.cleanup()
    
    assert test_db.execute("SELECT COUNT(*) FROM options_chain_snapshot").fetchone()[0] == 0
