
import sys
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Add src to path
sys.path.append(os.getcwd())

from pathlib import Path
from src.analysis.technical_aggregator import TechnicalAggregator
from src.analysis.indicator_store import IndicatorStore
from src.database.db_manager import DatabaseManager

def generate_mock_data(n=500, base=20000.0, trend=0.0005, volatility=0.005):
    """Generate mock OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
    
    # Random walk with trend
    returns = np.random.normal(trend, volatility, n)
    price_ratio = np.exp(np.cumsum(returns))
    close = base * price_ratio
    
    high = close * (1 + np.random.uniform(0, 0.01, n))
    low = close * (1 - np.random.uniform(0, 0.01, n))
    open_ = (high + low) / 2 + np.random.uniform(-0.002, 0.002, n) * close
    volume = np.random.randint(100000, 1000000, n)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df

def verify_manual():
    print("=== Phase 2 Manual Verification ===")
    agg = TechnicalAggregator()
    db = DatabaseManager(db_path=Path("/tmp/test_verify.db"))
    db.connect()
    db.initialise_schema()

    # 1. Test Graceful Degradation (NIFTY_IT - No Options)
    print("\n[1] Testing NIFTY_IT (No Options)...")
    df_it = generate_mock_data(n=500, base=35000.0)
    start_time = time.time()
    result_it = agg.analyze("NIFTY_IT", df_it)
    elapsed = time.time() - start_time
    
    print(f"Analysis completed in {elapsed:.4f}s")
    print(f"Overall Signal: {result_it.overall_signal}")
    print(f"Reasoning Length: {len(result_it.reasoning)} chars")
    print("--- REASONING PREVIEW ---")
    print("\n".join(result_it.reasoning.split('\n')[:5]) + "...")
    print("-------------------------")
    print(f"Warnings: {result_it.warnings}")

    # 2. Test Support/Resistance Sensibility
    print("\n[2] Testing S/R Levels (NIFTY50)...")
    df_n50 = generate_mock_data(n=500, base=22000.0)
    result_n50 = agg.analyze("NIFTY50", df_n50)
    
    print(f"Current Price: {df_n50.iloc[-1]['close']:.2f}")
    print(f"Support Levels: {result_n50.support_levels}")
    print(f"Resistance Levels: {result_n50.resistance_levels}")
    print(f"Immediate Support: {result_n50.immediate_support:.2f}")
    print(f"Immediate Resistance: {result_n50.immediate_resistance:.2f}")
    
    # 3. Test Alerts (Force some conditions)
    print("\n[3] Testing Alerts...")
    # Add a spike in volume and price to trigger something
    df_spike = df_n50.copy()
    df_spike.iloc[-1, df_spike.columns.get_loc('volume')] *= 10
    df_spike.iloc[-1, df_spike.columns.get_loc('close')] *= 1.05
    df_spike.iloc[-1, df_spike.columns.get_loc('high')] *= 1.06
    
    result_spike = agg.analyze("NIFTY50", df_spike)
    print(f"Alerts found: {len(result_spike.alerts)}")
    for alert in result_spike.alerts:
        print(f" - [{alert.severity}] {alert.type}: {alert.message}")

    # 4. Test DB Round-trip
    print("\n[4] Testing DB Round-trip...")
    # Need to add index_master first
    now_str = datetime.now().isoformat()
    db.execute(
        "INSERT INTO index_master (id, display_name, exchange, sector_category, created_at, updated_at) VALUES ('NIFTY50', 'NIFTY 50', 'NSE', 'broad_market', ?, ?)",
        (now_str, now_str)
    )
    IndicatorStore.save_analysis(result_n50, db)
    loaded = IndicatorStore.get_latest_analysis("NIFTY50", db)
    if loaded:
        print("Successfully saved and loaded from DB.")
        print(f"Loaded Signal Strength: {loaded['signal_strength']}")
    else:
        print("FAILED to load from DB.")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    verify_manual()
