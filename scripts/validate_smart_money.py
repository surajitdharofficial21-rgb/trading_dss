import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.database.db_manager import DatabaseManager
from src.data.index_registry import get_registry
from src.data.historical_data import HistoricalDataManager
from src.analysis.indicators.smart_money import SmartMoneyIndicators

# Setup basic logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    # 1. Initialize dependencies
    db = DatabaseManager.instance()
    db.initialise_schema()
    registry = get_registry()
    hm = HistoricalDataManager(registry, db)
    smi = SmartMoneyIndicators()

    index_id = "NIFTY50"
    start_date = "2024-06-01"
    end_date = "2025-01-01"

    print(f"--- Validating Smart Money Indicators for {index_id} ---")
    
    # 2. Get data (try storage first, then download if empty)
    df = hm.get_stored_history(index_id, start_date, end_date)
    
    if df.empty:
        print(f"No stored data found for {index_id}. Downloading history...")
        # Download 1 year of data to cover the range
        df_full = hm.download_index_history(index_id, period="2y")
        # Filter for the requested range
        df = df_full[(df_full.index >= start_date) & (df_full.index <= end_date)]
        
    if df.empty:
        print("Failed to acquire data. check index registry or internet connection.")
        return

    print(f"Acquired {len(df)} bars of data.")

    # 3. Run SMFI
    smfi = smi.calculate_smfi(df)
    print(f"SMFI: {smfi.current_value:.2f} — Signal: {smfi.signal} — Trend: {smfi.trend}")

    # 4. Volume Shocks
    vsd = smi.detect_volume_shocks(df)
    print(f"Recent shocks (last 10 bars): {vsd.recent_shocks_count}")
    print(f"Dominant type: {vsd.dominant_shock_type}")
    if vsd.shocks:
        last_shock = vsd.shocks[-1]
        print(f"Last shock details: {last_shock.timestamp} | {last_shock.shock_type} | Z-Score: {last_shock.volume_zscore}")

    # 5. Smart Money Score (master indicator)
    # Note: Using default support/resistance for this validation if not provided
    score = smi.calculate_smart_money_score(df)
    print(f"Smart Money Score: {score.score:.1f} ({score.grade})")
    print(f"Bias: {score.smart_money_bias}")
    print(f"Key finding: {score.key_finding}")
    print(f"Insight: {score.actionable_insight}")
    print(f"Confidence: {score.confidence:.2f} (Data Completeness: {score.data_completeness:.1f})")

if __name__ == "__main__":
    main()
