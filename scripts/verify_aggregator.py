from src.analysis import TechnicalAggregator, IndicatorStore
from src.data.historical_data import HistoricalDataManager
from src.data.options_chain import OptionsChainFetcher
from src.database.db_manager import DatabaseManager
import os

from pathlib import Path

# Create data/db directory if it doesn't exist
os.makedirs("data/db", exist_ok=True)

from pathlib import Path
from src.data.index_registry import get_registry

# Create data/db directory if it doesn't exist
os.makedirs("data/db", exist_ok=True)

db = DatabaseManager(Path("data/db/trading.db"))
db.connect()
db.initialise_schema()

# Sync registry to DB
registry = get_registry()
print("Syncing registry to DB...")
registry.sync_to_db(db)

hm = HistoricalDataManager(db=db)
aggregator = TechnicalAggregator()
store = IndicatorStore()

# Ensure we have data for NIFTY50
idx_nifty = "NIFTY50"
try:
    df = hm.get_stored_history(idx_nifty, "2023-01-01", "2025-01-01")
    if df.empty:
        print(f"{idx_nifty} data not found in DB. Downloading directly...")
        df = hm.download_index_history(idx_nifty, period="2y")
        print(f"Downloaded {len(df)} rows for {idx_nifty}")
        # Save to DB manually to make it stick
        hm._save_dataframe_to_db(idx_nifty, df, timeframe="1d")
except Exception as e:
    print(f"Error getting {idx_nifty}: {e}")
    df = None

if df is None or df.empty:
    print("FATAL: Failed to get NIFTY50 data.")
    exit(1)

df_benchmark = df  # NIFTY is its own benchmark for now

# Get options chain (only during market hours)
try:
    fetcher = OptionsChainFetcher()
    chain = fetcher.get_options_chain("NIFTY")
except Exception as e:
    print(f"Options chain fetcher error (expected outside market hours): {e}")
    chain = None

result = aggregator.analyze(
    index_id=idx_nifty,
    price_df=df,
    options_chain=chain,
    vix_value=15.2,
    benchmark_df=df_benchmark
)

# 2. Verify all fields
print(f"\nNIFTY 50 Analysis:")
print(f"Signal: {result.overall_signal} (confidence: {result.overall_confidence:.2f})")
print(f"Votes: {result.votes}")
print(f"Bullish: {result.bullish_votes}, Bearish: {result.bearish_votes}, Neutral: {result.neutral_votes}")
print(f"Support: {result.support_levels}")
print(f"Resistance: {result.resistance_levels}")
print(f"SL distance: {result.suggested_stop_loss_distance:.2f}")
print(f"Alerts: {len(result.alerts)}")
print(f"Data completeness: {result.data_completeness:.2f}")
print("\n=== REASONING ===")
print(result.reasoning)

# 3. Save to DB
store.save_analysis(result, db)
loaded = store.get_latest_analysis(idx_nifty, db)
assert loaded is not None
print(f"\nSaved and loaded successfully: {loaded['technical_signal']}")

# 4. Test on index WITHOUT options
idx_it = "NIFTY_IT" # Correct ID from indices.json
try:
    df_it = hm.get_stored_history(idx_it, "2023-01-01", "2025-01-01")
    if df_it.empty:
        print(f"\n{idx_it} data not found in DB. Downloading directly...")
        df_it = hm.download_index_history(idx_it, period="2y")
        print(f"Downloaded {len(df_it)} rows for {idx_it}")
except Exception as e:
    print(f"Error getting {idx_it}: {e}")
    df_it = None

if df_it is not None and not df_it.empty:
    result_it = aggregator.analyze(index_id=idx_it, price_df=df_it)
    print(f"\n{idx_it} (no options): {result_it.overall_signal} (confidence: {result_it.overall_confidence:.2f})")
    print(f"Warnings: {result_it.warnings}")
else:
    print(f"\nCould not run analysis for {idx_it} due to missing data.")
