import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.analysis.indicators.smart_money import SmartMoneyIndicators, SMFIResult, VSDResult, BTDResult, OIMIResult, LAIResult, SmartMoneyScore

class TestSmartMoneyIndicators(unittest.TestCase):
    def setUp(self):
        self.indicators = SmartMoneyIndicators()
        
    def create_dummy_ohlcv(self, rows=100):
        dates = [datetime.now() - timedelta(days=i) for i in range(rows)]
        dates.reverse()
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.linspace(100, 110, rows),
            "high": np.linspace(101, 111, rows),
            "low": np.linspace(99, 109, rows),
            "close": np.linspace(100.5, 110.5, rows),
            "volume": [1000] * rows
        })
        return df

    def test_calculate_smfi_accumulation(self):
        # Create accumulation pattern: second half of the day move is always positive and large
        rows = 60
        df = self.create_dummy_ohlcv(rows)
        # Midpoint = (H+L)/2 = (101+99)/2 = 100
        # First half move = Midpoint - Open = 100 - 100 = 0
        # Second half move = Close - Midpoint = 102 - 100 = 2
        # SMFI delta = (2 - 0) * VolFactor = 2 * 1 = 2 (Consistently positive)
        
        df["open"] = 100.0
        df["high"] = 101.0
        df["low"] = 99.0
        df["close"] = 100.5 # Default is slight accumulation
        
        # Override last 10 bars to strong accumulation
        for i in range(rows-10, rows):
            df.loc[i, "open"] = 100.0
            df.loc[i, "high"] = 105.0
            df.loc[i, "low"] = 95.0
            df.loc[i, "close"] = 104.0 
            # Midpoint = 100
            # First half = 100-100 = 0
            # Second half = 104-100 = 4
            # Delta = 4
            
        res = self.indicators.calculate_smfi(df)
        self.assertIsInstance(res, SMFIResult)
        self.assertEqual(res.trend, "RISING")
        self.assertTrue(res.current_value > 70)
        self.assertEqual(res.signal, "ACCUMULATION")
        self.assertGreater(res.days_in_accumulation, 0)

    def test_calculate_smfi_distribution(self):
        rows = 60
        df = self.create_dummy_ohlcv(rows)
        
        # Distribution: Smart money sells at end of day
        # Midpoint = 100
        # First half = 100-96 = 4 (Retail/Dumb money bought early)
        # Second half = 96-100 = -4 (Smart money sold late)
        # Delta = -4 - 4 = -8
        for i in range(rows-20, rows):
            df.loc[i, "open"] = 96.0
            df.loc[i, "high"] = 105.0
            df.loc[i, "low"] = 95.0
            df.loc[i, "close"] = 96.0 
            
        res = self.indicators.calculate_smfi(df)
        self.assertEqual(res.trend, "FALLING")
        self.assertTrue(res.current_value < 30)
        self.assertEqual(res.signal, "DISTRIBUTION")

    def test_detect_volume_shocks(self):
        df = self.create_dummy_ohlcv(50)
        # Create a shock at the last bar
        df.loc[49, "volume"] = 5000 # 5x average
        df.loc[49, "close"] = df.loc[49, "open"] * 1.02 # Bullish shock
        
        res = self.indicators.detect_volume_shocks(df)
        self.assertIsInstance(res, VSDResult)
        self.assertTrue(res.current_bar_shock)
        self.assertEqual(res.shocks[-1].shock_type, "BULLISH_SHOCK")
        self.assertEqual(res.shocks[-1].significance, "EXTREME")
        
        # Test Climax
        df.loc[47, "volume"] = 5000
        res_climax = self.indicators.detect_volume_shocks(df)
        self.assertTrue(res_climax.is_climax)

    def test_detect_breakout_traps_genuine(self):
        df = self.create_dummy_ohlcv(20)
        resistance = 112.0
        # Breakout at last bar
        df.loc[19, "open"] = 111.0
        df.loc[19, "close"] = 115.0 # Well above resistance
        df.loc[19, "volume"] = 2000 # 2x average
        
        res = self.indicators.detect_breakout_traps(
            df, 
            support=90, 
            resistance=resistance,
            oi_data=pd.DataFrame({"oi_change": [10.0]*20}, index=df.index),
            rsi_data=pd.Series([50.0]*20, index=df.index)
        )
        self.assertTrue(res.breakout_detected)
        self.assertEqual(res.breakout_direction, "UP")
        self.assertTrue(res.is_genuine)
        self.assertEqual(res.recommendation, "VALID_BREAKOUT")

    def test_detect_breakout_traps_trap(self):
        df = self.create_dummy_ohlcv(20)
        resistance = 112.0
        # Breakout bar
        df.loc[18, "open"] = 111.0
        df.loc[18, "close"] = 113.0 # Barely above
        df.loc[18, "volume"] = 500  # Low volume
        # Next bar reverses
        df.loc[19, "close"] = 111.0 # Back inside
        
        res = self.indicators.detect_breakout_traps(df, support=90, resistance=resistance)
        self.assertTrue(res.breakout_detected)
        self.assertTrue(res.is_trap)
        self.assertEqual(res.trap_risk, "HIGH")

    def test_calculate_oimi(self):
        df = self.create_dummy_ohlcv(20)
        # Price up 5% over last 5 bars
        # OI up 10% over last 5 bars
        oi_history = []
        for i in range(20):
            oi_history.append({
                "timestamp": df.iloc[i]["timestamp"],
                "total_ce_oi": 10000 + i * 500,
                "total_pe_oi": 10000 + i * 500
            })
            
        res = self.indicators.calculate_oimi(df, oi_history)
        self.assertIsInstance(res, OIMIResult)
        self.assertGreater(res.oimi, 20)
        self.assertIn(res.signal, ["BULLISH", "STRONG_BULLISH"])
        self.assertEqual(res.price_oi_alignment, "ALIGNED")

    def test_calculate_lai(self):
        df = self.create_dummy_ohlcv(30)
        # High volume, low movement
        # ATR (~1.0 for dummy data)
        # VolFactor = 3000 / 1000 = 3
        # Expected move = ATR * 3 = 3.0
        # Actual move = 0.1
        # LAI = 3 / 0.1 = 30
        df.loc[29, "volume"] = 5000
        df.loc[29, "open"] = 100.0
        df.loc[29, "close"] = 100.1
        
        res = self.indicators.calculate_lai(df)
        self.assertIsInstance(res, LAIResult)
        self.assertTrue(res.absorption_detected)
        self.assertEqual(res.absorption_type, "BEARISH_ABSORPTION") # Price moved UP slightly (visible) -> someone selling into it (absorption)
        self.assertGreater(res.current_lai, 2.0)

    def test_smart_money_score_bullish(self):
        df = self.create_dummy_ohlcv(60)
        # Make everything bullish
        # 1. SMFI Acc
        for i in range(60-10, 60):
            df.loc[i, "open"] = 100.0
            df.loc[i, "high"] = 105.0
            df.loc[i, "low"] = 95.0
            df.loc[i, "close"] = 104.0 
        
        # 2. Volume Shock
        df.loc[59, "volume"] = 5000
        
        # 3. OI
        # Make OI increase exponentially
        oi_history = [{"timestamp": df.iloc[i]["timestamp"], "total_oi": 10000 * (1.1 ** i)} for i in range(60)]
        
        # 4. Price trend
        for i in range(60-5, 60):
            df.loc[i, "close"] = df.loc[i-1, "close"] * 1.02
            
        # 5. LAI (at current bar 59)
        df.loc[59, "volume"] = 10000
        df.loc[59, "open"] = df.loc[59, "close"] # Small movement
        df.loc[59, "close"] = df.loc[59, "open"] - 0.1 # Bullish absorption
        
        res = self.indicators.calculate_smart_money_score(df, oi_data=oi_history)
        self.assertIsInstance(res, SmartMoneyScore)
        self.assertGreater(res.score, 60)
        self.assertIn(res.grade, ["A", "A+"])
        self.assertEqual(res.smart_money_bias, "STRONGLY_BULLISH")

    def test_smart_money_score_missing_data(self):
        df = self.create_dummy_ohlcv(20) # Short data
        # No OI, no S/R
        res = self.indicators.calculate_smart_money_score(df)
        self.assertIsInstance(res, SmartMoneyScore)
        self.assertEqual(res.oimi_component, 0)
        self.assertEqual(res.btd_component, 0)
        # data_count should be < 3? 
        # SMFI needs 2 bars. VSD needs 20 bars. LAI needs 15 bars.
        # If rows = 20, we have SMFI, VSD (min 20), LAI (min 15).
        # We need rows < 15 to have data_count < 3.
        df_short = self.create_dummy_ohlcv(10)
        res_short = self.indicators.calculate_smart_money_score(df_short)
        self.assertLess(res_short.confidence, 0.5)

if __name__ == "__main__":
    unittest.main()
