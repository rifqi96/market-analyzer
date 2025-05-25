import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market_analyzer.config import MARKET_TYPES
from market_analyzer.indicators import (
    calculate_moving_averages, calculate_bollinger_bands,
    calculate_rsi, calculate_macd, calculate_all_indicators
)

class TestIndicators(unittest.TestCase):
    """Unit tests for the technical indicators module."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame with OHLCV data
        dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
        
        # Create realistic price data with a trend and some volatility
        base_price = 50000.0
        trend = np.linspace(0, 5000, 100)  # Upward trend
        noise = np.random.normal(0, 500, 100)  # Random noise
        
        prices = base_price + trend + noise
        
        self.df = pd.DataFrame({
            'open': prices - 100,
            'high': prices + 200,
            'low': prices - 200,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_calculate_moving_averages(self):
        """Test moving averages calculation."""
        df = self.df.copy()
        
        # Calculate moving averages
        df = calculate_moving_averages(df)
        
        # Check if the SMA columns are created
        self.assertIn('SMA_50', df.columns)
        self.assertIn('SMA_200', df.columns)
        
        # Check if the EMA columns are created - use dynamic check based on available columns
        ema_columns = [col for col in df.columns if col.startswith('EMA_')]
        self.assertTrue(len(ema_columns) > 0, "No EMA columns found")
        
        # Check if the SMA values are reasonable
        # First period-1 values should be NaN
        self.assertTrue(df['SMA_50'].iloc[:49].isna().all())
        
        # Test at least some NaN values are present in the beginning of SMA_200
        self.assertTrue(df['SMA_200'].iloc[:50].isna().any())
        
        # Remaining values should not be NaN for SMA_50
        self.assertFalse(df['SMA_50'].iloc[49:].isna().any())
        
        # SMA should be close to the price for a stable price series
        self.assertTrue(np.isclose(
            df['SMA_50'].iloc[-1],
            df['close'].iloc[-50:].mean()
        ))
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        df = self.df.copy()
        
        # Calculate Bollinger Bands
        df = calculate_bollinger_bands(df)
        
        # Check if the BB columns are created
        self.assertIn('BB_Upper', df.columns)
        self.assertIn('BB_Middle', df.columns)
        self.assertIn('BB_Lower', df.columns)
        
        # Check if the BB values are reasonable
        # First period-1 values should be NaN
        self.assertTrue(df['BB_Upper'].iloc[:19].isna().all())
        self.assertTrue(df['BB_Lower'].iloc[:19].isna().all())
        
        # Middle band should be the SMA
        window = 20  # Default window for BB
        self.assertTrue(np.isclose(
            df['BB_Middle'].iloc[-1],
            df['close'].iloc[-window:].mean()
        ))
        
        # Upper band should be above middle band
        self.assertTrue((df['BB_Upper'].iloc[19:] > df['BB_Middle'].iloc[19:]).all())
        
        # Lower band should be below middle band
        self.assertTrue((df['BB_Lower'].iloc[19:] < df['BB_Middle'].iloc[19:]).all())
    
    def test_calculate_rsi(self):
        """Test RSI calculation."""
        df = self.df.copy()
        
        # Calculate RSI
        df = calculate_rsi(df)
        
        # Check if the RSI column is created
        self.assertIn('RSI', df.columns)
        
        # Skip first few values which might be NaN
        rsi_values = df['RSI'].dropna()
        
        # RSI should be between 0 and 100
        self.assertTrue((rsi_values >= 0).all(), "RSI values must be >= 0")
        self.assertTrue((rsi_values <= 100).all(), "RSI values must be <= 100")
    
    def test_calculate_macd(self):
        """Test MACD calculation."""
        df = self.df.copy()
        
        # Calculate MACD
        df = calculate_macd(df)
        
        # Check if the MACD columns are created
        self.assertIn('MACD', df.columns)
        self.assertIn('MACD_Signal', df.columns)
        self.assertIn('MACD_Histogram', df.columns)
        
        # MACD should have some NaN values at the beginning due to EMA calculation
        # but we don't check exact indices as implementation may vary
        self.assertTrue(df['MACD'].iloc[:5].isna().any(), "MACD should have some NaN values at beginning")
        
        # Only non-NaN values for histogram calculation
        valid_indices = df['MACD_Histogram'].dropna().index
        if len(valid_indices) > 0:
            # MACD Histogram should be MACD - Signal
            self.assertTrue(np.allclose(
                df.loc[valid_indices, 'MACD_Histogram'],
                df.loc[valid_indices, 'MACD'] - df.loc[valid_indices, 'MACD_Signal'],
                rtol=1e-5, atol=1e-5
            ), "MACD Histogram should equal MACD - Signal")
    
    def test_calculate_all_indicators(self):
        """Test calculating all indicators at once."""
        df = self.df.copy()
        
        # Calculate all indicators for different market types
        for market_type in MARKET_TYPES.values():
            df_with_indicators = calculate_all_indicators(df, market_type=market_type)
            
            # Check for essential indicator columns (common across all market types)
            base_indicators = [
                'SMA_50', 'SMA_200', 
                'BB_Upper', 'BB_Middle', 'BB_Lower',
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram'
            ]
            
            for col in base_indicators:
                self.assertIn(col, df_with_indicators.columns, f"Missing indicator: {col}")
            
            # Also verify we have some EMA columns (exact names might vary by market)
            ema_columns = [col for col in df_with_indicators.columns if col.startswith('EMA_')]
            self.assertTrue(len(ema_columns) > 0, "No EMA columns found")

if __name__ == '__main__':
    unittest.main() 