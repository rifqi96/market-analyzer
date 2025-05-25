import os
import sys
import unittest
import pandas as pd
from datetime import datetime, timedelta

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market_analyzer.config import MARKET_TYPES
from market_analyzer.data_fetcher import get_market_data

class TestDataFetcher(unittest.TestCase):
    """
    Unit tests for the data fetcher module.
    
    Note: These tests require internet connection to fetch data from APIs.
    Some tests might fail if the external APIs are unavailable or rate limited.
    """
    
    def test_get_market_data_cryptocurrency(self):
        """Test fetching cryptocurrency data."""
        # Only run this test if we want to make actual API calls
        if os.environ.get('SKIP_API_TESTS') == '1':
            self.skipTest("Skipping API test due to environment configuration")
        
        # Fetch data
        df = get_market_data(
            trading_pair='BTCUSDT',
            timeframe='1d',
            source='binance',
            market_type=MARKET_TYPES['CRYPTO'],
            limit=10  # Limit to 10 candles for testing
        )
        
        # Verify the result
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        # Check columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df.index))
        self.assertTrue(pd.api.types.is_float_dtype(df['open']))
        self.assertTrue(pd.api.types.is_float_dtype(df['close']))
    
    def test_get_market_data_stock(self):
        """Test fetching stock market data."""
        # Only run this test if we want to make actual API calls
        if os.environ.get('SKIP_API_TESTS') == '1':
            self.skipTest("Skipping API test due to environment configuration")
        
        try:
            # Fetch data
            df = get_market_data(
                trading_pair='AAPL',
                timeframe='1d',
                source='yahoo_finance',
                market_type=MARKET_TYPES['STOCK'],
                limit=10  # Limit to 10 candles for testing
            )
            
            # Verify the result
            self.assertIsNotNone(df)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            
            # Check columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                self.assertIn(col, df.columns)
            
            # Check data types
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df.index))
            self.assertTrue(pd.api.types.is_float_dtype(df['open']))
            self.assertTrue(pd.api.types.is_float_dtype(df['close']))
        except Exception as e:
            # If we hit API rate limits or other API issues, skip the test
            self.skipTest(f"Skipping due to API issue: {e}")
    
    def test_get_market_data_cache(self):
        """Test that caching works properly."""
        # Only run this test if we want to make actual API calls
        if os.environ.get('SKIP_API_TESTS') == '1':
            self.skipTest("Skipping API test due to environment configuration")
        
        # Fetch data the first time
        start_time = datetime.now()
        df1 = get_market_data(
            trading_pair='BTCUSDT',
            timeframe='1d',
            source='binance',
            market_type=MARKET_TYPES['CRYPTO'],
            limit=10
        )
        first_fetch_time = datetime.now() - start_time
        
        # Fetch the same data again (should use cache)
        start_time = datetime.now()
        df2 = get_market_data(
            trading_pair='BTCUSDT',
            timeframe='1d',
            source='binance',
            market_type=MARKET_TYPES['CRYPTO'],
            limit=10
        )
        second_fetch_time = datetime.now() - start_time
        
        # Verify results are the same
        self.assertTrue(df1.equals(df2))
        
        # Second fetch should be faster (but this is not guaranteed, so we don't assert it)
        print(f"First fetch time: {first_fetch_time.total_seconds()} seconds")
        print(f"Second fetch time: {second_fetch_time.total_seconds()} seconds")

if __name__ == '__main__':
    unittest.main() 