#!/usr/bin/env python3
"""Test script to verify end_date parameter works correctly"""

from datetime import datetime
from market_analyzer.data_fetcher import get_market_data

def test_end_date():
    """Test that end_date parameter filters data correctly"""
    print("Testing end_date parameter implementation...")
    
    # Test with a specific historical date
    end_date_str = "2024-01-15"
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Fetch data with end_date
    print(f"\nFetching BTCUSDT data up to {end_date_str}...")
    df = get_market_data(
        trading_pair="BTCUSDT",
        timeframe="1d",
        source="binance",
        market_type="cryptocurrency",
        limit=30,
        analysis_date=end_date_str
    )
    
    if df is not None:
        print(f"✓ Successfully fetched {len(df)} rows")
        print(f"✓ Date range: {df.index.min()} to {df.index.max()}")
        
        # Verify no data after end_date
        data_after_end = df[df.index > end_date]
        if len(data_after_end) == 0:
            print(f"✓ Correctly filtered: No data after {end_date_str}")
        else:
            print(f"✗ ERROR: Found {len(data_after_end)} rows after {end_date_str}")
            print(f"  Latest date in data: {df.index.max()}")
    else:
        print("✗ ERROR: Failed to fetch data")
    
    # Test with Yahoo Finance
    print(f"\nFetching AAPL data up to {end_date_str}...")
    df = get_market_data(
        trading_pair="AAPL",
        timeframe="1d",
        source="yahoo_finance",
        market_type="stock",
        limit=30,
        analysis_date=end_date_str
    )
    
    if df is not None:
        print(f"✓ Successfully fetched {len(df)} rows")
        print(f"✓ Date range: {df.index.min()} to {df.index.max()}")
        
        # Verify no data after end_date
        data_after_end = df[df.index > end_date]
        if len(data_after_end) == 0:
            print(f"✓ Correctly filtered: No data after {end_date_str}")
        else:
            print(f"✗ ERROR: Found {len(data_after_end)} rows after {end_date_str}")
    else:
        print("✗ ERROR: Failed to fetch data")

if __name__ == "__main__":
    test_end_date()