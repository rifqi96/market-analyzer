"""
Test script for commodity market data fetching and analysis.
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from market_analyzer.config import (
    MARKET_TYPES, OUTPUT_DIR, get_asset_name, get_currency_symbol
)
from market_analyzer.data_fetcher import get_market_data
from market_analyzer.indicators import calculate_all_indicators
from market_analyzer.visualizer import setup_output_dirs, generate_all_charts
from market_analyzer.analyzer import analyze_all_timeframes

def setup_test_dirs(trading_pair):
    """Set up test directories for a trading pair."""
    test_dir = os.path.join('test_output', 'commodity', trading_pair.lower())
    test_charts_dir = os.path.join(test_dir, 'charts')
    
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(test_charts_dir, exist_ok=True)
    
    return test_dir, test_charts_dir

def test_commodity_data(trading_pair, timeframe, source):
    """Test commodity data fetching, visualization, and basic analysis."""
    print(f"\n{'='*60}")
    print(f"Testing {trading_pair} ({get_asset_name(trading_pair, MARKET_TYPES['COMMODITY'])})")
    print(f"{'='*60}")
    print(f"Timeframe: {timeframe}")
    print(f"Source: {source}")
    print(f"Currency Symbol: {get_currency_symbol(trading_pair, MARKET_TYPES['COMMODITY'])}")
    
    # Setup test directories
    test_dir, test_charts_dir = setup_test_dirs(trading_pair)
    
    try:
        # Step 1: Fetch data
        print(f"Fetching {timeframe} data...")
        df = get_market_data(
            trading_pair=trading_pair,
            timeframe=timeframe,
            source=source,
            market_type=MARKET_TYPES['COMMODITY']
        )
        
        if df is None or df.empty:
            print("ERROR: No data returned!")
            return False
        
        print(f"Successfully fetched {len(df)} rows of data.")
        print("\nData Sample:")
        print(df.tail(3))
        
        # Step 2: Calculate indicators
        print("\nCalculating technical indicators...")
        try:
            df_with_indicators = calculate_all_indicators(df, market_type=MARKET_TYPES['COMMODITY'])
            print(f"Successfully calculated indicators.")
            
            # Show sample of calculated indicators
            print("\nIndicators Sample:")
            indicator_cols = [col for col in df_with_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            if indicator_cols:
                print(df_with_indicators[indicator_cols].tail(3))
        except Exception as e:
            print(f"ERROR calculating indicators: {e}")
            return False
        
        # Step 3: Generate charts
        print("\nGenerating charts...")
        try:
            setup_output_dirs(charts_dir=test_charts_dir)
            generate_all_charts(
                df_with_indicators, 
                timeframe, 
                trading_pair=trading_pair, 
                market_type=MARKET_TYPES['COMMODITY'],
                charts_dir=test_charts_dir
            )
            print(f"Charts saved to: {test_charts_dir}")
        except Exception as e:
            print(f"ERROR generating charts: {e}")
        
        # Step 4: Simple analysis
        print("\nPerforming basic analysis...")
        try:
            dataframes = {timeframe: df_with_indicators}
            analysis = analyze_all_timeframes(
                dataframes, 
                trading_pair=trading_pair,
                market_type=MARKET_TYPES['COMMODITY']
            )
            
            print("\nAnalysis Summary:")
            for tf, results in analysis.items():
                print(f"\n{tf} Timeframe:")
                print(f"  Trend: {results.get('trend', 'Unknown')}")
                print(f"  Support Levels: {', '.join([str(x) for x in results.get('support_levels', [])][:3])}")
                print(f"  Resistance Levels: {', '.join([str(x) for x in results.get('resistance_levels', [])][:3])}")
                if 'price_prediction' in results:
                    print(f"  Price Prediction: {results['price_prediction']}")
        except Exception as e:
            print(f"ERROR performing analysis: {e}")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Main function for testing commodity market support."""
    parser = argparse.ArgumentParser(description="Test commodity market data fetching and analysis")
    parser.add_argument('--pair', help='Single trading pair to test (e.g., XAUUSD, GC=F)')
    parser.add_argument('--timeframe', default='1d', help='Timeframe (default: 1d)')
    parser.add_argument('--source', default='yahoo_finance', help='Data source (default: yahoo_finance)')
    
    args = parser.parse_args()
    
    print(f"Starting commodity market tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test pairs to process
    if args.pair:
        pairs = [args.pair]
    else:
        # Test a variety of commodity symbols
        pairs = [
            'XAUUSD',  # Gold spot price
            'GC=F',    # Gold futures
            'XAGUSD',  # Silver spot price
            'SI=F',    # Silver futures
            'CL=F'     # Crude oil futures
        ]
    
    # Run tests
    results = {}
    for pair in pairs:
        results[pair] = test_commodity_data(pair, args.timeframe, args.source)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for pair, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{pair}: {status}")
    
    # Overall result
    if all(results.values()):
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. See details above.")

if __name__ == "__main__":
    main() 