import os
import sys
import unittest
import pandas as pd
import numpy as np
import matplotlib
import tempfile
import shutil
from datetime import datetime, timedelta

# Force matplotlib to use a non-interactive backend for testing
matplotlib.use('Agg')

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market_analyzer.config import MARKET_TYPES
from market_analyzer.indicators import calculate_all_indicators
from market_analyzer.visualizer import (
    setup_output_dirs, plot_price_with_mas, plot_bollinger_bands,
    plot_rsi, plot_macd, create_comprehensive_chart, generate_all_charts
)

class TestVisualizer(unittest.TestCase):
    """Unit tests for the visualizer module."""
    
    def setUp(self):
        """Set up test data and temporary directory."""
        # Create a sample DataFrame with OHLCV data
        dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
        
        # Create realistic price data with a trend and some volatility
        base_price = 50000.0
        trend = np.linspace(0, 5000, 100)  # Upward trend
        noise = np.random.normal(0, 500, 100)  # Random noise
        
        prices = base_price + trend + noise
        
        df = pd.DataFrame({
            'open': prices - 100,
            'high': prices + 200,
            'low': prices - 200,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Calculate technical indicators
        self.df = calculate_all_indicators(df, market_type=MARKET_TYPES['CRYPTO'])
        
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_setup_output_dirs(self):
        """Test creating output directories."""
        charts_dir = os.path.join(self.test_dir, 'charts')
        
        # Setup output directories
        setup_output_dirs(charts_dir=charts_dir)
        
        # Check if directory was created
        self.assertTrue(os.path.exists(charts_dir))
        self.assertTrue(os.path.isdir(charts_dir))
    
    def test_plot_price_with_mas(self):
        """Test plotting price with moving averages."""
        filename = 'test_price_mas.png'
        charts_dir = self.test_dir
        
        # Generate the plot
        plot_price_with_mas(self.df, timeframe='1d', filename=filename, charts_dir=charts_dir)
        
        # Check if the file was created
        output_path = os.path.join(charts_dir, filename)
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)
    
    def test_plot_bollinger_bands(self):
        """Test plotting Bollinger Bands."""
        filename = 'test_bb.png'
        charts_dir = self.test_dir
        
        # Generate the plot
        plot_bollinger_bands(self.df, timeframe='1d', filename=filename, charts_dir=charts_dir)
        
        # Check if the file was created
        output_path = os.path.join(charts_dir, filename)
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)
    
    def test_plot_rsi(self):
        """Test plotting RSI."""
        filename = 'test_rsi.png'
        charts_dir = self.test_dir
        
        # Generate the plot
        plot_rsi(self.df, timeframe='1d', filename=filename, charts_dir=charts_dir)
        
        # Check if the file was created
        output_path = os.path.join(charts_dir, filename)
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)
    
    def test_plot_macd(self):
        """Test plotting MACD."""
        filename = 'test_macd.png'
        charts_dir = self.test_dir
        
        # Generate the plot
        plot_macd(self.df, timeframe='1d', filename=filename, charts_dir=charts_dir)
        
        # Check if the file was created
        output_path = os.path.join(charts_dir, filename)
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)
    
    def test_create_comprehensive_chart(self):
        """Test creating a comprehensive chart."""
        filename = 'test_comprehensive.png'
        charts_dir = self.test_dir
        
        # Generate the plot
        create_comprehensive_chart(self.df, timeframe='1d', filename=filename, charts_dir=charts_dir)
        
        # Check if the file was created
        output_path = os.path.join(charts_dir, filename)
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)
    
    def test_generate_all_charts(self):
        """Test generating all charts."""
        charts_dir = self.test_dir
        
        # Generate all charts
        generate_all_charts(self.df, timeframe='1d', trading_pair='BTCUSDT', charts_dir=charts_dir)
        
        # Check if all chart files were created
        expected_files = [
            'price_mas_1d.png',
            'bollinger_bands_1d.png',
            'rsi_1d.png',
            'macd_1d.png',
            'comprehensive_1d.png'
        ]
        
        for filename in expected_files:
            output_path = os.path.join(charts_dir, filename)
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(os.path.getsize(output_path) > 0)

if __name__ == '__main__':
    unittest.main() 