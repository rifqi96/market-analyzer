"""
Module for generating visualizations of market price data and technical indicators.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from .config import (
    CHARTS_DIR, MARKET_TYPES, DEFAULT_MARKET_TYPE,
    get_currency_symbol, get_asset_name
)

# Set up matplotlib
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams['figure.dpi'] = 300

def setup_output_dirs(charts_dir=CHARTS_DIR):
    """
    Ensure output directories exist.
    
    Args:
        charts_dir (str): Directory to store chart images
    """
    os.makedirs(charts_dir, exist_ok=True)

def plot_price_with_mas(df, timeframe, trading_pair='BTCUSDT', market_type=DEFAULT_MARKET_TYPE, filename=None, charts_dir=CHARTS_DIR):
    """
    Plot price with moving averages.
    
    Args:
        df (pandas.DataFrame): DataFrame with price and indicator data
        timeframe (str): Timeframe of the data
        trading_pair (str): Trading pair symbol (e.g. BTCUSDT, AAPL, EURUSD)
        market_type (str): Type of market (crypto, stock, forex)
        filename (str, optional): Output filename
        charts_dir (str): Directory to save charts
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Get asset name and currency symbol
    asset_name = get_asset_name(trading_pair, market_type)
    currency_symbol = get_currency_symbol(trading_pair, market_type)
    
    # Plot price
    ax.plot(df.index, df['close'], label=asset_name, linewidth=1.5)
    
    # Plot moving averages
    if 'SMA_50' in df.columns:
        ax.plot(df.index, df['SMA_50'], label='SMA 50', linewidth=1)
    if 'SMA_200' in df.columns:
        ax.plot(df.index, df['SMA_200'], label='SMA 200', linewidth=1)
    if 'EMA_8' in df.columns:
        ax.plot(df.index, df['EMA_8'], label='EMA 8', linewidth=1, linestyle='--')
    if 'EMA_21' in df.columns:
        ax.plot(df.index, df['EMA_21'], label='EMA 21', linewidth=1, linestyle='--')
    
    # Customize plot
    ax.set_title(f'{asset_name} Price with Moving Averages ({timeframe})', fontsize=16)
    ax.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot if filename provided
    if filename:
        plt.savefig(os.path.join(charts_dir, filename))
        print(f"Saved chart to {os.path.join(charts_dir, filename)}")
    
    # Show plot
    plt.close()

def plot_bollinger_bands(df, timeframe, trading_pair='BTCUSDT', market_type=DEFAULT_MARKET_TYPE, filename=None, charts_dir=CHARTS_DIR):
    """
    Plot price with Bollinger Bands.
    
    Args:
        df (pandas.DataFrame): DataFrame with price and indicator data
        timeframe (str): Timeframe of the data
        trading_pair (str): Trading pair symbol (e.g. BTCUSDT, AAPL, EURUSD)
        market_type (str): Type of market (crypto, stock, forex)
        filename (str, optional): Output filename
        charts_dir (str): Directory to save charts
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Get asset name and currency symbol
    asset_name = get_asset_name(trading_pair, market_type)
    currency_symbol = get_currency_symbol(trading_pair, market_type)
    
    # Plot price
    ax.plot(df.index, df['close'], label=asset_name, linewidth=1.5)
    
    # Plot Bollinger Bands
    if all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
        ax.plot(df.index, df['BB_Upper'], label='Upper Band', linestyle='--', alpha=0.7, color='gray')
        ax.plot(df.index, df['BB_Middle'], label='Middle Band', linestyle='-', alpha=0.7, color='blue')
        ax.plot(df.index, df['BB_Lower'], label='Lower Band', linestyle='--', alpha=0.7, color='gray')
        ax.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.1, color='gray')
    
    # Customize plot
    ax.set_title(f'{asset_name} Price with Bollinger Bands ({timeframe})', fontsize=16)
    ax.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot if filename provided
    if filename:
        plt.savefig(os.path.join(charts_dir, filename))
        print(f"Saved chart to {os.path.join(charts_dir, filename)}")
    
    # Show plot
    plt.close()

def plot_rsi(df, timeframe, trading_pair='BTCUSDT', market_type=DEFAULT_MARKET_TYPE, filename=None, charts_dir=CHARTS_DIR):
    """
    Plot RSI indicator.
    
    Args:
        df (pandas.DataFrame): DataFrame with price and indicator data
        timeframe (str): Timeframe of the data
        trading_pair (str): Trading pair symbol (e.g. BTCUSDT, AAPL, EURUSD)
        market_type (str): Type of market (crypto, stock, forex)
        filename (str, optional): Output filename
        charts_dir (str): Directory to save charts
    """
    if 'RSI' not in df.columns:
        print("RSI data not available")
        return
    
    # Get asset name
    asset_name = get_asset_name(trading_pair, market_type)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot RSI
    ax.plot(df.index, df['RSI'], label='RSI', linewidth=1.5, color='purple')
    
    # Add overbought/oversold levels
    ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax.axhline(y=50, color='k', linestyle='--', alpha=0.2)
    
    # Set y-axis range
    ax.set_ylim(0, 100)
    
    # Customize plot
    ax.set_title(f'{asset_name} RSI ({timeframe})', fontsize=16)
    ax.set_ylabel('RSI', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot if filename provided
    if filename:
        plt.savefig(os.path.join(charts_dir, filename))
        print(f"Saved chart to {os.path.join(charts_dir, filename)}")
    
    # Show plot
    plt.close()

def plot_macd(df, timeframe, trading_pair='BTCUSDT', market_type=DEFAULT_MARKET_TYPE, filename=None, charts_dir=CHARTS_DIR):
    """
    Plot MACD indicator.
    
    Args:
        df (pandas.DataFrame): DataFrame with price and indicator data
        timeframe (str): Timeframe of the data
        trading_pair (str): Trading pair symbol (e.g. BTCUSDT, AAPL, EURUSD)
        market_type (str): Type of market (crypto, stock, forex)
        filename (str, optional): Output filename
        charts_dir (str): Directory to save charts
    """
    if not all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
        print("MACD data not available")
        return
    
    # Get asset name
    asset_name = get_asset_name(trading_pair, market_type)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot MACD and Signal line
    ax.plot(df.index, df['MACD'], label='MACD', linewidth=1.5, color='blue')
    ax.plot(df.index, df['MACD_Signal'], label='Signal', linewidth=1, color='red')
    
    # Plot histogram as bar chart
    bar_width = 0.8
    if len(df.index) > 1:
        bar_width = (df.index[1] - df.index[0]).total_seconds() * 0.8 / 86400  # 80% of the interval
    ax.bar(df.index, df['MACD_Histogram'], width=bar_width, label='Histogram', 
           color=df['MACD_Histogram'].apply(lambda x: 'g' if x >= 0 else 'r'),
           alpha=0.5)
    
    # Add zero line
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Customize plot
    ax.set_title(f'{asset_name} MACD ({timeframe})', fontsize=16)
    ax.set_ylabel('MACD', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot if filename provided
    if filename:
        plt.savefig(os.path.join(charts_dir, filename))
        print(f"Saved chart to {os.path.join(charts_dir, filename)}")
    
    # Show plot
    plt.close()

def create_comprehensive_chart(df, timeframe, trading_pair='BTCUSDT', market_type=DEFAULT_MARKET_TYPE, filename=None, charts_dir=CHARTS_DIR):
    """
    Create a comprehensive chart with price and multiple indicators.
    
    Args:
        df (pandas.DataFrame): DataFrame with price and indicator data
        timeframe (str): Timeframe of the data
        trading_pair (str): Trading pair symbol (e.g. BTCUSDT, AAPL, EURUSD)
        market_type (str): Type of market (crypto, stock, forex)
        filename (str, optional): Output filename
        charts_dir (str): Directory to save charts
    """
    # Get asset name and currency symbol
    asset_name = get_asset_name(trading_pair, market_type)
    currency_symbol = get_currency_symbol(trading_pair, market_type)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16), 
                                        gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot 1: Price with MAs and Bollinger Bands
    ax1.plot(df.index, df['close'], label=asset_name, linewidth=1.5)
    
    # Plot moving averages
    if 'SMA_50' in df.columns:
        ax1.plot(df.index, df['SMA_50'], label='SMA 50', linewidth=1)
    if 'SMA_200' in df.columns:
        ax1.plot(df.index, df['SMA_200'], label='SMA 200', linewidth=1)
    if 'EMA_8' in df.columns:
        ax1.plot(df.index, df['EMA_8'], label='EMA 8', linewidth=1, linestyle='--')
    if 'EMA_21' in df.columns:
        ax1.plot(df.index, df['EMA_21'], label='EMA 21', linewidth=1, linestyle='--')
    
    # Plot Bollinger Bands
    if all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
        ax1.plot(df.index, df['BB_Upper'], label='BB Upper', linestyle='--', alpha=0.3, color='gray')
        ax1.plot(df.index, df['BB_Lower'], label='BB Lower', linestyle='--', alpha=0.3, color='gray')
        ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.1, color='gray')
    
    # Add volume bars
    if 'volume' in df.columns:
        # Scale volume to fit at the bottom of the price chart
        volume_scaled = df['volume'] / df['volume'].max() * df['close'].min() * 0.5
        ax1.bar(df.index, volume_scaled, width=0.8, alpha=0.3, color='blue', label='Volume')
    
    # Customize ax1
    ax1.set_title(f'{asset_name} {timeframe} Technical Analysis', fontsize=16)
    ax1.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.get_xticklabels(), rotation=0, visible=False)
    
    # Plot 2: MACD
    if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
        ax2.plot(df.index, df['MACD'], label='MACD', linewidth=1.5, color='blue')
        ax2.plot(df.index, df['MACD_Signal'], label='Signal', linewidth=1, color='red')
        
        # Plot histogram
        bar_width = 0.8
        if len(df.index) > 1:
            bar_width = (df.index[1] - df.index[0]).total_seconds() * 0.8 / 86400
        ax2.bar(df.index, df['MACD_Histogram'], width=bar_width, 
               color=df['MACD_Histogram'].apply(lambda x: 'g' if x >= 0 else 'r'),
               alpha=0.5)
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax2.set_ylabel('MACD', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.get_xticklabels(), rotation=0, visible=False)
    
    # Plot 3: RSI
    if 'RSI' in df.columns:
        ax3.plot(df.index, df['RSI'], label='RSI', linewidth=1.5, color='purple')
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax3.axhline(y=50, color='k', linestyle='--', alpha=0.2)
        ax3.set_ylim(0, 100)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('RSI', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # Add overall title
    fig.suptitle(f'{asset_name} Technical Analysis ({timeframe})', fontsize=20, y=0.92)
    
    # Tight layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save plot if filename provided
    if filename:
        plt.savefig(os.path.join(charts_dir, filename))
        print(f"Saved chart to {os.path.join(charts_dir, filename)}")
    
    # Show plot
    plt.close()

def generate_all_charts(df, timeframe, trading_pair='BTCUSDT', market_type=DEFAULT_MARKET_TYPE, charts_dir=CHARTS_DIR):
    """
    Generate all charts for the given timeframe.
    
    Args:
        df (pandas.DataFrame): DataFrame with price and indicator data
        timeframe (str): Timeframe of the data
        trading_pair (str): Trading pair symbol (e.g. BTCUSDT, AAPL)
        market_type (str): Type of market (crypto, stock, forex)
        charts_dir (str): Directory to save charts
    """
    # Ensure the charts directory exists
    setup_output_dirs(charts_dir=charts_dir)
    
    # Generate individual charts
    plot_price_with_mas(df, timeframe, trading_pair, market_type, f'{trading_pair.lower()}_price_mas_{timeframe}.png', charts_dir=charts_dir)
    plot_bollinger_bands(df, timeframe, trading_pair, market_type, f'{trading_pair.lower()}_bollinger_bands_{timeframe}.png', charts_dir=charts_dir)
    plot_rsi(df, timeframe, trading_pair, market_type, f'{trading_pair.lower()}_rsi_{timeframe}.png', charts_dir=charts_dir)
    plot_macd(df, timeframe, trading_pair, market_type, f'{trading_pair.lower()}_macd_{timeframe}.png', charts_dir=charts_dir)
    
    # Generate comprehensive chart
    create_comprehensive_chart(df, timeframe, trading_pair, market_type, f'{trading_pair.lower()}_comprehensive_{timeframe}.png', charts_dir=charts_dir)
    
    print(f"Generated all charts for {trading_pair} {timeframe} timeframe")
