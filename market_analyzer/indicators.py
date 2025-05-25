"""
Module for calculating technical indicators.
"""

import pandas as pd
import numpy as np
from .config import MARKET_TYPES

# Default indicator parameters for different market types
INDICATOR_PARAMS = {
    MARKET_TYPES['CRYPTO']: {
        'ma_periods': [20, 50, 100, 200],
        'ema_periods': [8, 21, 55],
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'rsi': {'window': 14},
        'bb': {'window': 20, 'num_std': 2},
        'atr': {'window': 14},
        'stoch': {'k_window': 14, 'd_window': 3},
        'sr_window': 20,
    },
    MARKET_TYPES['STOCK']: {
        'ma_periods': [10, 20, 50, 200],
        'ema_periods': [9, 21, 50],
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'rsi': {'window': 14},
        'bb': {'window': 20, 'num_std': 2},
        'atr': {'window': 14},
        'stoch': {'k_window': 14, 'd_window': 3},
        'sr_window': 15,
    },
    MARKET_TYPES['FOREX']: {
        'ma_periods': [10, 20, 50, 100, 200],  # Added 200 to ensure SMA_200 is calculated
        'ema_periods': [5, 10, 21],
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'rsi': {'window': 14},
        'bb': {'window': 20, 'num_std': 2.5},  # Higher std for forex volatility
        'atr': {'window': 14},
        'stoch': {'k_window': 14, 'd_window': 3},
        'sr_window': 10,
    }
}

def calculate_moving_averages(df, market_type=MARKET_TYPES['CRYPTO']):
    """
    Calculate various moving averages with market-specific parameters.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        market_type (str): Type of market for parameter calibration
        
    Returns:
        pandas.DataFrame: DataFrame with moving averages added
    """
    df = df.copy()
    
    # Get market-specific parameters
    params = INDICATOR_PARAMS.get(market_type, INDICATOR_PARAMS[MARKET_TYPES['CRYPTO']])
    ma_periods = params['ma_periods']
    ema_periods = params['ema_periods']
    
    # Simple Moving Averages
    for period in ma_periods:
        df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
    
    # Exponential Moving Averages
    for period in ema_periods:
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    return df

def calculate_macd(df, market_type=MARKET_TYPES['CRYPTO']):
    """
    Calculate MACD indicator with market-specific parameters.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        market_type (str): Type of market for parameter calibration
        
    Returns:
        pandas.DataFrame: DataFrame with MACD indicators added
    """
    df = df.copy()
    
    # Get market-specific parameters
    params = INDICATOR_PARAMS.get(market_type, INDICATOR_PARAMS[MARKET_TYPES['CRYPTO']])
    fast = params['macd']['fast']
    slow = params['macd']['slow']
    signal = params['macd']['signal']
    
    # Calculate MACD with market-specific parameters
    # Using min_periods parameter to ensure NaN values at the beginning
    df[f'EMA_{fast}'] = df['close'].ewm(span=fast, adjust=False, min_periods=fast).mean()
    df[f'EMA_{slow}'] = df['close'].ewm(span=slow, adjust=False, min_periods=slow).mean()
    df['MACD'] = df[f'EMA_{fast}'] - df[f'EMA_{slow}']
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False, min_periods=signal).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    return df

def calculate_rsi(df, market_type=MARKET_TYPES['CRYPTO']):
    """
    Calculate Relative Strength Index with market-specific parameters.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        market_type (str): Type of market for parameter calibration
        
    Returns:
        pandas.DataFrame: DataFrame with RSI added
    """
    df = df.copy()
    
    # Get market-specific parameters
    params = INDICATOR_PARAMS.get(market_type, INDICATOR_PARAMS[MARKET_TYPES['CRYPTO']])
    window = params['rsi']['window']
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # For forex or highly volatile markets, use SMA for first average then EMA
    if market_type in [MARKET_TYPES['FOREX'], MARKET_TYPES['CRYPTO']]:
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # After initial SMA, use more responsive formula for subsequent values
        # This is Wilder's smoothing method
        avg_gain = avg_gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
        avg_loss = avg_loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    else:
        # Standard RSI calculation
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def calculate_bollinger_bands(df, market_type=MARKET_TYPES['CRYPTO']):
    """
    Calculate Bollinger Bands with market-specific parameters.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        market_type (str): Type of market for parameter calibration
        
    Returns:
        pandas.DataFrame: DataFrame with Bollinger Bands added
    """
    df = df.copy()
    
    # Get market-specific parameters
    params = INDICATOR_PARAMS.get(market_type, INDICATOR_PARAMS[MARKET_TYPES['CRYPTO']])
    window = params['bb']['window']
    num_std = params['bb']['num_std']
    
    df['BB_Middle'] = df['close'].rolling(window=window).mean()
    df['BB_Std'] = df['close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * num_std)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * num_std)
    
    # Add Bandwidth for volatility measurement (more useful in forex)
    df['BB_Bandwidth'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    return df

def calculate_atr(df, market_type=MARKET_TYPES['CRYPTO']):
    """
    Calculate Average True Range with market-specific parameters.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        market_type (str): Type of market for parameter calibration
        
    Returns:
        pandas.DataFrame: DataFrame with ATR added
    """
    df = df.copy()
    
    # Get market-specific parameters
    params = INDICATOR_PARAMS.get(market_type, INDICATOR_PARAMS[MARKET_TYPES['CRYPTO']])
    window = params['atr']['window']
    
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # For forex, use Wilder's smoothing for ATR
    if market_type == MARKET_TYPES['FOREX']:
        df['ATR'] = true_range.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    else:
        df['ATR'] = true_range.rolling(window=window).mean()
    
    # Add ATR percentage (ATR relative to price)
    df['ATR_Pct'] = df['ATR'] / df['close'] * 100
    
    return df

def calculate_obv(df, market_type=MARKET_TYPES['CRYPTO']):
    """
    Calculate On-Balance Volume with market-specific adjustments.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        market_type (str): Type of market for parameter calibration
        
    Returns:
        pandas.DataFrame: DataFrame with OBV added
    """
    df = df.copy()
    
    # Standard OBV calculation
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # For stocks, add OBV with volume spike filtering (reduce impact of unusual volume)
    if market_type == MARKET_TYPES['STOCK']:
        # Calculate mean and standard deviation of volume
        vol_mean = df['volume'].rolling(window=20).mean()
        vol_std = df['volume'].rolling(window=20).std()
        
        # Cap volume at mean + 2 std to reduce outlier impact
        capped_volume = df['volume'].copy()
        capped_volume = capped_volume.where(capped_volume <= vol_mean + 2*vol_std, vol_mean + 2*vol_std)
        
        # Calculate filtered OBV
        df['OBV_Filtered'] = (np.sign(df['close'].diff()) * capped_volume).fillna(0).cumsum()
    
    return df

def calculate_stochastic(df, market_type=MARKET_TYPES['CRYPTO']):
    """
    Calculate Stochastic Oscillator with market-specific parameters.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        market_type (str): Type of market for parameter calibration
        
    Returns:
        pandas.DataFrame: DataFrame with Stochastic Oscillator added
    """
    df = df.copy()
    
    # Get market-specific parameters
    params = INDICATOR_PARAMS.get(market_type, INDICATOR_PARAMS[MARKET_TYPES['CRYPTO']])
    k_window = params['stoch']['k_window']
    d_window = params['stoch']['d_window']
    
    lowest_low = df['low'].rolling(window=k_window).min()
    highest_high = df['high'].rolling(window=k_window).max()
    
    df['Stoch_K'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=d_window).mean()
    
    return df

def calculate_support_resistance(df, market_type=MARKET_TYPES['CRYPTO']):
    """
    Identify potential support and resistance levels with market-specific parameters.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        market_type (str): Type of market for parameter calibration
        
    Returns:
        tuple: (support_levels, resistance_levels)
    """
    df = df.copy()
    
    # Get market-specific parameters
    params = INDICATOR_PARAMS.get(market_type, INDICATOR_PARAMS[MARKET_TYPES['CRYPTO']])
    window = params['sr_window']
    
    # Identify pivot highs
    df['Pivot_High'] = df['high'].rolling(window=window, center=True).apply(
        lambda x: 1 if x.max() == x.iloc[window//2] else 0, raw=False)
    
    # Identify pivot lows
    df['Pivot_Low'] = df['low'].rolling(window=window, center=True).apply(
        lambda x: 1 if x.min() == x.iloc[window//2] else 0, raw=False)
    
    # Get resistance levels from pivot highs
    resistance_levels = df[df['Pivot_High'] == 1]['high'].tolist()
    
    # Get support levels from pivot lows
    support_levels = df[df['Pivot_Low'] == 1]['low'].tolist()
    
    # Drop temporary columns
    df.drop(['Pivot_High', 'Pivot_Low'], axis=1, inplace=True)
    
    return support_levels, resistance_levels

def calculate_all_indicators(df, market_type=MARKET_TYPES['CRYPTO']):
    """
    Calculate all technical indicators with market-specific parameters.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        market_type (str): Type of market for parameter calibration
        
    Returns:
        pandas.DataFrame: DataFrame with all indicators added
    """
    df = calculate_moving_averages(df, market_type)
    df = calculate_macd(df, market_type)
    df = calculate_rsi(df, market_type)
    df = calculate_bollinger_bands(df, market_type)
    df = calculate_atr(df, market_type)
    df = calculate_obv(df, market_type)
    df = calculate_stochastic(df, market_type)
    
    return df
