"""
Module for analyzing market data and generating forecasts.
"""

import pandas as pd
import numpy as np
from .indicators import calculate_support_resistance
from .config import MARKET_TYPES
from datetime import datetime

def identify_trend(df, window=20, market_type=MARKET_TYPES['CRYPTO']):
    """
    Identify the current trend based on moving averages and price action.
    
    Args:
        df (pandas.DataFrame): DataFrame with price and indicator data
        window (int): Window size for trend identification
        market_type (str): Type of market for parameter calibration
        
    Returns:
        str: Trend direction ('bullish', 'bearish', or 'sideways')
    """
    # Get recent price data
    recent_df = df.iloc[-window:]
    
    # Check if price is above key moving averages
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        price_above_50ma = recent_df['close'].iloc[-1] > recent_df['SMA_50'].iloc[-1]
        price_above_200ma = recent_df['close'].iloc[-1] > recent_df['SMA_200'].iloc[-1]
        golden_cross = recent_df['SMA_50'].iloc[-1] > recent_df['SMA_200'].iloc[-1]
        
        # Calculate slope of moving averages
        sma50_slope = (recent_df['SMA_50'].iloc[-1] - recent_df['SMA_50'].iloc[0]) / window
        sma200_slope = (recent_df['SMA_200'].iloc[-1] - recent_df['SMA_200'].iloc[0]) / window
        
        # Calculate price slope
        price_slope = (recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]) / window
        
        # Determine trend
        if price_above_50ma and price_above_200ma and golden_cross and sma50_slope > 0 and price_slope > 0:
            return 'bullish'
        elif (not price_above_50ma) and (not price_above_200ma) and (not golden_cross) and sma50_slope < 0 and price_slope < 0:
            return 'bearish'
        else:
            # Check if price is ranging
            price_range = recent_df['high'].max() - recent_df['low'].min()
            price_avg = recent_df['close'].mean()
            
            # Different thresholds for different markets
            sideways_threshold = 0.05  # Default for crypto
            if market_type == MARKET_TYPES['FOREX']:
                sideways_threshold = 0.01  # Forex has smaller price movements
            elif market_type == MARKET_TYPES['STOCK']:
                sideways_threshold = 0.03  # Stocks are between crypto and forex
                
            if price_range / price_avg < sideways_threshold and abs(price_slope) < 0.001:
                return 'sideways'
            
            # If not clear, determine based on recent price action and slope
            return 'bullish' if price_slope > 0 else 'bearish'
    
    # If moving averages are not available, use simple price action
    price_slope = (recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]) / window
    return 'bullish' if price_slope > 0 else 'bearish' if price_slope < 0 else 'sideways'

def identify_support_resistance_zones(df, window=20, buffer_pct=0.02, market_type=MARKET_TYPES['CRYPTO']):
    """
    Identify key support and resistance zones.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        window (int): Window size for identifying pivot points
        buffer_pct (float): Percentage buffer for creating zones
        market_type (str): Type of market for parameter calibration
        
    Returns:
        tuple: (support_zones, resistance_zones)
    """
    # Adjust buffer percentage based on market type
    if market_type == MARKET_TYPES['FOREX']:
        buffer_pct = 0.005  # Smaller buffer for forex
    elif market_type == MARKET_TYPES['STOCK']:
        buffer_pct = 0.015  # Medium buffer for stocks
    
    # Get support and resistance levels
    support_levels, resistance_levels = calculate_support_resistance(df, market_type=market_type)
    
    # Create support zones by clustering nearby levels
    support_zones = []
    for level in sorted(support_levels):
        # Check if the level is close to an existing zone
        found_zone = False
        for i, zone in enumerate(support_zones):
            if abs(level - zone['center']) / zone['center'] < buffer_pct:
                # Update existing zone
                support_zones[i]['levels'].append(level)
                support_zones[i]['center'] = sum(support_zones[i]['levels']) / len(support_zones[i]['levels'])
                support_zones[i]['lower'] = support_zones[i]['center'] * (1 - buffer_pct)
                support_zones[i]['upper'] = support_zones[i]['center'] * (1 + buffer_pct)
                found_zone = True
                break
        
        if not found_zone:
            # Create new zone
            support_zones.append({
                'center': level,
                'lower': level * (1 - buffer_pct),
                'upper': level * (1 + buffer_pct),
                'levels': [level]
            })
    
    # Create resistance zones by clustering nearby levels
    resistance_zones = []
    for level in sorted(resistance_levels):
        # Check if the level is close to an existing zone
        found_zone = False
        for i, zone in enumerate(resistance_zones):
            if abs(level - zone['center']) / zone['center'] < buffer_pct:
                # Update existing zone
                resistance_zones[i]['levels'].append(level)
                resistance_zones[i]['center'] = sum(resistance_zones[i]['levels']) / len(resistance_zones[i]['levels'])
                resistance_zones[i]['lower'] = resistance_zones[i]['center'] * (1 - buffer_pct)
                resistance_zones[i]['upper'] = resistance_zones[i]['center'] * (1 + buffer_pct)
                found_zone = True
                break
        
        if not found_zone:
            # Create new zone
            resistance_zones.append({
                'center': level,
                'lower': level * (1 - buffer_pct),
                'upper': level * (1 + buffer_pct),
                'levels': [level]
            })
    
    # Sort zones
    support_zones = sorted(support_zones, key=lambda x: x['center'])
    resistance_zones = sorted(resistance_zones, key=lambda x: x['center'])
    
    return support_zones, resistance_zones

def identify_key_levels(df, current_price, market_type=MARKET_TYPES['CRYPTO']):
    """
    Identify key price levels (support and resistance) based on historical data.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        current_price (float): Current price for reference
        market_type (str): Type of market for parameter calibration
        
    Returns:
        dict: Dictionary with key price levels
    """
    # Identify support and resistance zones
    support_zones, resistance_zones = identify_support_resistance_zones(df, market_type=market_type)
    
    # Get nearest support level below current price
    nearest_support = None
    for zone in reversed(support_zones):
        if zone['center'] < current_price:
            nearest_support = zone['center']
            break
    
    # Get nearest resistance level above current price
    nearest_resistance = None
    for zone in resistance_zones:
        if zone['center'] > current_price:
            nearest_resistance = zone['center']
            break
    
    # Get psychological levels
    psych_levels = []
    
    # Round numbers based on price magnitude and market type
    if market_type == MARKET_TYPES['FOREX']:
        # For forex, use smaller steps
        if current_price < 1:
            step = 0.0025  # For pairs like EURUSD
        else:
            step = 0.25    # For pairs like USDJPY
    else:
        # For stocks and crypto
        if current_price < 10:
            step = 1
        elif current_price < 100:
            step = 10
        elif current_price < 1000:
            step = 100
        elif current_price < 10000:
            step = 1000
        else:
            step = 10000
    
    # Find nearby psychological levels
    for i in range(-5, 6):
        level = round(current_price / step) * step + i * step
        if level > 0:
            psych_levels.append(level)
    
    return {
        'current_price': current_price,
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance,
        'support_zones': support_zones,
        'resistance_zones': resistance_zones,
        'psychological_levels': sorted(psych_levels)
    }

def analyze_momentum(df, window=14, market_type=MARKET_TYPES['CRYPTO']):
    """
    Analyze momentum indicators to determine market strength.
    
    Args:
        df (pandas.DataFrame): DataFrame with price and indicator data
        window (int): Window for recent analysis
        market_type (str): Type of market for parameter calibration
        
    Returns:
        dict: Momentum analysis results
    """
    # Get recent data
    recent_df = df.iloc[-window:]
    
    # Adjust RSI thresholds based on market type
    rsi_oversold = 30
    rsi_overbought = 70
    
    if market_type == MARKET_TYPES['FOREX']:
        # Forex tends to have more narrow RSI ranges
        rsi_oversold = 35
        rsi_overbought = 65
    elif market_type == MARKET_TYPES['STOCK']:
        # Standard thresholds for stocks
        rsi_oversold = 30
        rsi_overbought = 70
    
    # RSI analysis
    rsi_bullish = False
    rsi_bearish = False
    if 'RSI' in recent_df.columns:
        current_rsi = recent_df['RSI'].iloc[-1]
        rsi_bullish = current_rsi > rsi_overbought and current_rsi < 80
        rsi_bearish = current_rsi < rsi_oversold and current_rsi > 20
        
        # Check for RSI divergence
        rsi_higher_high = recent_df['RSI'].iloc[-1] > recent_df['RSI'].iloc[-window//2]
        price_higher_high = recent_df['close'].iloc[-1] > recent_df['close'].iloc[-window//2]
        rsi_divergence = rsi_higher_high != price_higher_high
    else:
        current_rsi = None
        rsi_divergence = None
    
    # MACD analysis
    macd_bullish = False
    macd_bearish = False
    if all(col in recent_df.columns for col in ['MACD', 'MACD_Signal']):
        current_macd = recent_df['MACD'].iloc[-1]
        current_signal = recent_df['MACD_Signal'].iloc[-1]
        macd_bullish = current_macd > current_signal and current_macd > 0
        macd_bearish = current_macd < current_signal and current_macd < 0
        
        # Check for MACD crossover
        prev_macd = recent_df['MACD'].iloc[-2]
        prev_signal = recent_df['MACD_Signal'].iloc[-2]
        macd_bullish_cross = prev_macd <= prev_signal and current_macd > current_signal
        macd_bearish_cross = prev_macd >= prev_signal and current_macd < current_signal
    else:
        current_macd = None
        current_signal = None
        macd_bullish_cross = None
        macd_bearish_cross = None
    
    # Stochastic analysis
    stoch_bullish = False
    stoch_bearish = False
    if all(col in recent_df.columns for col in ['Stoch_K', 'Stoch_D']):
        current_k = recent_df['Stoch_K'].iloc[-1]
        current_d = recent_df['Stoch_D'].iloc[-1]
        stoch_bullish = current_k > current_d and current_k < 80
        stoch_bearish = current_k < current_d and current_k > 20
        
        # Check for Stochastic crossover
        prev_k = recent_df['Stoch_K'].iloc[-2]
        prev_d = recent_df['Stoch_D'].iloc[-2]
        stoch_bullish_cross = prev_k <= prev_d and current_k > current_d
        stoch_bearish_cross = prev_k >= prev_d and current_k < current_d
    else:
        current_k = None
        current_d = None
        stoch_bullish_cross = None
        stoch_bearish_cross = None
    
    # Combine momentum signals
    bullish_signals = sum([1 for x in [rsi_bullish, macd_bullish, stoch_bullish] if x])
    bearish_signals = sum([1 for x in [rsi_bearish, macd_bearish, stoch_bearish] if x])
    
    momentum = 'bullish' if bullish_signals > bearish_signals else 'bearish' if bearish_signals > bullish_signals else 'neutral'
    strength = abs(bullish_signals - bearish_signals) / max(1, bullish_signals + bearish_signals)
    
    return {
        'momentum': momentum,
        'strength': strength,
        'rsi': {
            'value': current_rsi,
            'bullish': rsi_bullish,
            'bearish': rsi_bearish,
            'divergence': rsi_divergence
        },
        'macd': {
            'value': current_macd,
            'signal': current_signal,
            'bullish': macd_bullish,
            'bearish': macd_bearish,
            'bullish_cross': macd_bullish_cross,
            'bearish_cross': macd_bearish_cross
        },
        'stochastic': {
            'k': current_k,
            'd': current_d,
            'bullish': stoch_bullish,
            'bearish': stoch_bearish,
            'bullish_cross': stoch_bullish_cross,
            'bearish_cross': stoch_bearish_cross
        }
    }

def generate_short_term_forecast(df, timeframe, market_type=MARKET_TYPES['CRYPTO']):
    """
    Generate short-term forecast based on technical analysis.
    
    Args:
        df (pandas.DataFrame): DataFrame with price and indicator data
        timeframe (str): Timeframe of the data
        market_type (str): Type of market for parameter calibration
        
    Returns:
        dict: Forecast results
    """
    # Get current price
    current_price = df['close'].iloc[-1]
    
    # Identify trend
    trend = identify_trend(df, market_type=market_type)
    
    # Identify key levels
    key_levels = identify_key_levels(df, current_price, market_type=market_type)
    
    # Analyze momentum
    momentum = analyze_momentum(df, market_type=market_type)
    
    # Generate price targets based on trend and momentum
    targets = {}
    
    # Adjust volatility calculation based on market type
    if market_type == MARKET_TYPES['FOREX']:
        # Forex has lower volatility
        volatility = df['close'].pct_change().std() * np.sqrt(7) * 0.8
    elif market_type == MARKET_TYPES['STOCK']:
        # Stocks have medium volatility
        volatility = df['close'].pct_change().std() * np.sqrt(7) * 1.0
    else:
        # Crypto has higher volatility
        volatility = df['close'].pct_change().std() * np.sqrt(7) * 1.2
    
    if trend == 'bullish' and momentum['momentum'] == 'bullish':
        # Strong bullish case
        targets['bullish'] = current_price * (1 + 2 * volatility)
        targets['base'] = current_price * (1 + volatility)
        targets['bearish'] = current_price * (1 - 0.5 * volatility)
        probability = {'bullish': 0.6, 'base': 0.3, 'bearish': 0.1}
    
    elif trend == 'bullish' and momentum['momentum'] != 'bullish':
        # Moderately bullish case
        targets['bullish'] = current_price * (1 + 1.5 * volatility)
        targets['base'] = current_price * (1 + 0.5 * volatility)
        targets['bearish'] = current_price * (1 - 0.5 * volatility)
        probability = {'bullish': 0.4, 'base': 0.4, 'bearish': 0.2}
    
    elif trend == 'bearish' and momentum['momentum'] == 'bearish':
        # Strong bearish case
        targets['bullish'] = current_price * (1 + 0.5 * volatility)
        targets['base'] = current_price * (1 - volatility)
        targets['bearish'] = current_price * (1 - 2 * volatility)
        probability = {'bullish': 0.1, 'base': 0.3, 'bearish': 0.6}
    
    elif trend == 'bearish' and momentum['momentum'] != 'bearish':
        # Moderately bearish case
        targets['bullish'] = current_price * (1 + 0.5 * volatility)
        targets['base'] = current_price * (1 - 0.5 * volatility)
        targets['bearish'] = current_price * (1 - 1.5 * volatility)
        probability = {'bullish': 0.2, 'base': 0.4, 'bearish': 0.4}
    
    else:  # 'sideways' trend
        # Ranging case
        targets['bullish'] = current_price * (1 + volatility)
        targets['base'] = current_price
        targets['bearish'] = current_price * (1 - volatility)
        probability = {'bullish': 0.3, 'base': 0.4, 'bearish': 0.3}
    
    # Adjust targets based on key levels
    if key_levels['nearest_resistance'] and targets['bullish'] > key_levels['nearest_resistance']:
        # Cap bullish target at nearest resistance plus a small buffer
        targets['bullish'] = min(targets['bullish'], key_levels['nearest_resistance'] * 1.05)
    
    if key_levels['nearest_support'] and targets['bearish'] < key_levels['nearest_support']:
        # Limit bearish target at nearest support minus a small buffer
        targets['bearish'] = max(targets['bearish'], key_levels['nearest_support'] * 0.95)
    
    # Generate forecast
    forecast = {
        'timeframe': timeframe,
        'current_price': current_price,
        'trend': trend,
        'momentum': momentum['momentum'],
        'targets': targets,
        'probability': probability,
        'key_levels': key_levels,
        'analysis': {
            'trend': trend,
            'momentum': momentum,
            'volatility': volatility
        }
    }
    
    return forecast

def generate_monthly_forecast(df, timeframe, market_type=MARKET_TYPES['CRYPTO']):
    """
    Generate monthly forecast based on technical analysis.
    
    Args:
        df (pandas.DataFrame): DataFrame with price and indicator data
        timeframe (str): Timeframe of the data
        market_type (str): Type of market for parameter calibration
        
    Returns:
        dict: Forecast results
    """
    # Get current price
    current_price = df['close'].iloc[-1]
    
    # Identify trend
    trend = identify_trend(df, market_type=market_type)
    
    # Identify key levels
    key_levels = identify_key_levels(df, current_price, market_type=market_type)
    
    # Analyze momentum
    momentum = analyze_momentum(df, market_type=market_type)
    
    # Generate price targets based on trend and momentum
    targets = {}
    
    # Adjust volatility calculation based on market type
    if market_type == MARKET_TYPES['FOREX']:
        # Forex has lower volatility
        volatility = df['close'].pct_change().std() * np.sqrt(30) * 0.7
    elif market_type == MARKET_TYPES['STOCK']:
        # Stocks have medium volatility
        volatility = df['close'].pct_change().std() * np.sqrt(30) * 0.9
    else:
        # Crypto has higher volatility
        volatility = df['close'].pct_change().std() * np.sqrt(30) * 1.1
    
    if trend == 'bullish' and momentum['momentum'] == 'bullish':
        # Strong bullish case
        targets['bullish'] = current_price * (1 + 3 * volatility)
        targets['base'] = current_price * (1 + 2 * volatility)
        targets['bearish'] = current_price * (1 - 0.5 * volatility)
        probability = {'bullish': 0.5, 'base': 0.3, 'bearish': 0.2}
    
    elif trend == 'bullish' and momentum['momentum'] != 'bullish':
        # Moderately bullish case
        targets['bullish'] = current_price * (1 + 2.5 * volatility)
        targets['base'] = current_price * (1 + 1.5 * volatility)
        targets['bearish'] = current_price * (1 - volatility)
        probability = {'bullish': 0.4, 'base': 0.4, 'bearish': 0.2}
    
    elif trend == 'bearish' and momentum['momentum'] == 'bearish':
        # Strong bearish case
        targets['bullish'] = current_price * (1 + volatility)
        targets['base'] = current_price * (1 - 2 * volatility)
        targets['bearish'] = current_price * (1 - 3 * volatility)
        probability = {'bullish': 0.2, 'base': 0.3, 'bearish': 0.5}
    
    elif trend == 'bearish' and momentum['momentum'] != 'bearish':
        # Moderately bearish case
        targets['bullish'] = current_price * (1 + volatility)
        targets['base'] = current_price * (1 - 1.5 * volatility)
        targets['bearish'] = current_price * (1 - 2.5 * volatility)
        probability = {'bullish': 0.2, 'base': 0.4, 'bearish': 0.4}
    
    else:  # 'sideways' trend
        # Ranging case
        targets['bullish'] = current_price * (1 + 1.5 * volatility)
        targets['base'] = current_price
        targets['bearish'] = current_price * (1 - 1.5 * volatility)
        probability = {'bullish': 0.3, 'base': 0.4, 'bearish': 0.3}
    
    # Generate monthly price range for each month
    # Get the last date from the dataframe to determine the analysis date
    last_date = df.index[-1] if isinstance(df.index[-1], pd.Timestamp) else pd.to_datetime(df.index[-1])
    current_month = last_date.month
    current_year = last_date.year
    
    # Generate month names from current month to end of year
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    remaining_months = []
    
    for i in range(current_month, 13):  # From current month to December
        remaining_months.append(month_names[i-1])
    
    # If no remaining months (December), project to next year
    if not remaining_months:
        remaining_months = month_names[:8]  # Jan-Aug of next year
    
    monthly_targets = {}
    prev_month_high = current_price
    prev_month_low = current_price
    
    for i, month in enumerate(remaining_months):
        # Calculate compounding factor
        factor = (i + 1) / len(remaining_months)
        
        # Calculate targets for this month
        if trend == 'bullish':
            month_bullish = current_price * (1 + factor * (targets['bullish'] / current_price - 1))
            month_bearish = current_price * (1 + factor * (targets['bearish'] / current_price - 1))
        else:
            month_bullish = current_price * (1 + factor * (targets['bullish'] / current_price - 1))
            month_bearish = current_price * (1 + factor * (targets['bearish'] / current_price - 1))
        
        # Add random walk component based on volatility
        month_volatility = volatility * np.sqrt((i + 1) / len(remaining_months))
        random_factor = 0.5  # Reduce randomness for smoother progression
        
        month_bullish *= (1 + random_factor * month_volatility * np.random.normal(0, 1))
        month_bearish *= (1 + random_factor * month_volatility * np.random.normal(0, 1))
        
        # Ensure bearish target is lower than bullish target
        month_bullish = max(month_bullish, month_bearish)
        month_bearish = min(month_bullish, month_bearish)
        
        # Add month to targets
        monthly_targets[month] = {
            'high': month_bullish,
            'low': month_bearish,
            'median': (month_bullish + month_bearish) / 2
        }
        
        # Update for next month
        prev_month_high = month_bullish
        prev_month_low = month_bearish
    
    # Generate forecast
    forecast = {
        'timeframe': timeframe,
        'current_price': current_price,
        'trend': trend,
        'momentum': momentum['momentum'],
        'year_end_targets': targets,
        'probability': probability,
        'monthly_targets': monthly_targets,
        'key_levels': key_levels,
        'analysis': {
            'trend': trend,
            'momentum': momentum,
            'volatility': volatility
        }
    }
    
    return forecast

def analyze_all_timeframes(dataframes, trading_pair="BTCUSDT", market_type=MARKET_TYPES['CRYPTO'], analysis_date=None):
    """
    Analyze data across all timeframes to generate comprehensive forecast.
    
    Args:
        dataframes (dict): Dict of DataFrames for different timeframes
        trading_pair (str): Trading pair being analyzed
        market_type (str): Type of market (crypto, stock, forex)
        analysis_date (str): Optional analysis date for historical analysis
        
    Returns:
        dict: Analysis results
    """
    # Ensure we have at least one timeframe to analyze
    if not dataframes:
        print("No data available for analysis")
        return {}
    
    # Initialize results dictionary
    # Use the latest data date as timestamp if analysis_date is provided
    if analysis_date and dataframes:
        # Get the latest date from any dataframe
        first_df = list(dataframes.values())[0]
        if not first_df.empty:
            latest_date = first_df.index[-1]
            if isinstance(latest_date, pd.Timestamp):
                timestamp = latest_date.isoformat()
            else:
                timestamp = pd.to_datetime(latest_date).isoformat()
        else:
            timestamp = datetime.now().isoformat()
    else:
        timestamp = datetime.now().isoformat()
    
    results = {
        'trading_pair': trading_pair,
        'market_type': market_type,
        'timestamp': timestamp,
        'timeframes': list(dataframes.keys()),
        'analysis_date': analysis_date if analysis_date else None
    }
    
    # Short-term forecast (using daily and/or hourly timeframes)
    if '1d' in dataframes:
        results['short_term_forecast'] = generate_short_term_forecast(
            dataframes['1d'], '1d', market_type=market_type
        )
    elif '4h' in dataframes:
        results['short_term_forecast'] = generate_short_term_forecast(
            dataframes['4h'], '4h', market_type=market_type
        )
    elif '1h' in dataframes:
        results['short_term_forecast'] = generate_short_term_forecast(
            dataframes['1h'], '1h', market_type=market_type
        )
    else:
        # Use the first available timeframe
        timeframe = list(dataframes.keys())[0]
        results['short_term_forecast'] = generate_short_term_forecast(
            dataframes[timeframe], timeframe, market_type=market_type
        )
    
    # Monthly forecast (using weekly and/or daily timeframes)
    if '1w' in dataframes:
        results['monthly_forecast'] = generate_monthly_forecast(
            dataframes['1w'], '1w', market_type=market_type
        )
    elif '1d' in dataframes:
        results['monthly_forecast'] = generate_monthly_forecast(
            dataframes['1d'], '1d', market_type=market_type
        )
    else:
        # Use the first available timeframe
        timeframe = list(dataframes.keys())[0]
        results['monthly_forecast'] = generate_monthly_forecast(
            dataframes[timeframe], timeframe, market_type=market_type
        )
    
    return results
