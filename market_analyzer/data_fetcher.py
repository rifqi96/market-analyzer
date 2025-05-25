"""
Module for fetching market data from various APIs.
"""

import abc
import requests
import pandas as pd
import os
import pickle
import hashlib
import time
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from .config import (
    BINANCE_BASE_URL, COINGECKO_BASE_URL, YAHOO_FINANCE_BASE_URL, ALPHA_VANTAGE_BASE_URL,
    MARKET_TYPES, DEFAULT_MARKET_TYPE, DEFAULT_PAIRS, TIMEFRAMES, API_KEYS, validate_trading_pair,
    MARKET_APIS, DATA_DIR
)

# Cache settings
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
CACHE_EXPIRY = {
    '1h': timedelta(hours=1),
    '4h': timedelta(hours=4),
    '1d': timedelta(days=1),
    '3d': timedelta(days=3),
    '1w': timedelta(days=7)
}

# API backoff settings
MAX_RETRIES = 3
INITIAL_BACKOFF = 2  # seconds
MAX_BACKOFF = 60  # seconds

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(trading_pair: str, timeframe: str, source: str, market_type: str, analysis_date: str = None) -> str:
    """Generate a unique cache key based on request parameters"""
    key_str = f"{trading_pair}_{timeframe}_{source}_{market_type}"
    if analysis_date:
        key_str += f"_{analysis_date}"
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cache_path(cache_key: str) -> str:
    """Get file path for cache key"""
    return os.path.join(CACHE_DIR, f"{cache_key}.pkl")

def save_to_cache(df: pd.DataFrame, trading_pair: str, timeframe: str, 
                 source: str, market_type: str, analysis_date: str = None) -> None:
    """Save data to cache"""
    cache_key = get_cache_key(trading_pair, timeframe, source, market_type, analysis_date)
    cache_path = get_cache_path(cache_key)
    
    # Create cache entry with metadata
    cache_data = {
        'df': df,
        'trading_pair': trading_pair,
        'timeframe': timeframe,
        'source': source,
        'market_type': market_type,
        'timestamp': datetime.now()
    }
    
    # Save to file
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Data saved to cache: {cache_path}")

def load_from_cache(trading_pair: str, timeframe: str, 
                   source: str, market_type: str, analysis_date: str = None) -> Optional[pd.DataFrame]:
    """Load data from cache if available and not expired"""
    cache_key = get_cache_key(trading_pair, timeframe, source, market_type, analysis_date)
    cache_path = get_cache_path(cache_key)
    
    # Check if cache file exists
    if not os.path.exists(cache_path):
        return None
    
    # Load cache data
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Check if cache is expired
        expiry = CACHE_EXPIRY.get(timeframe, timedelta(days=1))
        if datetime.now() - cache_data['timestamp'] > expiry:
            print(f"Cache expired for {trading_pair} {timeframe}")
            return None
        
        print(f"Using cached data for {trading_pair} {timeframe} from {cache_data['timestamp']}")
        return cache_data['df']
    
    except (IOError, pickle.PickleError) as e:
        print(f"Error loading from cache: {e}")
        return None

def retry_with_backoff(func):
    """Decorator to implement exponential backoff for API requests."""
    def wrapper(*args, **kwargs):
        retries = 0
        backoff = INITIAL_BACKOFF
        
        while retries < MAX_RETRIES:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                # Check if it's a rate limit error (HTTP 429)
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                    if retries == MAX_RETRIES - 1:
                        # Last retry failed, raise the exception
                        raise
                    
                    # Calculate backoff time with jitter
                    jitter = random.uniform(0, 0.1 * backoff)
                    sleep_time = backoff + jitter
                    sleep_time = min(sleep_time, MAX_BACKOFF)
                    
                    print(f"Rate limit hit. Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    
                    # Increase backoff for next retry
                    backoff *= 2
                    retries += 1
                else:
                    # Not a rate limit error, raise immediately
                    raise
        
        # If we get here, all retries failed
        raise Exception(f"Failed after {MAX_RETRIES} retries with exponential backoff")
    
    return wrapper

class DataFetcher(abc.ABC):
    """Abstract base class for all data fetchers"""
    
    def __init__(self, trading_pair: str, market_type: str = DEFAULT_MARKET_TYPE):
        self.trading_pair = trading_pair
        self.market_type = market_type
        self.validate_trading_pair()
        
    def validate_trading_pair(self) -> bool:
        """Validate the trading pair format based on market type"""
        if not validate_trading_pair(self.trading_pair, self.market_type):
            raise ValueError(f"Invalid trading pair format for {self.market_type}: {self.trading_pair}")
        return True
        
    @abc.abstractmethod
    def fetch_data(self, timeframe: str, limit: int, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Fetch data for the specified trading pair and timeframe
        
        Args:
            timeframe (str): Time interval for the data
            limit (int): Number of data points to retrieve
            end_date (Optional[datetime]): End date for historical analysis
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data or None if error
        """
        pass
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean a DataFrame to ensure consistent format
        
        Args:
            df (pandas.DataFrame): DataFrame to process
            
        Returns:
            pandas.DataFrame: Processed DataFrame
        """
        # Ensure the DataFrame has the required columns
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            # If some columns are missing, fill with close price
            for col in missing_columns:
                if col != 'volume':
                    df[col] = df['close'] if 'close' in df.columns else None
                else:
                    df[col] = 0  # Default volume
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            else:
                df.index = pd.to_datetime(df.index)
        
        return df


class BinanceFetcher(DataFetcher):
    """Fetcher for Binance cryptocurrency exchange data"""
    
    def fetch_data(self, timeframe: str, limit: int, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from Binance API.
        
        Args:
            timeframe (str): Time interval (e.g., '1h', '1d')
            limit (int): Number of candles to retrieve
            end_date (Optional[datetime]): End date for historical analysis
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data or None if error
        """
        endpoint = f"{BINANCE_BASE_URL}/klines"
        params = {
            'symbol': self.trading_pair,
            'interval': TIMEFRAMES.get(timeframe, {}).get('binance_code', timeframe),
            'limit': limit
        }
        
        # If end_date is provided, calculate endTime in milliseconds
        if end_date:
            end_timestamp = int(end_date.timestamp() * 1000)
            params['endTime'] = end_timestamp
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'quote_asset_volume', 'number_of_trades',
                             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
            
            # Set index to timestamp
            df.set_index('timestamp', inplace=True)
            
            # Keep only essential columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            print(f"Successfully fetched {len(df)} {timeframe} candles for {self.trading_pair} from Binance")
            return df
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Binance: {e}")
            return None


class CoinGeckoFetcher(DataFetcher):
    """Fetcher for CoinGecko cryptocurrency data"""
    
    def __init__(self, trading_pair: str, market_type: str = DEFAULT_MARKET_TYPE):
        super().__init__(trading_pair, market_type)
        # Extract coin ID from trading pair (assuming format like BTCUSDT -> bitcoin)
        # This is a simplification; in a real implementation, you'd have a mapping
        self.coin_id = self._get_coin_id(trading_pair)
    
    def _get_coin_id(self, trading_pair: str) -> str:
        """Extract coin ID from trading pair for CoinGecko API"""
        # Simple mapping for common pairs (would need to be expanded)
        coin_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'SOL': 'solana',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'DOGE': 'dogecoin',
            'DOT': 'polkadot',
        }
        
        # Extract base currency (first 3-4 letters usually)
        base = trading_pair[:3]
        if base in coin_map:
            return coin_map[base]
        
        # For others, default to bitcoin
        print(f"Warning: Could not map {trading_pair} to CoinGecko ID, using 'bitcoin' as fallback")
        return 'bitcoin'
    
    def fetch_data(self, timeframe: str, limit: int, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Fetch price data from CoinGecko.
        
        Args:
            timeframe (str): Time interval
            limit (int): Number of data points (converted to days)
            end_date (Optional[datetime]): End date for historical analysis
            
        Returns:
            pandas.DataFrame: DataFrame with price data or None if error
        """
        # Convert limit to days based on timeframe
        days = 30 if timeframe == '1h' else 120 if timeframe == '4h' else 365
        
        endpoint = f"{COINGECKO_BASE_URL}/coins/{self.coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily' if timeframe == '1d' else None
        }
        
        # If end_date is provided, use the range endpoint instead
        if end_date:
            endpoint = f"{COINGECKO_BASE_URL}/coins/{self.coin_id}/market_chart/range"
            # Calculate start date based on limit and timeframe
            hours_per_period = {'1h': 1, '4h': 4, '1d': 24, '3d': 72, '1w': 168}.get(timeframe, 24)
            start_date = end_date - timedelta(hours=hours_per_period * limit)
            params = {
                'vs_currency': 'usd',
                'from': int(start_date.timestamp()),
                'to': int(end_date.timestamp())
            }
        
        try:
            response = requests.get(endpoint, params={k: v for k, v in params.items() if v is not None})
            response.raise_for_status()
            
            data = response.json()
            
            # Extract price data
            prices = data['prices']
            volumes = data['total_volumes']
            
            # Create DataFrame
            df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            
            # Convert timestamp
            df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
            df_volumes['timestamp'] = pd.to_datetime(df_volumes['timestamp'], unit='ms')
            
            # Merge dataframes
            df = pd.merge(df_prices, df_volumes, on='timestamp')
            df.set_index('timestamp', inplace=True)
            
            # Rename columns
            df.rename(columns={'price': 'close'}, inplace=True)
            
            # Since CoinGecko doesn't provide OHLC, we'll use close for all price columns
            df['open'] = df['close']
            df['high'] = df['close']
            df['low'] = df['close']
            
            # Filter by end_date if provided (in case API doesn't respect 'to' parameter)
            if end_date:
                df = df[df.index <= end_date]
            
            print(f"Successfully fetched {len(df)} {timeframe} candles for {self.trading_pair} from CoinGecko")
            return df
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from CoinGecko: {e}")
            return None


class YahooFinanceFetcher(DataFetcher):
    """Fetcher for Yahoo Finance stock data"""
    
    @retry_with_backoff
    def _make_yahoo_request(self, endpoint, params):
        """Make a request to Yahoo Finance API with retry logic."""
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    def fetch_data(self, timeframe: str, limit: int, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from Yahoo Finance API.
        
        Args:
            timeframe (str): Time interval
            limit (int): Number of data points
            end_date (Optional[datetime]): End date for historical analysis
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data or None if error
        """
        # Get interval code for Yahoo Finance
        interval = TIMEFRAMES.get(timeframe, {}).get('yahoo_code', '1d')
        
        # Calculate period based on limit and timeframe
        # This is a simplification; real implementation would be more sophisticated
        period = '1y'
        if timeframe in ['1h', '4h']:
            period = '60d'
        elif timeframe == '1w':
            period = '5y'
        
        # If end_date is provided, use period1 and period2 instead of range
        if end_date:
            # Calculate start date based on timeframe and limit
            timeframe_multipliers = {
                '1h': timedelta(hours=1),
                '4h': timedelta(hours=4),
                '1d': timedelta(days=1),
                '1w': timedelta(weeks=1)
            }
            multiplier = timeframe_multipliers.get(timeframe, timedelta(days=1))
            start_date = end_date - (multiplier * limit)
            
            endpoint = f"{YAHOO_FINANCE_BASE_URL}/{self.trading_pair}"
            params = {
                'interval': interval,
                'period1': int(start_date.timestamp()),
                'period2': int(end_date.timestamp())
            }
        else:
            endpoint = f"{YAHOO_FINANCE_BASE_URL}/{self.trading_pair}"
            params = {
                'interval': interval,
                'range': period
            }
        
        try:
            # Use the retry wrapper for the API request
            data = self._make_yahoo_request(endpoint, params)
            
            # Check if the response contains chart data
            if 'chart' not in data:
                raise ValueError(f"Unexpected Yahoo Finance API response format: {list(data.keys())}")
            
            chart_data = data['chart']
            
            # Check for errors in response
            if 'error' in chart_data:
                raise ValueError(f"Yahoo Finance API error: {chart_data['error']}")
            
            if not chart_data.get('result'):
                raise ValueError("No results found in Yahoo Finance API response")
            
            result = chart_data['result'][0]
            
            # Check if we have timestamp and price data
            if 'timestamp' not in result or 'indicators' not in result:
                raise ValueError("Missing required data in Yahoo Finance API response")
            
            # Extract timestamp and indicators
            timestamps = result['timestamp']
            
            if not timestamps:
                raise ValueError("No timestamps found in Yahoo Finance API response")
            
            quote = result['indicators']['quote'][0]
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps, unit='s'),
                'open': quote.get('open', []),
                'high': quote.get('high', []),
                'low': quote.get('low', []),
                'close': quote.get('close', []),
                'volume': quote.get('volume', [])
            })
            
            # Set index and clean up
            df.set_index('timestamp', inplace=True)
            df.dropna(inplace=True)
            
            # Verify we have data after cleanup
            if df.empty:
                raise ValueError("No valid data after processing Yahoo Finance API response")
            
            print(f"Successfully fetched {len(df)} {timeframe} candles for {self.trading_pair} from Yahoo Finance")
            return df
            
        except (requests.exceptions.RequestException, ValueError, KeyError, IndexError) as e:
            print(f"Error fetching data from Yahoo Finance: {e}")
            return None


class AlphaVantageForexFetcher(DataFetcher):
    """Specialized fetcher for forex data from Alpha Vantage"""
    
    def __init__(self, trading_pair: str, market_type: str = DEFAULT_MARKET_TYPE):
        super().__init__(trading_pair, market_type)
        self.api_key = API_KEYS.get('alpha_vantage', '')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required for forex data")
        
        # Extract from_currency and to_currency from trading pair
        # Format expected: EURUSD (first 3 chars = from, last 3 chars = to)
        if len(trading_pair) < 6:
            raise ValueError(f"Invalid forex pair format: {trading_pair}. Expected format: EURUSD")
        
        self.from_currency = trading_pair[:3]
        self.to_currency = trading_pair[3:6]
    
    def fetch_data(self, timeframe: str, limit: int, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Fetch forex data from Alpha Vantage API.
        
        Args:
            timeframe (str): Time interval
            limit (int): Number of data points
            end_date (Optional[datetime]): End date for historical analysis (Note: Alpha Vantage has limited support)
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data or None if error
        """
        # Map timeframe to Alpha Vantage interval
        interval_map = {
            '1h': '60min',
            '4h': '240min',
            '1d': 'daily',
            '1w': 'weekly'
        }
        
        # Determine which endpoint to use based on timeframe
        if timeframe in ['1h', '4h']:
            function = 'FX_INTRADAY'
            interval = interval_map.get(timeframe)
            endpoint = f"{ALPHA_VANTAGE_BASE_URL}"
            params = {
                'function': function,
                'from_symbol': self.from_currency,
                'to_symbol': self.to_currency,
                'interval': interval,
                'outputsize': 'full',
                'apikey': self.api_key
            }
        else:  # daily, weekly
            function = f"FX_{interval_map.get(timeframe, 'daily').upper()}"
            endpoint = f"{ALPHA_VANTAGE_BASE_URL}"
            params = {
                'function': function,
                'from_symbol': self.from_currency,
                'to_symbol': self.to_currency,
                'outputsize': 'full',
                'apikey': self.api_key
            }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response formats
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                raise ValueError(f"Unexpected API response format: {list(data.keys())}")
            
            # Convert to DataFrame
            df_dict = data[time_series_key]
            df = pd.DataFrame.from_dict(df_dict, orient='index')
            
            # Rename columns (format depends on the function)
            if function == 'FX_INTRADAY':
                df.rename(columns={
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close'
                }, inplace=True)
            else:
                df.rename(columns={
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close'
                }, inplace=True)
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Convert data types
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            # Add volume (not provided by Alpha Vantage for forex)
            df['volume'] = 0
            
            # Sort by date in descending order
            df.sort_index(ascending=False, inplace=True)
            
            # Filter by end_date if provided
            if end_date:
                df = df[df.index <= end_date]
            
            # Limit rows if needed
            if limit and len(df) > limit:
                df = df.head(limit)
            
            # Sort back to ascending order for analysis
            df.sort_index(inplace=True)
            
            print(f"Successfully fetched {len(df)} {timeframe} candles for {self.trading_pair} from Alpha Vantage")
            return df
            
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"Error fetching data from Alpha Vantage: {e}")
            return None


class AlphaVantageStockFetcher(DataFetcher):
    """Specialized fetcher for stock data from Alpha Vantage"""
    
    def __init__(self, trading_pair: str, market_type: str = DEFAULT_MARKET_TYPE):
        super().__init__(trading_pair, market_type)
        self.api_key = API_KEYS.get('alpha_vantage', '')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required for stock data")
        
        # For stock symbols, the trading pair is the symbol itself (e.g., AAPL, MSFT)
        self.symbol = trading_pair
    
    def fetch_data(self, timeframe: str, limit: int, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Alpha Vantage API.
        
        Args:
            timeframe (str): Time interval
            limit (int): Number of data points
            end_date (Optional[datetime]): End date for historical analysis (Note: Alpha Vantage has limited support)
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data or None if error
        """
        # Map timeframe to Alpha Vantage interval
        interval_map = {
            '1h': '60min',
            '4h': '240min',
            '1d': 'daily',
            '1w': 'weekly'
        }
        
        # Determine which endpoint to use based on timeframe
        if timeframe in ['1h', '4h']:
            function = 'TIME_SERIES_INTRADAY'
            interval = interval_map.get(timeframe)
            endpoint = f"{ALPHA_VANTAGE_BASE_URL}"
            params = {
                'function': function,
                'symbol': self.symbol,
                'interval': interval,
                'outputsize': 'full',
                'apikey': self.api_key
            }
        elif timeframe == '1d':
            function = 'TIME_SERIES_DAILY'
            endpoint = f"{ALPHA_VANTAGE_BASE_URL}"
            params = {
                'function': function,
                'symbol': self.symbol,
                'outputsize': 'full',
                'apikey': self.api_key
            }
        elif timeframe == '1w':
            function = 'TIME_SERIES_WEEKLY'
            endpoint = f"{ALPHA_VANTAGE_BASE_URL}"
            params = {
                'function': function,
                'symbol': self.symbol,
                'apikey': self.api_key
            }
        else:
            raise ValueError(f"Unsupported timeframe for Alpha Vantage stock data: {timeframe}")
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for error response
            if 'Error Message' in data:
                raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
            
            # Handle different response formats
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                raise ValueError(f"Unexpected API response format: {list(data.keys())}")
            
            # Convert to DataFrame
            df_dict = data[time_series_key]
            df = pd.DataFrame.from_dict(df_dict, orient='index')
            
            # Rename columns (format depends on the function)
            df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }, inplace=True)
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Convert data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            # Sort by date in descending order
            df.sort_index(ascending=False, inplace=True)
            
            # Filter by end_date if provided
            if end_date:
                df = df[df.index <= end_date]
            
            # Limit rows if needed
            if limit and len(df) > limit:
                df = df.head(limit)
            
            # Sort back to ascending order for analysis
            df.sort_index(inplace=True)
            
            print(f"Successfully fetched {len(df)} {timeframe} candles for {self.symbol} from Alpha Vantage")
            return df
            
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"Error fetching data from Alpha Vantage: {e}")
            return None


class DataFetcherFactory:
    """Factory for creating appropriate data fetchers based on market type"""
    
    @staticmethod
    def create_fetcher(trading_pair: str, market_type: str = DEFAULT_MARKET_TYPE) -> DataFetcher:
        """
        Create appropriate data fetcher for the given trading pair and market type
        
        Args:
            trading_pair (str): Trading pair to fetch data for
            market_type (str): Type of market (crypto, stock, forex, commodity)
            
        Returns:
            DataFetcher: Appropriate data fetcher instance
        """
        if market_type == MARKET_TYPES['CRYPTO']:
            return BinanceFetcher(trading_pair, market_type)
        elif market_type == MARKET_TYPES['STOCK']:
            # Try Alpha Vantage first if API key is available
            if API_KEYS.get('alpha_vantage'):
                return AlphaVantageStockFetcher(trading_pair, market_type)
            # Fall back to Yahoo Finance if no API key
            return YahooFinanceFetcher(trading_pair, market_type)
        elif market_type == MARKET_TYPES['FOREX']:
            # Try specialized forex fetcher first if API key is available
            if API_KEYS.get('alpha_vantage'):
                return AlphaVantageForexFetcher(trading_pair, market_type)
            # Fall back to Yahoo Finance if no API key
            return YahooFinanceFetcher(trading_pair, market_type)
        elif market_type == MARKET_TYPES['COMMODITY']:
            # For precious metals in XAU/XAG format, use Alpha Vantage forex fetcher if available
            if trading_pair.startswith('XAU') or trading_pair.startswith('XAG'):
                if API_KEYS.get('alpha_vantage'):
                    return AlphaVantageForexFetcher(trading_pair, market_type)
            # For futures contracts or other commodities, use Yahoo Finance
            return YahooFinanceFetcher(trading_pair, market_type)
        else:
            raise ValueError(f"Unsupported market type: {market_type}")


def get_market_data(trading_pair: str = None, timeframe: str = '1d', 
                    source: str = 'binance', market_type: str = DEFAULT_MARKET_TYPE, 
                    limit: int = None, analysis_date: str = None) -> Optional[pd.DataFrame]:
    """
    Wrapper function to get market data for any trading pair from the preferred source.
    Falls back to alternative source if the primary source fails.
    
    Args:
        trading_pair (str): Trading pair to fetch data for (e.g., 'BTCUSDT', 'AAPL')
        timeframe (str): Time interval for the data
        source (str): Primary data source
        market_type (str): Type of market (crypto, stock, forex, commodity)
        limit (int): Optional number of candles to retrieve (overrides default based on timeframe)
        analysis_date (str): Optional date to analyze as if it were "today" (format: YYYY-MM-DD)
        
    Returns:
        pandas.DataFrame: DataFrame with price data
    """
    # For backward compatibility - if no trading pair is provided, use default for market type
    if trading_pair is None:
        trading_pair = DEFAULT_PAIRS.get(market_type, 'BTCUSDT')
    
    # Parse analysis_date if provided
    end_date = None
    if analysis_date:
        try:
            end_date = datetime.strptime(analysis_date, '%Y-%m-%d')
            # Set to end of day for the analysis date
            end_date = end_date.replace(hour=23, minute=59, second=59)
        except ValueError:
            raise ValueError(f"Invalid analysis_date format: {analysis_date}. Expected format: YYYY-MM-DD")
    
    # Check cache first
    cached_df = load_from_cache(trading_pair, timeframe, source, market_type, analysis_date)
    if cached_df is not None:
        return cached_df
    
    # Determine available sources for this market type
    available_sources = [api['name'] for api in MARKET_APIS.get(market_type, [])]
    
    if not available_sources:
        raise ValueError(f"No data sources available for market type: {market_type}")
    
    # If specified source not available for this market type, use first available
    if source not in available_sources:
        source = available_sources[0]
        print(f"Source '{source}' not available for {market_type}, using {source} instead")
    
    # Get appropriate limits based on timeframe if not explicitly provided
    if limit is None:
        limit = 365  # default
        if timeframe in TIMEFRAMES:
            limit = TIMEFRAMES[timeframe].get('limit', 365)
    
    # Try primary source first
    try:
        if market_type == MARKET_TYPES['CRYPTO']:
            if source == 'binance':
                fetcher = BinanceFetcher(trading_pair, market_type)
            else:  # coingecko
                fetcher = CoinGeckoFetcher(trading_pair, market_type)
        elif market_type == MARKET_TYPES['STOCK']:
            if source == 'yahoo_finance':
                fetcher = YahooFinanceFetcher(trading_pair, market_type)
            elif source == 'alpha_vantage' and API_KEYS.get('alpha_vantage'):
                fetcher = AlphaVantageStockFetcher(trading_pair, market_type)
            else:
                fetcher = YahooFinanceFetcher(trading_pair, market_type)
        elif market_type == MARKET_TYPES['FOREX']:
            if source == 'alpha_vantage' and API_KEYS.get('alpha_vantage'):
                fetcher = AlphaVantageForexFetcher(trading_pair, market_type)
            else:
                fetcher = YahooFinanceFetcher(trading_pair, market_type)
        elif market_type == MARKET_TYPES['COMMODITY']:
            if source == 'yahoo_finance':
                fetcher = YahooFinanceFetcher(trading_pair, market_type)
            elif source == 'alpha_vantage' and API_KEYS.get('alpha_vantage'):
                # For XAU/XAG precious metals, use the Forex fetcher
                if trading_pair.startswith('XAU') or trading_pair.startswith('XAG'):
                    fetcher = AlphaVantageForexFetcher(trading_pair, market_type)
                else:
                    fetcher = YahooFinanceFetcher(trading_pair, market_type)
            else:
                fetcher = YahooFinanceFetcher(trading_pair, market_type)
        else:
            raise ValueError(f"Unsupported market type: {market_type}")
        
        df = fetcher.fetch_data(timeframe, limit, end_date)
        if df is not None:
            # Save to cache before returning
            save_to_cache(df, trading_pair, timeframe, source, market_type, analysis_date)
            return df
    except Exception as e:
        print(f"Error with primary source {source}: {e}")
    
    # If primary source fails, try alternatives
    for alt_source in [s for s in available_sources if s != source]:
        try:
            print(f"Trying alternative source: {alt_source}")
            if alt_source == 'binance':
                fetcher = BinanceFetcher(trading_pair, market_type)
            elif alt_source == 'coingecko':
                fetcher = CoinGeckoFetcher(trading_pair, market_type)
            elif alt_source == 'yahoo_finance':
                fetcher = YahooFinanceFetcher(trading_pair, market_type)
            elif alt_source == 'alpha_vantage' and API_KEYS.get('alpha_vantage'):
                if market_type == MARKET_TYPES['FOREX']:
                    fetcher = AlphaVantageForexFetcher(trading_pair, market_type)
                elif market_type == MARKET_TYPES['STOCK']:
                    fetcher = AlphaVantageStockFetcher(trading_pair, market_type)
                elif market_type == MARKET_TYPES['COMMODITY']:
                    # For XAU/XAG precious metals, use the Forex fetcher
                    if trading_pair.startswith('XAU') or trading_pair.startswith('XAG'):
                        fetcher = AlphaVantageForexFetcher(trading_pair, market_type)
                    else:
                        continue
                else:
                    continue
            else:
                continue
                
            df = fetcher.fetch_data(timeframe, limit, end_date)
            if df is not None:
                # Save to cache before returning (using the alternative source)
                save_to_cache(df, trading_pair, timeframe, alt_source, market_type, analysis_date)
                return df
        except Exception as e:
            print(f"Error with alternative source {alt_source}: {e}")
    
    # If all sources fail
    raise Exception(f"Failed to fetch data for {trading_pair} from all available sources")


# For backward compatibility
def get_bitcoin_data(timeframe: str = '1d', source: str = 'binance') -> pd.DataFrame:
    """
    Legacy function to get Bitcoin data for backward compatibility.
    
    Args:
        timeframe (str): Time interval for the data
        source (str): Primary data source ('binance' or 'coingecko')
        
    Returns:
        pandas.DataFrame: DataFrame with price data
    """
    print("Warning: get_bitcoin_data is deprecated, use get_market_data instead")
    return get_market_data('BTCUSDT', timeframe, source, MARKET_TYPES['CRYPTO'])
