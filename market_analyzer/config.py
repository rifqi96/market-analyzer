"""
Configuration settings for the Market Analyzer package.
"""

import os
import pathlib

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')

# Market Types
MARKET_TYPES = {
    'CRYPTO': 'cryptocurrency',
    'STOCK': 'stock',
    'FOREX': 'forex',
    'COMMODITY': 'commodity'
}

DEFAULT_MARKET_TYPE = MARKET_TYPES['CRYPTO']

# API URLs
BINANCE_BASE_URL = 'https://api.binance.com/api/v3'
COINGECKO_BASE_URL = 'https://api.coingecko.com/api/v3'
YAHOO_FINANCE_BASE_URL = 'https://query1.finance.yahoo.com/v8/finance/chart'
ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query'

# Default Trading Pairs by Market Type
DEFAULT_PAIRS = {
    MARKET_TYPES['CRYPTO']: 'BTCUSDT',
    MARKET_TYPES['STOCK']: 'AAPL',
    MARKET_TYPES['FOREX']: 'EURUSD',
    MARKET_TYPES['COMMODITY']: 'XAUUSD'  # Gold/USD as default
}

# Timeframes configuration
TIMEFRAMES = {
    '1h': {'interval': '1h', 'limit': 720, 'binance_code': '1h', 'yahoo_code': '60m'},     # 30 days
    '4h': {'interval': '4h', 'limit': 720, 'binance_code': '4h', 'yahoo_code': '1h'},      # 120 days
    '1d': {'interval': '1d', 'limit': 365, 'binance_code': '1d', 'yahoo_code': '1d'},      # 1 year
    '3d': {'interval': '3d', 'limit': 200, 'binance_code': '3d', 'yahoo_code': '1d'},      # ~1.5 years
    '1w': {'interval': '1w', 'limit': 200, 'binance_code': '1w', 'yahoo_code': '1wk'},     # ~4 years
}

# Market-specific API configurations
MARKET_APIS = {
    MARKET_TYPES['CRYPTO']: [
        {'name': 'binance', 'base_url': BINANCE_BASE_URL, 'requires_key': False},
        {'name': 'coingecko', 'base_url': COINGECKO_BASE_URL, 'requires_key': False}
    ],
    MARKET_TYPES['STOCK']: [
        {'name': 'yahoo_finance', 'base_url': YAHOO_FINANCE_BASE_URL, 'requires_key': False},
        {'name': 'alpha_vantage', 'base_url': ALPHA_VANTAGE_BASE_URL, 'requires_key': True}
    ],
    MARKET_TYPES['FOREX']: [
        {'name': 'alpha_vantage', 'base_url': ALPHA_VANTAGE_BASE_URL, 'requires_key': True}
    ],
    MARKET_TYPES['COMMODITY']: [
        {'name': 'yahoo_finance', 'base_url': YAHOO_FINANCE_BASE_URL, 'requires_key': False},
        {'name': 'alpha_vantage', 'base_url': ALPHA_VANTAGE_BASE_URL, 'requires_key': True}
    ]
}

# API Keys (should be moved to environment variables in production)
API_KEYS = {
    'alpha_vantage': os.environ.get('ALPHA_VANTAGE_API_KEY', '')
}

# Trading Pair formats
def validate_trading_pair(pair, market_type=DEFAULT_MARKET_TYPE):
    """Validate trading pair format based on market type."""
    if market_type == MARKET_TYPES['CRYPTO']:
        # Basic validation for crypto pairs (e.g., BTCUSDT, ETHBTC)
        return len(pair) >= 5
    elif market_type == MARKET_TYPES['STOCK']:
        # Basic validation for stock tickers (1-6 alphanumeric characters)
        return 1 <= len(pair) <= 6
    elif market_type == MARKET_TYPES['FOREX']:
        # Basic validation for forex pairs (usually 6 characters like EURUSD)
        return len(pair) >= 6
    elif market_type == MARKET_TYPES['COMMODITY']:
        # Validation for commodity symbols (e.g., XAUUSD, GC=F)
        return len(pair) >= 4
    return False

# Currency symbol mapping
CURRENCY_SYMBOLS = {
    'USD': '$',
    'USDT': '$',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'CNY': '¥',
    'IDR': 'Rp',
    'KRW': '₩',
    'INR': '₹',
    'RUB': '₽',
    'TRY': '₺',
    'BRL': 'R$',
    'CAD': 'C$',
    'AUD': 'A$',
    'NZD': 'NZ$',
    'CHF': 'Fr',
    'SGD': 'S$',
    'HKD': 'HK$',
    'ZAR': 'R',
    'SEK': 'kr',
    'NOK': 'kr',
    'DKK': 'kr',
    # Add more currency symbols as needed
}

def get_currency_symbol(trading_pair, market_type=DEFAULT_MARKET_TYPE):
    """
    Determine the appropriate currency symbol based on trading pair and market type.
    
    Args:
        trading_pair (str): Trading pair to analyze
        market_type (str): Type of market (crypto, stock, forex, commodity)
        
    Returns:
        str: Currency symbol to use in reports
    """
    # Default currency symbol
    default_symbol = '$'
    
    if market_type == MARKET_TYPES['CRYPTO']:
        # For crypto, extract quote currency (e.g., BTCUSDT -> USDT)
        quote_currencies = ['USDT', 'USD', 'EUR', 'GBP', 'JPY', 'BTC', 'ETH']
        for quote in quote_currencies:
            if trading_pair.endswith(quote):
                return CURRENCY_SYMBOLS.get(quote, default_symbol)
        
        # If no match, return default
        return default_symbol
    
    elif market_type == MARKET_TYPES['STOCK']:
        # For stocks, usually use $ but could be customized based on exchange
        # This could be expanded with a mapping of exchanges to currency symbols
        return default_symbol
    
    elif market_type == MARKET_TYPES['FOREX']:
        # For forex, extract quote currency (e.g., EURUSD -> USD, USDJPY -> JPY)
        if len(trading_pair) >= 6:
            quote_currency = trading_pair[3:6]
            return CURRENCY_SYMBOLS.get(quote_currency, default_symbol)
        
        # If no match, return default
        return default_symbol
    
    elif market_type == MARKET_TYPES['COMMODITY']:
        # For commodities, use the quote currency symbol if available
        # Handle different commodity formats
        
        # Precious metals in XAU/XAG format (e.g., XAUUSD -> USD)
        if trading_pair.startswith('XAU') or trading_pair.startswith('XAG'):
            if len(trading_pair) >= 6:
                quote_currency = trading_pair[3:6]
                return CURRENCY_SYMBOLS.get(quote_currency, default_symbol)
        
        # Futures contracts (e.g., GC=F, SI=F)
        commodity_currency_map = {
            'GC=F': '$',    # Gold Futures (USD)
            'SI=F': '$',    # Silver Futures (USD)
            'HG=F': '$',    # Copper Futures (USD)
            'PL=F': '$',    # Platinum Futures (USD)
            'PA=F': '$',    # Palladium Futures (USD)
            'CL=F': '$',    # Crude Oil Futures (USD)
            'BZ=F': '$',    # Brent Crude Oil Futures (USD)
            'NG=F': '$'     # Natural Gas Futures (USD)
        }
        
        return commodity_currency_map.get(trading_pair, default_symbol)
    
    # Default fallback
    return default_symbol

def get_asset_name(trading_pair, market_type=DEFAULT_MARKET_TYPE):
    """
    Get a proper display name for the trading pair based on market type.
    
    Args:
        trading_pair (str): Trading pair to analyze
        market_type (str): Type of market (crypto, stock, forex, commodity)
        
    Returns:
        str: Proper display name for the asset or pair
    """
    if market_type == MARKET_TYPES['CRYPTO']:
        # Common crypto names
        crypto_names = {
            'BTC': 'Bitcoin',
            'ETH': 'Ethereum',
            'SOL': 'Solana',
            'XRP': 'Ripple',
            'ADA': 'Cardano',
            'DOGE': 'Dogecoin',
            'DOT': 'Polkadot',
            'AVAX': 'Avalanche',
            'MATIC': 'Polygon',
            'LINK': 'Chainlink'
        }
        
        # Try to extract base currency (e.g., BTCUSDT -> BTC)
        for base in crypto_names:
            if trading_pair.startswith(base):
                return crypto_names[base]
        
        # If no specific name, return the trading pair
        return trading_pair
    
    elif market_type == MARKET_TYPES['STOCK']:
        # For stocks, the trading pair is usually the ticker symbol
        return trading_pair
    
    elif market_type == MARKET_TYPES['FOREX']:
        # For forex, format as BASE/QUOTE (e.g., EURUSD -> EUR/USD)
        if len(trading_pair) >= 6:
            base = trading_pair[:3]
            quote = trading_pair[3:6]
            return f"{base}/{quote}"
        
        # If not standard format, return as is
        return trading_pair
    
    elif market_type == MARKET_TYPES['COMMODITY']:
        # Comprehensive commodity names
        commodity_names = {
            # Precious Metals (Spot)
            'XAU': 'Gold',
            'XAUUSD': 'Gold/USD',
            'XAUEUR': 'Gold/EUR',
            'XAUGBP': 'Gold/GBP',
            'XAG': 'Silver',
            'XAGUSD': 'Silver/USD',
            'XAGEUR': 'Silver/EUR',
            'XAGGBP': 'Silver/GBP',
            'XPT': 'Platinum',
            'XPTUSD': 'Platinum/USD',
            'XPD': 'Palladium',
            'XPDUSD': 'Palladium/USD',
            
            # Metal Futures
            'GC=F': 'Gold Futures',
            'MGC=F': 'Micro Gold Futures',
            'SI=F': 'Silver Futures',
            'SIL=F': 'Micro Silver Futures',
            'HG=F': 'Copper Futures',
            'PL=F': 'Platinum Futures',
            'PA=F': 'Palladium Futures',
            
            # Energy Futures
            'CL=F': 'Crude Oil Futures',
            'MCL=F': 'Micro Crude Oil Futures',
            'BZ=F': 'Brent Crude Oil Futures',
            'NG=F': 'Natural Gas Futures',
            'RB=F': 'Gasoline RBOB Futures',
            'HO=F': 'Heating Oil Futures',
            
            # Agricultural Futures
            'ZC=F': 'Corn Futures',
            'ZW=F': 'Wheat Futures',
            'ZS=F': 'Soybean Futures',
            'ZM=F': 'Soybean Meal Futures',
            'ZL=F': 'Soybean Oil Futures',
            'KC=F': 'Coffee Futures',
            'SB=F': 'Sugar Futures',
            'CT=F': 'Cotton Futures',
            'CC=F': 'Cocoa Futures',
            'OJ=F': 'Orange Juice Futures',
            
            # ETFs
            'GLD': 'SPDR Gold Shares',
            'IAU': 'iShares Gold Trust',
            'SLV': 'iShares Silver Trust',
            'PALL': 'Aberdeen Standard Physical Palladium Shares',
            'PPLT': 'Aberdeen Standard Physical Platinum Shares',
            'USO': 'United States Oil Fund',
            'BNO': 'United States Brent Oil Fund'
        }
        
        # Return full name if available
        if trading_pair in commodity_names:
            return commodity_names[trading_pair]
        
        # For XAU/XAG pairs not explicitly defined
        if trading_pair.startswith('XAU'):
            quote = trading_pair[3:6] if len(trading_pair) >= 6 else ''
            return f"Gold/{quote}" if quote else "Gold"
        elif trading_pair.startswith('XAG'):
            quote = trading_pair[3:6] if len(trading_pair) >= 6 else ''
            return f"Silver/{quote}" if quote else "Silver"
        
        # For unrecognized futures or other pairs
        if '=' in trading_pair:
            return f"{trading_pair} Futures"
        
        # Return the original if no match
        return trading_pair
    
    # Default fallback
    return trading_pair
