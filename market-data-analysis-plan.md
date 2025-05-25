# Multi-Market Analysis Plan

## 1. API Selection

For retrieving market data across different asset classes, we'll use the following APIs:

### Cryptocurrency APIs

- **Primary API: Binance API**
  - **Endpoint**: `https://api.binance.com/api/v3/klines`
  - **Advantages**:
    - High-quality data with minimal downtime
    - Multiple timeframe options
    - No authentication required for historical data

- **Backup API: CoinGecko API**
  - **Endpoint**: `https://api.coingecko.com/api/v3/coins/bitcoin/market_chart`
  - **Advantages**:
    - Free tier available with reasonable rate limits
    - Provides additional market metrics

### Stock Market APIs

- **Primary API: Yahoo Finance API**
  - **Endpoint**: `https://query1.finance.yahoo.com/v8/finance/chart`
  - **Advantages**:
    - Comprehensive stock data
    - No authentication required (unofficial API)
    - Global market coverage

- **Backup API: Alpha Vantage API**
  - **Endpoint**: `https://www.alphavantage.co/query`
  - **Advantages**:
    - Reliable data source
    - Official API with documentation
    - Provides fundamental data as well
  - **Limitations**:
    - Requires API key
    - Rate limits (5 requests per minute for free tier)

### Forex APIs

- **Primary API: Alpha Vantage API**
  - **Endpoint**: `https://www.alphavantage.co/query`
  - **Advantages**:
    - Reliable forex data source
    - Supports multiple timeframes
  - **Limitations**:
    - Requires API key
    - Rate limits apply

### Commodity APIs

- **Primary API: Yahoo Finance API**
  - **Endpoint**: `https://query1.finance.yahoo.com/v8/finance/chart`
  - **Advantages**:
    - Comprehensive data for major commodities (Gold, Silver, Oil, etc.)
    - Access to futures contracts data
    - No authentication required
  - **Supported Symbols**:
    - Gold Futures: `GC=F`
    - Gold ETF: `GLD`
    - Silver Futures: `SI=F`
    - Crude Oil: `CL=F`

- **Backup API: Alpha Vantage API**
  - **Endpoint**: `https://www.alphavantage.co/query`
  - **Advantages**:
    - Reliable commodity data
    - Common format for precious metals (e.g., XAUUSD for Gold/USD)
  - **Limitations**:
    - Requires API key
    - Rate limits apply

## 2. Timeframe Selection

For effective market analysis across different asset classes, we'll analyze multiple timeframes:

### Short-Term Analysis (Next Week Prediction)

- **Primary timeframes**: 1-hour, 4-hour, and daily charts
- **Rationale**:
  - 1-hour charts capture intraday movements
  - 4-hour charts reduce noise while preserving important patterns
  - Daily charts provide broader context
- **Historical period**: 30-90 days of historical data

### Medium-Term Analysis (Monthly Prediction)

- **Primary timeframes**: Daily, 3-day, and weekly charts
- **Rationale**:
  - Daily charts show detailed market behavior
  - 3-day charts bridge the gap between daily and weekly perspectives
  - Weekly charts highlight major support/resistance levels
- **Historical period**: 180-365 days for pattern recognition

## 3. Technical Indicators Selection

We implement the following technical indicators across all market types:

### Trend Indicators

- **Moving Averages (MA)**: 20, 50, 100, and 200-period SMAs (calibrated by market type)
- **Exponential Moving Averages (EMA)**: Different period settings based on market type
- **Moving Average Convergence Divergence (MACD)**: (12, 26, 9) settings

### Momentum Indicators

- **Relative Strength Index (RSI)**: 14-period setting with 30/70 overbought/oversold levels
- **Stochastic Oscillator**: (14, 3, 3) settings

### Volatility Indicators

- **Bollinger Bands**: (20, 2) settings for volatility expansion/contraction

## 4. Market-Specific Optimizations

### Cryptocurrency

- Higher volatility thresholds
- Support for USDT, BTC, and ETH trading pairs
- Common base currencies (BTC, ETH, SOL, etc.) mapped to full names

### Stock Market

- Medium volatility thresholds
- Support for stock tickers from global exchanges
- Price formatting using appropriate currency symbols

### Forex

- Lower volatility thresholds
- Support for standard currency pairs (e.g., EURUSD, USDJPY)
- Proper display formatting as BASE/QUOTE (e.g., EUR/USD)
- Appropriate currency symbol selection based on quote currency

### Commodities

- Medium to high volatility thresholds
- Support for futures contracts and ETFs
- Special handling for precious metals (Gold, Silver)
- Support for energy commodities (Oil, Natural Gas)
- Proper commodity naming and currency symbol display

## 5. Report Generation

Our analysis generates the following reports with market-specific customizations:

1. **Short-Term Forecast Report**:
   - Uses proper asset name (Bitcoin, AAPL, EUR/USD, Gold, etc.)
   - Uses appropriate currency symbol (₿, $, €, etc.)
   - Weekly price targets with probabilities
   - Key support and resistance levels
   - Current trend and momentum analysis

2. **Monthly Forecast Report**:
   - Monthly price projections
   - Year-end targets
   - Technical analysis summary
   - Market-specific insights

3. **Combined Technical Analysis Report**:
   - Executive summary
   - Short and medium-term forecasts
   - Comprehensive technical analysis
   - Market-appropriate conclusions

## 6. Data Pipeline Components

1. **Data Fetchers**:
   - Factory pattern for creating market-specific data fetchers
   - Fallback mechanisms when primary source fails
   - Intelligent caching to minimize API calls
   - Standardized output format across markets

2. **Technical Analysis**:
   - Market-calibrated indicator parameters
   - Support/resistance detection tuned by market type
   - Volatility analysis adjusted for different markets

3. **Visualization**:
   - Market-specific chart styling and annotations
   - Price formatting with proper currency symbols
   - Dynamic chart directory organization by market type

4. **Reporting**:
   - Dynamic report generation with market-appropriate titles
   - Currency symbol selection based on trading pair
   - File organization by market type and trading pair

## 7. Implementation Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Fetchers  │────▶│    Analyzers    │────▶│  Visualizers    │────▶│    Reporters    │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │                       │
        ▼                       ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Market Type    │     │    Technical    │     │ Chart           │     │  Report         │
│  Adapters       │     │    Indicators   │     │ Generators      │     │  Generators     │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │                       │
        ▼                       ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Caching        │     │  Market-Specific │     │  Output         │     │  Market-Specific│
│  System         │     │  Parameters      │     │  Organization   │     │  Reports        │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 8. Output Structure

```
output/
├── cryptocurrency/
│   ├── BTCUSDT/
│   │   ├── charts/
│   │   ├── data/
│   │   └── reports/
├── stock/
│   ├── AAPL/
│   │   ├── charts/
│   │   ├── data/
│   │   └── reports/
├── forex/
│   ├── EURUSD/
│   │   ├── charts/
│   │   ├── data/
│   │   └── reports/
└── commodity/
    ├── XAUUSD/
    │   ├── charts/
    │   ├── data/
    │   └── reports/
```

## 9. Recent Enhancements

1. **Dynamic Currency Symbol Support**:
   - Automatically selects correct currency symbol (e.g., $, €, £, ¥, Rp)
   - Based on trading pair and market type
   - Proper formatting in reports and visualizations

2. **Asset Name Generation**:
   - Full names for common cryptocurrencies (e.g., Bitcoin, Ethereum)
   - Standard ticker display for stocks
   - BASE/QUOTE format for forex pairs (e.g., EUR/USD)
   - Full commodity names (Gold, Silver, Crude Oil)

3. **Multi-Market API Support**:
   - Alpha Vantage implementation for stocks, forex, and commodities
   - Yahoo Finance integration for stocks and commodity futures
   - Fallback mechanisms when primary sources hit rate limits
   - Market-specific error handling

4. **Report Template Improvements**:
   - Dynamic naming in reports based on analyzed asset
   - Market-appropriate formatting and terminology
   - Proper file naming based on trading pair

5. **Commodity Market Support**:
   - Added support for precious metals (Gold, Silver)
   - Added support for energy commodities (Oil, Natural Gas)
   - Support for both futures contracts and ETFs
   - Proper display of commodity names and pricing
