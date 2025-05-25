"""
Module for generating reports based on technical analysis.
"""

import os
import json
from datetime import datetime
import pandas as pd
from .config import (
    DATA_DIR, OUTPUT_DIR, MARKET_TYPES, DEFAULT_MARKET_TYPE,
    get_currency_symbol, get_asset_name
)

def save_analysis_json(analysis, data_dir=DATA_DIR, filename='analysis_results.json'):
    """
    Save analysis results to JSON file.
    
    Args:
        analysis (dict): Analysis results
        data_dir (str): Directory to save the file
        filename (str): Output filename
    """
    # Ensure output directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Convert DataFrame values to lists for JSON serialization
    def prepare_for_json(obj):
        if isinstance(obj, dict):
            return {k: prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [prepare_for_json(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    # Prepare analysis for JSON
    analysis_json = prepare_for_json(analysis)
    
    # Add metadata
    analysis_json['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'version': '0.1.0'
    }
    
    # Save to file
    output_path = os.path.join(data_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(analysis_json, f, indent=2)
    
    print(f"Saved analysis results to {output_path}")
    
    return output_path

def generate_short_term_report(analysis, output_dir=OUTPUT_DIR, filename='short_term_report.md'):
    """
    Generate short-term forecast report in Markdown format.
    
    Args:
        analysis (dict): Analysis results
        output_dir (str): Directory to save the report
        filename (str): Output filename
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get short-term forecast
    forecast = analysis['short_term_forecast']
    if not forecast:
        print("Short-term forecast not available")
        return None
    
    # Get trading pair and market type from analysis
    trading_pair = analysis.get('trading_pair', 'BTCUSDT')
    market_type = analysis.get('market_type', DEFAULT_MARKET_TYPE)
    
    # Get appropriate currency symbol and asset name
    currency_symbol = get_currency_symbol(trading_pair, market_type)
    asset_name = get_asset_name(trading_pair, market_type)
    
    # Format price values with correct currency symbol
    current_price = f"{currency_symbol}{forecast['current_price']:,.2f}"
    bullish_target = f"{currency_symbol}{forecast['targets']['bullish']:,.2f}"
    base_target = f"{currency_symbol}{forecast['targets']['base']:,.2f}"
    bearish_target = f"{currency_symbol}{forecast['targets']['bearish']:,.2f}"
    
    # Key levels
    nearest_support = f"{currency_symbol}{forecast['key_levels']['nearest_support']:,.2f}" if forecast['key_levels']['nearest_support'] else "None"
    nearest_resistance = f"{currency_symbol}{forecast['key_levels']['nearest_resistance']:,.2f}" if forecast['key_levels']['nearest_resistance'] else "None"
    
    # Generate report
    report = f"""# {asset_name} Short-Term Price Forecast

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Current Market Status

- **Current Price**: {current_price}
- **Market Trend**: {forecast['trend'].capitalize()}
- **Momentum**: {forecast['momentum'].capitalize()}
- **Nearest Support**: {nearest_support}
- **Nearest Resistance**: {nearest_resistance}

## 1-Week Price Targets

| Scenario | Price Target | Probability |
|----------|--------------|------------|
| Bullish  | {bullish_target} | {forecast['probability']['bullish']*100:.0f}% |
| Base Case | {base_target} | {forecast['probability']['base']*100:.0f}% |
| Bearish  | {bearish_target} | {forecast['probability']['bearish']*100:.0f}% |

## Technical Analysis Summary

"""
    
    # Add trend analysis
    report += f"### Trend Analysis\n\n"
    report += f"The current trend is **{forecast['trend'].upper()}** based on price action and moving average analysis. "
    
    if forecast['trend'] == 'bullish':
        report += "Price is trading above key moving averages and showing strength. "
    elif forecast['trend'] == 'bearish':
        report += "Price is trading below key moving averages and showing weakness. "
    else:
        report += "Price is consolidating in a range with no clear direction. "
    
    # Add momentum analysis
    report += f"\n\n### Momentum Analysis\n\n"
    momentum = forecast['analysis']['momentum']
    
    # RSI
    if momentum['rsi']['value'] is not None:
        rsi_value = momentum['rsi']['value']
        report += f"- **RSI**: {rsi_value:.2f} - "
        if rsi_value > 70:
            report += "Overbought territory, suggesting potential reversal or consolidation.\n"
        elif rsi_value < 30:
            report += "Oversold territory, suggesting potential reversal or bounce.\n"
        elif rsi_value > 50:
            report += "Bullish momentum with strength above the centerline.\n"
        else:
            report += "Bearish momentum with weakness below the centerline.\n"
    
    # MACD
    if momentum['macd']['value'] is not None:
        macd_value = momentum['macd']['value']
        signal_value = momentum['macd']['signal']
        
        report += f"- **MACD**: {macd_value:.2f} (Signal: {signal_value:.2f}) - "
        if momentum['macd']['bullish_cross']:
            report += "Bullish crossover, suggesting potential upward momentum.\n"
        elif momentum['macd']['bearish_cross']:
            report += "Bearish crossover, suggesting potential downward momentum.\n"
        elif macd_value > signal_value:
            report += "MACD above signal line, indicating bullish momentum.\n"
        else:
            report += "MACD below signal line, indicating bearish momentum.\n"
    
    # Add key levels analysis
    report += f"\n\n### Key Price Levels\n\n"
    
    # Psychological levels
    psych_levels = forecast['key_levels']['psychological_levels']
    report += f"**Psychological Levels**: "
    report += ", ".join([f"{currency_symbol}{level:,.0f}" for level in psych_levels if abs(level - forecast['current_price']) / forecast['current_price'] < 0.1])
    
    # Support zones
    report += f"\n\n**Support Zones**:\n"
    support_zones = forecast['key_levels']['support_zones']
    for i, zone in enumerate(sorted(support_zones, key=lambda x: -x['center'])):
        if i >= 3:  # Show only top 3 support zones
            break
        report += f"- {currency_symbol}{zone['center']:,.2f} ({currency_symbol}{zone['lower']:,.2f} - {currency_symbol}{zone['upper']:,.2f})\n"
    
    # Resistance zones
    report += f"\n**Resistance Zones**:\n"
    resistance_zones = forecast['key_levels']['resistance_zones']
    for i, zone in enumerate(sorted(resistance_zones, key=lambda x: x['center'])):
        if i >= 3:  # Show only top 3 resistance zones
            break
        report += f"- {currency_symbol}{zone['center']:,.2f} ({currency_symbol}{zone['lower']:,.2f} - {currency_symbol}{zone['upper']:,.2f})\n"
    
    # Add volatility analysis
    report += f"\n\n### Volatility Analysis\n\n"
    volatility = forecast['analysis']['volatility']
    report += f"Weekly volatility: {volatility * 100:.2f}%\n\n"
    
    # Add conclusion
    report += f"\n## Conclusion\n\n"
    
    # Generate conclusion based on trend and momentum
    if forecast['trend'] == 'bullish' and forecast['momentum'] == 'bullish':
        report += "The technical analysis indicates a **STRONG BULLISH** outlook for {asset_name} in the next week. "
        report += f"With a strong trend and positive momentum, price could test {currency_symbol}{bullish_target} if market conditions remain favorable. "
        report += f"Key resistance at {nearest_resistance} should be monitored for potential breakout or rejection."
    
    elif forecast['trend'] == 'bullish' and forecast['momentum'] != 'bullish':
        report += "The technical analysis indicates a **MODERATELY BULLISH** outlook for {asset_name} in the next week. "
        report += f"While the overall trend remains positive, momentum is showing mixed signals. "
        report += f"Price could test {currency_symbol}{base_target} if momentum improves, but may face resistance at {nearest_resistance}."
    
    elif forecast['trend'] == 'bearish' and forecast['momentum'] == 'bearish':
        report += "The technical analysis indicates a **STRONG BEARISH** outlook for {asset_name} in the next week. "
        report += f"With a downward trend and negative momentum, price could test {currency_symbol}{bearish_target} if selling pressure continues. "
        report += f"Key support at {nearest_support} should be monitored for potential breakdown or bounce."
    
    elif forecast['trend'] == 'bearish' and forecast['momentum'] != 'bearish':
        report += "The technical analysis indicates a **MODERATELY BEARISH** outlook for {asset_name} in the next week. "
        report += f"While the overall trend is negative, momentum is showing mixed signals. "
        report += f"Price could test {currency_symbol}{base_target} if downward pressure persists, but may find support at {nearest_support}."
    
    else:  # 'sideways' trend
        report += "The technical analysis indicates a **NEUTRAL** outlook for {asset_name} in the next week. "
        report += f"Price is likely to consolidate between {currency_symbol}{bearish_target} and {currency_symbol}{bullish_target} with no clear directional bias. "
        report += "Traders should watch for breakouts from this range for potential trading opportunities."
    
    # Save report to file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Saved short-term report to {output_path}")
    
    return output_path

def generate_monthly_report(analysis, output_dir=OUTPUT_DIR, filename='monthly_report.md'):
    """
    Generate monthly forecast report in Markdown format.
    
    Args:
        analysis (dict): Analysis results
        output_dir (str): Directory to save the report
        filename (str): Output filename
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get monthly forecast
    forecast = analysis['monthly_forecast']
    if not forecast:
        print("Monthly forecast not available")
        return None
    
    # Get trading pair and market type from analysis
    trading_pair = analysis.get('trading_pair', 'BTCUSDT')
    market_type = analysis.get('market_type', DEFAULT_MARKET_TYPE)
    
    # Get appropriate currency symbol and asset name
    currency_symbol = get_currency_symbol(trading_pair, market_type)
    asset_name = get_asset_name(trading_pair, market_type)
    
    # Format price values with correct currency symbol
    current_price = f"{currency_symbol}{forecast['current_price']:,.2f}"
    bullish_target = f"{currency_symbol}{forecast['year_end_targets']['bullish']:,.2f}"
    base_target = f"{currency_symbol}{forecast['year_end_targets']['base']:,.2f}"
    bearish_target = f"{currency_symbol}{forecast['year_end_targets']['bearish']:,.2f}"
    
    # Key levels
    nearest_support = f"{currency_symbol}{forecast['key_levels']['nearest_support']:,.2f}" if forecast['key_levels']['nearest_support'] else "None"
    nearest_resistance = f"{currency_symbol}{forecast['key_levels']['nearest_resistance']:,.2f}" if forecast['key_levels']['nearest_resistance'] else "None"
    
    # Generate report
    report = f"""# {asset_name} Monthly Price Forecast (Through End of 2025)

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Current Market Status

- **Current Price**: {current_price}
- **Market Trend**: {forecast['trend'].capitalize()}
- **Momentum**: {forecast['momentum'].capitalize()}
- **Nearest Support**: {nearest_support}
- **Nearest Resistance**: {nearest_resistance}

## Year-End Price Targets

| Scenario | Price Target | Probability |
|----------|--------------|------------|
| Bullish  | {bullish_target} | {forecast['probability']['bullish']*100:.0f}% |
| Base Case | {base_target} | {forecast['probability']['base']*100:.0f}% |
| Bearish  | {bearish_target} | {forecast['probability']['bearish']*100:.0f}% |

## Monthly Price Projections

| Month | Low | Median | High |
|-------|-----|--------|------|
"""
    
    # Add monthly targets
    for month, targets in forecast['monthly_targets'].items():
        report += f"| {month} | {currency_symbol}{targets['low']:,.2f} | {currency_symbol}{targets['median']:,.2f} | {currency_symbol}{targets['high']:,.2f} |\n"
    
    # Add technical analysis summary
    report += f"\n## Technical Analysis Summary\n\n"
    
    # Add trend analysis
    report += f"### Trend Analysis\n\n"
    report += f"The current trend is **{forecast['trend'].upper()}** based on price action and moving average analysis. "
    
    if forecast['trend'] == 'bullish':
        report += "Price is trading above key moving averages and showing strength, suggesting potential for continued upward movement. "
    elif forecast['trend'] == 'bearish':
        report += "Price is trading below key moving averages and showing weakness, suggesting potential for continued downward movement. "
    else:
        report += "Price is consolidating in a range with no clear direction, suggesting a period of uncertainty. "
    
    # Add momentum analysis
    report += f"\n\n### Momentum Analysis\n\n"
    momentum = forecast['analysis']['momentum']
    
    # RSI
    if momentum['rsi']['value'] is not None:
        rsi_value = momentum['rsi']['value']
        report += f"- **RSI**: {rsi_value:.2f} - "
        if rsi_value > 70:
            report += "Overbought territory, suggesting potential reversal or consolidation in the near term.\n"
        elif rsi_value < 30:
            report += "Oversold territory, suggesting potential reversal or bounce in the near term.\n"
        elif rsi_value > 50:
            report += "Bullish momentum with strength above the centerline, supporting a positive outlook.\n"
        else:
            report += "Bearish momentum with weakness below the centerline, supporting a negative outlook.\n"
    
    # MACD
    if momentum['macd']['value'] is not None:
        macd_value = momentum['macd']['value']
        signal_value = momentum['macd']['signal']
        
        report += f"- **MACD**: {macd_value:.2f} (Signal: {signal_value:.2f}) - "
        if momentum['macd']['bullish_cross']:
            report += "Bullish crossover, suggesting potential upward momentum in the medium term.\n"
        elif momentum['macd']['bearish_cross']:
            report += "Bearish crossover, suggesting potential downward momentum in the medium term.\n"
        elif macd_value > signal_value:
            report += "MACD above signal line, indicating bullish momentum in the medium term.\n"
        else:
            report += "MACD below signal line, indicating bearish momentum in the medium term.\n"
    
    # Add volatility analysis
    report += f"\n\n### Volatility Analysis\n\n"
    volatility = forecast['analysis']['volatility']
    report += f"Monthly volatility: {volatility * 100:.2f}%\n\n"
    
    # Add key levels analysis
    report += f"\n\n### Key Price Levels\n\n"
    
    # Psychological levels
    psych_levels = forecast['key_levels']['psychological_levels']
    report += f"**Psychological Levels**: "
    report += ", ".join([f"{currency_symbol}{level:,.0f}" for level in psych_levels])
    
    # Add conclusion
    report += f"\n\n## Conclusion\n\n"
    
    # Generate conclusion based on trend and momentum
    if forecast['trend'] == 'bullish' and forecast['momentum'] == 'bullish':
        report += "The technical analysis indicates a **STRONG BULLISH** outlook for {asset_name} through the end of 2025. "
        report += f"With a strong trend and positive momentum, price could rise toward {currency_symbol}{bullish_target} by year-end if market conditions remain favorable. "
        report += "Key psychological and resistance levels should be monitored for potential breakouts or rejections along the way."
    
    elif forecast['trend'] == 'bullish' and forecast['momentum'] != 'bullish':
        report += "The technical analysis indicates a **MODERATELY BULLISH** outlook for {asset_name} through the end of 2025. "
        report += f"While the overall trend remains positive, momentum is showing mixed signals. "
        report += f"Price could approach {currency_symbol}{base_target} by year-end if momentum improves, but may face challenges at key resistance levels."
    
    elif forecast['trend'] == 'bearish' and forecast['momentum'] == 'bearish':
        report += "The technical analysis indicates a **STRONG BEARISH** outlook for {asset_name} through the end of 2025. "
        report += f"With a downward trend and negative momentum, price could decline toward {currency_symbol}{bearish_target} by year-end if selling pressure continues. "
        report += "Key psychological and support levels should be monitored for potential breakdowns or bounces along the way."
    
    elif forecast['trend'] == 'bearish' and forecast['momentum'] != 'bearish':
        report += "The technical analysis indicates a **MODERATELY BEARISH** outlook for {asset_name} through the end of 2025. "
        report += f"While the overall trend is negative, momentum is showing mixed signals. "
        report += f"Price could move toward {currency_symbol}{base_target} by year-end if downward pressure persists, but may find support at key levels."
    
    else:  # 'sideways' trend
        report += "The technical analysis indicates a **NEUTRAL** outlook for {asset_name} through the end of 2025. "
        report += f"Price is likely to fluctuate between {currency_symbol}{bearish_target} and {currency_symbol}{bullish_target} with no clear directional bias. "
        report += "Traders should watch for breakouts from key levels for potential long-term trading opportunities."
    
    # Add disclaimer
    report += f"\n\n## Disclaimer\n\n"
    report += "This forecast is based on technical analysis and historical price patterns. "
    report += "It should not be considered as financial advice. "
    report += "Cryptocurrency markets are highly volatile and unpredictable. "
    report += "Past performance is not indicative of future results. "
    report += "Always do your own research and consider your risk tolerance before making investment decisions."
    
    # Save report to file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Saved monthly report to {output_path}")
    
    return output_path

def generate_combined_report(analysis, output_dir=OUTPUT_DIR, filename='bitcoin_technical_analysis.md'):
    """
    Generate a comprehensive report combining short-term and monthly forecasts.
    
    Args:
        analysis (dict): Analysis results
        output_dir (str): Directory to save the report
        filename (str): Output filename
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get trading pair and market type from analysis
    trading_pair = analysis.get('trading_pair', 'BTCUSDT')
    market_type = analysis.get('market_type', DEFAULT_MARKET_TYPE)
    
    # Get appropriate currency symbol and asset name
    currency_symbol = get_currency_symbol(trading_pair, market_type)
    asset_name = get_asset_name(trading_pair, market_type)
    
    # Generate report
    report = f"""# {asset_name} Technical Analysis Report

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

This report provides a comprehensive technical analysis of {asset_name}'s price action, including short-term (1-week) and medium-term (through year-end) forecasts based on various technical indicators and price patterns.

"""
    
    # Add executive summary
    report += f"## Executive Summary\n\n"
    
    short_term_forecast = analysis['short_term_forecast']
    monthly_forecast = analysis['monthly_forecast']
    
    if short_term_forecast:
        current_price = f"{currency_symbol}{short_term_forecast['current_price']:,.2f}"
        short_trend = short_term_forecast['trend']
        short_momentum = short_term_forecast['momentum']
        
        report += f"- **Current Price**: {current_price}\n"
        report += f"- **Short-Term Trend (1-Week)**: {short_trend.capitalize()}\n"
        report += f"- **Short-Term Momentum**: {short_momentum.capitalize()}\n"
        
        if monthly_forecast:
            long_trend = monthly_forecast['trend']
            long_momentum = monthly_forecast['momentum']
            
            report += f"- **Medium-Term Trend (Year-End)**: {long_trend.capitalize()}\n"
            report += f"- **Medium-Term Momentum**: {long_momentum.capitalize()}\n"
        
        # Add price targets summary
        report += f"\n### Price Targets\n\n"
        
        # Short-term targets
        report += "**1-Week Forecast**:\n"
        report += f"- Bullish Scenario: {currency_symbol}{short_term_forecast['targets']['bullish']:,.2f} ({short_term_forecast['probability']['bullish']*100:.0f}% probability)\n"
        report += f"- Base Case: {currency_symbol}{short_term_forecast['targets']['base']:,.2f} ({short_term_forecast['probability']['base']*100:.0f}% probability)\n"
        report += f"- Bearish Scenario: {currency_symbol}{short_term_forecast['targets']['bearish']:,.2f} ({short_term_forecast['probability']['bearish']*100:.0f}% probability)\n"
        
        # Medium-term targets
        if monthly_forecast:
            report += "\n**Year-End Forecast**:\n"
            report += f"- Bullish Scenario: {currency_symbol}{monthly_forecast['year_end_targets']['bullish']:,.2f} ({monthly_forecast['probability']['bullish']*100:.0f}% probability)\n"
            report += f"- Base Case: {currency_symbol}{monthly_forecast['year_end_targets']['base']:,.2f} ({monthly_forecast['probability']['base']*100:.0f}% probability)\n"
            report += f"- Bearish Scenario: {currency_symbol}{monthly_forecast['year_end_targets']['bearish']:,.2f} ({monthly_forecast['probability']['bearish']*100:.0f}% probability)\n"
    
    # Add short-term forecast
    if short_term_forecast:
        report += f"\n\n## Short-Term Forecast (1-Week)\n\n"
        
        # Key levels
        nearest_support = f"{currency_symbol}{short_term_forecast['key_levels']['nearest_support']:,.2f}" if short_term_forecast['key_levels']['nearest_support'] else "None"
        nearest_resistance = f"{currency_symbol}{short_term_forecast['key_levels']['nearest_resistance']:,.2f}" if short_term_forecast['key_levels']['nearest_resistance'] else "None"
        
        report += f"### Key Levels\n\n"
        report += f"- **Nearest Support**: {nearest_support}\n"
        report += f"- **Nearest Resistance**: {nearest_resistance}\n"
        
        # Technical analysis
        report += f"\n### Technical Analysis\n\n"
        
        # Trend
        report += f"#### Trend Analysis\n\n"
        report += f"The current trend is **{short_term_forecast['trend'].upper()}** based on price action and moving average analysis. "
        
        if short_term_forecast['trend'] == 'bullish':
            report += "Price is trading above key moving averages and showing strength. "
        elif short_term_forecast['trend'] == 'bearish':
            report += "Price is trading below key moving averages and showing weakness. "
        else:
            report += "Price is consolidating in a range with no clear direction. "
        
        # Momentum
        report += f"\n\n#### Momentum Analysis\n\n"
        momentum = short_term_forecast['analysis']['momentum']
        
        # RSI
        if momentum['rsi']['value'] is not None:
            rsi_value = momentum['rsi']['value']
            report += f"- **RSI**: {rsi_value:.2f} - "
            if rsi_value > 70:
                report += "Overbought territory, suggesting potential reversal or consolidation.\n"
            elif rsi_value < 30:
                report += "Oversold territory, suggesting potential reversal or bounce.\n"
            elif rsi_value > 50:
                report += "Bullish momentum with strength above the centerline.\n"
            else:
                report += "Bearish momentum with weakness below the centerline.\n"
        
        # MACD
        if momentum['macd']['value'] is not None:
            macd_value = momentum['macd']['value']
            signal_value = momentum['macd']['signal']
            
            report += f"- **MACD**: {macd_value:.2f} (Signal: {signal_value:.2f}) - "
            if momentum['macd']['bullish_cross']:
                report += "Bullish crossover, suggesting potential upward momentum.\n"
            elif momentum['macd']['bearish_cross']:
                report += "Bearish crossover, suggesting potential downward momentum.\n"
            elif macd_value > signal_value:
                report += "MACD above signal line, indicating bullish momentum.\n"
            else:
                report += "MACD below signal line, indicating bearish momentum.\n"
        
        # Short-term conclusion
        report += f"\n### Short-Term Outlook\n\n"
        
        # Generate conclusion based on trend and momentum
        if short_term_forecast['trend'] == 'bullish' and short_term_forecast['momentum'] == 'bullish':
            report += f"The technical analysis indicates a **STRONG BULLISH** outlook for {asset_name} in the next week. "
            report += f"With a strong trend and positive momentum, price could test {currency_symbol}{short_term_forecast['targets']['bullish']:,.2f} if market conditions remain favorable. "
            report += f"Key resistance at {nearest_resistance} should be monitored for potential breakout or rejection."
        
        elif short_term_forecast['trend'] == 'bullish' and short_term_forecast['momentum'] != 'bullish':
            report += f"The technical analysis indicates a **MODERATELY BULLISH** outlook for {asset_name} in the next week. "
            report += f"While the overall trend remains positive, momentum is showing mixed signals. "
            report += f"Price could test {currency_symbol}{short_term_forecast['targets']['base']:,.2f} if momentum improves, but may face resistance at {nearest_resistance}."
        
        elif short_term_forecast['trend'] == 'bearish' and short_term_forecast['momentum'] == 'bearish':
            report += f"The technical analysis indicates a **STRONG BEARISH** outlook for {asset_name} in the next week. "
            report += f"With a downward trend and negative momentum, price could test {currency_symbol}{short_term_forecast['targets']['bearish']:,.2f} if selling pressure continues. "
            report += f"Key support at {nearest_support} should be monitored for potential breakdown or bounce."
        
        elif short_term_forecast['trend'] == 'bearish' and short_term_forecast['momentum'] != 'bearish':
            report += f"The technical analysis indicates a **MODERATELY BEARISH** outlook for {asset_name} in the next week. "
            report += f"While the overall trend is negative, momentum is showing mixed signals. "
            report += f"Price could test {currency_symbol}{short_term_forecast['targets']['base']:,.2f} if downward pressure persists, but may find support at {nearest_support}."
        
        else:  # 'sideways' trend
            report += f"The technical analysis indicates a **NEUTRAL** outlook for {asset_name} in the next week. "
            report += f"Price is likely to consolidate between {currency_symbol}{short_term_forecast['targets']['bearish']:,.2f} and {currency_symbol}{short_term_forecast['targets']['bullish']:,.2f} with no clear directional bias. "
            report += f"Traders should watch for breakouts from this range for potential trading opportunities."
    
    # Add monthly forecast
    if monthly_forecast:
        report += f"\n\n## Medium-Term Forecast (Through Year-End)\n\n"
        
        # Monthly price projections
        report += f"### Monthly Price Projections\n\n"
        report += "| Month | Low | Median | High |\n"
        report += "|-------|-----|--------|------|\n"
        
        # Add monthly targets
        for month, targets in monthly_forecast['monthly_targets'].items():
            report += f"| {month} | {currency_symbol}{targets['low']:,.2f} | {currency_symbol}{targets['median']:,.2f} | {currency_symbol}{targets['high']:,.2f} |\n"
        
        # Medium-term conclusion
        report += f"\n### Medium-Term Outlook\n\n"
        
        # Generate conclusion based on trend and momentum
        if monthly_forecast['trend'] == 'bullish' and monthly_forecast['momentum'] == 'bullish':
            report += f"The technical analysis indicates a **STRONG BULLISH** outlook for {asset_name} through the end of 2025. "
            report += f"With a strong trend and positive momentum, price could rise toward {currency_symbol}{monthly_forecast['year_end_targets']['bullish']:,.2f} by year-end if market conditions remain favorable. "
            report += "Key psychological and resistance levels should be monitored for potential breakouts or rejections along the way."
        
        elif monthly_forecast['trend'] == 'bullish' and monthly_forecast['momentum'] != 'bullish':
            report += f"The technical analysis indicates a **MODERATELY BULLISH** outlook for {asset_name} through the end of 2025. "
            report += f"While the overall trend remains positive, momentum is showing mixed signals. "
            report += f"Price could approach {currency_symbol}{monthly_forecast['year_end_targets']['base']:,.2f} by year-end if momentum improves, but may face challenges at key resistance levels."
        
        elif monthly_forecast['trend'] == 'bearish' and monthly_forecast['momentum'] == 'bearish':
            report += f"The technical analysis indicates a **STRONG BEARISH** outlook for {asset_name} through the end of 2025. "
            report += f"With a downward trend and negative momentum, price could decline toward {currency_symbol}{monthly_forecast['year_end_targets']['bearish']:,.2f} by year-end if selling pressure continues. "
            report += "Key psychological and support levels should be monitored for potential breakdowns or bounces along the way."
        
        elif monthly_forecast['trend'] == 'bearish' and monthly_forecast['momentum'] != 'bearish':
            report += f"The technical analysis indicates a **MODERATELY BEARISH** outlook for {asset_name} through the end of 2025. "
            report += f"While the overall trend is negative, momentum is showing mixed signals. "
            report += f"Price could move toward {currency_symbol}{monthly_forecast['year_end_targets']['base']:,.2f} by year-end if downward pressure persists, but may find support at key levels."
        
        else:  # 'sideways' trend
            report += f"The technical analysis indicates a **NEUTRAL** outlook for {asset_name} through the end of 2025. "
            report += f"Price is likely to fluctuate between {currency_symbol}{monthly_forecast['year_end_targets']['bearish']:,.2f} and {currency_symbol}{monthly_forecast['year_end_targets']['bullish']:,.2f} with no clear directional bias. "
            report += "Traders should watch for breakouts from key levels for potential long-term trading opportunities."
    
    # Add disclaimer
    report += f"\n\n## Disclaimer\n\n"
    report += "This forecast is based on technical analysis and historical price patterns. "
    report += "It should not be considered as financial advice. "
    report += "Cryptocurrency markets are highly volatile and unpredictable. "
    report += "Past performance is not indicative of future results. "
    report += "Always do your own research and consider your risk tolerance before making investment decisions."
    
    # Save report to file - update to use trading pair in filename
    output_filename = filename
    if filename == 'bitcoin_technical_analysis.md':
        # Use trading pair for the filename if default is being used
        pair_for_filename = trading_pair.lower()
        output_filename = f"{pair_for_filename}_technical_analysis.md"
    
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Saved combined report to {output_path}")
    
    return output_path
