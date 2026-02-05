"""
Generate daily OHLC (Open, High, Low, Close) data from existing hourly price data.
This data is needed for proper stop loss calculation using intraday highs/lows.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def main():
    print("=" * 80)
    print("GENERATING DAILY OHLC DATA FROM HOURLY PRICES")
    print("=" * 80)
    
    # Load existing hourly price data
    print("\nLoading price_history.csv...")
    df = pd.read_csv('price_history.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    print(f"Loaded {len(df):,} hourly records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique coins: {df['coin'].nunique()}")
    
    # Check for missing hours per day per coin
    print("\nChecking data quality...")
    hours_per_day = df.groupby(['date', 'coin']).size()
    incomplete_days = hours_per_day[hours_per_day < 24]
    print(f"Days with incomplete data (< 24 hours): {len(incomplete_days)}")
    
    # Sort by timestamp to ensure correct open/close calculation
    df = df.sort_values(['coin', 'timestamp'])
    
    # Calculate daily OHLC from hourly data
    print("\nCalculating daily OHLC...")
    daily_ohlc = df.groupby(['date', 'coin']).agg(
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('volume', 'sum'),
        hours_count=('price', 'count')
    ).reset_index()
    
    # Calculate additional fields for backtest
    daily_ohlc['daily_return'] = (daily_ohlc['close'] / daily_ohlc['open'] - 1) * 100
    
    # For stop loss calculation:
    # Long position: max adverse excursion = (low - open) / open * 100
    # Short position: max adverse excursion = (high - open) / open * 100
    daily_ohlc['long_mae'] = (daily_ohlc['low'] / daily_ohlc['open'] - 1) * 100  # Most negative for longs
    daily_ohlc['short_mae'] = (daily_ohlc['high'] / daily_ohlc['open'] - 1) * 100  # Most negative for shorts (inverted)
    
    # Max favorable excursion
    daily_ohlc['long_mfe'] = (daily_ohlc['high'] / daily_ohlc['open'] - 1) * 100
    daily_ohlc['short_mfe'] = (daily_ohlc['open'] / daily_ohlc['low'] - 1) * 100  # Inverted for short
    
    print(f"Generated {len(daily_ohlc):,} daily OHLC records")
    
    # Reorder columns
    daily_ohlc = daily_ohlc[['date', 'coin', 'open', 'high', 'low', 'close', 'volume', 
                             'hours_count', 'daily_return', 'long_mae', 'short_mae', 
                             'long_mfe', 'short_mfe']]
    
    # Sort by date and coin
    daily_ohlc = daily_ohlc.sort_values(['date', 'coin'])
    
    # Save to CSV
    output_file = 'daily_ohlc.csv'
    daily_ohlc.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved to {output_file}")
    print(f"  Total rows: {len(daily_ohlc):,}")
    print(f"  Date range: {daily_ohlc['date'].min()} to {daily_ohlc['date'].max()}")
    print(f"  Unique coins: {daily_ohlc['coin'].nunique()}")
    print(f"  Unique days: {daily_ohlc['date'].nunique()}")
    
    # Show statistics
    print("\n" + "=" * 80)
    print("DATA STATISTICS")
    print("=" * 80)
    
    print("\nDaily Return Statistics:")
    print(f"  Mean:   {daily_ohlc['daily_return'].mean():.3f}%")
    print(f"  Std:    {daily_ohlc['daily_return'].std():.3f}%")
    print(f"  Min:    {daily_ohlc['daily_return'].min():.2f}%")
    print(f"  Max:    {daily_ohlc['daily_return'].max():.2f}%")
    
    print("\nLong Position MAE (Max Adverse Excursion):")
    print(f"  Mean:   {daily_ohlc['long_mae'].mean():.3f}%")
    print(f"  5th %:  {daily_ohlc['long_mae'].quantile(0.05):.2f}%")
    print(f"  Min:    {daily_ohlc['long_mae'].min():.2f}%")
    
    print("\nShort Position MAE (Max Adverse Excursion):")
    print(f"  Mean:   {daily_ohlc['short_mae'].mean():.3f}%")
    print(f"  95th %: {daily_ohlc['short_mae'].quantile(0.95):.2f}%")
    print(f"  Max:    {daily_ohlc['short_mae'].max():.2f}%")
    
    # Show sample data
    print("\n" + "=" * 80)
    print("SAMPLE DATA (BTC)")
    print("=" * 80)
    btc_sample = daily_ohlc[daily_ohlc['coin'] == 'BTC'].head(10)
    print(btc_sample.to_string(index=False))
    
    # Check for any coins with missing days
    print("\n" + "=" * 80)
    print("DATA COVERAGE CHECK")
    print("=" * 80)
    
    all_dates = set(daily_ohlc['date'].unique())
    coin_coverage = daily_ohlc.groupby('coin')['date'].nunique()
    max_days = len(all_dates)
    
    coins_missing_days = coin_coverage[coin_coverage < max_days]
    print(f"\nCoins with incomplete coverage (< {max_days} days): {len(coins_missing_days)}")
    if len(coins_missing_days) > 0:
        print("Top 10 coins with most missing days:")
        missing_counts = max_days - coins_missing_days
        print(missing_counts.nlargest(10))

if __name__ == "__main__":
    main()
