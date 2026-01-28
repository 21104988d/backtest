"""
Analyze extreme funding rates and evaluate performance after extreme events.

Data source: Hyperliquid MAINNET only.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict


def load_funding_data(filepath: str = 'funding_history.csv') -> pd.DataFrame:
    """Load funding rate history data."""
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['datetime', 'coin']).reset_index(drop=True)
    return df


def identify_extreme_funding_per_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each hour, identify coins with the most extreme funding rates.
    
    Returns:
        DataFrame with columns: hour, coin, funding_rate, rank, extreme_type
    """
    # Round to nearest hour
    df['hour'] = df['datetime'].dt.floor('h')
    
    # Get the latest funding rate for each coin in each hour
    hourly_df = df.groupby(['hour', 'coin']).agg({
        'funding_rate': 'last',
        'timestamp': 'last'
    }).reset_index()
    
    # For each hour, rank coins by funding rate (both positive and negative extremes)
    extreme_records = []
    
    for hour in hourly_df['hour'].unique():
        hour_data = hourly_df[hourly_df['hour'] == hour].copy()
        
        # Sort by funding rate and get top/bottom N
        hour_data = hour_data.sort_values('funding_rate')
        
        # Most negative funding rates (top 5)
        most_negative = hour_data.head(5).copy()
        most_negative['extreme_type'] = 'negative'
        most_negative['rank'] = range(1, len(most_negative) + 1)
        
        # Most positive funding rates (top 5)
        most_positive = hour_data.tail(5).copy()
        most_positive['extreme_type'] = 'positive'
        most_positive['rank'] = range(1, len(most_positive) + 1)
        
        extreme_records.append(most_negative)
        extreme_records.append(most_positive)
    
    extreme_df = pd.concat(extreme_records, ignore_index=True)
    extreme_df = extreme_df.sort_values('hour').reset_index(drop=True)
    
    return extreme_df


def fetch_price_data(coin: str, start_time: int, end_time: int, interval: str = '1h') -> pd.DataFrame:
    """
    Fetch price data from Hyperliquid MAINNET for performance calculation.
    
    Args:
        coin: Trading pair symbol
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
        interval: Candle interval (default: '1h')
    """
    import requests
    
    url = "https://api.hyperliquid.xyz/info"  # Mainnet API endpoint
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
            if not df.empty and 't' in df.columns:
                df['datetime'] = pd.to_datetime(df['t'], unit='ms')
                df['open'] = df['o'].astype(float)
                df['high'] = df['h'].astype(float)
                df['low'] = df['l'].astype(float)
                df['close'] = df['c'].astype(float)
                df['volume'] = df['v'].astype(float)
                return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching price data for {coin}: {e}")
        return pd.DataFrame()


def calculate_performance_after_extreme(extreme_df: pd.DataFrame, funding_df: pd.DataFrame, 
                                        hours_forward: int = 1) -> pd.DataFrame:
    """
    Calculate price performance N hours after extreme funding events.
    
    Args:
        extreme_df: DataFrame with extreme funding events
        funding_df: Original funding data
        hours_forward: Number of hours to measure forward performance
    
    Returns:
        DataFrame with performance metrics added
    """
    import requests
    import time
    
    results = []
    
    # Group by unique coin-hour combinations to avoid redundant API calls
    unique_events = extreme_df[['hour', 'coin']].drop_duplicates()
    
    print(f"Calculating performance for {len(unique_events)} unique events...")
    
    for idx, row in unique_events.iterrows():
        coin = row['coin']
        event_time = row['hour']
        
        # Calculate time window
        start_time = int(event_time.timestamp() * 1000)
        end_time = int((event_time + timedelta(hours=hours_forward + 1)).timestamp() * 1000)
        
        # Fetch price data
        price_df = fetch_price_data(coin, start_time, end_time, interval='1h')
        
        if not price_df.empty and len(price_df) >= 2:
            initial_price = price_df.iloc[0]['close']
            final_price = price_df.iloc[min(hours_forward, len(price_df)-1)]['close']
            
            # Calculate returns
            pct_return = ((final_price - initial_price) / initial_price) * 100
            
            # Store results
            event_data = extreme_df[(extreme_df['hour'] == event_time) & 
                                   (extreme_df['coin'] == coin)].iloc[0].to_dict()
            event_data['initial_price'] = initial_price
            event_data['final_price'] = final_price
            event_data['return_pct'] = pct_return
            results.append(event_data)
        
        # Rate limiting
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(unique_events)} events...")
            time.sleep(0.5)
    
    performance_df = pd.DataFrame(results)
    return performance_df


def analyze_extreme_performance(performance_df: pd.DataFrame) -> None:
    """Generate analysis and statistics on extreme funding performance."""
    
    print("\n" + "="*80)
    print("EXTREME FUNDING RATE PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Overall statistics
    print("\n1. Overall Performance Statistics:")
    print("-" * 80)
    print(f"Total events analyzed: {len(performance_df)}")
    
    if 'return_pct' in performance_df.columns:
        print(f"Average return: {performance_df['return_pct'].mean():.4f}%")
        print(f"Median return: {performance_df['return_pct'].median():.4f}%")
        print(f"Std deviation: {performance_df['return_pct'].std():.4f}%")
        print(f"Win rate: {(performance_df['return_pct'] > 0).sum() / len(performance_df) * 100:.2f}%")
    
    # Performance by extreme type
    print("\n2. Performance by Extreme Type:")
    print("-" * 80)
    
    for extreme_type in ['negative', 'positive']:
        subset = performance_df[performance_df['extreme_type'] == extreme_type]
        if not subset.empty and 'return_pct' in subset.columns:
            print(f"\n{extreme_type.upper()} Funding Rate Extremes:")
            print(f"  Count: {len(subset)}")
            print(f"  Avg Return: {subset['return_pct'].mean():.4f}%")
            print(f"  Median Return: {subset['return_pct'].median():.4f}%")
            print(f"  Win Rate: {(subset['return_pct'] > 0).sum() / len(subset) * 100:.2f}%")
            print(f"  Best: {subset['return_pct'].max():.4f}%")
            print(f"  Worst: {subset['return_pct'].min():.4f}%")
    
    # Performance by rank
    print("\n3. Performance by Extremity Rank:")
    print("-" * 80)
    
    if 'rank' in performance_df.columns and 'return_pct' in performance_df.columns:
        for rank in sorted(performance_df['rank'].unique()):
            subset = performance_df[performance_df['rank'] == rank]
            print(f"\nRank {rank} (Most Extreme):")
            print(f"  Avg Return: {subset['return_pct'].mean():.4f}%")
            print(f"  Win Rate: {(subset['return_pct'] > 0).sum() / len(subset) * 100:.2f}%")
    
    # Top performing coins
    print("\n4. Top Performing Coins (by average return):")
    print("-" * 80)
    
    if 'return_pct' in performance_df.columns:
        coin_performance = performance_df.groupby('coin')['return_pct'].agg(['mean', 'count', 'std'])
        coin_performance = coin_performance[coin_performance['count'] >= 3]  # At least 3 events
        coin_performance = coin_performance.sort_values('mean', ascending=False)
        print(coin_performance.head(10))


def save_extreme_events(extreme_df: pd.DataFrame, filename: str = 'extreme_funding_events.csv'):
    """Save extreme funding events to CSV."""
    extreme_df.to_csv(filename, index=False)
    print(f"\n✓ Extreme funding events saved to {filename}")


def main():
    """Main execution function."""
    print("="*80)
    print("ANALYZING EXTREME FUNDING RATES")
    print("="*80)
    
    # Load data
    print("\nStep 1: Loading funding rate data...")
    df = load_funding_data()
    print(f"Loaded {len(df):,} funding rate records")
    
    # Identify extreme funding rates
    print("\nStep 2: Identifying extreme funding rates per hour...")
    extreme_df = identify_extreme_funding_per_hour(df)
    print(f"Identified {len(extreme_df):,} extreme funding events")
    
    # Save extreme events
    save_extreme_events(extreme_df)
    
    # Calculate performance (optional - requires price data)
    print("\nStep 3: Calculating performance after extreme events...")
    print("Note: This requires fetching price data and may take a while...")
    
    user_input = input("Do you want to fetch price data and calculate performance? (y/n): ")
    
    if user_input.lower() == 'y':
        performance_df = calculate_performance_after_extreme(extreme_df, df, hours_forward=1)
        
        if not performance_df.empty:
            # Save performance data
            performance_df.to_csv('extreme_funding_performance.csv', index=False)
            print("✓ Performance data saved to extreme_funding_performance.csv")
            
            # Analyze performance
            analyze_extreme_performance(performance_df)
    else:
        print("Skipping performance calculation.")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
