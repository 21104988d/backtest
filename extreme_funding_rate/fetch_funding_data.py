"""
Fetch hourly funding rate history from Hyperliquid MAINNET for all trading pairs.

This script only fetches data from the mainnet environment, not testnet.
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json


def get_all_perpetuals() -> List[str]:
    """Get all available perpetual markets from Hyperliquid MAINNET."""
    url = "https://api.hyperliquid.xyz/info"  # Mainnet API endpoint
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "type": "meta"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Extract perpetual symbols
        universe = data.get('universe', [])
        symbols = [item['name'] for item in universe if item.get('name')]
        
        print(f"Found {len(symbols)} perpetual markets")
        return symbols
    except Exception as e:
        print(f"Error fetching perpetuals: {e}")
        return []


def fetch_funding_history(coin: str, start_time: int, end_time: int = None) -> List[Dict[str, Any]]:
    """
    Fetch funding rate history for a specific coin from MAINNET.
    
    Args:
        coin: The coin symbol (e.g., 'BTC', 'ETH')
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds (optional, defaults to now)
    
    Returns:
        List of funding rate records from mainnet
    """
    url = "https://api.hyperliquid.xyz/info"  # Mainnet API endpoint
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "type": "fundingHistory",
        "coin": coin,
        "startTime": start_time
    }
    
    if end_time:
        payload["endTime"] = end_time
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Error fetching funding history for {coin}: {e}")
        return []


def fetch_all_funding_rates(symbols: List[str], days_back: int = 30) -> pd.DataFrame:
    """
    Fetch funding rate history for all symbols.
    
    Args:
        symbols: List of trading pair symbols
        days_back: Number of days of history to fetch
    
    Returns:
        DataFrame with columns: timestamp, coin, funding_rate, premium
    """
    # Calculate start time (days_back days ago)
    end_time = int(time.time() * 1000)
    start_time = end_time - (days_back * 24 * 60 * 60 * 1000)
    
    all_data = []
    
    for i, symbol in enumerate(symbols):
        print(f"Fetching funding history for {symbol} ({i+1}/{len(symbols)})...")
        
        funding_history = fetch_funding_history(symbol, start_time, end_time)
        
        for record in funding_history:
            all_data.append({
                'timestamp': record.get('time', 0),
                'coin': symbol,
                'funding_rate': float(record.get('fundingRate', 0)),
                'premium': float(record.get('premium', 0)) if record.get('premium') else None
            })
        
        # Be respectful with API rate limits
        time.sleep(0.2)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    if not df.empty:
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values(['datetime', 'coin'])
        df = df.reset_index(drop=True)
    
    return df


def main():
    """Main execution function."""
    print("="*60)
    print("Fetching Hyperliquid Funding Rate History")
    print("="*60)
    
    # Get all perpetual markets
    print("\nStep 1: Fetching available perpetual markets...")
    symbols = get_all_perpetuals()
    
    if not symbols:
        print("No symbols found. Exiting.")
        return
    
    print(f"\nFound symbols: {', '.join(symbols[:10])}..." if len(symbols) > 10 else f"\nFound symbols: {', '.join(symbols)}")
    
    # Fetch funding rate history
    print("\nStep 2: Fetching funding rate history...")
    days_back = 30  # Using 30 days for comprehensive backtest
    df = fetch_all_funding_rates(symbols, days_back=days_back)
    
    if df.empty:
        print("No data fetched. Exiting.")
        return
    
    # Save to CSV
    output_file = 'funding_history.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Data saved to {output_file}")
    
    # Display summary statistics
    print("\n" + "="*60)
    print("Data Summary")
    print("="*60)
    print(f"Total records: {len(df):,}")
    print(f"Unique coins: {df['coin'].nunique()}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"\nFunding Rate Statistics:")
    print(df['funding_rate'].describe())
    print(f"\nSample data:")
    print(df.head(10))


if __name__ == "__main__":
    main()
