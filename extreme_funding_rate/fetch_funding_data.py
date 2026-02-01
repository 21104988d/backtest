"""
Fetch hourly funding rate history from Hyperliquid MAINNET for all trading pairs.

This script only fetches data from the mainnet environment, not testnet.
"""
import os
import requests
import pandas as pd
import time
import argparse
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


def fetch_funding_history(coin: str, start_time: int, end_time: int = None, max_retries: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch funding rate history for a specific coin from MAINNET.
    
    Args:
        coin: The coin symbol (e.g., 'BTC', 'ETH')
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds (optional, defaults to now)
        max_retries: Maximum number of retries for rate limit errors
    
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
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers)
            
            # Handle rate limiting with exponential backoff
            if response.status_code == 429:
                wait_time = (2 ** attempt) * 2  # 2, 4, 8, 16, 32 seconds
                print(f"  Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait_time = (2 ** attempt) * 2
                print(f"  Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            print(f"Error fetching funding history for {coin}: {e}")
            return []
        except Exception as e:
            print(f"Error fetching funding history for {coin}: {e}")
            return []
    
    print(f"  Max retries reached for {coin}, skipping this chunk...")
    return []


def fetch_funding_history_chunked(
    coin: str,
    start_time: int,
    end_time: int,
    chunk_hours: int = 500,
    sleep_seconds: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Fetch funding history in time windows to avoid large responses.

    Chunk size is defined in hours (default 500). Each request covers a window
    of up to chunk_hours, which typically yields ~500 hourly records.
    """
    all_records: List[Dict[str, Any]] = []
    window_ms = chunk_hours * 60 * 60 * 1000

    window_start = start_time
    chunk_count = 0
    while window_start < end_time:
        window_end = min(window_start + window_ms, end_time)
        data = fetch_funding_history(coin, window_start, window_end)
        if data:
            all_records.extend(data)
        window_start = window_end
        chunk_count += 1
        # Longer sleep between chunks to avoid rate limits
        time.sleep(sleep_seconds)

    return all_records


def fetch_all_funding_rates(
    symbols: List[str],
    days_back: int = 3650,
    chunk_hours: int = 500,
    output_file: str = 'funding_history.csv',
) -> pd.DataFrame:
    """
    Fetch funding rate history for all symbols.
    
    Args:
        symbols: List of trading pair symbols
        days_back: Number of days of history to fetch
        output_file: File to save incremental progress
    
    Returns:
        DataFrame with columns: timestamp, coin, funding_rate, premium
    """
    # Calculate start time (days_back days ago)
    end_time = int(time.time() * 1000)
    start_time = end_time - (days_back * 24 * 60 * 60 * 1000)
    
    all_data = []
    
    # Check if we have existing data to resume from
    completed_symbols = set()
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            completed_symbols = set(existing_df['coin'].unique())
            all_data = existing_df.to_dict('records')
            print(f"  Resuming from existing file with {len(completed_symbols)} completed symbols")
        except Exception as e:
            print(f"  Could not load existing file: {e}")
    
    for i, symbol in enumerate(symbols):
        if symbol in completed_symbols:
            print(f"Skipping {symbol} ({i+1}/{len(symbols)}) - already fetched")
            continue
            
        print(f"Fetching funding history for {symbol} ({i+1}/{len(symbols)})...")
        
        funding_history = fetch_funding_history_chunked(
            symbol,
            start_time,
            end_time,
            chunk_hours=chunk_hours,
        )
        
        for record in funding_history:
            all_data.append({
                'timestamp': record.get('time', 0),
                'coin': symbol,
                'funding_rate': float(record.get('fundingRate', 0)),
                'premium': float(record.get('premium', 0)) if record.get('premium') else None
            })
        
        print(f"  → Fetched {len(funding_history)} records for {symbol}")
        
        # Save progress every 10 symbols
        if (i + 1) % 10 == 0:
            df_temp = pd.DataFrame(all_data)
            if not df_temp.empty:
                df_temp['datetime'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
                df_temp.to_csv(output_file, index=False)
                print(f"  💾 Progress saved: {len(df_temp):,} records")
        
        # Be respectful with API rate limits - longer delay between symbols
        time.sleep(1.5)
    
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
    
    parser = argparse.ArgumentParser(description="Fetch Hyperliquid funding history")
    parser.add_argument("--days-back", type=int, default=3650, help="How many days of history to fetch")
    parser.add_argument("--chunk-hours", type=int, default=500, help="Chunk window size in hours")
    parser.add_argument("--output", type=str, default="funding_history.csv", help="Output filename")
    args = parser.parse_args()

    output_file = args.output
    
    # Fetch funding rate history
    print("\nStep 2: Fetching funding rate history...")
    df = fetch_all_funding_rates(
        symbols,
        days_back=args.days_back,
        chunk_hours=args.chunk_hours,
        output_file=output_file,
    )
    
    if df.empty:
        print("No data fetched. Exiting.")
        return
    
    # Final save to CSV
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
