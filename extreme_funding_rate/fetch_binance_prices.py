"""
Fetch historical price data from Binance to fill gaps in price_history.csv

This script:
1. Analyzes current price coverage
2. Maps Hyperliquid symbols to Binance symbols
3. Fetches missing hourly candle data from Binance Futures
4. Merges with existing data and saves to price_history.csv
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"
RATE_LIMIT_DELAY = 0.2  # seconds between requests
MAX_CANDLES_PER_REQUEST = 1500  # Binance limit
PRICE_HISTORY_FILE = "price_history.csv"

# Symbol mapping: Hyperliquid -> Binance
# Most symbols are the same, but some need mapping
SYMBOL_MAPPING = {
    'kPEPE': 'PEPE',
    'kBONK': 'BONK',
    'kFLOKI': 'FLOKI',
    'kSHIB': 'SHIB',
    'kLUNC': 'LUNC',
    '1000PEPE': 'PEPE',
    # Add more mappings as needed
}

# Coins that likely don't exist on Binance (Hyperliquid-only or very new)
SKIP_COINS = {
    'RLB', 'HPOS', 'FRIEND', 'UNIBOT', 'OX', 'SHIA', 'NFTI', 'CANTO',
    'MYRO', 'PANDORA', 'JELLY', 'STRAX', 'ORBS', 'BNT', 'LOOM', 'BLZ',
    'CYBER', 'BADGER', 'ILV', 'NTRN', 'LISTA', 'RDNT', 'CATI',
    # Hyperliquid specific
    'HYPE', 'PURR', 'JEFF', 'SOPH', 'MON', 'RAGE', 'VERT', 'FARM',
    'TARO', 'VINE', 'CATBAL', 'TRUMP', 'MELANIA', 'ANIME', 'BERA',
    'IP', 'TST', 'RESOLV', 'LAYER', 'KAITO', 'ASTER', 'DOOD',
    '0G', 'HEMI', 'YZY', 'STBL', 'USDT0', 'PROMPT', 'MOVE', 'USUAL',
    'SKR', 'WLFI', 'PROVE', 'PZP', 'GAME'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_binance_symbol(hl_symbol):
    """Convert Hyperliquid symbol to Binance symbol"""
    # Check mapping first
    if hl_symbol in SYMBOL_MAPPING:
        return SYMBOL_MAPPING[hl_symbol] + 'USDT'
    return hl_symbol + 'USDT'


def check_binance_symbol_exists(symbol):
    """Check if a symbol exists on Binance Futures"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/exchangeInfo"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        symbols = [s['symbol'] for s in data['symbols']]
        return symbol in symbols
    except Exception as e:
        print(f"Error checking symbol {symbol}: {e}")
        return False


def fetch_binance_klines(symbol, start_time, end_time, interval='1h'):
    """
    Fetch klines (candlestick data) from Binance Futures
    
    Args:
        symbol: Binance symbol (e.g., 'BTCUSDT')
        start_time: Start timestamp in ms
        end_time: End timestamp in ms
        interval: Candle interval ('1h' for hourly)
    
    Returns:
        List of candles or None if error
    """
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': MAX_CANDLES_PER_REQUEST
    }
    
    try:
        response = requests.get(BINANCE_FUTURES_URL, params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400:
            # Symbol might not exist
            return None
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"Request error: {e}")
        return None


def fetch_all_klines(symbol, start_date, end_date):
    """
    Fetch all klines between start and end date, handling pagination
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    all_candles = []
    
    current_start = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    while current_start < end_ms:
        candles = fetch_binance_klines(symbol, current_start, end_ms)
        
        if candles is None or len(candles) == 0:
            break
            
        all_candles.extend(candles)
        
        # Move to next batch
        last_candle_time = candles[-1][0]
        current_start = last_candle_time + 1
        
        # Rate limit
        time.sleep(RATE_LIMIT_DELAY)
        
        # Progress indicator
        if len(all_candles) % 5000 == 0:
            print(f"    Fetched {len(all_candles)} candles...")
    
    if len(all_candles) == 0:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['price'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    return df[['timestamp', 'price', 'volume']]


# =============================================================================
# MAIN LOGIC
# =============================================================================

def main():
    print("=" * 80)
    print("BINANCE PRICE DATA FETCHER")
    print("=" * 80)
    
    # Load existing data
    print("\n1. Loading existing data...")
    
    funding = pd.read_csv('funding_history.csv')
    funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)
    
    price_file = Path(PRICE_HISTORY_FILE)
    if price_file.exists():
        price = pd.read_csv(PRICE_HISTORY_FILE)
        price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
        print(f"   Existing price records: {len(price):,}")
    else:
        price = pd.DataFrame(columns=['timestamp', 'coin', 'price'])
        price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
        print("   No existing price file, starting fresh")
    
    # Analyze coverage
    print("\n2. Analyzing coverage gaps...")
    
    funding_coverage = funding.groupby('coin').agg({
        'datetime': ['min', 'max']
    }).reset_index()
    funding_coverage.columns = ['coin', 'funding_start', 'funding_end']
    
    if len(price) > 0:
        price_coverage = price.groupby('coin').agg({
            'timestamp': ['min', 'max']
        }).reset_index()
        price_coverage.columns = ['coin', 'price_start', 'price_end']
    else:
        price_coverage = pd.DataFrame(columns=['coin', 'price_start', 'price_end'])
    
    coverage = funding_coverage.merge(price_coverage, on='coin', how='left')
    
    # Identify coins to fetch
    coins_to_fetch = []
    
    for _, row in coverage.iterrows():
        coin = row['coin']
        
        # Skip coins not on Binance
        if coin in SKIP_COINS:
            continue
        
        funding_start = row['funding_start']
        
        if pd.isna(row['price_start']):
            # No price data at all
            coins_to_fetch.append({
                'coin': coin,
                'start': funding_start,
                'end': row['funding_end'],
                'type': 'full'
            })
        else:
            # Check if there's a gap before current price data
            price_start = row['price_start']
            if funding_start < price_start - timedelta(hours=1):
                coins_to_fetch.append({
                    'coin': coin,
                    'start': funding_start,
                    'end': price_start - timedelta(hours=1),
                    'type': 'gap'
                })
    
    print(f"   Coins to fetch: {len(coins_to_fetch)}")
    
    # Sort by priority (gap type first, then by date range)
    coins_to_fetch.sort(key=lambda x: (x['type'] != 'gap', x['start']))
    
    # First, check which symbols exist on Binance
    print("\n3. Checking Binance symbol availability...")
    
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        response = requests.get(url, timeout=10)
        binance_symbols = {s['symbol'] for s in response.json()['symbols']}
        print(f"   Binance has {len(binance_symbols)} futures symbols")
    except Exception as e:
        print(f"   Error fetching Binance symbols: {e}")
        binance_symbols = set()
    
    # Filter to coins that exist on Binance
    valid_coins = []
    for item in coins_to_fetch:
        binance_symbol = get_binance_symbol(item['coin'])
        if binance_symbol in binance_symbols:
            item['binance_symbol'] = binance_symbol
            valid_coins.append(item)
        else:
            print(f"   Skipping {item['coin']} - not on Binance ({binance_symbol})")
    
    print(f"\n   Valid coins to fetch: {len(valid_coins)}")
    
    if len(valid_coins) == 0:
        print("\n   No coins to fetch!")
        return
    
    # Show fetch plan
    print("\n4. Fetch plan:")
    total_days = 0
    for item in valid_coins[:20]:  # Show first 20
        days = (item['end'] - item['start']).days
        total_days += days
        print(f"   {item['coin']:<10} {item['binance_symbol']:<12} {str(item['start'])[:10]} to {str(item['end'])[:10]} ({days} days) [{item['type']}]")
    
    if len(valid_coins) > 20:
        for item in valid_coins[20:]:
            days = (item['end'] - item['start']).days
            total_days += days
        print(f"   ... and {len(valid_coins) - 20} more")
    
    print(f"\n   Total: {total_days:,} coin-days = ~{total_days * 24:,} hourly candles")
    
    # Estimate time
    estimated_requests = total_days * 24 / MAX_CANDLES_PER_REQUEST
    estimated_time = estimated_requests * RATE_LIMIT_DELAY / 60
    print(f"   Estimated time: ~{estimated_time:.0f} minutes")
    
    # Confirm
    print("\n5. Starting fetch...")
    
    # Fetch data
    new_data = []
    successful = 0
    failed = 0
    
    for i, item in enumerate(valid_coins):
        coin = item['coin']
        binance_symbol = item['binance_symbol']
        start = item['start']
        end = item['end']
        
        print(f"\n   [{i+1}/{len(valid_coins)}] Fetching {coin} ({binance_symbol})...")
        print(f"       Period: {str(start)[:10]} to {str(end)[:10]}")
        
        df = fetch_all_klines(binance_symbol, start, end)
        
        if df is not None and len(df) > 0:
            df['coin'] = coin
            new_data.append(df)
            print(f"       ✓ Got {len(df):,} candles")
            successful += 1
        else:
            print(f"       ✗ Failed to fetch")
            failed += 1
        
        # Save progress periodically
        if (i + 1) % 10 == 0 and len(new_data) > 0:
            print(f"\n   Saving progress ({i+1} coins processed)...")
            _save_progress(price, new_data)
    
    # Final save
    print("\n6. Saving final data...")
    
    if len(new_data) > 0:
        new_df = pd.concat(new_data, ignore_index=True)
        print(f"   New records fetched: {len(new_df):,}")
        
        # Combine with existing
        combined = pd.concat([price, new_df], ignore_index=True)
        
        # Remove duplicates (same coin, same timestamp)
        combined = combined.drop_duplicates(subset=['coin', 'timestamp'], keep='last')
        
        # Sort
        combined = combined.sort_values(['coin', 'timestamp'])
        
        # Save
        combined.to_csv(PRICE_HISTORY_FILE, index=False)
        print(f"   Total records in {PRICE_HISTORY_FILE}: {len(combined):,}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   New records: {sum(len(df) for df in new_data):,}")


def _save_progress(original_price, new_data_list):
    """Save progress to avoid losing data on errors"""
    if len(new_data_list) == 0:
        return
    
    new_df = pd.concat(new_data_list, ignore_index=True)
    combined = pd.concat([original_price, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['coin', 'timestamp'], keep='last')
    combined = combined.sort_values(['coin', 'timestamp'])
    combined.to_csv(PRICE_HISTORY_FILE, index=False)


if __name__ == "__main__":
    main()
