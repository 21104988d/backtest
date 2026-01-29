"""
Fetch 6 months of funding rate data and 7 months of price data for backtesting.
This will allow cross-checking with the existing 3-month data.
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASEDIR)

# Hyperliquid API
BASE_URL = "https://api.hyperliquid.xyz/info"

# Date ranges for 6-month backtest
# API currently supports ~210 days of 1h candles, so keep price lookback within that.
END_DATE = datetime(2026, 1, 28, 7, 0, 0)  # Jan 28, 2026 07:00 UTC
START_FUNDING = END_DATE - timedelta(days=180)  # 6 months back (approx)
START_PRICE = END_DATE - timedelta(days=210)  # 7 months back (beta lookback), within API limit

# Rate limiting / retries
RATE_LIMIT_DELAY = 0.25
MAX_RETRIES = 3

def get_all_coins():
    """Get list of all perpetual coins."""
    response = requests.post(BASE_URL, json={"type": "meta"})
    data = response.json()
    coins = [asset['name'] for asset in data['universe']]
    print(f"Found {len(coins)} coins")
    return coins

def fetch_funding_history(coin, start_time, end_time):
    """Fetch funding rate history for a coin."""
    all_data = []
    current_start = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    while current_start < end_ts:
        payload = {
            "type": "fundingHistory",
            "coin": coin,
            "startTime": current_start,
            "endTime": end_ts
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(BASE_URL, json=payload, timeout=30)
                if response.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, list) or not data:
                    return all_data
                all_data.extend(data)
                last_time = max(int(d['time']) for d in data)
                if last_time <= current_start:
                    return all_data
                current_start = last_time + 1
                time.sleep(RATE_LIMIT_DELAY)
                break
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    return all_data
                time.sleep(1)
    
    return all_data

def fetch_candles(coin, start_time, end_time, interval="1h"):
    """Fetch OHLC candles for a coin."""
    all_candles = []
    current_start = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    while current_start < end_ts:
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ts
            }
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(BASE_URL, json=payload, timeout=30)
                if response.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, list) or not data:
                    return all_candles
                all_candles.extend(data)
                last_time = max(int(d['t']) for d in data)
                if last_time <= current_start:
                    return all_candles
                current_start = last_time + 1
                time.sleep(RATE_LIMIT_DELAY)
                break
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    return all_candles
                time.sleep(1)
    
    return all_candles

def main():
    print("="*70)
    print("FETCHING 6-MONTH DATA FOR BACKTEST")
    print("="*70)
    print(f"Funding period: {START_FUNDING} to {END_DATE}")
    print(f"Price period: {START_PRICE} to {END_DATE} (includes beta lookback)")
    print("="*70)
    
    # Get all coins
    coins = get_all_coins()
    
    # =========================================================================
    # STEP 1: Fetch funding data (6 months)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: Fetching 6-month funding history")
    print("="*70)
    
    all_funding = []
    for i, coin in enumerate(coins):
        print(f"[{i+1}/{len(coins)}] Fetching funding for {coin}...", end=" ")
        data = fetch_funding_history(coin, START_FUNDING, END_DATE)
        
        for record in data:
            all_funding.append({
                'coin': coin,
                'datetime': pd.to_datetime(record['time'], unit='ms'),
                'funding_rate': float(record['fundingRate'])
            })
        
        print(f"{len(data)} records")
        time.sleep(RATE_LIMIT_DELAY)
    
    funding_df = pd.DataFrame(all_funding)
    funding_df.to_csv('funding_history_6months.csv', index=False)
    print(f"\n✅ Saved funding_history_6months.csv")
    print(f"   Records: {len(funding_df)}")
    print(f"   Coins: {funding_df['coin'].nunique()}")
    print(f"   Period: {funding_df['datetime'].min()} to {funding_df['datetime'].max()}")
    
    # Get coins that have funding data
    coins_with_funding = funding_df['coin'].unique().tolist()
    
    # =========================================================================
    # STEP 2: Fetch price data (7 months for beta lookback)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: Fetching 7-month price data (includes beta lookback)")
    print("="*70)
    
    all_prices = []
    failed_coins = []
    
    # Always include BTC first
    priority_coins = ['BTC'] + [c for c in coins_with_funding if c != 'BTC']
    
    for i, coin in enumerate(priority_coins):
        print(f"[{i+1}/{len(priority_coins)}] Fetching prices for {coin}...", end=" ")
        candles = fetch_candles(coin, START_PRICE, END_DATE)
        
        if candles:
            for c in candles:
                all_prices.append({
                    'coin': coin,
                    'timestamp': pd.to_datetime(c['t'], unit='ms'),
                    'price': float(c['c'])  # Close price
                })
            print(f"{len(candles)} candles")
        else:
            failed_coins.append(coin)
            print("FAILED")
        
        time.sleep(RATE_LIMIT_DELAY)
    
    price_df = pd.DataFrame(all_prices)
    price_df.to_csv('price_cache_6months.csv', index=False)
    print(f"\n✅ Saved price_cache_6months.csv")
    print(f"   Records: {len(price_df)}")
    if not price_df.empty:
        print(f"   Coins: {price_df['coin'].nunique()}")
        print(f"   Period: {price_df['timestamp'].min()} to {price_df['timestamp'].max()}")
    else:
        print("   Coins: 0")
        print("   Period: N/A")
    
    if failed_coins:
        print(f"\n⚠️  Failed to fetch price for {len(failed_coins)} coins:")
        print(f"   {failed_coins[:20]}{'...' if len(failed_coins) > 20 else ''}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Funding data: {len(funding_df)} records, {funding_df['coin'].nunique()} coins")
    if not price_df.empty:
        print(f"Price data: {len(price_df)} records, {price_df['coin'].nunique()} coins")
    else:
        print("Price data: 0 records, 0 coins")
    print(f"\nFiles created:")
    print(f"  - funding_history_6months.csv")
    print(f"  - price_cache_6months.csv")
    print("="*70)

if __name__ == "__main__":
    main()
