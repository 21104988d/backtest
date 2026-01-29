#!/usr/bin/env python3
"""
Fetch 3 months of funding data and 4 months of price data.
- Funding: 3 months for backtest period
- Price: 4 months (3 months backtest + 1 month lookback for beta calculation)

Uses rate limiting to avoid API errors.
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import sys
import os

# Configuration
BACKTEST_MONTHS = 3
BETA_LOOKBACK_DAYS = 30
RATE_LIMIT_DELAY = 0.25  # 250ms between requests

def get_all_perpetuals():
    """Get all available perpetual markets from Hyperliquid MAINNET."""
    url = "https://api.hyperliquid.xyz/info"
    payload = {"type": "meta"}
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        universe = data.get('universe', [])
        symbols = [item['name'] for item in universe if item.get('name')]
        
        print(f"Found {len(symbols)} perpetual markets")
        return sorted(symbols)
    except Exception as e:
        print(f"Error fetching perpetuals: {e}")
        return []

def fetch_funding_history(coin, start_time_ms, end_time_ms, max_retries=3):
    """Fetch funding rate history for a coin."""
    url = "https://api.hyperliquid.xyz/info"
    payload = {
        "type": "fundingHistory",
        "coin": coin,
        "startTime": start_time_ms,
        "endTime": end_time_ms
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 429:
                wait_time = 5 * (attempt + 1)
                print(f" [Rate limited, waiting {wait_time}s]", end='', flush=True)
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return []
    return []

def fetch_candles(coin, start_time, end_time, max_retries=3):
    """Fetch hourly candles for a coin from Hyperliquid."""
    url = 'https://api.hyperliquid.xyz/info'
    
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    req_data = {
        'type': 'candleSnapshot',
        'req': {
            'coin': coin,
            'interval': '1h',
            'startTime': start_ms,
            'endTime': end_ms
        }
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=req_data, timeout=30)
            
            if resp.status_code == 429:
                wait_time = 5 * (attempt + 1)
                print(f" [Rate limited, waiting {wait_time}s]", end='', flush=True)
                time.sleep(wait_time)
                continue
            
            data = resp.json()
            
            if isinstance(data, list) and len(data) > 0:
                records = []
                for candle in data:
                    ts = pd.to_datetime(candle['t'], unit='ms')
                    close_price = float(candle['c'])
                    records.append({'timestamp': ts, 'coin': coin, 'price': close_price})
                return records
            return []
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return []
    return []

def main():
    print("=" * 70)
    print("FETCHING 3 MONTHS OF FUNDING DATA + 4 MONTHS OF PRICE DATA")
    print("=" * 70)
    
    # Calculate date ranges
    now = datetime.utcnow()
    
    # Funding: 3 months back
    funding_end = now.replace(minute=0, second=0, microsecond=0)
    funding_start = funding_end - timedelta(days=BACKTEST_MONTHS * 30)
    
    # Price: 4 months back (3 months backtest + 1 month beta lookback)
    price_end = funding_end + timedelta(hours=1)  # Include current hour
    price_start = funding_start - timedelta(days=BETA_LOOKBACK_DAYS)
    
    print(f"\nFunding period: {funding_start} to {funding_end} ({BACKTEST_MONTHS} months)")
    print(f"Price period: {price_start} to {price_end} (includes {BETA_LOOKBACK_DAYS} days lookback)")
    
    # Step 1: Get all coins
    print("\n" + "=" * 70)
    print("STEP 1: FETCHING AVAILABLE COINS")
    print("=" * 70)
    all_coins = get_all_perpetuals()
    
    if not all_coins:
        print("ERROR: Could not fetch coin list")
        return
    
    # Ensure BTC is included for hedging
    if 'BTC' not in all_coins:
        all_coins.append('BTC')
    
    print(f"Total coins: {len(all_coins)}")
    
    # Step 2: Fetch funding data
    print("\n" + "=" * 70)
    print("STEP 2: FETCHING 3 MONTHS OF FUNDING DATA")
    print("=" * 70)
    
    funding_start_ms = int(funding_start.timestamp() * 1000)
    funding_end_ms = int(funding_end.timestamp() * 1000)
    
    all_funding = []
    success_count = 0
    fail_count = 0
    
    for i, coin in enumerate(all_coins):
        progress = (i + 1) / len(all_coins) * 100
        bar_len = 40
        filled = int(bar_len * (i + 1) / len(all_coins))
        bar = '█' * filled + '░' * (bar_len - filled)
        
        print(f"\r[{bar}] {progress:5.1f}% | {i+1}/{len(all_coins)} | {coin:<12}", end='', flush=True)
        
        records = fetch_funding_history(coin, funding_start_ms, funding_end_ms)
        
        if records:
            for r in records:
                all_funding.append({
                    'timestamp': r.get('time', 0),
                    'coin': coin,
                    'funding_rate': float(r.get('fundingRate', 0)),
                    'premium': float(r.get('premium', 0)) if r.get('premium') else None
                })
            success_count += 1
            print(f" | {len(records)} records", end='', flush=True)
        else:
            fail_count += 1
            print(f" | FAILED", end='', flush=True)
        
        time.sleep(RATE_LIMIT_DELAY)
    
    print(f"\n\n✓ Funding data: {success_count} coins succeeded, {fail_count} failed")
    
    # Create and save funding DataFrame
    funding_df = pd.DataFrame(all_funding)
    if not funding_df.empty:
        funding_df['datetime'] = pd.to_datetime(funding_df['timestamp'], unit='ms')
        funding_df = funding_df.sort_values(['datetime', 'coin']).reset_index(drop=True)
        funding_df.to_csv('funding_history.csv', index=False)
        print(f"💾 Saved funding_history.csv: {len(funding_df):,} records, {funding_df['coin'].nunique()} coins")
        print(f"   Date range: {funding_df['datetime'].min()} to {funding_df['datetime'].max()}")
    else:
        print("ERROR: No funding data fetched!")
        return
    
    # Step 3: Get coins that have funding data
    coins_with_funding = funding_df['coin'].unique().tolist()
    if 'BTC' not in coins_with_funding:
        coins_with_funding.append('BTC')
    
    print(f"\nCoins with funding data: {len(coins_with_funding)}")
    
    # Step 4: Fetch price data
    print("\n" + "=" * 70)
    print("STEP 3: FETCHING 4 MONTHS OF PRICE DATA")
    print("=" * 70)
    
    all_prices = []
    success_count = 0
    fail_count = 0
    
    for i, coin in enumerate(coins_with_funding):
        progress = (i + 1) / len(coins_with_funding) * 100
        bar_len = 40
        filled = int(bar_len * (i + 1) / len(coins_with_funding))
        bar = '█' * filled + '░' * (bar_len - filled)
        
        print(f"\r[{bar}] {progress:5.1f}% | {i+1}/{len(coins_with_funding)} | {coin:<12}", end='', flush=True)
        
        records = fetch_candles(coin, price_start, price_end)
        
        if records:
            all_prices.extend(records)
            success_count += 1
            print(f" | {len(records)} candles", end='', flush=True)
        else:
            fail_count += 1
            print(f" | FAILED", end='', flush=True)
        
        time.sleep(RATE_LIMIT_DELAY)
    
    print(f"\n\n✓ Price data: {success_count} coins succeeded, {fail_count} failed")
    
    # Create and save price DataFrame
    price_df = pd.DataFrame(all_prices)
    if not price_df.empty:
        price_df = price_df.sort_values(['timestamp', 'coin']).reset_index(drop=True)
        price_df.to_csv('price_cache_with_beta_history.csv', index=False)
        print(f"💾 Saved price_cache_with_beta_history.csv: {len(price_df):,} records, {price_df['coin'].nunique()} coins")
        print(f"   Date range: {price_df['timestamp'].min()} to {price_df['timestamp'].max()}")
    else:
        print("ERROR: No price data fetched!")
        return
    
    # Summary
    print("\n" + "=" * 70)
    print("DATA FETCH COMPLETE")
    print("=" * 70)
    print(f"\n📊 Funding Data:")
    print(f"   Records: {len(funding_df):,}")
    print(f"   Coins: {funding_df['coin'].nunique()}")
    print(f"   Period: {funding_df['datetime'].min()} to {funding_df['datetime'].max()}")
    
    print(f"\n📊 Price Data:")
    print(f"   Records: {len(price_df):,}")
    print(f"   Coins: {price_df['coin'].nunique()}")
    print(f"   Period: {price_df['timestamp'].min()} to {price_df['timestamp'].max()}")
    
    # Check for coins in funding but not in price
    funding_coins = set(funding_df['coin'].unique())
    price_coins = set(price_df['coin'].unique())
    missing = funding_coins - price_coins
    
    if missing:
        print(f"\n⚠️ {len(missing)} coins in funding but not in price data:")
        print(f"   {sorted(list(missing))[:20]}...")

if __name__ == "__main__":
    main()
