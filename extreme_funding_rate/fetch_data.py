#!/usr/bin/env python3
"""
Unified data fetcher.

Fetch funding + price data for a configurable backtest period and write CSVs
for fast re-runs.
"""
import argparse
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from config import load_config

BASE_URL = "https://api.hyperliquid.xyz/info"
RATE_LIMIT_DELAY = 0.25
MAX_RETRIES = 3
FUNDING_PAGE_LIMIT = 500
CANDLE_PAGE_HOURS = 500  # 500 hourly candles per request
MAX_CANDLE_DAYS = 210    # API candle availability limit (~210 days)


def get_all_perpetuals():
    """Get all available perpetual markets from Hyperliquid MAINNET."""
    payload = {"type": "meta"}
    try:
        response = requests.post(BASE_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        universe = data.get('universe', [])
        symbols = [item['name'] for item in universe if item.get('name')]
        print(f"Found {len(symbols)} perpetual markets")
        return sorted(symbols)
    except Exception as e:
        print(f"Error fetching perpetuals: {e}")
        return []


def fetch_funding_history_paginated(coin, start_ms, end_ms):
    """Fetch funding rate history, paging in 500-record chunks."""
    all_records = []
    current_end = end_ms

    while current_end > start_ms:
        payload = {
            "type": "fundingHistory",
            "coin": coin,
            "startTime": start_ms,
            "endTime": current_end
        }

        data = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(BASE_URL, json=payload, timeout=30)
                if response.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                response.raise_for_status()
                data = response.json()
                break
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    data = []
                time.sleep(1)

        if not isinstance(data, list) or not data:
            break

        all_records.extend(data)

        times = [int(d['time']) for d in data if 'time' in d]
        if not times:
            break
        earliest = min(times)

        if earliest <= start_ms:
            break

        # Move window backward for next 500 records
        current_end = earliest - 1
        time.sleep(RATE_LIMIT_DELAY)

        if len(data) < FUNDING_PAGE_LIMIT:
            break

    return all_records


def fetch_candles_chunked(coin, start_time, end_time, chunk_hours=CANDLE_PAGE_HOURS):
    """Fetch hourly candles in chunked windows."""
    records = []
    current_start = start_time

    while current_start < end_time:
        chunk_end = min(current_start + timedelta(hours=chunk_hours), end_time)

        req_data = {
            'type': 'candleSnapshot',
            'req': {
                'coin': coin,
                'interval': '1h',
                'startTime': int(current_start.timestamp() * 1000),
                'endTime': int(chunk_end.timestamp() * 1000)
            }
        }

        data = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(BASE_URL, json=req_data, timeout=30)
                if resp.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    data = []
                time.sleep(1)

        if not isinstance(data, list) or not data:
            current_start = chunk_end
            time.sleep(RATE_LIMIT_DELAY)
            continue

        for candle in data:
            ts = pd.to_datetime(candle['t'], unit='ms')
            close_price = float(candle['c'])
            records.append({'timestamp': ts, 'coin': coin, 'price': close_price})

        current_start = chunk_end
        time.sleep(RATE_LIMIT_DELAY)

    return records


def resolve_output_names(months, beta_lookback_days, prefix=None):
    if prefix:
        return f"{prefix}_funding.csv", f"{prefix}_prices.csv"
    if months == 3 and beta_lookback_days == 30:
        return "funding_history.csv", "price_cache_with_beta_history.csv"
    return f"funding_history_{months}months.csv", f"price_cache_{months}months.csv"


def main():
    parser = argparse.ArgumentParser(description="Fetch funding + price data with chunked requests.")
    parser.add_argument("--months", type=int, default=None, help="Backtest period in months (default: from .env MONTHS_BACK)")
    parser.add_argument("--beta-lookback-days", type=int, default=30, help="Beta lookback days (default: 30)")
    parser.add_argument("--end-date", type=str, default=None, help="End date UTC, e.g. 2026-01-28T07:00:00")
    parser.add_argument("--prefix", type=str, default=None, help="Optional output prefix")
    args = parser.parse_args()

    config = load_config()
    if args.months is None:
        args.months = int(config.get('months_back', 3))

    global RATE_LIMIT_DELAY
    RATE_LIMIT_DELAY = float(config.get('fetch_delay', RATE_LIMIT_DELAY))

    end_date = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    if args.end_date:
        end_date = datetime.fromisoformat(args.end_date)

    funding_end = end_date
    funding_start = funding_end - timedelta(days=args.months * 30)
    price_start = funding_start - timedelta(days=args.beta_lookback_days)
    price_end = funding_end + timedelta(hours=1)

    earliest_price_start = funding_end - timedelta(days=MAX_CANDLE_DAYS)
    if price_start < earliest_price_start:
        print(f"⚠️ Price data limited to last {MAX_CANDLE_DAYS} days by API.")
        print(f"   Requested price start: {price_start}")
        price_start = earliest_price_start
        print(f"   Using price start: {price_start}")

    funding_out, price_out = resolve_output_names(args.months, args.beta_lookback_days, args.prefix)

    print("=" * 70)
    print("FETCHING FUNDING + PRICE DATA")
    print("=" * 70)
    print(f"Funding period: {funding_start} to {funding_end} ({args.months} months)")
    print(f"Price period: {price_start} to {price_end} (lookback {args.beta_lookback_days} days)")
    print(f"Output: {funding_out}, {price_out}")

    all_coins = get_all_perpetuals()
    if not all_coins:
        print("ERROR: Could not fetch coin list")
        return
    if 'BTC' not in all_coins:
        all_coins.append('BTC')

    # Funding data
    print("\n" + "=" * 70)
    print("STEP 1: FETCHING FUNDING HISTORY")
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

        records = fetch_funding_history_paginated(coin, funding_start_ms, funding_end_ms)
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
            print(" | FAILED", end='', flush=True)

    print(f"\n\n✓ Funding data: {success_count} coins succeeded, {fail_count} failed")

    funding_df = pd.DataFrame(all_funding)
    if funding_df.empty:
        print("ERROR: No funding data fetched!")
        return

    funding_df['datetime'] = pd.to_datetime(funding_df['timestamp'], unit='ms')
    funding_df = funding_df.sort_values(['datetime', 'coin']).reset_index(drop=True)
    funding_df.to_csv(funding_out, index=False)
    print(f"💾 Saved {funding_out}: {len(funding_df):,} records, {funding_df['coin'].nunique()} coins")
    print(f"   Date range: {funding_df['datetime'].min()} to {funding_df['datetime'].max()}")

    # Price data
    print("\n" + "=" * 70)
    print("STEP 2: FETCHING PRICE DATA")
    print("=" * 70)

    coins_with_funding = funding_df['coin'].unique().tolist()
    if 'BTC' not in coins_with_funding:
        coins_with_funding.append('BTC')

    all_prices = []
    success_count = 0
    fail_count = 0

    for i, coin in enumerate(coins_with_funding):
        progress = (i + 1) / len(coins_with_funding) * 100
        bar_len = 40
        filled = int(bar_len * (i + 1) / len(coins_with_funding))
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r[{bar}] {progress:5.1f}% | {i+1}/{len(coins_with_funding)} | {coin:<12}", end='', flush=True)

        records = fetch_candles_chunked(coin, price_start, price_end)
        if records:
            all_prices.extend(records)
            success_count += 1
            print(f" | {len(records)} candles", end='', flush=True)
        else:
            fail_count += 1
            print(" | FAILED", end='', flush=True)

    print(f"\n\n✓ Price data: {success_count} coins succeeded, {fail_count} failed")

    # Retry failed coins with smaller chunks (helps with API gaps)
    failed_coins = [coin for coin in coins_with_funding if coin not in set([r['coin'] for r in all_prices])]
    if failed_coins:
        for retry_chunk in [200, 100]:
            if not failed_coins:
                break
            print(f"\nRetrying {len(failed_coins)} failed coins with chunk={retry_chunk}h...")
            still_failed = []
            for coin in failed_coins:
                records = fetch_candles_chunked(coin, price_start, price_end, chunk_hours=retry_chunk)
                if records:
                    all_prices.extend(records)
                else:
                    still_failed.append(coin)
            failed_coins = still_failed

    # Save failed coins list for review
    if failed_coins:
        pd.DataFrame({'coin': failed_coins}).to_csv('price_fetch_failed_coins.csv', index=False)
        print(f"⚠️  Still failed coins: {len(failed_coins)} (saved to price_fetch_failed_coins.csv)")

    price_df = pd.DataFrame(all_prices)
    if price_df.empty:
        print("ERROR: No price data fetched!")
        return

    price_df = price_df.sort_values(['timestamp', 'coin']).reset_index(drop=True)
    price_df.to_csv(price_out, index=False)
    print(f"💾 Saved {price_out}: {len(price_df):,} records, {price_df['coin'].nunique()} coins")
    print(f"   Date range: {price_df['timestamp'].min()} to {price_df['timestamp'].max()}")

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

    funding_coins = set(funding_df['coin'].unique())
    price_coins = set(price_df['coin'].unique())
    missing = funding_coins - price_coins
    if missing:
        print(f"\n⚠️ {len(missing)} coins in funding but not in price data:")
        print(f"   {sorted(list(missing))[:20]}...")


if __name__ == "__main__":
    main()
