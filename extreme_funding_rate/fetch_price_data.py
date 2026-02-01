#!/usr/bin/env python3
"""
Fetch hourly price data from Hyperliquid MAINNET for each coin in the funding summary.

Uses funding_history_summary.csv to align price time range per coin. Note: the
candleSnapshot API appears to be limited (~210 days). The script clamps the
requested start time accordingly to avoid empty responses.
"""
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

BASE_URL = "https://api.hyperliquid.xyz/info"
DEFAULT_CHUNK_HOURS = 500
DEFAULT_SLEEP_SECONDS = 2.0
DEFAULT_MAX_CANDLE_DAYS = 210
MAX_RETRIES = 5


def parse_datetime(value: str) -> datetime:
    return pd.to_datetime(value, utc=True).to_pydatetime()


def load_coin_ranges(summary_path: Path, fallback_funding_path: Path) -> Dict[str, Dict[str, datetime]]:
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if not {"coin", "start", "end"}.issubset(summary.columns):
            raise ValueError("Summary file must include coin,start,end columns")
        ranges = {}
        for _, row in summary.iterrows():
            if pd.isna(row["coin"]) or pd.isna(row["start"]) or pd.isna(row["end"]):
                continue
            ranges[str(row["coin"]) ] = {
                "start": parse_datetime(row["start"]),
                "end": parse_datetime(row["end"]),
            }
        return ranges

    if not fallback_funding_path.exists():
        raise FileNotFoundError("No funding summary or funding history found to derive coin ranges.")

    funding = pd.read_csv(fallback_funding_path)
    if "datetime" in funding.columns:
        funding["datetime"] = pd.to_datetime(funding["datetime"], errors="coerce", utc=True)
    else:
        funding["datetime"] = pd.to_datetime(funding["timestamp"], unit="ms", errors="coerce", utc=True)

    ranges = {}
    grouped = funding.dropna(subset=["datetime"]).groupby("coin")["datetime"]
    for coin, series in grouped:
        ranges[str(coin)] = {
            "start": series.min().to_pydatetime(),
            "end": series.max().to_pydatetime(),
        }
    return ranges


def fetch_candles_chunked(coin: str, start_time: datetime, end_time: datetime, chunk_hours: int, sleep_seconds: float) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    current_start = start_time

    while current_start < end_time:
        chunk_end = min(current_start + timedelta(hours=chunk_hours), end_time)
        req_data = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "1h",
                "startTime": int(current_start.timestamp() * 1000),
                "endTime": int(chunk_end.timestamp() * 1000),
            },
        }

        data = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(BASE_URL, json=req_data, timeout=30)
                if resp.status_code == 429:
                    wait_time = (2 ** attempt) * 2
                    print(f"  Rate limited, waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}...")
                    time.sleep(wait_time)
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
            time.sleep(sleep_seconds)
            continue

        for candle in data:
            ts = pd.to_datetime(candle["t"], unit="ms", utc=True)
            records.append({
                "timestamp": ts.isoformat(),
                "coin": coin,
                "price": float(candle["c"]),
            })

        current_start = chunk_end
        time.sleep(sleep_seconds)

    return records


def main():
    parser = argparse.ArgumentParser(description="Fetch hourly price data aligned to funding history ranges.")
    parser.add_argument("--summary-file", default="funding_history_summary.csv", help="Funding summary file with start/end per coin")
    parser.add_argument("--funding-file", default="funding_history.csv", help="Funding history file (fallback if summary missing)")
    parser.add_argument("--output", default="price_history.csv", help="Output CSV for price data")
    parser.add_argument("--chunk-hours", type=int, default=DEFAULT_CHUNK_HOURS, help="Hours per candle request chunk")
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS, help="Delay between API calls")
    parser.add_argument("--max-candle-days", type=int, default=DEFAULT_MAX_CANDLE_DAYS, help="Max candle history available from API")
    parser.add_argument("--coins", type=str, default=None, help="Comma-separated coin list to fetch (optional)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    summary_path = Path(args.summary_file)
    funding_path = Path(args.funding_file)
    output_path = Path(args.output)

    coin_ranges = load_coin_ranges(summary_path, funding_path)
    if args.coins:
        wanted = {c.strip() for c in args.coins.split(",") if c.strip()}
        coin_ranges = {c: v for c, v in coin_ranges.items() if c in wanted}

    completed = set()
    all_records: List[Dict[str, str]] = []
    if args.resume and output_path.exists():
        existing = pd.read_csv(output_path)
        completed = set(existing["coin"].unique())
        all_records = existing.to_dict("records")
        print(f"Resuming from {output_path} with {len(completed)} coins already fetched")

    print(f"Total coins to fetch: {len(coin_ranges)}")

    for idx, (coin, rng) in enumerate(coin_ranges.items(), start=1):
        if coin in completed:
            print(f"Skipping {coin} ({idx}/{len(coin_ranges)}) - already fetched")
            continue

        funding_start = rng["start"]
        funding_end = rng["end"]
        latest_allowed_start = funding_end - timedelta(days=args.max_candle_days)
        price_start = max(funding_start, latest_allowed_start)
        price_end = funding_end + timedelta(hours=1)

        print(f"Fetching {coin} ({idx}/{len(coin_ranges)})")
        print(f"  Funding range: {funding_start} -> {funding_end}")
        if price_start > funding_start:
            print(f"  Price start clamped to {price_start} (API limit {args.max_candle_days}d)")
        print(f"  Price range:   {price_start} -> {price_end}")

        records = fetch_candles_chunked(
            coin,
            start_time=price_start,
            end_time=price_end,
            chunk_hours=args.chunk_hours,
            sleep_seconds=args.sleep_seconds,
        )

        print(f"  → Fetched {len(records)} candles for {coin}")
        all_records.extend(records)

        if idx % 10 == 0:
            pd.DataFrame(all_records).to_csv(output_path, index=False)
            print(f"  💾 Progress saved: {len(all_records):,} records")

    pd.DataFrame(all_records).to_csv(output_path, index=False)
    print(f"\n✓ Price data saved to {output_path}")


if __name__ == "__main__":
    main()
