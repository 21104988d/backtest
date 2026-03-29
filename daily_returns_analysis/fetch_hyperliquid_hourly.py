#!/usr/bin/env python3
"""Fetch Hyperliquid hourly candles for all tradable perpetuals.

This script builds the recent-history segment of the two-stage dataset:
1) Pull all tradable symbols from Hyperliquid meta endpoint.
2) Fetch hourly candles in chunked windows for each symbol.
3) Optionally merge with an external historical backfill CSV.

Output schema is compatible with fetch_ohlc_data.py expectations:
  timestamp, coin, price, volume
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests


BASE_URL = "https://api.hyperliquid.xyz/info"
DEFAULT_CHUNK_HOURS = 500
DEFAULT_MAX_CANDLE_DAYS = 210
DEFAULT_RATE_DELAY_SEC = 0.25
MAX_RETRIES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Hyperliquid hourly candles for all tradable assets.")
    parser.add_argument(
        "--output",
        default="price_history_hyperliquid_recent.csv",
        help="Output CSV for recent Hyperliquid hourly prices.",
    )
    parser.add_argument(
        "--final-output",
        default="price_history.csv",
        help="Final merged output CSV (recent + optional backfill).",
    )
    parser.add_argument(
        "--history-backfill",
        default=None,
        help="Optional historical backfill CSV path with timestamp,coin,price,volume columns.",
    )
    parser.add_argument(
        "--chunk-hours",
        type=int,
        default=DEFAULT_CHUNK_HOURS,
        help="Number of hours per candleSnapshot request.",
    )
    parser.add_argument(
        "--max-candle-days",
        type=int,
        default=DEFAULT_MAX_CANDLE_DAYS,
        help="Maximum historical days to request from Hyperliquid API.",
    )
    parser.add_argument(
        "--rate-delay-sec",
        type=float,
        default=DEFAULT_RATE_DELAY_SEC,
        help="Delay between API requests.",
    )
    parser.add_argument(
        "--coins",
        type=str,
        default=None,
        help="Optional comma-separated coin allowlist.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by skipping coins already present in --output.",
    )
    return parser.parse_args()


def post_info(payload: dict) -> requests.Response:
    return requests.post(BASE_URL, json=payload, timeout=30)


def get_all_perpetuals() -> List[str]:
    payload = {"type": "meta"}
    response = post_info(payload)
    response.raise_for_status()
    data = response.json()
    universe = data.get("universe", [])
    coins = sorted({row.get("name") for row in universe if row.get("name")})
    if not coins:
        raise ValueError("No tradable perpetual symbols returned by Hyperliquid meta endpoint.")
    return coins


def fetch_candles_chunked(
    coin: str,
    start_time: datetime,
    end_time: datetime,
    chunk_hours: int,
    rate_delay_sec: float,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    current_start = start_time

    while current_start < end_time:
        chunk_end = min(current_start + timedelta(hours=chunk_hours), end_time)
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "1h",
                "startTime": int(current_start.timestamp() * 1000),
                "endTime": int(chunk_end.timestamp() * 1000),
            },
        }

        data = []
        for attempt in range(MAX_RETRIES):
            try:
                response = post_info(payload)
                if response.status_code == 429:
                    wait_sec = 2 ** attempt
                    time.sleep(wait_sec)
                    continue
                response.raise_for_status()
                data = response.json()
                break
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    data = []
                time.sleep(1)

        for candle in data if isinstance(data, list) else []:
            ts = pd.to_datetime(candle.get("t"), unit="ms", utc=True)
            records.append(
                {
                    "timestamp": ts.isoformat(),
                    "coin": coin,
                    "price": float(candle.get("c", 0.0)),
                    "volume": float(candle.get("v", 0.0) or 0.0),
                    "source": "hyperliquid_recent",
                }
            )

        current_start = chunk_end
        time.sleep(rate_delay_sec)

    return records


def normalize_input_schema(df: pd.DataFrame, source_tag: str) -> pd.DataFrame:
    required = {"timestamp", "coin", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    normalized = df.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["timestamp", "coin", "price"])

    if "volume" not in normalized.columns:
        normalized["volume"] = 0.0
    normalized["source"] = source_tag
    return normalized[["timestamp", "coin", "price", "volume", "source"]]


def merge_with_backfill(recent_df: pd.DataFrame, backfill_path: Path | None) -> pd.DataFrame:
    merged = recent_df.copy()
    if backfill_path is None:
        return merged
    if not backfill_path.exists():
        raise FileNotFoundError(f"Backfill file not found: {backfill_path}")

    backfill_df = pd.read_csv(backfill_path)
    backfill_norm = normalize_input_schema(backfill_df, source_tag="historical_backfill")

    merged = pd.concat([backfill_norm, recent_df], ignore_index=True)
    merged = merged.sort_values(["coin", "timestamp", "source"])
    merged = merged.drop_duplicates(subset=["timestamp", "coin"], keep="last")
    return merged


def main() -> None:
    args = parse_args()

    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    recent_start = now_utc - timedelta(days=args.max_candle_days)

    print("=" * 80)
    print("FETCH HYPERLIQUID HOURLY CANDLES (RECENT WINDOW)")
    print("=" * 80)
    print(f"Window: {recent_start} -> {now_utc}")

    coins = get_all_perpetuals()
    if args.coins:
        allowlist = {coin.strip() for coin in args.coins.split(",") if coin.strip()}
        coins = [coin for coin in coins if coin in allowlist]
    print(f"Tradable symbols to fetch: {len(coins)}")

    output_path = Path(args.output)

    completed = set()
    all_records: List[Dict[str, object]] = []
    if args.resume and output_path.exists():
        existing = pd.read_csv(output_path)
        if "coin" in existing.columns:
            completed = set(existing["coin"].dropna().unique())
            all_records = existing.to_dict("records")
            print(f"Resume mode: skipping {len(completed)} already fetched symbols")

    for idx, coin in enumerate(coins, start=1):
        if coin in completed:
            print(f"[{idx}/{len(coins)}] Skip {coin} (already fetched)")
            continue

        print(f"[{idx}/{len(coins)}] Fetch {coin}")
        coin_records = fetch_candles_chunked(
            coin=coin,
            start_time=recent_start,
            end_time=now_utc,
            chunk_hours=args.chunk_hours,
            rate_delay_sec=args.rate_delay_sec,
        )
        print(f"    {len(coin_records)} candles")
        all_records.extend(coin_records)

        if idx % 10 == 0 and all_records:
            pd.DataFrame(all_records).to_csv(output_path, index=False)
            print(f"    checkpoint saved: {len(all_records):,} rows")

    recent_df = pd.DataFrame(all_records)
    if recent_df.empty:
        raise ValueError("No candle data fetched. Check API access and retry.")

    recent_df = normalize_input_schema(recent_df, source_tag="hyperliquid_recent")
    recent_df = recent_df.sort_values(["coin", "timestamp"]).reset_index(drop=True)
    recent_df.to_csv(output_path, index=False)
    print(f"Saved recent dataset: {output_path} ({len(recent_df):,} rows)")

    backfill_path = Path(args.history_backfill) if args.history_backfill else None
    merged = merge_with_backfill(recent_df, backfill_path)
    merged = merged.sort_values(["timestamp", "coin"]).reset_index(drop=True)

    final_output = Path(args.final_output)
    merged_to_save = merged.copy()
    merged_to_save["timestamp"] = pd.to_datetime(merged_to_save["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    merged_to_save.to_csv(final_output, index=False)

    print(f"Saved final merged dataset: {final_output} ({len(merged_to_save):,} rows)")
    print(f"Date range: {merged_to_save['timestamp'].min()} -> {merged_to_save['timestamp'].max()}")
    print(f"Coins: {merged_to_save['coin'].nunique()}")


if __name__ == "__main__":
    main()
