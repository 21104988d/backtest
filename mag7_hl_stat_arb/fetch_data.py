#!/usr/bin/env python3
"""Fetch daily close prices for MAG7 stocks and their Hyperliquid perp counterparts.

Two data sources:
  - Yahoo Finance (yfinance) for real stock daily closes
  - Hyperliquid candleSnapshot API for xyz:SYMBOL perp daily closes

Output:
  data/yahoo_mag7_daily.csv        -- Yahoo Finance daily closes
  data/hl_mag7_daily.csv           -- Hyperliquid perp daily closes
  data/aligned_mag7_daily.csv      -- Merged on common trading dates with both prices

Usage:
  python fetch_data.py
  python fetch_data.py --start 2024-01-01 --end 2025-03-28 --coins AAPL,MSFT,NVDA
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import yfinance as yf
import pandas as pd

HL_API_URL = "https://api.hyperliquid.xyz/info"

MAG7 = {
    "AAPL": "xyz:AAPL",
    "MSFT": "xyz:MSFT",
    "NVDA": "xyz:NVDA",
    "GOOGL": "xyz:GOOGL",
    "AMZN": "xyz:AMZN",
    "META": "xyz:META",
    "TSLA": "xyz:TSLA",
}

MAX_RETRIES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MAG7 daily data from Yahoo + Hyperliquid.")
    parser.add_argument("--start", default="2024-09-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--coins", default=None, help="Comma-separated subset of MAG7 tickers")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch even if cached files exist")
    return parser.parse_args()


def hl_post(payload: dict) -> object:
    req = Request(
        HL_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(MAX_RETRIES):
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    return None


def fetch_hl_daily(
    hl_symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[dict]:
    """Fetch daily candles from Hyperliquid for a single symbol."""
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": hl_symbol,
            "interval": "1d",
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    data = hl_post(payload)
    if not isinstance(data, list):
        return []

    rows = []
    for x in data:
        ts_ms = int(x["t"])
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "timestamp_ms": ts_ms,
            "open": float(x.get("o", x.get("c", 0))),
            "high": float(x.get("h", x.get("c", 0))),
            "low": float(x.get("l", x.get("c", 0))),
            "close": float(x["c"]),
            "volume": float(x.get("v", 0) or 0),
        })

    rows.sort(key=lambda r: r["timestamp_ms"])
    dedup: dict[str, dict] = {}
    for r in rows:
        dedup[r["date"]] = r
    return list(dedup.values())


def fetch_yahoo_daily(
    ticker: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Fetch daily OHLCV from Yahoo Finance via yfinance."""
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return []

    # Flatten MultiIndex columns if present (yfinance >= 0.2 returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    rows = []
    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
        rows.append({
            "date": date_str,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]) if "Volume" in row and pd.notna(row["Volume"]) else 0.0,
        })
    rows.sort(key=lambda r: r["date"])
    return rows


def main() -> None:
    args = parse_args()

    end_date = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = args.start

    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) + 86_400_000

    ticker_map = dict(MAG7)
    if args.coins:
        subset = {c.strip().upper() for c in args.coins.split(",")}
        ticker_map = {k: v for k, v in MAG7.items() if k in subset}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yahoo_cache = out_dir / "yahoo_mag7_daily.csv"
    hl_cache = out_dir / "hl_mag7_daily.csv"
    aligned_out = out_dir / "aligned_mag7_daily.csv"

    # ── Yahoo Finance ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FETCHING YAHOO FINANCE DAILY CLOSES")
    print(f"{'='*70}")
    print(f"Period: {start_date} → {end_date}")
    print(f"Tickers: {list(ticker_map.keys())}")

    yahoo_records: list[dict] = []
    for ticker, hl_sym in ticker_map.items():
        if not args.refresh and yahoo_cache.exists():
            pass  # reload later from file, skip re-fetch logic per ticker
        print(f"  {ticker} (Yahoo Finance)...", end=" ", flush=True)
        rows = fetch_yahoo_daily(ticker, start_date, end_date)
        for r in rows:
            r["ticker"] = ticker
        yahoo_records.extend(rows)
        print(f"{len(rows)} bars")
        time.sleep(0.3)

    yahoo_df = pd.DataFrame(yahoo_records)
    if yahoo_df.empty:
        raise RuntimeError("No Yahoo Finance data fetched. Check tickers and date range.")

    yahoo_df = yahoo_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    yahoo_df.to_csv(yahoo_cache, index=False)
    print(f"\nSaved: {yahoo_cache} ({len(yahoo_df):,} rows)")

    # ── Hyperliquid Perps ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FETCHING HYPERLIQUID PERP DAILY CLOSES")
    print(f"{'='*70}")
    print(f"Period: {start_date} → {end_date}")

    hl_records: list[dict] = []
    for ticker, hl_sym in ticker_map.items():
        print(f"  {hl_sym} (Hyperliquid)...", end=" ", flush=True)
        rows = fetch_hl_daily(hl_sym, start_ms, end_ms)
        for r in rows:
            r["ticker"] = ticker
            r["hl_symbol"] = hl_sym
        hl_records.extend(rows)
        print(f"{len(rows)} bars")
        time.sleep(0.25)

    hl_df = pd.DataFrame(hl_records)
    if hl_df.empty:
        raise RuntimeError("No Hyperliquid data fetched. Check symbols and API.")

    hl_df = hl_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    hl_df.to_csv(hl_cache, index=False)
    print(f"\nSaved: {hl_cache} ({len(hl_df):,} rows)")

    # ── Align ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ALIGNING YAHOO + HYPERLIQUID ON COMMON DATES")
    print(f"{'='*70}")

    aligned_rows: list[dict] = []
    for ticker in ticker_map:
        y = yahoo_df[yahoo_df["ticker"] == ticker][["date", "close"]].rename(columns={"close": "yahoo_close"})
        h = hl_df[hl_df["ticker"] == ticker][["date", "close"]].rename(columns={"close": "hl_close"})
        merged = pd.merge(y, h, on="date", how="inner")
        merged["ticker"] = ticker
        n_before = max(len(y), len(h))
        print(f"  {ticker}: Yahoo={len(y)}, HL={len(h)}, Aligned={len(merged)} (dropped {n_before - len(merged)} unmatched dates)")
        aligned_rows.append(merged)

    aligned_df = pd.concat(aligned_rows, ignore_index=True)
    aligned_df = aligned_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    aligned_df["log_basis"] = aligned_df["hl_close"] / aligned_df["yahoo_close"] - 1.0

    aligned_df.to_csv(aligned_out, index=False)
    print(f"\nSaved: {aligned_out} ({len(aligned_df):,} rows)")
    print(f"Date range: {aligned_df['date'].min()} → {aligned_df['date'].max()}")
    print(f"Tickers: {sorted(aligned_df['ticker'].unique())}")

    # ── Basis summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("BASIS SUMMARY (HL price / Yahoo price - 1)")
    print(f"{'='*70}")
    for ticker in sorted(ticker_map.keys()):
        sub = aligned_df[aligned_df["ticker"] == ticker]["log_basis"]
        print(
            f"  {ticker:6s}: mean={sub.mean()*100:+.3f}%  "
            f"std={sub.std()*100:.3f}%  "
            f"min={sub.min()*100:+.3f}%  "
            f"max={sub.max()*100:+.3f}%"
        )


if __name__ == "__main__":
    main()
