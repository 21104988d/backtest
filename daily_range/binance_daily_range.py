#!/usr/bin/env python3
"""Fetch BTCUSDT daily candles from Binance and plot their range distribution."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import matplotlib

# Use a non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def resolve_output_path(path: Path) -> Path:
    """Return an absolute filesystem path for the requested output artifact."""
    return path if path.is_absolute() else Path.cwd() / path


def build_percentage_bins(series: pd.Series, bin_size: float, fallback_bins: int) -> Sequence[float] | int:
    """Create histogram bins either via a fixed width or an integer fallback."""
    if bin_size > 0:
        max_value = float(series.max()) if not series.empty else 0.0
        max_edge = max(bin_size, math.ceil(max_value / bin_size) * bin_size) if max_value > 0 else bin_size
        bins = np.arange(0, max_edge + bin_size, bin_size)
        if len(bins) < 2:
            bins = np.array([0.0, bin_size])
        return bins
    return max(1, fallback_bins)


def fetch_daily_klines(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 1000) -> List[list]:
    """Download all available klines for the given symbol and interval."""
    session = requests.Session()
    params = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": 0}
    klines: List[list] = []

    # Step through the dataset in 1000-candle slices until the exchange stops returning data.
    while True:
        response = session.get(BINANCE_KLINES_URL, params=params, timeout=15)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        klines.extend(batch)
        params["startTime"] = batch[-1][0] + 1
        if len(batch) < limit:
            break
        time.sleep(0.1)

    return klines


def klines_to_dataframe(klines: Iterable[Iterable]) -> pd.DataFrame:
    """Convert raw kline rows to a typed DataFrame with a daily range percentage column."""
    df = pd.DataFrame(klines, columns=KLINE_COLUMNS)
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.tz_convert(None)
    price_range = df["high"] - df["low"]
    # Express day-level moves relative to the opening price to remove the absolute-price effect.
    open_non_zero = df["open"].replace(0, np.nan)
    df["daily_range_pct"] = (price_range / open_non_zero) * 100.0
    df["high_from_open_pct"] = ((df["high"] - df["open"]) / open_non_zero) * 100.0
    df["open_to_low_pct"] = ((df["open"] - df["low"]) / open_non_zero) * 100.0
    return df


def plot_distribution(
    series: pd.Series,
    bins: Sequence[float] | int,
    title: str,
    x_label: str,
    ci_lines: Mapping[str, float] | None = None,
    value_suffix: str = "",
) -> plt.Figure:
    """Plot a histogram for the provided series and optional percentile markers."""
    clean_series = series.dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    counts, _, _ = ax.hist(clean_series, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.3)
    if ci_lines:
        # Overlay percentile markers so it is easy to spot high-range outliers.
        color_cycle = {"90%": "#ff7f0e", "99%": "#d62728"}
        y_max = counts.max() if len(counts) else 0.0
        for label, value in ci_lines.items():
            display_label = f"{label} quantile ({value:.2f}{value_suffix})"
            ax.axvline(
                value,
                color=color_cycle.get(label, "#2ca02c"),
                linestyle="--",
                linewidth=1.5,
                label=display_label,
            )
            if y_max > 0:
                ax.text(
                    value,
                    y_max * 0.95,
                    f"{value:.2f}{value_suffix}",
                    color=color_cycle.get(label, "#2ca02c"),
                    rotation=90,
                    va="top",
                    ha="right",
                    bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 2},
                )
        ax.legend()
    return fig


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the daily range distribution for a Binance symbol.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair to download, e.g. BTCUSDT.")
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Histogram bin count (ignored when --bin-size is provided).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "daily_range_distribution.png",
        help="Path for the output image (PNG).",
    )
    parser.add_argument(
        "--high-output",
        type=Path,
        default=Path(__file__).resolve().parent / "daily_high_distribution.png",
        help="Path for the daily high move distribution image (PNG).",
    )
    parser.add_argument(
        "--low-output",
        type=Path,
        default=Path(__file__).resolve().parent / "daily_low_distribution.png",
        help="Path for the daily low move distribution image (PNG).",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.5,
        help="Histogram bin width (in percentage points); bins will be 0, bin_size, 2*bin_size, ...",
    )
    parser.add_argument(
        "--price-bins",
        type=int,
        default=60,
        help="Histogram bin count for the high/low move distributions (used when --bin-size <= 0).",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    print(f"Fetching {args.symbol} daily candles from Binance...")
    try:
        klines = fetch_daily_klines(symbol=args.symbol)
    except requests.HTTPError as exc:
        print(f"Binance API request failed: {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Unexpected network error: {exc}", file=sys.stderr)
        return 1

    if not klines:
        print("Binance returned no data.", file=sys.stderr)
        return 1

    df = klines_to_dataframe(klines)
    date_start = df["open_time"].min().date()
    date_end = df["open_time"].max().date()
    print(f"Received {len(df)} daily candles from {date_start} to {date_end}.")
    range_series = df["daily_range_pct"].dropna()
    high_series = df["high_from_open_pct"].dropna()
    low_series = df["open_to_low_pct"].dropna()

    if range_series.empty or high_series.empty or low_series.empty:
        print("No valid percentage-based metrics were computed.", file=sys.stderr)
        return 1

    ci_percentiles = {"90%": 0.90, "99%": 0.99}
    range_ci_values = {label: float(range_series.quantile(q)) for label, q in ci_percentiles.items()}
    high_ci_values = {label: float(high_series.quantile(q)) for label, q in ci_percentiles.items()}
    low_ci_values = {label: float(low_series.quantile(q)) for label, q in ci_percentiles.items()}

    print(
        "Daily range stats (% of open): "
        f"min={range_series.min():.2f}%, "
        f"mean={range_series.mean():.2f}%, "
        f"median={range_series.median():.2f}%, "
        f"max={range_series.max():.2f}%, "
        f"90%={range_ci_values['90%']:.2f}%, "
        f"99%={range_ci_values['99%']:.2f}%"
    )
    print(
        "Daily high move stats (% of open): "
        f"min={high_series.min():.2f}%, "
        f"mean={high_series.mean():.2f}%, "
        f"median={high_series.median():.2f}%, "
        f"max={high_series.max():.2f}%, "
        f"90%={high_ci_values['90%']:.2f}%, "
        f"99%={high_ci_values['99%']:.2f}%"
    )
    print(
        "Daily low move stats (% of open): "
        f"min={low_series.min():.2f}%, "
        f"mean={low_series.mean():.2f}%, "
        f"median={low_series.median():.2f}%, "
        f"max={low_series.max():.2f}%, "
        f"90%={low_ci_values['90%']:.2f}%, "
        f"99%={low_ci_values['99%']:.2f}%"
    )

    range_bins = build_percentage_bins(range_series, args.bin_size, args.bins)
    high_bins = build_percentage_bins(high_series, args.bin_size, args.price_bins)
    low_bins = build_percentage_bins(low_series, args.bin_size, args.price_bins)

    range_fig = plot_distribution(
        range_series,
        bins=range_bins,
        title=f"{args.symbol} Daily Range Percentage Distribution",
        x_label="Daily Range (%)",
        ci_lines=range_ci_values,
        value_suffix="%",
    )

    range_output_path = resolve_output_path(args.output)
    range_output_path.parent.mkdir(parents=True, exist_ok=True)
    range_fig.savefig(range_output_path, dpi=150, bbox_inches="tight")
    plt.close(range_fig)
    print(f"Saved range histogram to {range_output_path}")

    high_fig = plot_distribution(
        high_series,
        bins=high_bins,
        title=f"{args.symbol} Daily High Move Distribution",
        x_label="(High - Open) / Open (%)",
        ci_lines=high_ci_values,
        value_suffix="%",
    )
    high_output_path = resolve_output_path(args.high_output)
    high_output_path.parent.mkdir(parents=True, exist_ok=True)
    high_fig.savefig(high_output_path, dpi=150, bbox_inches="tight")
    plt.close(high_fig)
    print(f"Saved high move histogram to {high_output_path}")

    low_fig = plot_distribution(
        low_series,
        bins=low_bins,
        title=f"{args.symbol} Daily Low Move Distribution",
        x_label="(Open - Low) / Open (%)",
        ci_lines=low_ci_values,
        value_suffix="%",
    )
    low_output_path = resolve_output_path(args.low_output)
    low_output_path.parent.mkdir(parents=True, exist_ok=True)
    low_fig.savefig(low_output_path, dpi=150, bbox_inches="tight")
    plt.close(low_fig)
    print(f"Saved low move histogram to {low_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
