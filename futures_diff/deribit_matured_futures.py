"""Utilities for fetching Deribit matured futures metadata and settlement history.

This module replaces the previous Jupyter notebook so it can be executed directly
without requiring a notebook kernel.  It exposes two helpers, `fetch_matured_futures`
and `fetch_future_settlements`, plus a small CLI that prints summary tables for a
set of currencies.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable, Optional
FUTURE_NAME_PATTERN = re.compile(
    r"^(?P<underlying>[A-Z]+)-(?P<day>\d{1,2})(?P<month>[A-Z]{3})(?P<year>\d{2})$"
)


import pandas as pd
import requests

API_URL = "https://www.deribit.com/api/v2/public/get_instruments"
SETTLEMENTS_URL = "https://www.deribit.com/api/v2/public/get_last_settlements_by_currency"
TRADINGVIEW_URL = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
PERPETUAL_INSTRUMENT = {
    "BTC": "BTC-PERPETUAL",
    "ETH": "ETH-PERPETUAL",
}

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", None)


def plot_close_diff_lines(
    currency: str,
    perpetual_name: Optional[str],
    traces: list[tuple[str, pd.DataFrame]],
    export_path: Optional[Path],
) -> None:
    if not traces:
        return

    import matplotlib

    if "DISPLAY" not in os.environ:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for instrument, df in traces:
        if df.empty or "close_diff" not in df.columns or "hours_to_maturity" not in df.columns:
            continue
        ordered = df.sort_values("hours_to_maturity")
        plt.plot(
            ordered["hours_to_maturity"],
            ordered["close_diff"],
            label=instrument,
        )

    plt.xlabel("Hours relative to maturity (0 = maturity)")
    plt.ylabel("Close price difference (future - perpetual)")
    title_parts = [currency.upper(), "futures close diff"]
    if perpetual_name:
        title_parts.append(f"vs {perpetual_name}")
    plt.title(" ".join(title_parts))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    default_dir = Path(__file__).resolve().parent
    output_dir = export_path if export_path else default_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_label = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d%H%M%S")
    perp_fragment = (perpetual_name or "PERPETUAL").replace("/", "_").replace(" ", "_")
    filename = f"{currency.upper()}_{perp_fragment}_close_diff.png"
    output_file = output_dir / filename
    plt.savefig(output_file)
    print(f"Close diff plot saved to {output_file}")

    backend = matplotlib.get_backend().lower()
    if backend not in {"agg", "pdf", "ps", "svg", "cairo"}:
        plt.show()
    plt.close()


def plot_price_comparison_lines(
    currency: str,
    matured_name: str,
    matured_prices: pd.DataFrame,
    reference_name: str,
    reference_prices: pd.DataFrame,
    export_path: Optional[Path],
) -> None:
    if matured_prices.empty or reference_prices.empty:
        return

    import matplotlib

    if "DISPLAY" not in os.environ:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    matur_sorted = matured_prices.sort_values("timestamp")
    plt.plot(
        matur_sorted["timestamp"],
        matur_sorted["close"],
        label=matured_name,
        linewidth=2,
    )

    ref_sorted = reference_prices.sort_values("timestamp")
    plt.plot(
        ref_sorted["timestamp"],
        ref_sorted["close"],
        label=reference_name,
        linestyle="--",
        linewidth=1.5,
    )

    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("Close price")
    plt.title(
        f"{currency.upper()} price comparison: {matured_name} vs {reference_name}"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()

    default_dir = Path(__file__).resolve().parent
    output_dir = export_path if export_path else default_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matur_fragment = matured_name.replace("/", "_").replace(" ", "_")
    ref_fragment = reference_name.replace("/", "_").replace(" ", "_")
    filename = f"{currency.upper()}_{matur_fragment}_vs_{ref_fragment}_price.png"
    output_file = output_dir / filename
    plt.savefig(output_file)
    print(f"Price comparison plot saved to {output_file}")

    backend = matplotlib.get_backend().lower()
    if backend not in {"agg", "pdf", "ps", "svg", "cairo"}:
        plt.show()
    plt.close()


def plot_reference_diff_lines(
    currency: str,
    reference_name: str,
    traces: list[tuple[str, pd.DataFrame]],
    export_path: Optional[Path],
) -> None:
    if not traces:
        return

    import matplotlib

    if "DISPLAY" not in os.environ:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for instrument, df in traces:
        if df.empty or "close_diff" not in df.columns or "hours_to_maturity" not in df.columns:
            continue
        ordered = df.sort_values("hours_to_maturity")
        plt.plot(
            ordered["hours_to_maturity"],
            ordered["close_diff"],
            marker="o",
            label=instrument,
        )

    plt.xlabel("Hours relative to maturity (0 = maturity)")
    plt.ylabel("Close price difference (matured - reference)")
    plt.title(
        f"{currency.upper()} futures close diff vs {reference_name}"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    default_dir = Path(__file__).resolve().parent
    output_dir = export_path if export_path else default_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_fragment = reference_name.replace("/", "_").replace(" ", "_")
    filename = f"{currency.upper()}_{ref_fragment}_reference_close_diff.png"
    output_file = output_dir / filename
    plt.savefig(output_file)
    print(f"Reference close diff plot saved to {output_file}")

    backend = matplotlib.get_backend().lower()
    if backend not in {"agg", "pdf", "ps", "svg", "cairo"}:
        plt.show()
    plt.close()


def fetch_matured_futures(currency: str = "BTC") -> pd.DataFrame:
    """Fetch expired (matured) futures instruments for a given currency."""
    params = {
        "currency": currency.upper(),
        "kind": "future",
        "include_expired": True,
    }
    response = requests.get(API_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()

    if "error" in payload and payload["error"] is not None:
        raise RuntimeError(f"Deribit API returned an error: {payload['error']}")

    records = payload.get("result", [])
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame

    timestamp_mappings = {
        "creation_timestamp": "creation_time",
        "expiration_timestamp": "expiration_time",
        "settlement_timestamp": "settlement_time",
    }

    for source_col, target_col in timestamp_mappings.items():
        if source_col in frame:
            frame[target_col] = pd.to_datetime(
                frame[source_col], unit="ms", utc=True, errors="coerce"
            )

    if "expiration_time" in frame:
        frame = frame[frame["expiration_time"].notna()]
        utc_now = pd.Timestamp.now(tz="UTC")
        frame = frame[frame["expiration_time"] <= utc_now]

    if "is_active" in frame:
        frame = frame[frame["is_active"].fillna(False) == False]

    desired_order = [
        "instrument_name",
        "base_currency",
        "quote_currency",
        "creation_time",
        "expiration_time",
        "settlement_time",
        "strike",
        "tick_size",
        "min_trade_amount",
        "settlement_period",
        "option_type",
    ]

    available_columns = [col for col in desired_order if col in frame.columns]
    return (
        frame[available_columns]
        .sort_values("expiration_time", ascending=False)
        .reset_index(drop=True)
    )


def fetch_active_futures(currency: str = "BTC") -> pd.DataFrame:
    """Fetch currently active futures instruments for a given currency."""

    params = {
        "currency": currency.upper(),
        "kind": "future",
        "include_expired": False,
    }
    response = requests.get(API_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()

    if "error" in payload and payload["error"] is not None:
        raise RuntimeError(f"Deribit API returned an error: {payload['error']}")

    records = payload.get("result", [])
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame

    timestamp_mappings = {
        "creation_timestamp": "creation_time",
        "expiration_timestamp": "expiration_time",
        "settlement_timestamp": "settlement_time",
    }

    for source_col, target_col in timestamp_mappings.items():
        if source_col in frame:
            frame[target_col] = pd.to_datetime(
                frame[source_col], unit="ms", utc=True, errors="coerce"
            )

    utc_now = pd.Timestamp.now(tz="UTC")
    if "expiration_time" in frame:
        frame = frame[frame["expiration_time"].notna()]
        frame = frame[frame["expiration_time"] > utc_now]

    if "is_active" in frame:
        frame = frame[frame["is_active"].fillna(False) == True]

    desired_order = [
        "instrument_name",
        "base_currency",
        "quote_currency",
        "creation_time",
        "expiration_time",
        "settlement_time",
        "strike",
        "tick_size",
        "min_trade_amount",
        "settlement_period",
        "option_type",
    ]

    available_columns = [col for col in desired_order if col in frame.columns]
    return (
        frame[available_columns]
        .sort_values("creation_time", ascending=True)
        .reset_index(drop=True)
    )


def select_longest_history_future(active_futures: pd.DataFrame) -> Optional[pd.Series]:
    """Pick the active future with the earliest creation timestamp."""

    if active_futures.empty:
        return None

    df = active_futures.copy()
    if "creation_time" in df.columns:
        df = df[df["creation_time"].notna()]
        if df.empty:
            return None
        df_sorted = df.sort_values("creation_time", ascending=True)
    else:
        df_sorted = df.sort_values("expiration_time", ascending=True)
    return df_sorted.iloc[0]


def fetch_future_settlements(
    currency: str = "BTC",
    count: int = 100,
    max_pages: Optional[int] = 1,
    settlement_type: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch historical settlement or delivery records for expired futures."""

    all_records: list[dict] = []
    continuation: Optional[str] = None

    page = 0
    while True:
        params = {
            "currency": currency.upper(),
            "instrument_type": "future",
            "count": count,
        }
        if continuation:
            params["continuation"] = continuation
        if settlement_type:
            params["type"] = settlement_type

        response = requests.get(SETTLEMENTS_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()

        if "error" in payload and payload["error"] is not None:
            raise RuntimeError(f"Deribit API returned an error: {payload['error']}")

        result = payload.get("result", {})
        settlements = result.get("settlements", [])
        if not isinstance(settlements, list):
            raise TypeError("Unexpected response structure: 'settlements' is not a list")

        all_records.extend(settlements)
        continuation = result.get("continuation")
        page += 1

        should_stop = continuation is None
        if max_pages is not None and page >= max_pages:
            should_stop = True

        if should_stop:
            break

    frame = pd.DataFrame.from_records(all_records)
    if frame.empty:
        return frame

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="coerce")
    return frame.sort_values("timestamp", ascending=False).reset_index(drop=True)


def fetch_hourly_prices(
    instrument_name: str,
    expiration_time: pd.Timestamp,
    hours_before: int = 240,
    close_only: bool = False,
) -> pd.DataFrame:
    """Fetch hourly OHLCV data for an instrument from expiration - N hours to expiration."""

    if hours_before <= 0:
        hours_before = 240

    if expiration_time.tzinfo is None:
        end_ts = expiration_time.tz_localize("UTC")
    else:
        end_ts = expiration_time.tz_convert("UTC")

    start_ts = end_ts - pd.Timedelta(hours=hours_before)

    params = {
        "instrument_name": instrument_name,
        "resolution": "60",
        "start_timestamp": int(start_ts.timestamp() * 1000),
        "end_timestamp": int(end_ts.timestamp() * 1000),
    }

    response = requests.get(TRADINGVIEW_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()

    result = payload.get("result", {})
    ticks = result.get("ticks", [])
    status = result.get("status")

    if status != "ok" or not ticks:
        columns = ["instrument_name", "timestamp", "close"] if close_only else [
            "instrument_name",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "cost",
        ]
        return pd.DataFrame(columns=columns)

    columns = {
        "timestamp": pd.to_datetime(result.get("ticks", []), unit="ms", utc=True),
        "open": result.get("open", []),
        "high": result.get("high", []),
        "low": result.get("low", []),
        "close": result.get("close", []),
        "volume": result.get("volume", []),
        "cost": result.get("cost", []),
    }

    frame = pd.DataFrame(columns)
    frame.insert(0, "instrument_name", instrument_name)

    if close_only:
        frame = frame[["instrument_name", "timestamp", "close"]]

    return frame


def _infer_expiration_from_name(instrument_name: str) -> Optional[pd.Timestamp]:
    """Infer the expiration date from a Deribit future instrument name."""

    match = FUTURE_NAME_PATTERN.match(instrument_name.upper())
    if not match:
        return None

    day = int(match.group("day"))
    month = match.group("month")
    year = int(match.group("year"))

    try:
        parsed = datetime.strptime(f"{day:02d}{month}{year:02d}", "%d%b%y")
    except ValueError:
        return None

    return pd.Timestamp(parsed, tz="UTC")


def summarize_settled_matured_futures(settlements: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of futures with settlements whose expiry is in the past."""

    if settlements.empty:
        return settlements.iloc[0:0]

    df = settlements.copy()
    df["type"] = df["type"].astype(str)
    df = df[df["type"].str.lower() == "settlement"]
    if df.empty:
        return df

    df["expiration_date"] = df["instrument_name"].apply(_infer_expiration_from_name)
    df = df[df["expiration_date"].notna()]
    if df.empty:
        return df.iloc[0:0]

    today = pd.Timestamp.utcnow().normalize()
    df = df[df["expiration_date"] <= today]
    if df.empty:
        return df.iloc[0:0]

    df_sorted = df.sort_values("timestamp")
    latest_idx = df_sorted.groupby("instrument_name")["timestamp"].idxmax()
    latest = df_sorted.loc[latest_idx].copy()
    counts = df.groupby("instrument_name").size().rename("settlement_count")
    latest = latest.join(counts, on="instrument_name")

    summary = latest[
        [
            "instrument_name",
            "expiration_date",
            "timestamp",
            "settlement_count",
            "index_price",
            "mark_price",
        ]
    ].rename(
        columns={
            "timestamp": "last_settlement_time",
            "index_price": "last_index_price",
            "mark_price": "last_mark_price",
        }
    )

    return summary.sort_values("expiration_date", ascending=False).reset_index(drop=True)


def _print_dataframe(df: pd.DataFrame, title: str, rows: int) -> None:
    print(title)
    if df.empty:
        print("  (no rows)\n")
        return
    if rows is not None and rows > 0:
        with pd.option_context("display.max_rows", rows):
            print(df.head(rows))
    else:
        with pd.option_context("display.max_rows", None):
            print(df)
    print()


def main(
    currencies: Iterable[str],
    count: int,
    max_pages: Optional[int],
    rows: int,
    show_instruments: bool,
    show_history: bool,
    hours_before: int,
    export_dir: Optional[Path],
    close_only: bool,
    include_perp_spread: bool,
    plot_spread: bool,
    plot_reference_comparison: bool,
) -> None:
    effective_plot_spread = plot_spread and include_perp_spread
    if plot_spread and not include_perp_spread:
        print("Plotting requested but --include-perp-spread is required; skipping plots.\n")

    for currency in currencies:
        print(f"=== {currency.upper()} ===")

        if show_instruments:
            matured_df = fetch_matured_futures(currency)
            _print_dataframe(matured_df, "Matured instruments", rows)

        settlements_df = fetch_future_settlements(currency, count=count, max_pages=max_pages)
        if show_history:
            _print_dataframe(settlements_df, "Settlement history", rows)

        matured_from_settlements = summarize_settled_matured_futures(settlements_df)
        _print_dataframe(matured_from_settlements, "Settled & matured futures", rows)

        if matured_from_settlements.empty:
            continue

        plot_traces: list[tuple[str, pd.DataFrame]] = []
        reference_diff_traces: list[tuple[str, pd.DataFrame]] = []

        reference_future = None
        reference_instrument_name: Optional[str] = None
        if plot_reference_comparison:
            active_futures = fetch_active_futures(currency)
            reference_future = select_longest_history_future(active_futures)
            if reference_future is None:
                print("  No active future with historical data available for reference comparison.\n")
            else:
                reference_instrument_name = reference_future["instrument_name"]
                creation_time = reference_future.get("creation_time")
                creation_info = ""
                if isinstance(creation_time, pd.Timestamp):
                    creation_info = creation_time.tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")
                elif creation_time:
                    creation_info = str(creation_time)
                creation_suffix = f" (created {creation_info})" if creation_info else ""
                print(
                    f"  Reference active future for comparison: {reference_instrument_name}{creation_suffix}"
                )

        export_path: Optional[Path] = None
        if export_dir:
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir.absolute()

        perpetual_name = (
            PERPETUAL_INSTRUMENT.get(currency.upper()) if include_perp_spread else None
        )
        if include_perp_spread and not perpetual_name:
            print(
                f"  No perpetual instrument mapping configured for currency {currency.upper()}."
            )
        for _, row in matured_from_settlements.iterrows():
            instrument_name = row["instrument_name"]
            expiration_date = row["expiration_date"]

            reference_view_for_export: Optional[pd.DataFrame] = None

            prices_df = fetch_hourly_prices(
                instrument_name,
                expiration_date,
                hours_before=hours_before,
                close_only=close_only,
            )
            _print_dataframe(
                prices_df,
                (
                    f"Hourly close prices for {instrument_name} (last {hours_before}h)"
                    if close_only
                    else f"Hourly prices for {instrument_name} (last {hours_before}h)"
                ),
                rows,
            )

            spread_view = None
            if perpetual_name:
                perp_df = fetch_hourly_prices(
                    perpetual_name,
                    expiration_date,
                    hours_before=hours_before,
                    close_only=True,
                )
                if not perp_df.empty and not prices_df.empty and "close" in prices_df.columns:
                    future_for_merge = prices_df.rename(columns={"close": "future_close"}).copy()
                    if "instrument_name" in future_for_merge.columns:
                        future_for_merge = future_for_merge.rename(
                            columns={"instrument_name": "future_instrument"}
                        )
                    perp_close = perp_df[["timestamp", "close"]].rename(
                        columns={"close": "perpetual_close"}
                    )
                    spread_df = future_for_merge.merge(perp_close, on="timestamp", how="inner")
                    if spread_df.empty:
                        spread_view = spread_df
                    else:
                        if "future_instrument" in spread_df.columns:
                            spread_view = spread_df[
                                [
                                    "future_instrument",
                                    "timestamp",
                                    "future_close",
                                    "perpetual_close",
                                ]
                            ].rename(columns={"future_instrument": "instrument_name"})
                        else:
                            spread_view = spread_df[[
                                "timestamp",
                                "future_close",
                                "perpetual_close",
                            ]]
                        spread_view = spread_view.assign(
                            close_diff=
                            spread_view["future_close"] - spread_view["perpetual_close"],
                            hours_to_maturity=(
                                spread_view["timestamp"] - expiration_date
                            ).dt.total_seconds()
                            / 3600.0,
                        )

            if spread_view is not None:
                spread_title = (
                    f"Close price spread vs {perpetual_name}"
                    if perpetual_name
                    else "Close price spread (perpetual instrument unavailable)"
                )
                _print_dataframe(spread_view, spread_title, rows)

                if effective_plot_spread and not spread_view.empty:
                    plot_traces.append((instrument_name, spread_view.copy()))

            if (
                plot_reference_comparison
                and prices_df is not None
                and not prices_df.empty
                and reference_instrument_name
            ):
                reference_prices = fetch_hourly_prices(
                    reference_instrument_name,
                    expiration_date,
                    hours_before=hours_before,
                    close_only=True,
                )
                if reference_prices.empty:
                    print(
                        f"  Reference future {reference_instrument_name} has no overlapping data for {instrument_name}."
                    )
                else:
                    plot_price_comparison_lines(
                        currency,
                        instrument_name,
                        prices_df,
                        reference_instrument_name,
                        reference_prices,
                        export_path,
                    )

                    future_for_merge = prices_df.rename(columns={"close": "future_close"}).copy()
                    if "instrument_name" in future_for_merge.columns:
                        future_for_merge = future_for_merge.rename(
                            columns={"instrument_name": "future_instrument"}
                        )
                    reference_close = reference_prices[["timestamp", "close"]].rename(
                        columns={"close": "reference_close"}
                    )
                    reference_spread = future_for_merge.merge(
                        reference_close, on="timestamp", how="inner"
                    )
                    if not reference_spread.empty:
                        if "future_instrument" in reference_spread.columns:
                            reference_view = reference_spread[
                                [
                                    "future_instrument",
                                    "timestamp",
                                    "future_close",
                                    "reference_close",
                                ]
                            ].rename(columns={"future_instrument": "instrument_name"})
                        else:
                            reference_view = reference_spread[[
                                "timestamp",
                                "future_close",
                                "reference_close",
                            ]]
                            reference_view.insert(0, "instrument_name", instrument_name)

                        reference_view = reference_view.assign(
                            close_diff=reference_view["future_close"]
                            - reference_view["reference_close"],
                            hours_to_maturity=(
                                reference_view["timestamp"] - expiration_date
                            ).dt.total_seconds()
                            / 3600.0,
                        )

                        _print_dataframe(
                            reference_view,
                            f"Close price diff vs {reference_instrument_name}",
                            rows,
                        )

                        reference_diff_traces.append((instrument_name, reference_view.copy()))
                        reference_view_for_export = reference_view

            if export_path and not prices_df.empty:
                suffix = "close" if close_only else "ohlc"
                file_path = (
                    export_path
                    / f"{instrument_name}_hourly_{hours_before}h_{suffix}.csv"
                )
                prices_df.to_csv(file_path, index=False)

                if spread_view is not None and not spread_view.empty:
                    spread_file = (
                        export_path
                        / f"{instrument_name}_hourly_{hours_before}h_spread.csv"
                    )
                    spread_view.to_csv(spread_file, index=False)

                if (
                    plot_reference_comparison
                    and reference_instrument_name
                    and reference_view_for_export is not None
                    and not reference_view_for_export.empty
                ):
                    ref_file = (
                        export_path
                        / f"{instrument_name}_hourly_{hours_before}h_reference_spread.csv"
                    )
                    reference_view_for_export.to_csv(ref_file, index=False)

        if export_path:
            print(f"Hourly price CSV files saved to {export_path}\n")

        if effective_plot_spread:
            plot_close_diff_lines(currency, perpetual_name, plot_traces, export_path)

        if plot_reference_comparison and reference_instrument_name:
            plot_reference_diff_lines(
                currency,
                reference_instrument_name,
                reference_diff_traces,
                export_path,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--currencies",
        nargs="+",
        default=["BTC", "ETH"],
        help="List of currencies to query (default: BTC ETH)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of settlement rows to request per API call (default: 100)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum number of settlement pages to fetch per currency (<=0 to fetch all).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Rows of each DataFrame preview to print (default: 10)",
    )
    parser.add_argument(
        "--show-instruments",
        action="store_true",
        help="Also print the direct matured instruments listing.",
    )
    parser.add_argument(
        "--show-history",
        action="store_true",
        help="Also print the raw settlement history table.",
    )
    parser.add_argument(
        "--hours-before",
        type=int,
        default=240,
        help="Hours before expiration to include when fetching hourly prices (default: 240).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Directory to write hourly price CSV files (optional).",
    )
    parser.add_argument(
        "--close-only",
        action="store_true",
        help="Only include close price (timestamp + close) in hourly output.",
    )
    parser.add_argument(
        "--include-perp-spread",
        action="store_true",
        help="Also fetch perpetual close prices and compute spreads.",
    )
    parser.add_argument(
        "--plot-spread",
        action="store_true",
        help="Plot the close diff lines for each matured instrument (requires --include-perp-spread).",
    )
    parser.add_argument(
        "--plot-reference-comparison",
        action="store_true",
        help="Plot price comparisons between each matured future and the longest-history active future.",
    )

    args = parser.parse_args()
    max_pages = args.max_pages if args.max_pages > 0 else None
    main(
        args.currencies,
        count=args.count,
        max_pages=max_pages,
        rows=args.rows,
        show_instruments=args.show_instruments,
        show_history=args.show_history,
        hours_before=args.hours_before,
        export_dir=args.export_dir,
        close_only=args.close_only,
        include_perp_spread=args.include_perp_spread,
        plot_spread=args.plot_spread,
        plot_reference_comparison=args.plot_reference_comparison,
    )
