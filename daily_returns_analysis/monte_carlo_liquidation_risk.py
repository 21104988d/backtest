#!/usr/bin/env python3
"""Estimate liquidation/ruin risk for the selected stat-arb portfolio via Monte Carlo."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    DAILY_OHLC_FILE,
    INITIAL_CAPITAL,
    MIN_ASSETS_PER_DAY,
    POSITION_SIZE_FIXED,
    ROUND_TRIP_FEE_PCT,
    SLIPPAGE_BPS_PER_SIDE,
    SPREAD_BPS_PER_SIDE,
    STOP_EXTRA_SLIPPAGE_BPS,
)
from train_test_sl_search import apply_sl_return, resolve_sl_pct


def parse_float_csv(raw: str) -> List[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def parse_int_csv(raw: str) -> List[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo liquidation risk simulation")
    parser.add_argument("--n", type=int, default=5, help="Top/bottom N assets traded")
    parser.add_argument(
        "--sl-mode",
        choices=["none", "fixed", "dynamic"],
        default="fixed",
        help="Stop-loss mode for strategy replay",
    )
    parser.add_argument("--sl-param", type=float, default=1.68, help="Stop-loss parameter")
    parser.add_argument(
        "--nonoverlap-windows",
        default="hypothesis_168_nonoverlap_windows.csv",
        help="Windows CSV used to define strict non-overlap OOS dates",
    )
    parser.add_argument(
        "--horizons",
        default="252,365,730,1095",
        help="Comma-separated MC horizons in trading days",
    )
    parser.add_argument(
        "--floor-ratios",
        default="0.0,0.1,0.25,0.5",
        help="Comma-separated equity floors as ratio of initial capital",
    )
    parser.add_argument("--paths", type=int, default=50000, help="Number of Monte Carlo paths")
    parser.add_argument("--chunk-size", type=int, default=2500, help="Chunk size for simulation")
    parser.add_argument("--block-size", type=int, default=5, help="Block size for block bootstrap")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-prefix",
        default="monte_carlo_liquidation_risk",
        help="Output prefix for generated artifacts",
    )
    return parser.parse_args()


def bps_to_pct(value_bps: float) -> float:
    return value_bps / 100.0


def load_ohlc() -> pd.DataFrame:
    ohlc_df = pd.read_csv(DAILY_OHLC_FILE)
    ohlc_df["date"] = pd.to_datetime(ohlc_df["date"]).dt.date
    ohlc_df = ohlc_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["daily_return", "open", "high", "low", "close"]
    )
    return ohlc_df


def parse_nonoverlap_test_dates(
    windows_csv: Path,
    all_dates: Sequence,
    n: int,
    sl_mode: str,
    sl_param: float,
) -> Set:
    windows = pd.read_csv(windows_csv)
    sl_match = np.isclose(windows["sl_param"].astype(float).to_numpy(), float(sl_param), atol=1e-9)
    rows = windows[(windows["n"] == n) & (windows["sl_mode"] == sl_mode) & sl_match]
    if rows.empty:
        raise ValueError(
            "No rows found in non-overlap windows CSV for requested strategy: "
            f"n={n}, mode={sl_mode}, param={sl_param}"
        )

    test_ranges = [
        (
            pd.to_datetime(row["window_test_start"]).date(),
            pd.to_datetime(row["window_test_end"]).date(),
        )
        for _, row in rows.iterrows()
    ]

    selected_dates: Set = set()
    for d in all_dates:
        if any(start <= d <= end for start, end in test_ranges):
            selected_dates.add(d)
    return selected_dates


def compute_daily_pnl_series(
    ohlc_df: pd.DataFrame,
    all_dates: Sequence,
    active_dates: Set,
    n: int,
    sl_mode: str,
    sl_param: float,
    spread_bps_per_side: float,
    slippage_bps_per_side: float,
    stop_extra_slippage_bps: float,
) -> pd.DataFrame:
    required_assets = max(MIN_ASSETS_PER_DAY, n * 2)
    base_execution_cost_pct = 2 * (bps_to_pct(spread_bps_per_side) + bps_to_pct(slippage_bps_per_side))
    stop_extra_cost_pct = bps_to_pct(stop_extra_slippage_bps)

    by_date = {d: g for d, g in ohlc_df.groupby("date", sort=False)}

    equity = float(INITIAL_CAPITAL)
    rows: List[Dict[str, float]] = []

    for idx in range(1, len(all_dates)):
        signal_date = all_dates[idx - 1]
        trade_date = all_dates[idx]

        if signal_date not in active_dates or trade_date not in active_dates:
            continue

        signal_day = by_date.get(signal_date)
        trade_day = by_date.get(trade_date)
        if signal_day is None or trade_day is None:
            continue
        if len(signal_day) < required_assets:
            continue

        top_n = signal_day.nlargest(n, "daily_return")[["coin", "daily_return"]].values.tolist()
        bottom_n = signal_day.nsmallest(n, "daily_return")[["coin", "daily_return"]].values.tolist()

        trade_lookup = {str(r["coin"]): r for _, r in trade_day.iterrows()}

        daily_pnl = 0.0
        traded_positions = 0

        for coin, prev_ret in top_n:
            row = trade_lookup.get(str(coin))
            if row is None:
                continue
            sl_pct = resolve_sl_pct(sl_mode, sl_param, float(prev_ret))
            gross_ret_pct, stopped = apply_sl_return(row, is_long=False, sl_pct=sl_pct)
            extra_cost_pct = stop_extra_cost_pct if stopped else 0.0
            net_ret_pct = gross_ret_pct - ROUND_TRIP_FEE_PCT - base_execution_cost_pct - extra_cost_pct
            daily_pnl += (net_ret_pct / 100.0) * POSITION_SIZE_FIXED
            traded_positions += 1

        for coin, prev_ret in bottom_n:
            row = trade_lookup.get(str(coin))
            if row is None:
                continue
            sl_pct = resolve_sl_pct(sl_mode, sl_param, float(prev_ret))
            gross_ret_pct, stopped = apply_sl_return(row, is_long=True, sl_pct=sl_pct)
            extra_cost_pct = stop_extra_cost_pct if stopped else 0.0
            net_ret_pct = gross_ret_pct - ROUND_TRIP_FEE_PCT - base_execution_cost_pct - extra_cost_pct
            daily_pnl += (net_ret_pct / 100.0) * POSITION_SIZE_FIXED
            traded_positions += 1

        if traded_positions == 0:
            continue

        prior_equity = equity
        equity = equity + daily_pnl
        daily_return_pct = (daily_pnl / prior_equity) * 100.0 if prior_equity > 0 else np.nan

        rows.append(
            {
                "trade_date": pd.Timestamp(trade_date),
                "daily_pnl": float(daily_pnl),
                "daily_return_pct": float(daily_return_pct),
                "positions": int(traded_positions),
                "equity": float(equity),
            }
        )

    if not rows:
        raise ValueError("No daily pnl rows were generated for the selected strategy/dates")

    return pd.DataFrame(rows)


def bootstrap_iid(samples: np.ndarray, rng: np.random.Generator, n_paths: int, horizon: int) -> np.ndarray:
    idx = rng.integers(0, len(samples), size=(n_paths, horizon), endpoint=False)
    return samples[idx]


def bootstrap_block(
    samples: np.ndarray,
    rng: np.random.Generator,
    n_paths: int,
    horizon: int,
    block_size: int,
) -> np.ndarray:
    block_size = max(1, int(block_size))
    if block_size > len(samples):
        block_size = len(samples)

    n_blocks = int(math.ceil(horizon / block_size))
    max_start = len(samples) - block_size + 1
    starts = rng.integers(0, max_start, size=(n_paths, n_blocks), endpoint=False)

    offsets = np.arange(block_size)
    take_idx = starts[:, :, None] + offsets[None, None, :]
    draws = samples[take_idx].reshape(n_paths, n_blocks * block_size)
    return draws[:, :horizon]


def run_monte_carlo(
    samples: np.ndarray,
    initial_capital: float,
    floor_ratios: Sequence[float],
    horizons: Sequence[int],
    n_paths: int,
    chunk_size: int,
    method: str,
    block_size: int,
    seed: int,
    dataset_name: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    floors = {ratio: initial_capital * ratio for ratio in floor_ratios}

    out_rows: List[Dict[str, float]] = []

    for horizon in horizons:
        hit_counts = {ratio: 0 for ratio in floor_ratios}
        final_equity_acc: List[np.ndarray] = []
        min_equity_acc: List[np.ndarray] = []

        for start in range(0, n_paths, chunk_size):
            this_chunk = min(chunk_size, n_paths - start)
            if method == "iid":
                chunk_draws = bootstrap_iid(samples, rng, this_chunk, horizon)
            elif method == "block":
                chunk_draws = bootstrap_block(samples, rng, this_chunk, horizon, block_size)
            else:
                raise ValueError(f"Unsupported method: {method}")

            equity_paths = initial_capital + np.cumsum(chunk_draws, axis=1, dtype=np.float64)
            final_equity = equity_paths[:, -1]
            min_equity = np.min(equity_paths, axis=1)

            for ratio, floor_value in floors.items():
                hit_counts[ratio] += int(np.sum(min_equity <= floor_value))

            final_equity_acc.append(final_equity)
            min_equity_acc.append(min_equity)

        final_all = np.concatenate(final_equity_acc)
        min_all = np.concatenate(min_equity_acc)

        row: Dict[str, float] = {
            "dataset": dataset_name,
            "method": method,
            "horizon_days": int(horizon),
            "paths": int(n_paths),
            "final_equity_p01": float(np.quantile(final_all, 0.01)),
            "final_equity_p05": float(np.quantile(final_all, 0.05)),
            "final_equity_p50": float(np.quantile(final_all, 0.50)),
            "final_equity_p95": float(np.quantile(final_all, 0.95)),
            "min_equity_p01": float(np.quantile(min_all, 0.01)),
            "min_equity_p50": float(np.quantile(min_all, 0.50)),
            "prob_final_le_0_pct": float((final_all <= 0).mean() * 100.0),
        }

        for ratio in floor_ratios:
            label = f"prob_hit_floor_{int(round(ratio * 100)):02d}pct_pct"
            row[label] = float(hit_counts[ratio] / n_paths * 100.0)

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def dataset_stats(df: pd.DataFrame, dataset_name: str) -> Dict[str, float]:
    return {
        "dataset": dataset_name,
        "days": int(len(df)),
        "mean_daily_pnl": float(df["daily_pnl"].mean()),
        "std_daily_pnl": float(df["daily_pnl"].std(ddof=0)),
        "worst_daily_pnl": float(df["daily_pnl"].min()),
        "best_daily_pnl": float(df["daily_pnl"].max()),
        "win_day_rate_pct": float((df["daily_pnl"] > 0).mean() * 100.0),
        "observed_min_equity": float(df["equity"].min()),
        "observed_final_equity": float(df["equity"].iloc[-1]),
    }


def build_markdown_report(
    summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    report_path: Path,
    visual_path: Path,
    floor_ratios: Sequence[float],
) -> None:
    def fmt(v: float) -> str:
        if pd.isna(v):
            return "nan"
        return f"{v:.4f}"

    floor_zero_col = "prob_hit_floor_00pct_pct"
    floor_10_col = "prob_hit_floor_10pct_pct"

    lines: List[str] = [
        "# Monte Carlo Liquidation Risk",
        "",
        "Ruin is defined as path equity touching the floor at any point within the horizon.",
        "The key liquidation metric requested is probability of touching 0 equity.",
        "",
        "## Dataset Stats",
    ]

    for _, row in stats_df.iterrows():
        lines.extend(
            [
                (
                    f"- {row['dataset']}: days={int(row['days'])}, "
                    f"mean daily pnl={fmt(row['mean_daily_pnl'])}, "
                    f"std={fmt(row['std_daily_pnl'])}, "
                    f"worst day={fmt(row['worst_daily_pnl'])}, "
                    f"observed min equity={fmt(row['observed_min_equity'])}"
                )
            ]
        )

    lines.extend(["", "## Monte Carlo Results (Key)"])

    focus = summary_df[
        [
            "dataset",
            "method",
            "horizon_days",
            floor_zero_col,
            floor_10_col,
            "prob_final_le_0_pct",
            "final_equity_p01",
            "final_equity_p50",
        ]
    ].sort_values(["dataset", "method", "horizon_days"])

    for _, row in focus.iterrows():
        lines.append(
            (
                f"- {row['dataset']} | {row['method']} | {int(row['horizon_days'])}d: "
                f"P(hit<=0)={fmt(row[floor_zero_col])}% | "
                f"P(hit<=10% init)={fmt(row[floor_10_col])}% | "
                f"P(final<=0)={fmt(row['prob_final_le_0_pct'])}% | "
                f"Final equity p01={fmt(row['final_equity_p01'])} | "
                f"p50={fmt(row['final_equity_p50'])}"
            )
        )

    floors_text = ", ".join(f"{ratio * 100:.0f}%" for ratio in floor_ratios)
    lines.extend(
        [
            "",
            "## Notes",
            f"- Floors tested for touch probability: {floors_text} of initial capital.",
            "- IID bootstrap resamples single days independently.",
            "- Block bootstrap preserves short-run clustering (volatility/regime streaks).",
            "",
            "## Artifact",
            f"- Visual summary: {visual_path.name}",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(summary_df: pd.DataFrame, output_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for dataset, color in [("full_history", "#1f77b4"), ("nonoverlap_oos", "#ff7f0e")]:
        for method, marker in [("iid", "o"), ("block", "s")]:
            subset = summary_df[(summary_df["dataset"] == dataset) & (summary_df["method"] == method)].sort_values("horizon_days")
            if subset.empty:
                continue
            label = f"{dataset}:{method}"

            axes[0, 0].plot(
                subset["horizon_days"],
                subset["prob_hit_floor_00pct_pct"],
                marker=marker,
                color=color,
                linewidth=2,
                label=label,
            )
            axes[0, 1].plot(
                subset["horizon_days"],
                subset["prob_hit_floor_10pct_pct"],
                marker=marker,
                color=color,
                linewidth=2,
                label=label,
            )
            axes[1, 0].plot(
                subset["horizon_days"],
                subset["final_equity_p01"],
                marker=marker,
                color=color,
                linewidth=2,
                label=label,
            )
            axes[1, 1].plot(
                subset["horizon_days"],
                subset["final_equity_p50"],
                marker=marker,
                color=color,
                linewidth=2,
                label=label,
            )

    axes[0, 0].set_title("P(Equity Touches 0)")
    axes[0, 1].set_title("P(Equity Touches 10% Initial)")
    axes[1, 0].set_title("Final Equity 1st Percentile")
    axes[1, 1].set_title("Final Equity Median")

    for ax in axes.flat:
        ax.set_xlabel("Horizon (days)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0, 0].set_ylabel("Probability %")
    axes[0, 1].set_ylabel("Probability %")
    axes[1, 0].set_ylabel("Equity")
    axes[1, 1].set_ylabel("Equity")

    fig.suptitle("Monte Carlo Liquidation Risk: N=5 Fixed 1.68", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    horizons = sorted(set(parse_int_csv(args.horizons)))
    floor_ratios = sorted(set(parse_float_csv(args.floor_ratios)))

    if args.paths <= 0:
        raise ValueError("--paths must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if any(h <= 0 for h in horizons):
        raise ValueError("All horizons must be > 0")
    if any(r < 0 or r > 1 for r in floor_ratios):
        raise ValueError("Floor ratios must be within [0, 1]")

    ohlc_df = load_ohlc()
    all_dates = sorted(ohlc_df["date"].unique())

    nonoverlap_dates = parse_nonoverlap_test_dates(
        windows_csv=Path(args.nonoverlap_windows),
        all_dates=all_dates,
        n=args.n,
        sl_mode=args.sl_mode,
        sl_param=args.sl_param,
    )

    daily_full = compute_daily_pnl_series(
        ohlc_df=ohlc_df,
        all_dates=all_dates,
        active_dates=set(all_dates),
        n=args.n,
        sl_mode=args.sl_mode,
        sl_param=args.sl_param,
        spread_bps_per_side=SPREAD_BPS_PER_SIDE,
        slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
        stop_extra_slippage_bps=STOP_EXTRA_SLIPPAGE_BPS,
    )

    daily_nonoverlap = compute_daily_pnl_series(
        ohlc_df=ohlc_df,
        all_dates=all_dates,
        active_dates=nonoverlap_dates,
        n=args.n,
        sl_mode=args.sl_mode,
        sl_param=args.sl_param,
        spread_bps_per_side=SPREAD_BPS_PER_SIDE,
        slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
        stop_extra_slippage_bps=STOP_EXTRA_SLIPPAGE_BPS,
    )

    datasets = {
        "full_history": daily_full,
        "nonoverlap_oos": daily_nonoverlap,
    }

    all_summary = []
    all_stats = []

    for dataset_name, daily_df in datasets.items():
        all_stats.append(dataset_stats(daily_df, dataset_name))
        samples = daily_df["daily_pnl"].to_numpy(dtype=np.float64)

        for method in ("iid", "block"):
            sim_df = run_monte_carlo(
                samples=samples,
                initial_capital=float(INITIAL_CAPITAL),
                floor_ratios=floor_ratios,
                horizons=horizons,
                n_paths=args.paths,
                chunk_size=args.chunk_size,
                method=method,
                block_size=args.block_size,
                seed=args.seed,
                dataset_name=dataset_name,
            )
            all_summary.append(sim_df)

    summary_df = pd.concat(all_summary, ignore_index=True)
    stats_df = pd.DataFrame(all_stats)

    out_prefix = Path(args.output_prefix)
    summary_csv = Path(f"{out_prefix}_summary.csv")
    stats_csv = Path(f"{out_prefix}_dataset_stats.csv")
    report_md = Path(f"{out_prefix}_report.md")
    visual_png = Path(f"{out_prefix}.png")
    full_daily_csv = Path(f"{out_prefix}_daily_full_history.csv")
    nonoverlap_daily_csv = Path(f"{out_prefix}_daily_nonoverlap_oos.csv")

    summary_df.to_csv(summary_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    daily_full.to_csv(full_daily_csv, index=False)
    daily_nonoverlap.to_csv(nonoverlap_daily_csv, index=False)

    plot_summary(summary_df, visual_png)
    build_markdown_report(summary_df, stats_df, report_md, visual_png, floor_ratios)

    print(f"saved_summary {summary_csv}")
    print(f"saved_stats {stats_csv}")
    print(f"saved_daily_full {full_daily_csv}")
    print(f"saved_daily_nonoverlap {nonoverlap_daily_csv}")
    print(f"saved_report {report_md}")
    print(f"saved_visual {visual_png}")


if __name__ == "__main__":
    main()
