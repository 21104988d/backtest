#!/usr/bin/env python3
"""Stress test: fee levels and bid-ask spread impact on MAG7 basis arb.

Tests a grid of:
  - Taker fee rates: 0 to 10 bps
  - Bid-ask spread (per round-trip): 0 to 50 bps

For each combination, re-runs the backtest and records net Sharpe and return.
Also performs sensitivity analysis per ticker.

Inputs (from fetch_data.py + backtest_basis_arb.py):
  data/aligned_mag7_daily.csv

Outputs:
  data/stress_fee_spread_heatmap.png    -- Heatmap of portfolio Sharpe vs fee+spread
  data/stress_fee_spread_results.csv    -- Raw grid results
  data/stress_fee_spread_report.md      -- Summary report with breakeven analysis

Usage:
  python stress_test_fees.py
  python stress_test_fees.py --z-window 20 --entry-z 2.0 --exit-z 0.5
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test fees and bid-ask spread for MAG7 basis arb.")
    parser.add_argument("--input", default="data/aligned_mag7_daily.csv")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--z-window", type=int, default=20)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--capital", type=float, default=10000.0)
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def std_sample(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return float("nan")
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))


def annualized_sharpe(returns: list[float], periods_per_year: float = 252.0) -> float:
    if len(returns) < 5:
        return float("nan")
    mu = mean(returns)
    sigma = std_sample(returns)
    if math.isnan(sigma) or sigma == 0.0:
        return float("nan")
    return (mu / sigma) * math.sqrt(periods_per_year)


def rolling_zscore(series: list[float], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(series)
    for i in range(window, len(series)):
        window_vals = series[i - window : i]
        mu = mean(window_vals)
        sigma = std_sample(window_vals)
        if math.isnan(sigma) or sigma == 0.0:
            out[i] = 0.0
        else:
            out[i] = (series[i] - mu) / sigma
    return out


def run_fast_backtest(
    basis: list[float],
    z_window: int,
    entry_z: float,
    exit_z: float,
    total_cost_per_trade: float,  # fee + half_spread (one full round-trip entry→exit)
) -> float:
    """Returns annualized net Sharpe. Streamlined for grid search."""
    zscores = rolling_zscore(basis, z_window)
    position = 0
    bar_returns: list[float] = []

    for i in range(1, len(basis)):
        z_prev = zscores[i - 1]
        if z_prev is None:
            bar_returns.append(0.0)
            continue

        prev_pos = position
        if prev_pos == 0:
            if z_prev >= entry_z:
                position = -1
            elif z_prev <= -entry_z:
                position = 1
        else:
            if abs(z_prev) <= exit_z:
                position = 0
            elif prev_pos == -1 and z_prev <= -entry_z:
                position = 1
            elif prev_pos == 1 and z_prev >= entry_z:
                position = -1

        basis_ret = basis[i] - basis[i - 1]
        gross = position * basis_ret
        turnover = abs(position - prev_pos)
        cost = turnover * total_cost_per_trade
        bar_returns.append(gross - cost)

    return annualized_sharpe(bar_returns)


def run_fast_backtest_full(
    basis: list[float],
    z_window: int,
    entry_z: float,
    exit_z: float,
    total_cost_per_trade: float,
    capital: float,
) -> tuple[float, float, float]:
    """Returns (sharpe, total_return_pct, max_drawdown_pct)."""
    zscores = rolling_zscore(basis, z_window)
    position = 0
    bar_returns: list[float] = []
    equity = [capital]

    for i in range(1, len(basis)):
        z_prev = zscores[i - 1]
        if z_prev is None:
            bar_returns.append(0.0)
            equity.append(equity[-1])
            continue

        prev_pos = position
        if prev_pos == 0:
            if z_prev >= entry_z:
                position = -1
            elif z_prev <= -entry_z:
                position = 1
        else:
            if abs(z_prev) <= exit_z:
                position = 0
            elif prev_pos == -1 and z_prev <= -entry_z:
                position = 1
            elif prev_pos == 1 and z_prev >= entry_z:
                position = -1

        basis_ret = basis[i] - basis[i - 1]
        gross = position * basis_ret
        turnover = abs(position - prev_pos)
        cost = turnover * total_cost_per_trade
        net = gross - cost
        bar_returns.append(net)
        equity.append(equity[-1] * (1.0 + net))

    sharpe = annualized_sharpe(bar_returns)

    total_ret = equity[-1] / capital - 1.0 if len(equity) > 1 else 0.0

    peak = equity[0]
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = (v / peak) - 1.0 if peak > 0 else 0.0
        mdd = min(mdd, dd)

    return sharpe, total_ret * 100, mdd * 100


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found. Run fetch_data.py first.")

    df = pd.read_csv(input_path)
    tickers = sorted(df["ticker"].unique())

    # Build per-ticker basis series
    ticker_basis: dict[str, list[float]] = {}
    for ticker in tickers:
        sub = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if len(sub) < args.z_window + 10:
            continue
        basis = (sub["hl_close"] / sub["yahoo_close"] - 1.0).tolist()
        ticker_basis[ticker] = basis

    print("=" * 70)
    print("STRESS TEST: FEE + BID-ASK SPREAD SENSITIVITY")
    print("=" * 70)
    print(f"Tickers: {list(ticker_basis.keys())}")

    # ── Grid definition ───────────────────────────────────────────────────────
    # Fee rates per side (bps)
    fee_bps = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0]
    # Bid-ask spread (round-trip, bps) — total slippage cost
    spread_bps = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]

    print(f"\nGrid: {len(fee_bps)} fee levels × {len(spread_bps)} spread levels = {len(fee_bps)*len(spread_bps)} combinations")
    print("Running grid search...\n")

    # Grid: for each (fee, spread), compute portfolio-average Sharpe
    grid_sharpe = np.full((len(fee_bps), len(spread_bps)), float("nan"))
    grid_return = np.full((len(fee_bps), len(spread_bps)), float("nan"))
    grid_mdd = np.full((len(fee_bps), len(spread_bps)), float("nan"))

    csv_rows: list[dict] = []

    for fi, fee in enumerate(fee_bps):
        for si, spread in enumerate(spread_bps):
            # Total cost per trade: 2× taker fee (entry + exit) + spread
            # Entry: 1 taker fee + half spread
            # Exit: 1 taker fee + half spread
            # Total: 2× fee + full spread (in fraction terms)
            total_cost = (fee / 10000) * 2 + (spread / 10000)

            sharpes, returns, mdds = [], [], []
            for ticker, basis in ticker_basis.items():
                s, r, m = run_fast_backtest_full(
                    basis=basis,
                    z_window=args.z_window,
                    entry_z=args.entry_z,
                    exit_z=args.exit_z,
                    total_cost_per_trade=total_cost,
                    capital=args.capital,
                )
                if not math.isnan(s):
                    sharpes.append(s)
                    returns.append(r)
                    mdds.append(m)

            if sharpes:
                avg_sharpe = sum(sharpes) / len(sharpes)
                avg_return = sum(returns) / len(returns)
                avg_mdd = sum(mdds) / len(mdds)
                grid_sharpe[fi, si] = avg_sharpe
                grid_return[fi, si] = avg_return
                grid_mdd[fi, si] = avg_mdd

                csv_rows.append({
                    "fee_bps": fee,
                    "spread_bps_roundtrip": spread,
                    "total_cost_bps": fee * 2 + spread,
                    "avg_sharpe": round(avg_sharpe, 3),
                    "avg_return_pct": round(avg_return, 3),
                    "avg_mdd_pct": round(avg_mdd, 3),
                    "n_tickers": len(sharpes),
                })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_out = out_dir / "stress_fee_spread_results.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved: {csv_out} ({len(csv_rows)} rows)")

    # ── Per-ticker detailed stress test ───────────────────────────────────────
    print("\nPer-ticker stress at baseline fee (4.5bps) + varying spread:")
    baseline_fee = 4.5
    print(f"  {'Ticker':6s}", end="")
    for sp in spread_bps:
        print(f"  {sp:4.0f}bps", end="")
    print()

    for ticker, basis in ticker_basis.items():
        print(f"  {ticker:6s}", end="")
        for sp in spread_bps:
            total = (baseline_fee / 10000) * 2 + (sp / 10000)
            s = run_fast_backtest(basis, args.z_window, args.entry_z, args.exit_z, total)
            marker = "✓" if s > 0.5 else ("~" if s > 0 else "✗")
            print(f"  {s:+5.2f}{marker}", end="")
        print()

    # ── Breakeven analysis ────────────────────────────────────────────────────
    print("\nBreakeven total cost (where avg Sharpe = 0):")
    breakeven_costs = []
    prev_row = None
    for row in sorted(csv_rows, key=lambda r: r["total_cost_bps"]):
        if prev_row is not None and prev_row["avg_sharpe"] > 0 >= row["avg_sharpe"]:
            # Linear interpolation
            x1, y1 = prev_row["total_cost_bps"], prev_row["avg_sharpe"]
            x2, y2 = row["total_cost_bps"], row["avg_sharpe"]
            be = x1 - y1 * (x2 - x1) / (y2 - y1) if (y2 - y1) != 0 else float("nan")
            breakeven_costs.append(be)
        prev_row = row

    if breakeven_costs:
        be_avg = sum(breakeven_costs) / len(breakeven_costs)
        print(f"  Approximate breakeven: {be_avg:.1f} bps total round-trip cost")
        print(f"  (At HL taker 4.5bps × 2 = 9bps, breakeven spread allowance: {max(0, be_avg - 9.0):.1f}bps)")
    else:
        print("  No clear crossover in tested range")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Sharpe heatmap
    im1 = axes[0].imshow(
        grid_sharpe,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=2.0,
    )
    axes[0].set_xticks(range(len(spread_bps)))
    axes[0].set_xticklabels([f"{s:.0f}" for s in spread_bps], fontsize=8)
    axes[0].set_yticks(range(len(fee_bps)))
    axes[0].set_yticklabels([f"{f:.1f}" for f in fee_bps], fontsize=8)
    axes[0].set_xlabel("Bid-ask spread (round-trip bps)")
    axes[0].set_ylabel("Taker fee (bps per side)")
    axes[0].set_title("Portfolio Avg Annualized Sharpe")
    plt.colorbar(im1, ax=axes[0])

    for fi in range(len(fee_bps)):
        for si in range(len(spread_bps)):
            v = grid_sharpe[fi, si]
            if not math.isnan(v):
                axes[0].text(si, fi, f"{v:.2f}", ha="center", va="center", fontsize=7,
                             color="black" if -0.5 < v < 1.5 else "white")

    # Return heatmap
    im2 = axes[1].imshow(
        grid_return,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=-20,
        vmax=40,
    )
    axes[1].set_xticks(range(len(spread_bps)))
    axes[1].set_xticklabels([f"{s:.0f}" for s in spread_bps], fontsize=8)
    axes[1].set_yticks(range(len(fee_bps)))
    axes[1].set_yticklabels([f"{f:.1f}" for f in fee_bps], fontsize=8)
    axes[1].set_xlabel("Bid-ask spread (round-trip bps)")
    axes[1].set_ylabel("Taker fee (bps per side)")
    axes[1].set_title("Portfolio Avg Total Return (%)")
    plt.colorbar(im2, ax=axes[1])

    for fi in range(len(fee_bps)):
        for si in range(len(spread_bps)):
            v = grid_return[fi, si]
            if not math.isnan(v):
                axes[1].text(si, fi, f"{v:.1f}%", ha="center", va="center", fontsize=6,
                             color="black")

    fig.suptitle(
        f"MAG7 Basis Arb — Fee & Spread Stress Test\n"
        f"(z_window={args.z_window}, entry_z={args.entry_z}, exit_z={args.exit_z})",
        fontsize=11,
    )
    fig.tight_layout()
    heatmap_out = out_dir / "stress_fee_spread_heatmap.png"
    fig.savefig(heatmap_out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {heatmap_out}")

    # ── Markdown report ───────────────────────────────────────────────────────
    md_lines = [
        "# MAG7 Basis Arb — Fee & Spread Stress Test Report\n",
        f"**Strategy params**: z_window={args.z_window}, entry_z={args.entry_z}, exit_z={args.exit_z}\n",
        "\n## Fee + Spread Grid (Portfolio Avg Sharpe)\n",
        "| Fee (bps/side) \\ Spread RT (bps) |" + "".join(f" {s:.0f} |" for s in spread_bps),
        "|---|" + "".join("---|" for _ in spread_bps),
    ]
    for fi, fee in enumerate(fee_bps):
        row_str = f"| {fee:.1f} |"
        for si in range(len(spread_bps)):
            v = grid_sharpe[fi, si]
            row_str += f" {v:.2f} |" if not math.isnan(v) else " - |"
        md_lines.append(row_str)

    md_lines += [
        "\n## Interpretation\n",
        "- Green cells (Sharpe > 1.0): Strategy profitable after this cost level.",
        "- Yellow cells (0–1.0): Marginal profitability.",
        "- Red cells (< 0): Strategy unprofitable at this cost level.",
        "",
        f"### Trading Costs on Hyperliquid (Reference)",
        "| Fee Type | Rate | Notes |",
        "|----------|------|-------|",
        "| Taker fee (cross-margin) | 4.5 bps | Standard retail tier |",
        "| Maker fee | −1.1 bps | Rebate for limit orders |",
        "| Stock perp bid-ask spread | ~10–30 bps | Varies by liquidity and session |",
        "| Typical round-trip cost (taker) | ~19–39 bps | 2×4.5 fee + spread |",
        "",
        "### Breakeven Analysis\n",
    ]

    if breakeven_costs:
        md_lines += [
            f"- Breakeven total round-trip cost: ~**{be_avg:.1f} bps**",
            f"- At HL taker fee (9 bps round-trip), allowable spread: ~**{max(0, be_avg - 9.0):.1f} bps**",
            "",
            "> **Trading recommendation**: Use limit orders (maker) to reduce fees to −1.1 bps per side.",
            "> This converts 9 bps fee load to −2.2 bps, effectively expanding the cost budget for spread.",
        ]
    else:
        md_lines.append("- See heatmap for detailed breakeven visualization.")

    md_out = out_dir / "stress_fee_spread_report.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved: {md_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
