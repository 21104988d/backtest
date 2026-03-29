#!/usr/bin/env python3
"""Backtest basis arbitrage between MAG7 Yahoo Finance prices and Hyperliquid perps.

Strategy:
  For each stock (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA):
    1. Compute daily basis = hl_close / yahoo_close - 1
    2. Compute rolling z-score of basis (window = Z_WINDOW)
    3. Entry rules:
       - z >= ENTRY_Z  → short HL perp (HL is rich vs real stock)
       - z <= -ENTRY_Z → long HL perp  (HL is cheap vs real stock)
    4. Exit rule: |z| <= EXIT_Z → flat
    5. Fee drag applied at each turnover bar

Assumptions:
  - Only trading the HL perp leg (not executing real stock trades)
  - Position = +1 (long HL) or -1 (short HL) or 0 (flat)
  - Capital = $10,000 per pair, all-in per signal
  - Taker fee charged on entry + exit

Inputs (from fetch_data.py):
  data/aligned_mag7_daily.csv

Outputs:
  data/backtest_results.json     -- Per-ticker and portfolio metrics
  data/backtest_results.md       -- Human-readable summary
  data/backtest_signals_<TICK>.csv -- Per-ticker signal/equity series
  data/backtest_equity.png       -- Equity curves for all tickers

Usage:
  python backtest_basis_arb.py
  python backtest_basis_arb.py --z-window 20 --entry-z 1.5 --exit-z 0.5 --fee 0.00045
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MAG7 = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest MAG7 basis arb on HL vs Yahoo data.")
    parser.add_argument("--input", default="data/aligned_mag7_daily.csv", help="Aligned daily CSV")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--z-window", type=int, default=20, help="Rolling z-score window (bars)")
    parser.add_argument("--entry-z", type=float, default=2.0, help="Z-score threshold to enter")
    parser.add_argument("--exit-z", type=float, default=0.5, help="Z-score threshold to exit")
    parser.add_argument("--fee", type=float, default=0.00045, help="Taker fee rate per side")
    parser.add_argument("--capital", type=float, default=10000.0, help="Capital per pair in USD")
    return parser.parse_args()


# ── Pure-python stat helpers (no numpy dependency for core logic) ──────────────

def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def std_sample(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return float("nan")
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))


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


def max_drawdown(equity: list[float]) -> float:
    if not equity:
        return float("nan")
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = (v / peak) - 1.0 if peak > 0 else 0.0
        mdd = min(mdd, dd)
    return mdd


def annualized_sharpe(returns: list[float], periods_per_year: float = 252.0) -> float:
    if len(returns) < 2:
        return float("nan")
    mu = mean(returns)
    sigma = std_sample(returns)
    if math.isnan(sigma) or sigma == 0.0:
        return float("nan")
    return (mu / sigma) * math.sqrt(periods_per_year)


def winrate(returns: list[float]) -> float:
    wins = sum(1 for r in returns if r > 0)
    return wins / len(returns) if returns else float("nan")


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    dates: list[str],
    basis: list[float],
    z_window: int,
    entry_z: float,
    exit_z: float,
    fee_rate: float,
    capital: float,
) -> dict:
    """Run single-ticker basis-arb backtest. Returns metrics dict + series."""
    zscores = rolling_zscore(basis, z_window)

    position = [0] * len(dates)
    equity_gross = [capital] * len(dates)
    equity_net = [capital] * len(dates)
    bar_returns_gross: list[float] = []
    bar_returns_net: list[float] = []
    trade_log: list[dict] = []

    for i in range(1, len(dates)):
        z_prev = zscores[i - 1]
        z_curr = zscores[i]
        pos_prev = position[i - 1]

        if z_prev is None or z_curr is None:
            position[i] = 0
            equity_gross[i] = equity_gross[i - 1]
            equity_net[i] = equity_net[i - 1]
            bar_returns_gross.append(0.0)
            bar_returns_net.append(0.0)
            continue

        # Determine new position based on PREVIOUS z (no lookahead)
        if pos_prev == 0:
            if z_prev >= entry_z:
                new_pos = -1  # short HL (HL rich)
            elif z_prev <= -entry_z:
                new_pos = 1   # long HL (HL cheap)
            else:
                new_pos = 0
        else:
            if abs(z_prev) <= exit_z:
                new_pos = 0   # exit
            elif pos_prev == -1 and z_prev <= -entry_z:
                new_pos = 1   # flip
            elif pos_prev == 1 and z_prev >= entry_z:
                new_pos = -1  # flip
            else:
                new_pos = pos_prev  # hold

        position[i] = new_pos

        # Return for this bar: basis change × position
        # When long HL: profit when basis increases (HL rises vs Yahoo)
        # When short HL: profit when basis decreases
        basis_return = basis[i] - basis[i - 1]
        gross_ret = new_pos * basis_return

        # Fee: turnover × fee_rate (both sides = 2× if full round-trip at once)
        turnover = abs(new_pos - pos_prev)
        fee_cost = turnover * fee_rate  # one-side fee, scaled by position size

        net_ret = gross_ret - fee_cost

        equity_gross[i] = equity_gross[i - 1] * (1.0 + gross_ret)
        equity_net[i] = equity_net[i - 1] * (1.0 + net_ret)

        bar_returns_gross.append(gross_ret)
        bar_returns_net.append(net_ret)

        if turnover > 0:
            trade_log.append({
                "date": dates[i],
                "prev_pos": pos_prev,
                "new_pos": new_pos,
                "z_prev": round(z_prev, 4),
                "basis": round(basis[i], 6),
                "gross_ret": round(gross_ret, 6),
                "fee_cost": round(fee_cost, 6),
                "net_ret": round(net_ret, 6),
            })

    total_gross = equity_gross[-1] / capital - 1.0
    total_net = equity_net[-1] / capital - 1.0
    sharpe_gross = annualized_sharpe(bar_returns_gross)
    sharpe_net = annualized_sharpe(bar_returns_net)
    mdd_gross = max_drawdown(equity_gross)
    mdd_net = max_drawdown(equity_net)
    win_rate = winrate(bar_returns_net)
    n_trades = len(trade_log)

    return {
        "dates": dates,
        "basis": basis,
        "zscores": [z if z is not None else float("nan") for z in zscores],
        "position": position,
        "equity_gross": equity_gross,
        "equity_net": equity_net,
        "bar_returns_net": bar_returns_net,
        "trade_log": trade_log,
        "metrics": {
            "n_bars": len(dates),
            "n_trades": n_trades,
            "total_return_gross": round(total_gross * 100, 3),
            "total_return_net": round(total_net * 100, 3),
            "annualized_sharpe_gross": round(sharpe_gross, 3),
            "annualized_sharpe_net": round(sharpe_net, 3),
            "max_drawdown_gross_pct": round(mdd_gross * 100, 3),
            "max_drawdown_net_pct": round(mdd_net * 100, 3),
            "win_rate": round(win_rate * 100, 1),
        },
    }


def plot_all(results: dict[str, dict], out_path: Path) -> None:
    tickers = [t for t in results if "equity_net" in results[t]]
    n = len(tickers)
    cols = 2
    rows = math.ceil(n / cols) + 1  # +1 for portfolio

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes_flat = axes.flatten()

    portfolio_equity: list[float] | None = None

    for idx, ticker in enumerate(tickers):
        r = results[ticker]
        eq_net = r["equity_net"]
        eq_gross = r["equity_gross"]
        z = r["zscores"]
        pos = r["position"]
        dates_idx = list(range(len(r["dates"])))
        m = r["metrics"]

        ax = axes_flat[idx]
        ax.plot(dates_idx, eq_gross, color="#aaaaaa", linewidth=1.0, label="Gross", alpha=0.6)
        ax.plot(dates_idx, eq_net, color="#2a9d8f", linewidth=1.4, label="Net")
        ax.set_title(
            f"{ticker}  Sharpe={m['annualized_sharpe_net']:.2f}  "
            f"MDD={m['max_drawdown_net_pct']:.1f}%  "
            f"Ret={m['total_return_net']:.1f}%",
            fontsize=9,
        )
        ax.axhline(10000.0, color="#888888", linewidth=0.6)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)

        if portfolio_equity is None:
            portfolio_equity = list(eq_net)
        else:
            for j in range(min(len(portfolio_equity), len(eq_net))):
                portfolio_equity[j] += eq_net[j] - 10000.0

    # Portfolio equity = sum of individual net equities
    if portfolio_equity:
        ax_port = axes_flat[n]
        ax_port.plot(list(range(len(portfolio_equity))), portfolio_equity, color="#c0392b", linewidth=1.5)
        ax_port.set_title("Portfolio (sum of all pairs)", fontsize=9)
        ax_port.axhline(10000.0 * n, color="#888888", linewidth=0.6)
        ax_port.grid(alpha=0.2)

    # Hide unused axes
    for i in range(n + 1, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("MAG7 vs Hyperliquid Basis Arb — Equity Curves", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run fetch_data.py first."
        )

    df = pd.read_csv(input_path)
    required = {"ticker", "date", "yahoo_close", "hl_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {sorted(missing)}")

    tickers = sorted(df["ticker"].unique())

    print("=" * 70)
    print("MAG7 vs HYPERLIQUID BASIS ARB BACKTEST")
    print("=" * 70)
    print(f"Parameters: z_window={args.z_window}, entry_z={args.entry_z}, "
          f"exit_z={args.exit_z}, fee={args.fee*10000:.1f}bps")
    print(f"Tickers: {tickers}")
    print()

    all_results: dict[str, dict] = {}

    for ticker in tickers:
        sub = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if len(sub) < args.z_window + 10:
            print(f"  {ticker}: insufficient data ({len(sub)} bars), skipping")
            continue

        dates = sub["date"].tolist()
        yahoo_close = sub["yahoo_close"].tolist()
        hl_close = sub["hl_close"].tolist()

        # Basis: raw ratio
        basis = [hl_close[i] / yahoo_close[i] - 1.0 for i in range(len(dates))]

        result = run_backtest(
            dates=dates,
            basis=basis,
            z_window=args.z_window,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            fee_rate=args.fee,
            capital=args.capital,
        )

        m = result["metrics"]
        print(
            f"  {ticker:6s}: Bars={m['n_bars']:3d}  Trades={m['n_trades']:3d}  "
            f"Ret={m['total_return_net']:+6.1f}%  "
            f"Sharpe={m['annualized_sharpe_net']:+5.2f}  "
            f"MDD={m['max_drawdown_net_pct']:5.1f}%  "
            f"WinRate={m['win_rate']:.0f}%"
        )

        # Save per-ticker signals CSV
        sig_path = out_dir / f"backtest_signals_{ticker}.csv"
        with sig_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "yahoo_close", "hl_close", "basis", "zscore", "position", "equity_net"])
            writer.writeheader()
            for i, d in enumerate(result["dates"]):
                writer.writerow({
                    "date": d,
                    "yahoo_close": round(yahoo_close[i], 4),
                    "hl_close": round(hl_close[i], 4),
                    "basis": round(basis[i], 6),
                    "zscore": round(result["zscores"][i], 4) if not math.isnan(result["zscores"][i]) else "",
                    "position": result["position"][i],
                    "equity_net": round(result["equity_net"][i], 4),
                })

        # Store for aggregate (strip large series from JSON results)
        all_results[ticker] = {
            "equity_net": result["equity_net"],
            "equity_gross": result["equity_gross"],
            "zscores": result["zscores"],
            "position": result["position"],
            "dates": result["dates"],
            "metrics": m,
        }

    # ── Portfolio aggregate ───────────────────────────────────────────────────
    if all_results:
        all_rets = []
        for ticker, r in all_results.items():
            all_rets.append(r["equity_net"])

        min_len = min(len(x) for x in all_rets)
        portfolio_equity = [sum(r[i] for r in all_rets) for i in range(min_len)]
        portfolio_start = args.capital * len(all_results)

        port_returns = [
            portfolio_equity[i] / portfolio_equity[i - 1] - 1.0
            for i in range(1, len(portfolio_equity))
        ]
        port_total = portfolio_equity[-1] / portfolio_start - 1.0
        port_sharpe = annualized_sharpe(port_returns)
        port_mdd = max_drawdown(portfolio_equity)

        print(f"\n  {'PORTFOLIO':6s}: Total={port_total*100:+.1f}%  "
              f"Sharpe={port_sharpe:+.2f}  MDD={port_mdd*100:.1f}%")

        all_results["__portfolio__"] = {
            "metrics": {
                "total_return_net": round(port_total * 100, 3),
                "annualized_sharpe_net": round(port_sharpe, 3),
                "max_drawdown_net_pct": round(port_mdd * 100, 3),
                "n_pairs": len(all_results) - 1,
                "capital_total": portfolio_start,
            }
        }

    # ── Save JSON metrics ─────────────────────────────────────────────────────
    json_out = out_dir / "backtest_results.json"
    metrics_only = {
        k: v.get("metrics", {}) for k, v in all_results.items()
    }
    params = {
        "z_window": args.z_window,
        "entry_z": args.entry_z,
        "exit_z": args.exit_z,
        "fee_rate": args.fee,
        "capital_per_pair": args.capital,
    }
    json_out.write_text(
        json.dumps({"params": params, "results": metrics_only}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {json_out}")

    # ── Markdown report ───────────────────────────────────────────────────────
    md_lines = [
        "# MAG7 vs Hyperliquid Basis Arb — Backtest Results\n",
        f"**Parameters**: z_window={args.z_window}, entry_z={args.entry_z}, "
        f"exit_z={args.exit_z}, fee={args.fee*10000:.1f}bps taker\n",
        "\n## Per-Ticker Results\n",
        "| Ticker | Bars | Trades | Total Return | Ann. Sharpe | Max Drawdown | Win Rate |",
        "|--------|------|--------|-------------|-------------|--------------|----------|",
    ]
    for ticker in tickers:
        if ticker not in all_results:
            continue
        m = all_results[ticker]["metrics"]
        md_lines.append(
            f"| {ticker} | {m.get('n_bars','-')} | {m.get('n_trades','-')} "
            f"| {m.get('total_return_net',0):+.1f}% "
            f"| {m.get('annualized_sharpe_net',0):+.2f} "
            f"| {m.get('max_drawdown_net_pct',0):.1f}% "
            f"| {m.get('win_rate',0):.0f}% |"
        )

    if "__portfolio__" in all_results:
        pm = all_results["__portfolio__"]["metrics"]
        md_lines += [
            "\n## Portfolio Summary\n",
            f"| Total Return | Ann. Sharpe | Max Drawdown |",
            f"|-------------|-------------|--------------|",
            f"| {pm.get('total_return_net',0):+.1f}% | {pm.get('annualized_sharpe_net',0):+.2f} | {pm.get('max_drawdown_net_pct',0):.1f}% |",
        ]

    md_lines += [
        "\n## Strategy Notes\n",
        "- **Basis** = HL perp close / Yahoo Finance close − 1",
        "- **Signal**: Rolling z-score of basis. Short HL when rich, long HL when cheap.",
        "- **Only HL leg traded** (not executing actual stock buys/sells).",
        "- **Fee**: Taker fee on each side of trade (entry + exit).",
        "- **Limitation**: HL stock perps were launched ~Sept 2024; limited history.",
    ]

    md_out = out_dir / "backtest_results.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved: {md_out}")

    # ── Equity chart ──────────────────────────────────────────────────────────
    chart_out = out_dir / "backtest_equity.png"
    plot_all(all_results, chart_out)
    print(f"Saved: {chart_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
