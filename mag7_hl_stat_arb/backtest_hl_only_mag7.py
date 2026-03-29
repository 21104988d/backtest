#!/usr/bin/env python3
"""HL-only MAG7 daily close backtest.

Uses ONLY Hyperliquid daily closes from data/hl_mag7_daily.csv.
Strategy per ticker:
- Build hedge leg as equal-weight basket of the OTHER 6 MAG7 names
- Compute returns for ticker and basket
- Estimate rolling beta (past-only)
- Hedged return = r_ticker - beta * r_basket
- Build cumulative spread and rolling z-score
- Mean-reversion rules:
  * enter short spread when z >= entry_z
  * enter long spread when z <= -entry_z
  * exit when |z| <= exit_z
- Apply taker fee on turnover

Outputs:
- data/hl_only_backtest_results.json
- data/hl_only_backtest_results.md
- data/hl_only_backtest_equity.png
- data/hl_only_signals_<TICKER>.csv
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
    p = argparse.ArgumentParser(description="Backtest HL-only MAG7 stat-arb.")
    p.add_argument("--input", default="data/hl_mag7_daily.csv")
    p.add_argument("--output-dir", default="data")
    p.add_argument("--beta-window", type=int, default=30)
    p.add_argument("--z-window", type=int, default=20)
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--exit-z", type=float, default=0.5)
    p.add_argument("--fee", type=float, default=0.00045)
    p.add_argument("--capital", type=float, default=10000.0)
    return p.parse_args()


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def std_sample(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return float("nan")
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))


def annualized_sharpe(returns: list[float], periods_per_year: float = 252.0) -> float:
    if len(returns) < 2:
        return float("nan")
    mu = mean(returns)
    sigma = std_sample(returns)
    if math.isnan(sigma) or sigma == 0.0:
        return float("nan")
    return (mu / sigma) * math.sqrt(periods_per_year)


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


def ols_beta(y: list[float], x: list[float]) -> float:
    mx = mean(x)
    my = mean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    if sxx <= 0:
        return 0.0
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(len(y)))
    return sxy / sxx


def rolling_beta_past_only(y: list[float], x: list[float], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(y)
    for i in range(window, len(y)):
        out[i] = ols_beta(y[i - window : i], x[i - window : i])
    return out


def rolling_zscore(series: list[float], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(series)
    for i in range(window, len(series)):
        vals = series[i - window : i]
        mu = mean(vals)
        sd = std_sample(vals)
        out[i] = 0.0 if math.isnan(sd) or sd == 0.0 else (series[i] - mu) / sd
    return out


def run_strategy(
    dates: list[str],
    asset_ret: list[float],
    basket_ret: list[float],
    beta_window: int,
    z_window: int,
    entry_z: float,
    exit_z: float,
    fee: float,
    capital: float,
) -> dict:
    beta = rolling_beta_past_only(asset_ret, basket_ret, beta_window)

    hedged_ret = []
    for i in range(len(asset_ret)):
        b = beta[i]
        hedged_ret.append(0.0 if b is None else (asset_ret[i] - b * basket_ret[i]))

    spread = []
    level = 0.0
    for r in hedged_ret:
        level += r
        spread.append(level)

    z = rolling_zscore(spread, z_window)

    position = [0] * len(asset_ret)
    eq_gross = [capital] * len(asset_ret)
    eq_net = [capital] * len(asset_ret)
    bar_net = []
    trades = 0

    for i in range(1, len(asset_ret)):
        z_prev = z[i - 1]
        pos_prev = position[i - 1]

        if z_prev is None:
            position[i] = 0
            eq_gross[i] = eq_gross[i - 1]
            eq_net[i] = eq_net[i - 1]
            bar_net.append(0.0)
            continue

        if pos_prev == 0:
            if z_prev >= entry_z:
                pos = -1
            elif z_prev <= -entry_z:
                pos = 1
            else:
                pos = 0
        else:
            if abs(z_prev) <= exit_z:
                pos = 0
            elif pos_prev == -1 and z_prev <= -entry_z:
                pos = 1
            elif pos_prev == 1 and z_prev >= entry_z:
                pos = -1
            else:
                pos = pos_prev

        position[i] = pos
        turnover = abs(pos - pos_prev)
        if turnover > 0:
            trades += 1

        gross = pos * hedged_ret[i]
        cost = turnover * fee
        net = gross - cost

        eq_gross[i] = eq_gross[i - 1] * (1.0 + gross)
        eq_net[i] = eq_net[i - 1] * (1.0 + net)
        bar_net.append(net)

    total_net = eq_net[-1] / capital - 1.0
    return {
        "dates": dates,
        "beta": [float("nan") if b is None else b for b in beta],
        "spread": spread,
        "zscore": [float("nan") if v is None else v for v in z],
        "position": position,
        "equity_net": eq_net,
        "equity_gross": eq_gross,
        "metrics": {
            "bars": len(asset_ret),
            "trades": trades,
            "total_return_net_pct": round(total_net * 100, 3),
            "ann_sharpe_net": round(annualized_sharpe(bar_net), 3),
            "max_drawdown_net_pct": round(max_drawdown(eq_net) * 100, 3),
        },
    }


def plot_equity(results: dict[str, dict], out_file: Path, capital: float) -> None:
    tickers = [t for t in results.keys() if t != "__portfolio__"]
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axs = axes.flatten()

    port_eq = None
    for i, t in enumerate(tickers):
        r = results[t]
        x = list(range(len(r["equity_net"])))
        axs[i].plot(x, r["equity_gross"], color="#999999", linewidth=1.0, alpha=0.6, label="Gross")
        axs[i].plot(x, r["equity_net"], color="#1f77b4", linewidth=1.4, label="Net")
        m = r["metrics"]
        axs[i].set_title(f"{t} | Sharpe={m['ann_sharpe_net']:.2f} | Ret={m['total_return_net_pct']:+.1f}%")
        axs[i].axhline(capital, color="#666", linewidth=0.6)
        axs[i].grid(alpha=0.25)

        if port_eq is None:
            port_eq = r["equity_net"].copy()
        else:
            for j in range(min(len(port_eq), len(r["equity_net"]))):
                port_eq[j] += (r["equity_net"][j] - capital)

    if port_eq is not None:
        idx = len(tickers)
        axs[idx].plot(range(len(port_eq)), port_eq, color="#d62728", linewidth=1.6)
        axs[idx].axhline(capital * len(tickers), color="#666", linewidth=0.6)
        axs[idx].set_title("Portfolio (sum of 7 strategies)")
        axs[idx].grid(alpha=0.25)

    for k in range(len(tickers) + 1, len(axs)):
        axs[k].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input: {input_path}")

    df = pd.read_csv(input_path)
    need = {"date", "ticker", "close"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # pivot to close matrix and keep complete rows across all MAG7
    close_mat = df[df["ticker"].isin(MAG7)].pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    close_mat = close_mat.sort_index()
    close_mat = close_mat.dropna(subset=MAG7)

    if len(close_mat) < max(args.beta_window + 5, args.z_window + 5):
        raise RuntimeError(f"Not enough aligned bars across all MAG7: {len(close_mat)}")

    dates = close_mat.index.tolist()
    returns = close_mat.pct_change().dropna()
    ret_dates = returns.index.tolist()

    results: dict[str, dict] = {}

    print("=" * 72)
    print("HL-ONLY MAG7 DAILY CLOSE BACKTEST")
    print("=" * 72)
    print(f"Aligned bars (all 7 names): {len(close_mat)}")
    print(f"Return bars: {len(returns)}")
    print(f"Date range: {dates[0]} -> {dates[-1]}")
    print(
        f"Params: beta_window={args.beta_window}, z_window={args.z_window}, "
        f"entry_z={args.entry_z}, exit_z={args.exit_z}, fee={args.fee*10000:.2f} bps"
    )
    print()

    for ticker in MAG7:
        others = [t for t in MAG7 if t != ticker]
        asset_ret = returns[ticker].tolist()
        basket_ret = returns[others].mean(axis=1).tolist()

        out = run_strategy(
            dates=ret_dates,
            asset_ret=asset_ret,
            basket_ret=basket_ret,
            beta_window=args.beta_window,
            z_window=args.z_window,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            fee=args.fee,
            capital=args.capital,
        )
        results[ticker] = out
        m = out["metrics"]
        print(
            f"{ticker:6s}: trades={m['trades']:2d}  ret={m['total_return_net_pct']:+6.2f}%  "
            f"sharpe={m['ann_sharpe_net']:+5.2f}  mdd={m['max_drawdown_net_pct']:6.2f}%"
        )

        sig_out = out_dir / f"hl_only_signals_{ticker}.csv"
        with sig_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["date", "position", "zscore", "spread", "beta", "equity_net", "equity_gross"],
            )
            w.writeheader()
            for i in range(len(out["dates"])):
                z = out["zscore"][i]
                b = out["beta"][i]
                w.writerow(
                    {
                        "date": out["dates"][i],
                        "position": out["position"][i],
                        "zscore": "" if math.isnan(z) else round(z, 6),
                        "spread": round(out["spread"][i], 8),
                        "beta": "" if math.isnan(b) else round(b, 6),
                        "equity_net": round(out["equity_net"][i], 6),
                        "equity_gross": round(out["equity_gross"][i], 6),
                    }
                )

    # portfolio summary
    min_len = min(len(results[t]["equity_net"]) for t in MAG7)
    portfolio_eq = [0.0] * min_len
    for i in range(min_len):
        portfolio_eq[i] = sum(results[t]["equity_net"][i] for t in MAG7)

    p_ret = portfolio_eq[-1] / (args.capital * len(MAG7)) - 1.0
    p_bar = [portfolio_eq[i] / portfolio_eq[i - 1] - 1.0 for i in range(1, len(portfolio_eq))]
    p_sharpe = annualized_sharpe(p_bar)
    p_mdd = max_drawdown(portfolio_eq)

    results["__portfolio__"] = {
        "metrics": {
            "total_return_net_pct": round(p_ret * 100, 3),
            "ann_sharpe_net": round(p_sharpe, 3),
            "max_drawdown_net_pct": round(p_mdd * 100, 3),
            "capital_total": args.capital * len(MAG7),
            "pairs": len(MAG7),
        }
    }

    print("-" * 72)
    print(
        f"PORTFOLIO: ret={p_ret*100:+.2f}%  sharpe={p_sharpe:+.2f}  "
        f"mdd={p_mdd*100:.2f}%"
    )

    # save json
    json_out = out_dir / "hl_only_backtest_results.json"
    metrics_only = {k: v["metrics"] for k, v in results.items()}
    json_out.write_text(
        json.dumps(
            {
                "params": {
                    "beta_window": args.beta_window,
                    "z_window": args.z_window,
                    "entry_z": args.entry_z,
                    "exit_z": args.exit_z,
                    "fee": args.fee,
                    "capital": args.capital,
                },
                "aligned_bars_all_mag7": len(close_mat),
                "results": metrics_only,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # save markdown
    md_out = out_dir / "hl_only_backtest_results.md"
    lines = [
        "# HL-only MAG7 Backtest Results\n",
        "Uses only Hyperliquid daily closes from `hl_mag7_daily.csv`.\n",
        f"**Aligned bars (all 7):** {len(close_mat)}  ",
        f"**Date range:** {dates[0]} -> {dates[-1]}  ",
        f"**Params:** beta_window={args.beta_window}, z_window={args.z_window}, "
        f"entry_z={args.entry_z}, exit_z={args.exit_z}, fee={args.fee*10000:.2f} bps\n",
        "\n## Per-ticker\n",
        "| Ticker | Trades | Net Return | Ann Sharpe | Max Drawdown |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in MAG7:
        m = results[t]["metrics"]
        lines.append(
            f"| {t} | {m['trades']} | {m['total_return_net_pct']:+.2f}% | "
            f"{m['ann_sharpe_net']:+.2f} | {m['max_drawdown_net_pct']:.2f}% |"
        )

    pm = results["__portfolio__"]["metrics"]
    lines += [
        "\n## Portfolio\n",
        f"- Net return: **{pm['total_return_net_pct']:+.2f}%**",
        f"- Annualized Sharpe: **{pm['ann_sharpe_net']:+.2f}**",
        f"- Max drawdown: **{pm['max_drawdown_net_pct']:.2f}%**",
    ]
    md_out.write_text("\n".join(lines), encoding="utf-8")

    # save figure
    fig_out = out_dir / "hl_only_backtest_equity.png"
    plot_equity(results, fig_out, args.capital)

    print(f"Saved: {json_out}")
    print(f"Saved: {md_out}")
    print(f"Saved: {fig_out}")


if __name__ == "__main__":
    main()
