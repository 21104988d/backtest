import csv
import json
import math
import subprocess
import time
import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent
PYTHON = "/Users/leeisaackaiyui/Desktop/backtest/.venv/bin/python"
BACKTEST_SCRIPT = BASE / "rolling_beta_stat_arb.py"
WINDOWS = [40, 60, 90, 120]
BARS_PER_YEAR_4H = 2190.0


def symbol_to_filename(symbol: str) -> str:
    return symbol.lower().replace(":", "_").replace("/", "-")


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def std_sample(vals):
    n = len(vals)
    if n < 2:
        return float("nan")
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))


def sharpe_annualized(rets, bars_per_year):
    if len(rets) < 2:
        return float("nan")
    s = std_sample(rets)
    if not s or math.isnan(s) or s <= 0:
        return float("nan")
    return (mean(rets) / s) * math.sqrt(bars_per_year)


def max_drawdown_from_returns(rets):
    if not rets:
        return float("nan")
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    for r in rets:
        eq *= 1.0 + r
        peak = max(peak, eq)
        dd = (eq / peak) - 1.0 if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
    return mdd


def load_strategy_rets(path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(float(row["strategy_ret_net"]))
    return out


def run_one_window(window, end_ms, asset_symbol, btc_symbol):
    cmd = [
        PYTHON,
        str(BACKTEST_SCRIPT),
        "--asset",
        asset_symbol,
        "--btc",
        btc_symbol,
        "--interval",
        "4h",
        "--start-ms",
        "0",
        "--end-ms",
        str(end_ms),
        "--beta-window",
        str(window),
        "--taker-fee-rate",
        "0.00045",
    ]
    subprocess.run(cmd, cwd=str(BASE), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="ETH")
    parser.add_argument("--btc", default="BTC")
    args = parser.parse_args()

    rows = []
    end_ms = int(time.time() * 1000)
    asset_prefix = symbol_to_filename(args.asset)

    for w in WINDOWS:
        run_one_window(w, end_ms, args.asset, args.btc)

        metrics_path = BASE / f"{asset_prefix}_4h_rolling_beta_metrics.json"
        signals_path = BASE / f"{asset_prefix}_4h_rolling_beta_signals.csv"
        report_path = BASE / f"{asset_prefix}_4h_rolling_beta_report.md"

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rets = load_strategy_rets(signals_path)

        split = int(len(rets) * 0.7)
        is_rets = rets[:split]
        oos_rets = rets[split:]

        is_sh = sharpe_annualized(is_rets, BARS_PER_YEAR_4H)
        oos_sh = sharpe_annualized(oos_rets, BARS_PER_YEAR_4H)
        is_mdd = max_drawdown_from_returns(is_rets)
        oos_mdd = max_drawdown_from_returns(oos_rets)

        sharpe_decay = is_sh - oos_sh
        dd_stability_gap = abs(abs(oos_mdd) - abs(is_mdd))

        robust_score = oos_sh - (0.5 * max(0.0, sharpe_decay)) - (2.0 * abs(oos_mdd)) - dd_stability_gap

        rows.append(
            {
                "beta_window": w,
                "net_return_full": metrics.get("strategy_total_return_net", float("nan")),
                "net_sharpe_full": metrics.get("strategy_sharpe_365_net", float("nan")),
                "max_drawdown_full": metrics.get("max_drawdown_net", float("nan")),
                "is_sharpe": is_sh,
                "oos_sharpe": oos_sh,
                "is_mdd": is_mdd,
                "oos_mdd": oos_mdd,
                "sharpe_decay_is_minus_oos": sharpe_decay,
                "drawdown_stability_gap": dd_stability_gap,
                "robust_score": robust_score,
            }
        )

        suffix = f"{asset_prefix}_4h_bw{w}"
        (BASE / f"{suffix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (BASE / f"{suffix}_signals.csv").write_text(signals_path.read_text(encoding="utf-8"), encoding="utf-8")
        (BASE / f"{suffix}_report.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    rows_sorted = sorted(rows, key=lambda x: x["robust_score"], reverse=True)

    out_csv = BASE / "grid_4h_beta_windows_oos.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        w.writeheader()
        w.writerows(rows_sorted)

    best = rows_sorted[0]
    lines = [
        "# 4H Beta Window Grid (OOS Robustness)",
        "",
        "Split: 70% in-sample / 30% out-of-sample on 4H strategy net returns.",
        "Ranking objective: maximize OOS Sharpe with penalties for OOS drawdown and IS->OOS instability.",
        "",
        "| beta_window | net_return_full | net_sharpe_full | max_drawdown_full | is_sharpe | oos_sharpe | is_mdd | oos_mdd | sharpe_decay_is_minus_oos | drawdown_stability_gap | robust_score |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in rows_sorted:
        lines.append(
            f"| {r['beta_window']} | {r['net_return_full']:.4%} | {r['net_sharpe_full']:.4f} | {r['max_drawdown_full']:.4%} | {r['is_sharpe']:.4f} | {r['oos_sharpe']:.4f} | {r['is_mdd']:.4%} | {r['oos_mdd']:.4%} | {r['sharpe_decay_is_minus_oos']:.4f} | {r['drawdown_stability_gap']:.4%} | {r['robust_score']:.4f} |"
        )

    lines.extend(
        [
            "",
            f"## Selected Robust Window: {best['beta_window']}",
            f"- OOS Sharpe: {best['oos_sharpe']:.4f}",
            f"- OOS Max Drawdown: {best['oos_mdd']:.4%}",
            f"- IS->OOS Sharpe Decay: {best['sharpe_decay_is_minus_oos']:.4f}",
            f"- Drawdown Stability Gap: {best['drawdown_stability_gap']:.4%}",
        ]
    )

    out_md = BASE / "grid_4h_beta_windows_oos.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    print(f"best_window {best['beta_window']}")


if __name__ == "__main__":
    main()
