import csv
import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TIMEFRAMES = ["1h", "4h", "1d"]


def symbol_to_filename(symbol: str) -> str:
    return symbol.lower().replace(":", "_").replace("/", "-")


def read_signals(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "date": row["date"],
                    "equity_net": float(row["equity_curve_net"]),
                    "equity_gross": float(row["equity_curve_gross"]),
                }
            )
    return rows


def read_metrics(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def max_drawdown_series(equity):
    out = []
    peak = equity[0] if equity else 1.0
    for v in equity:
        peak = max(peak, v)
        out.append((v / peak) - 1.0 if peak > 0 else 0.0)
    return out


def tf_days(tf: str):
    if tf.endswith("h"):
        return float(tf[:-1]) / 24.0
    if tf.endswith("d"):
        return float(tf[:-1])
    if tf.endswith("w"):
        return float(tf[:-1]) * 7.0
    return float("nan")


def score(metrics):
    net_ret = metrics.get("strategy_total_return_net", 0.0)
    sharpe = metrics.get("strategy_sharpe_365_net", 0.0)
    mdd_abs = abs(metrics.get("max_drawdown_net", 0.0))
    fee_drag = metrics.get("total_fee_drag_return", 0.0)
    return (0.45 * sharpe) + (0.45 * net_ret) - (0.25 * mdd_abs) - (0.10 * fee_drag)


def recommend(all_metrics):
    ranked = sorted(all_metrics.items(), key=lambda kv: score(kv[1]), reverse=True)
    best_tf, best_m = ranked[0]
    second_tf, second_m = ranked[1]

    best_ret = best_m.get("strategy_total_return_net", 0.0)
    best_sh = best_m.get("strategy_sharpe_365_net", 0.0)
    best_mdd = abs(best_m.get("max_drawdown_net", 0.0))

    second_ret = second_m.get("strategy_total_return_net", 0.0)
    second_sh = second_m.get("strategy_sharpe_365_net", 0.0)
    second_mdd = abs(second_m.get("max_drawdown_net", 0.0))

    lead = (
        f"Top score: {best_tf.upper()} (net={best_ret:.2%}, sharpe={best_sh:.3f}, mdd={best_mdd:.2%}) "
        f"vs {second_tf.upper()} (net={second_ret:.2%}, sharpe={second_sh:.3f}, mdd={second_mdd:.2%})."
    )

    if best_tf == "1h":
        return lead + " Choose 1H only if you can handle higher monitoring and frequent recalibration."
    if best_tf == "4h":
        return lead + " 4H is the balanced choice between responsiveness and robustness."
    return lead + " 1D is the robust/low-maintenance choice when stability is the priority."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="ETH")
    parser.add_argument("--btc", default="BTC")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    asset_prefix = symbol_to_filename(args.asset)
    sig_paths = {tf: base / f"{asset_prefix}_{tf}_rolling_beta_signals.csv" for tf in TIMEFRAMES}
    met_paths = {tf: base / f"{asset_prefix}_{tf}_rolling_beta_metrics.json" for tf in TIMEFRAMES}

    files = list(sig_paths.values()) + list(met_paths.values())
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

    signals = {tf: read_signals(p) for tf, p in sig_paths.items()}
    metrics = {tf: read_metrics(p) for tf, p in met_paths.items()}

    colors = {"1h": "#d62828", "4h": "#2a9d8f", "1d": "#1d3557"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    for tf in TIMEFRAMES:
        s = signals[tf]
        x = list(range(len(s)))
        e = [r["equity_net"] for r in s]
        axes[0].plot(x, e, linewidth=2.0, color=colors[tf], label=f"{tf.upper()} Net")

    axes[0].axhline(1.0, color="#666666", linewidth=0.7)
    axes[0].set_title(f"{args.asset} vs {args.btc} Net Equity: 1H vs 4H vs 1D")
    axes[0].set_ylabel("Equity")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    for tf in TIMEFRAMES:
        s = signals[tf]
        x = list(range(len(s)))
        e = [r["equity_net"] for r in s]
        d = max_drawdown_series(e)
        axes[1].plot(x, d, linewidth=1.4, color=colors[tf], label=f"{tf.upper()} Drawdown")

    axes[1].axhline(0.0, color="#666666", linewidth=0.7)
    axes[1].set_title("Underwater Curve (Drawdown)")
    axes[1].set_xlabel("Bar index")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    out_png = base / f"{asset_prefix}_1d_vs_4h_fee_comparison.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    rec = recommend(metrics)
    out_md = base / f"{asset_prefix}_1d_vs_4h_evaluation.md"
    lines = [
        "# 1H vs 4H vs 1D Evaluation (Fee-Aware)",
        "",
        f"- Fee model (taker): {metrics['1d'].get('taker_fee_rate', float('nan')):.5f}",
        "",
    ]

    for tf in TIMEFRAMES:
        m = metrics[tf]
        beta_window = m.get("rolling_beta_window", 0)
        beta_horizon_days = beta_window * tf_days(tf)
        lines.extend(
            [
                f"## {tf.upper()}",
                f"- Net return: {m.get('strategy_total_return_net', float('nan')):.2%}",
                f"- Net Sharpe: {m.get('strategy_sharpe_365_net', float('nan')):.4f}",
                f"- Max drawdown net: {m.get('max_drawdown_net', float('nan')):.2%}",
                f"- Avg holding: {m.get('avg_holding_bars', float('nan')):.2f} bars ({m.get('avg_holding_days', float('nan')):.2f} days)",
                f"- Rolling-beta points: {m.get('rolling_beta_points', 0)} / window {beta_window} ({m.get('rolling_beta_coverage', float('nan')):.2%} coverage)",
                f"- Effective beta lookback horizon: ~{beta_horizon_days:.2f} days",
                "",
            ]
        )

    lines.append(f"## Recommendation: {rec}")
    summary = "\n".join(lines) + "\n"
    out_md.write_text(summary, encoding="utf-8")

    print(f"wrote {out_png.name}")
    print(f"wrote {out_md.name}")


if __name__ == "__main__":
    main()
