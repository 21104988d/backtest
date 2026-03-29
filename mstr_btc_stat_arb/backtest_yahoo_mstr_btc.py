import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def std_sample(vals):
    n = len(vals)
    if n < 2:
        return float("nan")
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))


def corr(x, y):
    if len(x) < 2:
        return float("nan")
    mx = mean(x)
    my = mean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(len(x)))
    return sxy / math.sqrt(sxx * syy)


def ols_alpha_beta(y, x):
    mx = mean(x)
    my = mean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(len(y)))
    b = sxy / sxx
    a = my - b * mx
    return a, b


def rolling_beta_past_only(asset_ret, btc_ret, window):
    out = [None] * len(asset_ret)
    for i in range(window, len(asset_ret)):
        _, b = ols_alpha_beta(asset_ret[i - window : i], btc_ret[i - window : i])
        out[i] = b
    return out


def max_drawdown(eq):
    peak = eq[0]
    mdd = 0.0
    for v in eq:
        peak = max(peak, v)
        dd = (v / peak) - 1.0 if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
    return mdd


def fetch_close_series(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return []
    out = []
    for idx, row in df.iterrows():
        out.append((idx.strftime("%Y-%m-%d"), float(row["Close"])))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="MSTR")
    parser.add_argument("--btc", default="BTC-USD")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-03-23")
    parser.add_argument("--beta-window", type=int, default=60)
    parser.add_argument("--z-window", type=int, default=30)
    parser.add_argument("--z-entry", type=float, default=2.0)
    parser.add_argument("--z-exit", type=float, default=0.5)
    parser.add_argument("--fee-rate", type=float, default=0.00045)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    prefix = f"yahoo_{args.asset.lower()}_vs_btc"

    a_rows = fetch_close_series(args.asset, args.start, args.end)
    b_rows = fetch_close_series(args.btc, args.start, args.end)
    if not a_rows or not b_rows:
        raise RuntimeError("Failed to fetch Yahoo data")

    a_map = {d: c for d, c in a_rows}
    b_map = {d: c for d, c in b_rows}
    dates = sorted(set(a_map) & set(b_map))
    if len(dates) < args.beta_window + 40:
        raise RuntimeError("Not enough aligned rows")

    a_close = [a_map[d] for d in dates]
    b_close = [b_map[d] for d in dates]

    a_ret = [a_close[i] / a_close[i - 1] - 1.0 for i in range(1, len(a_close))]
    b_ret = [b_close[i] / b_close[i - 1] - 1.0 for i in range(1, len(b_close))]
    ret_dates = dates[1:]

    log_a = [math.log(v) for v in a_close]
    log_b = [math.log(v) for v in b_close]
    a0, b0 = ols_alpha_beta(log_a, log_b)
    spread = [log_a[i] - (a0 + b0 * log_b[i]) for i in range(len(dates))]

    rb = rolling_beta_past_only(a_ret, b_ret, args.beta_window)
    hedged_ret = [0.0 if rb[i] is None else (a_ret[i] - rb[i] * b_ret[i]) for i in range(len(a_ret))]

    z = [None] * len(dates)
    for i in range(args.z_window - 1, len(dates)):
        w = spread[i - args.z_window + 1 : i + 1]
        m = mean(w)
        s = std_sample(w)
        z[i] = (spread[i] - m) / s if s and not math.isnan(s) and s > 0 else None

    z_ret = z[1:]
    pos = [0] * len(ret_dates)
    cur = 0
    for i in range(len(ret_dates)):
        zi = z_ret[i]
        if zi is None:
            pos[i] = cur
            continue
        if cur == 0:
            if zi >= args.z_entry:
                cur = -1
            elif zi <= -args.z_entry:
                cur = 1
        else:
            if abs(zi) <= args.z_exit:
                cur = 0
        pos[i] = cur

    gross = [0.0] * len(ret_dates)
    for i in range(1, len(ret_dates)):
        gross[i] = pos[i - 1] * hedged_ret[i]

    net = [0.0] * len(ret_dates)
    fee_cost = [0.0] * len(ret_dates)
    for i in range(len(ret_dates)):
        turnover = 0.0
        if i >= 1:
            prev = pos[i - 2] if i >= 2 else 0
            curr = pos[i - 1]
            turnover = abs(curr - prev)
        fee_cost[i] = turnover * args.fee_rate
        net[i] = gross[i] - fee_cost[i]

    eq = []
    v = 1.0
    for r in net:
        v *= 1.0 + r
        eq.append(v)

    bars_per_year = 365.0
    sh = float("nan")
    s = std_sample(net[1:])
    if s and not math.isnan(s) and s > 0:
        sh = (mean(net[1:]) / s) * math.sqrt(bars_per_year)

    metrics = {
        "source": "Yahoo Finance",
        "asset": args.asset,
        "btc": args.btc,
        "start": dates[0],
        "end": dates[-1],
        "rows_aligned": len(dates),
        "returns_corr": corr(a_ret, b_ret),
        "price_beta_static": b0,
        "rolling_beta_window": args.beta_window,
        "rolling_beta_last": rb[-1],
        "strategy_total_return_net": eq[-1] - 1.0,
        "strategy_sharpe_365_net": sh,
        "max_drawdown_net": max_drawdown(eq),
        "signal_count": sum(abs(x) for x in pos),
        "z_entry": args.z_entry,
        "z_exit": args.z_exit,
    }

    (base / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(eq)))
    ax.plot(x, eq, label="Strategy Equity (Net)", color="#1d3557", linewidth=1.8)
    ax.axhline(1.0, color="#666", linewidth=0.8)
    ax.set_title(f"Yahoo {args.asset} vs BTC Stat-Arb (Daily, start {args.start})")
    ax.set_xlabel("Bar index")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    chart = base / f"{prefix}_equity.png"
    fig.savefig(chart, dpi=160)
    plt.close(fig)

    md = "\n".join(
        [
            f"# Yahoo {args.asset} vs BTC Stat-Arb (Daily)",
            "",
            f"- Period: {metrics['start']} to {metrics['end']}",
            f"- Rows aligned: {metrics['rows_aligned']}",
            f"- Return correlation: {metrics['returns_corr']:.4f}",
            f"- Static price beta: {metrics['price_beta_static']:.4f}",
            f"- Rolling beta window: {metrics['rolling_beta_window']}",
            f"- Last rolling beta: {metrics['rolling_beta_last']:.4f}",
            f"- Net return: {metrics['strategy_total_return_net']:.2%}",
            f"- Net Sharpe(365): {metrics['strategy_sharpe_365_net']:.4f}",
            f"- Max drawdown: {metrics['max_drawdown_net']:.2%}",
            f"- Signal count: {metrics['signal_count']}",
            f"- Thresholds: z_entry={metrics['z_entry']}, z_exit={metrics['z_exit']}",
            f"- Chart: {chart.name}",
        ]
    ) + "\n"
    (base / f"{prefix}_report.md").write_text(md, encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"wrote {prefix}_metrics.json")
    print(f"wrote {prefix}_report.md")
    print(f"wrote {chart.name}")


if __name__ == "__main__":
    main()
