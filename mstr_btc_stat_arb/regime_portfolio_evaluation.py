import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


def ols_alpha_beta(y, x):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        return float("nan"), float("nan")
    mx = x.mean()
    my = y.mean()
    sxx = ((x - mx) ** 2).sum()
    if sxx <= 0:
        return float(my), 0.0
    sxy = ((x - mx) * (y - my)).sum()
    b = sxy / sxx
    a = my - b * mx
    return float(a), float(b)


def rolling_beta_past_only(asset_ret, btc_ret, window):
    out = np.full(len(asset_ret), np.nan)
    for i in range(window, len(asset_ret)):
        _, b = ols_alpha_beta(asset_ret[i - window : i], btc_ret[i - window : i])
        out[i] = b
    return out


def sharpe_365(returns):
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0 or np.isnan(sd):
        return float("nan")
    return float((r.mean() / sd) * math.sqrt(365.0))


def max_drawdown(returns):
    r = pd.Series(returns).fillna(0.0)
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def total_return(returns):
    r = pd.Series(returns).fillna(0.0)
    if len(r) == 0:
        return float("nan")
    return float((1.0 + r).prod() - 1.0)


def fetch_yahoo_close(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = pd.DataFrame({"date": pd.to_datetime(df.index), "close": close.astype(float).to_numpy()})
    return out


def build_strategy_df(asset_close, btc_close, beta_window, z_window, z_entry, z_exit, fee_rate):
    df = (
        asset_close.merge(btc_close, on="date", how="inner", suffixes=("_asset", "_btc"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(df) < beta_window + z_window + 10:
        return pd.DataFrame()

    a_close = df["close_asset"].values
    b_close = df["close_btc"].values

    a_ret = (a_close[1:] / a_close[:-1]) - 1.0
    b_ret = (b_close[1:] / b_close[:-1]) - 1.0

    log_a = np.log(a_close)
    log_b = np.log(b_close)
    a0, b0 = ols_alpha_beta(log_a, log_b)
    spread = log_a - (a0 + b0 * log_b)

    rb = rolling_beta_past_only(a_ret, b_ret, beta_window)
    hedged_ret = np.zeros(len(a_ret))
    mask = ~np.isnan(rb)
    hedged_ret[mask] = a_ret[mask] - rb[mask] * b_ret[mask]

    z = np.full(len(df), np.nan)
    for i in range(z_window - 1, len(df)):
        w = spread[i - z_window + 1 : i + 1]
        s = float(np.std(w, ddof=1)) if len(w) > 1 else np.nan
        if s and not np.isnan(s) and s > 0:
            z[i] = (spread[i] - float(np.mean(w))) / s

    z_ret = z[1:]
    pos = np.zeros(len(z_ret), dtype=int)
    cur = 0
    for i, zi in enumerate(z_ret):
        if np.isnan(zi):
            pos[i] = cur
            continue
        if cur == 0:
            if zi >= z_entry:
                cur = -1
            elif zi <= -z_entry:
                cur = 1
        else:
            if abs(zi) <= z_exit:
                cur = 0
        pos[i] = cur

    gross = np.zeros(len(pos))
    for i in range(1, len(pos)):
        gross[i] = pos[i - 1] * hedged_ret[i]

    fee = np.zeros(len(pos))
    net = np.zeros(len(pos))
    for i in range(len(pos)):
        turnover = 0.0
        if i >= 1:
            prev = pos[i - 2] if i >= 2 else 0
            curr = pos[i - 1]
            turnover = abs(curr - prev)
        fee[i] = turnover * fee_rate
        net[i] = gross[i] - fee[i]

    out = pd.DataFrame(
        {
            "date": df["date"].iloc[1:].values,
            "asset_close": a_close[1:],
            "btc_close": b_close[1:],
            "a_ret": a_ret,
            "b_ret": b_ret,
            "hedged_ret": hedged_ret,
            "position": pos,
            "strategy_ret_net": net,
        }
    )
    return out


def assign_regimes_leak_safe(df, vol_window=30, quantile_lookback=252, min_history=126):
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["rolling_vol"] = out["hedged_ret"].rolling(vol_window).std(ddof=1)

    q33 = np.full(len(out), np.nan)
    q66 = np.full(len(out), np.nan)
    rv = out["rolling_vol"].values

    for i in range(len(out)):
        if i < min_history:
            continue
        s = max(0, i - quantile_lookback)
        hist = rv[s:i]
        hist = hist[~np.isnan(hist)]
        if len(hist) < min_history:
            continue
        q33[i] = float(np.quantile(hist, 0.33))
        q66[i] = float(np.quantile(hist, 0.66))

    out["vol_q33_past"] = q33
    out["vol_q66_past"] = q66

    regime = np.array(["unknown"] * len(out), dtype=object)
    high = (out["rolling_vol"] >= out["vol_q66_past"]) & out["vol_q66_past"].notna()
    low = (out["rolling_vol"] <= out["vol_q33_past"]) & out["vol_q33_past"].notna()
    mid = (~high) & (~low) & out["vol_q33_past"].notna() & out["vol_q66_past"].notna()
    regime[high.values] = "high"
    regime[low.values] = "low"
    regime[mid.values] = "mid"
    out["vol_regime"] = regime

    return out


def evaluate_portfolios(df):
    out = df.copy()
    base = out["strategy_ret_net"].fillna(0.0)

    is_high = out["vol_regime"] == "high"
    is_low = out["vol_regime"] == "low"
    is_mid = out["vol_regime"] == "mid"

    out["ret_baseline"] = base
    out["ret_high_only"] = np.where(is_high, base, 0.0)
    out["ret_low_only"] = np.where(is_low, base, 0.0)
    out["ret_high_low_only"] = np.where(is_high | is_low, base, 0.0)
    out["ret_regime_weighted"] = np.where(is_high, base, np.where(is_mid, 0.5 * base, np.where(is_low, 0.2 * base, 0.0)))

    methods = [
        "ret_baseline",
        "ret_high_only",
        "ret_low_only",
        "ret_high_low_only",
        "ret_regime_weighted",
    ]

    rows = []
    for m in methods:
        r = out[m]
        rows.append(
            {
                "portfolio": m,
                "total_return": total_return(r),
                "sharpe_365": sharpe_365(r),
                "max_drawdown": max_drawdown(r),
                "active_ratio": float((r != 0.0).mean()),
            }
        )

    by_regime = []
    for reg in ["high", "mid", "low"]:
        s = out[out["vol_regime"] == reg]["strategy_ret_net"]
        if len(s) == 0:
            continue
        by_regime.append(
            {
                "vol_regime": reg,
                "bars": int(len(s)),
                "mean_ret": float(s.mean()),
                "hit_rate": float((s > 0).mean()),
                "total_return": total_return(s),
                "sharpe_365": sharpe_365(s),
                "max_drawdown": max_drawdown(s),
            }
        )

    return out, pd.DataFrame(rows), pd.DataFrame(by_regime)


def render_equity(df, out_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    series = [
        ("ret_baseline", "baseline"),
        ("ret_high_only", "high_only"),
        ("ret_low_only", "low_only"),
        ("ret_high_low_only", "high_low_only"),
        ("ret_regime_weighted", "regime_weighted"),
    ]
    for col, lbl in series:
        eq = (1.0 + df[col].fillna(0.0)).cumprod()
        ax.plot(df["date"], eq, label=lbl, linewidth=1.5)
    ax.axhline(1.0, color="#444", linewidth=0.8)
    ax.set_title("Full-History Portfolio Evaluation by Volatility Regime")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def render_price_with_position_markers(df, out_path):
    # Non-normalized view: actual asset price with open/close position events.
    d = df.sort_values("date").copy()
    prev_pos = d["position"].shift(1).fillna(0)
    curr_pos = d["position"].fillna(0)

    opens = d[(prev_pos == 0) & (curr_pos != 0)]
    closes = d[(prev_pos != 0) & (curr_pos == 0)]
    flips = d[(prev_pos != 0) & (curr_pos != 0) & (prev_pos != curr_pos)]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(d["date"], d["asset_close"], color="#1d3557", linewidth=1.4, label="MSTR price")
    ax1.scatter(opens["date"], opens["asset_close"], marker="^", color="#2a9d8f", s=50, label="open")
    ax1.scatter(closes["date"], closes["asset_close"], marker="v", color="#e76f51", s=50, label="close")
    ax1.scatter(flips["date"], flips["asset_close"], marker="D", color="#f4a261", s=32, label="flip")
    ax1.set_title("MSTR Price (Non-normalized) With Position Open/Close Indicators")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("MSTR Price")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    raw_cum = d["strategy_ret_net"].fillna(0.0).cumsum()
    ax2.plot(d["date"], raw_cum, color="#6a4c93", linewidth=1.2, alpha=0.9, label="cumulative raw return")
    ax2.set_ylabel("Cumulative Raw Return (sum of daily returns)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run(args):
    base = Path(__file__).resolve().parent

    asset = fetch_yahoo_close(args.asset, args.start, args.end)
    btc = fetch_yahoo_close(args.btc, args.start, args.end)
    if asset.empty or btc.empty:
        raise RuntimeError("Failed to fetch Yahoo data")

    strat = build_strategy_df(
        asset_close=asset,
        btc_close=btc,
        beta_window=args.beta_window,
        z_window=args.z_window,
        z_entry=args.z_entry,
        z_exit=args.z_exit,
        fee_rate=args.fee_rate,
    )
    if strat.empty:
        raise RuntimeError("Not enough data for strategy evaluation")

    reg = assign_regimes_leak_safe(
        strat,
        vol_window=args.vol_window,
        quantile_lookback=args.quantile_lookback,
        min_history=args.min_history,
    )

    eval_df, perf_df, regime_df = evaluate_portfolios(reg)

    out_eval = base / "regime_portfolio_full_history_series.csv"
    out_perf = base / "regime_portfolio_full_history_performance.csv"
    out_regime = base / "regime_portfolio_regime_stats.csv"
    out_chart = base / "regime_portfolio_full_history_equity.png"
    out_chart_price = base / "regime_portfolio_price_with_positions.png"
    out_report = base / "regime_portfolio_full_history_report.md"
    out_summary = base / "regime_portfolio_full_history_summary.json"

    eval_df.to_csv(out_eval, index=False)
    perf_df.to_csv(out_perf, index=False)
    regime_df.to_csv(out_regime, index=False)
    render_equity(eval_df, out_chart)
    render_price_with_position_markers(eval_df, out_chart_price)

    report = []
    report.append("# Regime Portfolio Evaluation (Full History)")
    report.append("")
    report.append(f"- Period: {eval_df['date'].min().date()} to {eval_df['date'].max().date()}")
    report.append("- Regime logic: leakage-safe rolling volatility quantiles from past data only")
    report.append(f"- Vol window: {args.vol_window}, quantile lookback: {args.quantile_lookback}, min history: {args.min_history}")
    report.append(f"- Strategy params: beta_window={args.beta_window}, z_entry={args.z_entry}, z_exit={args.z_exit}, z_window={args.z_window}")
    report.append("")
    report.append("## Portfolio Performance")
    for _, r in perf_df.sort_values("total_return", ascending=False).iterrows():
        report.append(
            f"- {r['portfolio']}: Return={r['total_return']:.2%}, Sharpe={r['sharpe_365']:.4f}, MaxDD={r['max_drawdown']:.2%}, Active={r['active_ratio']:.2%}"
        )
    report.append("")
    report.append("## Regime Contribution (Baseline Returns)")
    for _, r in regime_df.iterrows():
        report.append(
            f"- {r['vol_regime']}: bars={int(r['bars'])}, mean_ret={r['mean_ret']:.5f}, hit_rate={r['hit_rate']:.2%}, Return={r['total_return']:.2%}, Sharpe={r['sharpe_365']:.4f}, MaxDD={r['max_drawdown']:.2%}"
        )
    report.append("")
    report.append("## Artifacts")
    report.append(f"- Series CSV: {out_eval.name}")
    report.append(f"- Performance CSV: {out_perf.name}")
    report.append(f"- Regime stats CSV: {out_regime.name}")
    report.append(f"- Equity chart: {out_chart.name}")
    report.append(f"- Non-normalized price+position chart: {out_chart_price.name}")

    out_report.write_text("\n".join(report) + "\n", encoding="utf-8")

    summary = {
        "report": out_report.name,
        "performance_csv": out_perf.name,
        "regime_stats_csv": out_regime.name,
        "series_csv": out_eval.name,
        "chart": out_chart.name,
        "price_position_chart": out_chart_price.name,
        "performance": perf_df.to_dict(orient="records"),
        "regime_stats": regime_df.to_dict(orient="records"),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="MSTR")
    parser.add_argument("--btc", default="BTC-USD")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-03-23")
    parser.add_argument("--beta-window", type=int, default=120)
    parser.add_argument("--z-window", type=int, default=30)
    parser.add_argument("--z-entry", type=float, default=2.25)
    parser.add_argument("--z-exit", type=float, default=0.75)
    parser.add_argument("--fee-rate", type=float, default=0.00045)
    parser.add_argument("--vol-window", type=int, default=30)
    parser.add_argument("--quantile-lookback", type=int, default=252)
    parser.add_argument("--min-history", type=int, default=126)
    args = parser.parse_args()
    run(args)
