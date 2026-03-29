import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import optimize_oos_20_regime as core


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


def fit_ou_ar1(series):
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    if len(x) < 30:
        return {"kappa": np.nan, "half_life": np.nan, "b": np.nan}

    x0 = x[:-1]
    x1 = x[1:]
    mx0 = x0.mean()
    mx1 = x1.mean()
    sxx = ((x0 - mx0) ** 2).sum()
    if sxx <= 0:
        return {"kappa": np.nan, "half_life": np.nan, "b": np.nan}

    sxy = ((x0 - mx0) * (x1 - mx1)).sum()
    b = float(sxy / sxx)
    kappa = np.nan
    half_life = np.nan

    if 0.0 < b < 1.0:
        kappa = -math.log(b)
        if kappa > 0:
            half_life = math.log(2.0) / kappa

    return {"kappa": float(kappa), "half_life": float(half_life), "b": b}


def rolling_ou_metrics(series, lookback):
    s = pd.Series(series).to_numpy(dtype=float)
    kappa = np.full(len(s), np.nan)
    hl = np.full(len(s), np.nan)
    b = np.full(len(s), np.nan)

    for i in range(lookback, len(s)):
        w = s[i - lookback : i]
        out = fit_ou_ar1(w)
        kappa[i] = out["kappa"]
        hl[i] = out["half_life"]
        b[i] = out["b"]

    return kappa, hl, b


def regime_change_indices(reg_series):
    x = pd.Series(reg_series).astype(str)
    prev = x.shift(1)
    return list(x.index[(x != prev) & prev.notna()])


def to_num_days(dt_series):
    # matplotlib can plot pandas datetime directly; this helper is unused but kept for explicitness.
    return pd.to_datetime(dt_series)


def build_distribution_plot(df, col, title, out_path):
    regs = ["high", "mid", "low", "unknown"]
    data = [df[df["vol_regime"] == r][col].dropna().values for r in regs]

    fig, ax = plt.subplots(figsize=(10.5, 5))
    bp = ax.boxplot(data, tick_labels=regs, showfliers=False, patch_artist=True)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#7f7f7f"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)

    # Overlay jittered points for shape visibility.
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        x = i + rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(x, vals, s=8, alpha=0.25, color=colors[i - 1])

    ax.set_title(title)
    ax.set_xlabel("Regime")
    ax.set_ylabel(col)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run(args):
    base = Path(__file__).resolve().parent

    asset = core.fetch_yahoo_close(args.asset, args.start, args.end)
    btc = core.fetch_yahoo_close(args.btc, args.start, args.end)
    if asset.empty or btc.empty:
        raise RuntimeError("Failed to fetch Yahoo data")

    df = (
        asset.merge(btc, on="date", how="inner", suffixes=("_asset", "_btc"))
        .sort_values("date")
        .reset_index(drop=True)
    )

    a_close = df["close_asset"].to_numpy(dtype=float)
    b_close = df["close_btc"].to_numpy(dtype=float)

    a_ret = (a_close[1:] / a_close[:-1]) - 1.0
    b_ret = (b_close[1:] / b_close[:-1]) - 1.0
    ret_dates = pd.to_datetime(df["date"].iloc[1:].to_numpy())

    beta = rolling_beta_past_only(a_ret, b_ret, args.beta_window)
    hedged = np.where(np.isnan(beta), 0.0, a_ret - beta * b_ret)

    reg_df = core.assign_regimes_leak_safe(
        pd.DataFrame(
            {
                "date": ret_dates,
                "hedged_ret": hedged,
                "strategy_ret_net": np.zeros(len(hedged), dtype=float),
            }
        ),
        vol_window=args.vol_window,
        quantile_lookback=args.quantile_lookback,
        min_history=args.min_history,
    )

    # Spread for OU estimation.
    log_a = np.log(a_close)
    log_b = np.log(b_close)
    a0, b0 = ols_alpha_beta(log_a, log_b)
    spread = log_a - (a0 + b0 * log_b)
    spread_ret = spread[1:]  # align with ret_dates

    ou_kappa, ou_hl, ou_b = rolling_ou_metrics(spread_ret, args.ou_lookback)

    out = pd.DataFrame(
        {
            "date": ret_dates,
            "vol_regime": reg_df["vol_regime"].to_numpy(),
            "rolling_beta": beta,
            "ou_kappa": ou_kappa,
            "ou_half_life": ou_hl,
            "ou_b": ou_b,
            "hedged_ret": hedged,
        }
    )

    out_csv = base / "ou_beta_regime_diagnostics_series.csv"
    out_csv.write_text(out.to_csv(index=False), encoding="utf-8")

    # Time-series plot with vertical regime-change lines.
    changes = regime_change_indices(out["vol_regime"])
    reg_colors = {"high": "#d62728", "mid": "#1f77b4", "low": "#2ca02c", "unknown": "#7f7f7f"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(out["date"], out["rolling_beta"], color="#1f5c99", linewidth=1.4, label="rolling beta")
    axes[0].axhline(float(pd.Series(out["rolling_beta"]).mean(skipna=True)), color="#555", linestyle="--", linewidth=0.9)
    axes[0].set_title("Rolling Beta with Regime Change Boundaries")
    axes[0].set_ylabel("Beta")
    axes[0].grid(alpha=0.2)

    axes[1].plot(out["date"], out["ou_kappa"], color="#8c564b", linewidth=1.2, label="OU kappa")
    axes[1].set_title("Rolling OU Kappa with Regime Change Boundaries")
    axes[1].set_ylabel("OU Kappa")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.2)

    for idx in changes:
        dt = out["date"].iloc[idx]
        reg = str(out["vol_regime"].iloc[idx])
        c = reg_colors.get(reg, "#999")
        axes[0].axvline(dt, color=c, alpha=0.16, linewidth=0.8)
        axes[1].axvline(dt, color=c, alpha=0.16, linewidth=0.8)

    fig.tight_layout()
    ts_png = base / "ou_beta_regime_timeseries.png"
    fig.savefig(ts_png, dpi=170)
    plt.close(fig)

    # Distribution plots.
    beta_dist_png = base / "beta_distribution_by_regime.png"
    kappa_dist_png = base / "ou_kappa_distribution_by_regime.png"
    hl_dist_png = base / "ou_half_life_distribution_by_regime.png"

    build_distribution_plot(out, "rolling_beta", "Rolling Beta Distribution by Regime", beta_dist_png)
    build_distribution_plot(out, "ou_kappa", "Rolling OU Kappa Distribution by Regime", kappa_dist_png)

    # Half-life can have extreme tail; clip for visibility but keep raw in csv.
    out_hl_plot = out.copy()
    out_hl_plot["ou_half_life"] = out_hl_plot["ou_half_life"].clip(upper=args.half_life_plot_cap)
    build_distribution_plot(out_hl_plot, "ou_half_life", f"Rolling OU Half-Life Distribution by Regime (clipped at {args.half_life_plot_cap})", hl_dist_png)

    # Summary table.
    rows = []
    for reg in ["high", "mid", "low", "unknown"]:
        d = out[out["vol_regime"] == reg]
        rows.append(
            {
                "regime": reg,
                "bars": int(len(d)),
                "beta_mean": float(d["rolling_beta"].mean()),
                "beta_std": float(d["rolling_beta"].std(ddof=1)) if len(d) > 1 else float("nan"),
                "kappa_mean": float(d["ou_kappa"].mean()),
                "kappa_std": float(d["ou_kappa"].std(ddof=1)) if len(d) > 1 else float("nan"),
                "hl_mean": float(d["ou_half_life"].mean()),
                "hl_median": float(d["ou_half_life"].median()),
            }
        )

    summary = {
        "asset": args.asset,
        "btc": args.btc,
        "start": str(out["date"].min().date()),
        "end": str(out["date"].max().date()),
        "beta_window": args.beta_window,
        "ou_lookback": args.ou_lookback,
        "regime_definition": {
            "vol_window": args.vol_window,
            "quantile_lookback": args.quantile_lookback,
            "min_history": args.min_history,
        },
        "summary_by_regime": rows,
        "artifacts": {
            "series_csv": out_csv.name,
            "timeseries_plot": ts_png.name,
            "beta_distribution_plot": beta_dist_png.name,
            "ou_kappa_distribution_plot": kappa_dist_png.name,
            "ou_half_life_distribution_plot": hl_dist_png.name,
        },
    }

    out_json = base / "ou_beta_regime_diagnostics_summary.json"
    out_md = base / "ou_beta_regime_diagnostics_report.md"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Rolling OU/Beta Regime Diagnostics")
    lines.append("")
    lines.append(f"- Pair: {args.asset} vs {args.btc}")
    lines.append(f"- Period: {summary['start']} to {summary['end']}")
    lines.append(f"- Beta window: {args.beta_window}")
    lines.append(f"- OU lookback bars: {args.ou_lookback}")
    lines.append("")
    lines.append("## Summary by Regime")
    for r in rows:
        lines.append(
            f"- {r['regime']}: bars={r['bars']}, beta_mean={r['beta_mean']:.4f}, beta_std={r['beta_std']:.4f}, kappa_mean={r['kappa_mean']:.5f}, hl_median={r['hl_median']:.2f}"
        )

    lines.append("")
    lines.append(f"- Time-series plot: {ts_png.name}")
    lines.append(f"- Beta distribution: {beta_dist_png.name}")
    lines.append(f"- OU kappa distribution: {kappa_dist_png.name}")
    lines.append(f"- OU half-life distribution: {hl_dist_png.name}")
    lines.append(f"- Series CSV: {out_csv.name}")
    lines.append(f"- JSON: {out_json.name}")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="MSTR")
    parser.add_argument("--btc", default="BTC-USD")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-03-23")

    parser.add_argument("--beta-window", type=int, default=40)
    parser.add_argument("--ou-lookback", type=int, default=120)

    parser.add_argument("--vol-window", type=int, default=40)
    parser.add_argument("--quantile-lookback", type=int, default=252)
    parser.add_argument("--min-history", type=int, default=84)

    parser.add_argument("--half-life-plot-cap", type=float, default=120.0)
    run(parser.parse_args())
