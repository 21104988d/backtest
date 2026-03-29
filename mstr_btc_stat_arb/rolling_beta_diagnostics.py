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


def safe_corr(x, y):
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 5:
        return float("nan")
    return float(df["x"].corr(df["y"]))


def series_stats(s):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
            "iqr": float("nan"),
            "cv": float("nan"),
        }

    mean = float(s.mean())
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    p10 = float(s.quantile(0.10))
    p50 = float(s.quantile(0.50))
    p90 = float(s.quantile(0.90))
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    cv = abs(std / mean) if mean != 0 else float("nan")

    return {
        "count": int(len(s)),
        "mean": mean,
        "std": std,
        "min": float(s.min()),
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "max": float(s.max()),
        "iqr": iqr,
        "cv": float(cv) if not np.isnan(cv) else float("nan"),
    }


def break_count(beta_series, threshold_sigma=2.0):
    s = pd.Series(beta_series).dropna()
    if len(s) < 10:
        return {"count": 0, "rate": float("nan"), "threshold": float("nan")}
    d = s.diff().dropna()
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    thr = threshold_sigma * sd
    cnt = int((d.abs() > thr).sum()) if sd > 0 else 0
    return {
        "count": cnt,
        "rate": float(cnt / len(d)) if len(d) else float("nan"),
        "threshold": thr,
    }


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
    if len(df) < max(args.beta_windows) + 50:
        raise RuntimeError("Not enough aligned rows")

    a_close = df["close_asset"].to_numpy()
    b_close = df["close_btc"].to_numpy()
    a_ret = (a_close[1:] / a_close[:-1]) - 1.0
    b_ret = (b_close[1:] / b_close[:-1]) - 1.0

    # Build a reference regime labeling frame using a fixed strategy stream.
    ref_strat = core.build_strategy_df(
        asset_close=asset,
        btc_close=btc,
        beta_window=args.regime_base_beta_window,
        z_window=args.z_window,
        z_entry=args.regime_base_z_entry,
        z_exit=args.regime_base_z_exit,
        fee_rate=args.fee_rate,
    )
    reg_df = core.assign_regimes_leak_safe(
        ref_strat,
        vol_window=args.vol_window,
        quantile_lookback=args.quantile_lookback,
        min_history=args.min_history,
    )

    ret_dates = pd.to_datetime(df["date"].iloc[1:].values)
    res_rows = []

    for bw in args.beta_windows:
        rb = rolling_beta_past_only(a_ret, b_ret, bw)
        hedged = a_ret - np.nan_to_num(rb, nan=0.0) * b_ret

        tmp = pd.DataFrame(
            {
                "date": ret_dates,
                "beta": rb,
                "asset_ret": a_ret,
                "btc_ret": b_ret,
                "hedged_ret": hedged,
            }
        )
        tmp = tmp.merge(reg_df[["date", "vol_regime"]], on="date", how="left")

        beta_st = series_stats(tmp["beta"])
        br = break_count(tmp["beta"], threshold_sigma=args.break_sigma)
        raw_corr = safe_corr(tmp["asset_ret"], tmp["btc_ret"])
        hedged_corr = safe_corr(tmp["hedged_ret"], tmp["btc_ret"])

        regime_stats = {}
        for reg in ["high", "mid", "low", "unknown"]:
            sub = tmp[tmp["vol_regime"] == reg]
            regime_stats[reg] = {
                "bars": int(len(sub)),
                "beta_mean": float(sub["beta"].mean()) if len(sub) else float("nan"),
                "beta_std": float(sub["beta"].std(ddof=1)) if len(sub) > 1 else float("nan"),
                "hedged_corr_to_btc": safe_corr(sub["hedged_ret"], sub["btc_ret"]),
            }

        period_stats = {}
        periods = {
            "full": (pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")),
            "pre_2024": (pd.Timestamp("1900-01-01"), pd.Timestamp("2024-01-01")),
            "y2024_2025": (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01")),
            "y2026_plus": (pd.Timestamp("2026-01-01"), pd.Timestamp("2100-01-01")),
        }
        for name, (st, en) in periods.items():
            x = tmp[(tmp["date"] >= st) & (tmp["date"] < en)]
            period_stats[name] = {
                "bars": int(len(x)),
                "beta_mean": float(x["beta"].mean()) if len(x) else float("nan"),
                "beta_std": float(x["beta"].std(ddof=1)) if len(x) > 1 else float("nan"),
                "hedged_corr_to_btc": safe_corr(x["hedged_ret"], x["btc_ret"]),
            }

        res_rows.append(
            {
                "beta_window": bw,
                "beta_stats": beta_st,
                "breaks": br,
                "raw_corr_asset_btc": raw_corr,
                "hedged_corr_to_btc": hedged_corr,
                "regime_stats": regime_stats,
                "period_stats": period_stats,
            }
        )

        # Save per-window timeseries for inspection.
        out_ts = base / f"rolling_beta_diag_bw{bw}_series.csv"
        tmp.to_csv(out_ts, index=False)

        # Rolling beta chart.
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(tmp["date"], tmp["beta"], color="#1f5c99", linewidth=1.5, label="rolling beta")
        ax.axhline(float(tmp["beta"].mean(skipna=True)), color="#aa3a3a", linestyle="--", linewidth=1.1, label="mean beta")
        ax.set_title(f"Rolling Beta (window={bw})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Beta")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(base / f"rolling_beta_diag_bw{bw}.png", dpi=160)
        plt.close(fig)

    # Compare windows in one table.
    summary_table = []
    for r in res_rows:
        summary_table.append(
            {
                "beta_window": r["beta_window"],
                "beta_mean": r["beta_stats"]["mean"],
                "beta_std": r["beta_stats"]["std"],
                "beta_cv": r["beta_stats"]["cv"],
                "beta_p10": r["beta_stats"]["p10"],
                "beta_p90": r["beta_stats"]["p90"],
                "beta_break_rate": r["breaks"]["rate"],
                "raw_corr_asset_btc": r["raw_corr_asset_btc"],
                "hedged_corr_to_btc": r["hedged_corr_to_btc"],
                "hedge_corr_reduction": (
                    abs(r["raw_corr_asset_btc"]) - abs(r["hedged_corr_to_btc"])
                    if not np.isnan(r["raw_corr_asset_btc"]) and not np.isnan(r["hedged_corr_to_btc"])
                    else float("nan")
                ),
            }
        )

    summary_df = pd.DataFrame(summary_table).sort_values("beta_window").reset_index(drop=True)
    summary_df.to_csv(base / "rolling_beta_diag_summary.csv", index=False)

    best_idx = summary_df["hedge_corr_reduction"].astype(float).idxmax()
    best_window = int(summary_df.loc[best_idx, "beta_window"])

    result = {
        "asset": args.asset,
        "btc": args.btc,
        "start": str(pd.to_datetime(df["date"].iloc[0]).date()),
        "end": str(pd.to_datetime(df["date"].iloc[-1]).date()),
        "rows": int(len(df)),
        "beta_windows": args.beta_windows,
        "break_sigma": args.break_sigma,
        "best_window_by_hedge_corr_reduction": best_window,
        "summary_rows": summary_table,
        "details": res_rows,
    }

    (base / "rolling_beta_diag_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    md_lines = []
    md_lines.append("# Rolling Beta Diagnostics")
    md_lines.append("")
    md_lines.append(f"- Pair: {args.asset} vs {args.btc}")
    md_lines.append(f"- Period: {result['start']} to {result['end']}")
    md_lines.append(f"- Rows: {result['rows']}")
    md_lines.append(f"- Beta windows evaluated: {args.beta_windows}")
    md_lines.append("")
    md_lines.append("## Summary (across beta windows)")
    for _, x in summary_df.iterrows():
        md_lines.append(
            f"- bw={int(x['beta_window'])}: beta_mean={x['beta_mean']:.4f}, beta_std={x['beta_std']:.4f}, break_rate={x['beta_break_rate']:.2%}, raw_corr={x['raw_corr_asset_btc']:.4f}, hedged_corr={x['hedged_corr_to_btc']:.4f}, reduction={x['hedge_corr_reduction']:.4f}"
        )

    md_lines.append("")
    md_lines.append(f"- Best window by corr reduction: {best_window}")
    md_lines.append("- Artifacts: rolling_beta_diag_summary.csv, rolling_beta_diag_summary.json, rolling_beta_diag_bw*.png, rolling_beta_diag_bw*_series.csv")
    (base / "rolling_beta_diag_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="MSTR")
    parser.add_argument("--btc", default="BTC-USD")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-03-23")
    parser.add_argument("--beta-windows", default="40,60,90,120")
    parser.add_argument("--z-window", type=int, default=30)
    parser.add_argument("--fee-rate", type=float, default=0.00045)

    parser.add_argument("--regime-base-beta-window", type=int, default=90)
    parser.add_argument("--regime-base-z-entry", type=float, default=1.75)
    parser.add_argument("--regime-base-z-exit", type=float, default=1.0)
    parser.add_argument("--vol-window", type=int, default=40)
    parser.add_argument("--quantile-lookback", type=int, default=252)
    parser.add_argument("--min-history", type=int, default=84)

    parser.add_argument("--break-sigma", type=float, default=2.0)
    args = parser.parse_args()
    args.beta_windows = [int(x.strip()) for x in args.beta_windows.split(",") if x.strip()]

    run(args)
