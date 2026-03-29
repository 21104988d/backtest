import argparse
import json
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
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 10:
        return float("nan")
    return float(df["x"].corr(df["y"]))


def assign_regimes_leak_safe(hedged_ret, vol_window=40, quantile_lookback=252, min_history=84):
    out = pd.DataFrame({"hedged_ret": hedged_ret})
    out["rolling_vol"] = out["hedged_ret"].rolling(vol_window).std(ddof=1)

    q33 = np.full(len(out), np.nan)
    q66 = np.full(len(out), np.nan)
    rv = out["rolling_vol"].to_numpy()

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

    reg = np.array(["unknown"] * len(out), dtype=object)
    high = (out["rolling_vol"] >= q66) & ~np.isnan(q66)
    low = (out["rolling_vol"] <= q33) & ~np.isnan(q33)
    mid = (~high) & (~low) & ~np.isnan(q33) & ~np.isnan(q66)
    reg[high] = "high"
    reg[low] = "low"
    reg[mid] = "mid"

    out["vol_regime"] = reg
    return out


def ridge_fit(X, y, alpha):
    if len(X) == 0:
        return None
    xtx = X.T @ X
    reg = np.eye(xtx.shape[0])
    reg[0, 0] = 0.0
    w = np.linalg.pinv(xtx + alpha * reg) @ (X.T @ y)
    return w


def rmse(a, b):
    x = pd.Series(a).astype(float)
    y = pd.Series(b).astype(float)
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(z) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((z["x"] - z["y"]) ** 2)))


def mae(a, b):
    x = pd.Series(a).astype(float)
    y = pd.Series(b).astype(float)
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(z) == 0:
        return float("nan")
    return float(np.mean(np.abs(z["x"] - z["y"])))


def regime_metric_table(df):
    out = {}
    for reg in ["high", "mid", "low", "unknown"]:
        x = df[df["vol_regime"] == reg]
        out[reg] = {
            "bars": int(len(x)),
            "corr_asset_btc": safe_corr(x["asset_ret"], x["btc_ret"]),
            "corr_truehedge_btc": safe_corr(x["hedged_true"], x["btc_ret"]),
            "corr_modelhedge_btc": safe_corr(x["hedged_model"], x["btc_ret"]),
            "beta_mae": mae(x["beta_true"], x["beta_pred"]),
            "beta_rmse": rmse(x["beta_true"], x["beta_pred"]),
            "beta_mean_true": float(x["beta_true"].mean()) if len(x) else float("nan"),
            "beta_mean_pred": float(x["beta_pred"].mean()) if len(x) else float("nan"),
        }
    return out


def period_metric_table(df):
    periods = {
        "full": (pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")),
        "pre_2024": (pd.Timestamp("1900-01-01"), pd.Timestamp("2024-01-01")),
        "y2024_2025": (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01")),
        "y2026_plus": (pd.Timestamp("2026-01-01"), pd.Timestamp("2100-01-01")),
    }
    out = {}
    for name, (st, en) in periods.items():
        x = df[(df["date"] >= st) & (df["date"] < en)]
        out[name] = {
            "bars": int(len(x)),
            "corr_asset_btc": safe_corr(x["asset_ret"], x["btc_ret"]),
            "corr_truehedge_btc": safe_corr(x["hedged_true"], x["btc_ret"]),
            "corr_modelhedge_btc": safe_corr(x["hedged_model"], x["btc_ret"]),
            "beta_mae": mae(x["beta_true"], x["beta_pred"]),
            "beta_rmse": rmse(x["beta_true"], x["beta_pred"]),
        }
    return out


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
    if len(df) < args.beta_window + args.min_train_bars + 50:
        raise RuntimeError("Not enough aligned rows")

    a_close = df["close_asset"].to_numpy()
    b_close = df["close_btc"].to_numpy()
    a_ret = (a_close[1:] / a_close[:-1]) - 1.0
    b_ret = (b_close[1:] / b_close[:-1]) - 1.0
    ret_dates = pd.to_datetime(df["date"].iloc[1:].to_numpy())

    beta_true = rolling_beta_past_only(a_ret, b_ret, args.beta_window)
    hedged_true = a_ret - np.nan_to_num(beta_true, nan=0.0) * b_ret

    reg = assign_regimes_leak_safe(
        hedged_true,
        vol_window=args.vol_window,
        quantile_lookback=args.quantile_lookback,
        min_history=args.min_history,
    )

    work = pd.DataFrame(
        {
            "date": ret_dates,
            "asset_ret": a_ret,
            "btc_ret": b_ret,
            "beta_true": beta_true,
            "hedged_true": hedged_true,
            "rolling_vol": reg["rolling_vol"].to_numpy(),
            "vol_regime": reg["vol_regime"].to_numpy(),
        }
    )

    # Feature engineering for dynamic beta model.
    work["beta_l1"] = work["beta_true"].shift(1)
    work["beta_diff_5"] = work["beta_true"] - work["beta_true"].shift(5)
    work["beta_mean_20"] = work["beta_true"].rolling(20).mean()
    work["beta_std_20"] = work["beta_true"].rolling(20).std(ddof=1)
    work["time_idx"] = np.arange(len(work), dtype=float) / max(1, len(work) - 1)
    work["reg_high"] = (work["vol_regime"] == "high").astype(float)
    work["reg_mid"] = (work["vol_regime"] == "mid").astype(float)
    work["reg_low"] = (work["vol_regime"] == "low").astype(float)

    feature_cols = [
        "intercept",
        "time_idx",
        "beta_l1",
        "beta_diff_5",
        "beta_mean_20",
        "beta_std_20",
        "rolling_vol",
        "reg_high",
        "reg_mid",
        "reg_low",
    ]

    work["intercept"] = 1.0
    beta_pred = np.full(len(work), np.nan)
    slope_series = np.full(len(work), np.nan)

    # Predict beta_t using information available at t-1.
    for t in range(args.min_train_bars + 1, len(work)):
        # Target beta_true[j], features from row j-1.
        y_idx = np.arange(1, t)
        x_idx = y_idx - 1

        X = work.loc[x_idx, feature_cols].copy()
        y = work.loc[y_idx, "beta_true"].copy()

        train = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True).rename("y")], axis=1).dropna()
        if len(train) < args.min_train_bars:
            continue

        X_train = train[feature_cols].to_numpy(dtype=float)
        y_train = train["y"].to_numpy(dtype=float)

        w = ridge_fit(X_train, y_train, alpha=args.ridge_alpha)
        if w is None:
            continue

        x_pred = work.loc[t - 1, feature_cols]
        if x_pred.isna().any():
            continue

        p = float(np.dot(x_pred.to_numpy(dtype=float), w))

        # Clamp predictions to empirical rolling-beta range for robustness.
        lo = float(np.nanquantile(y_train, args.pred_clip_low_q))
        hi = float(np.nanquantile(y_train, args.pred_clip_high_q))
        p = min(max(p, lo), hi)

        beta_pred[t] = p
        slope_series[t] = w[feature_cols.index("time_idx")]

    work["beta_pred"] = beta_pred
    work["slope_coef_time"] = slope_series
    work["hedged_model"] = work["asset_ret"] - work["beta_pred"] * work["btc_ret"]

    eval_df = work.dropna(subset=["beta_true", "beta_pred", "hedged_model", "btc_ret"]).copy()
    if eval_df.empty:
        raise RuntimeError("No valid rows for dynamic beta evaluation")

    overall = {
        "bars": int(len(eval_df)),
        "beta_mae": mae(eval_df["beta_true"], eval_df["beta_pred"]),
        "beta_rmse": rmse(eval_df["beta_true"], eval_df["beta_pred"]),
        "corr_asset_btc": safe_corr(eval_df["asset_ret"], eval_df["btc_ret"]),
        "corr_truehedge_btc": safe_corr(eval_df["hedged_true"], eval_df["btc_ret"]),
        "corr_modelhedge_btc": safe_corr(eval_df["hedged_model"], eval_df["btc_ret"]),
    }
    overall["truehedge_corr_reduction"] = abs(overall["corr_asset_btc"]) - abs(overall["corr_truehedge_btc"])
    overall["modelhedge_corr_reduction"] = abs(overall["corr_asset_btc"]) - abs(overall["corr_modelhedge_btc"])

    slope_valid = eval_df["slope_coef_time"].dropna()
    slope_stats = {
        "count": int(len(slope_valid)),
        "mean": float(slope_valid.mean()) if len(slope_valid) else float("nan"),
        "std": float(slope_valid.std(ddof=1)) if len(slope_valid) > 1 else float("nan"),
        "p10": float(slope_valid.quantile(0.10)) if len(slope_valid) else float("nan"),
        "p50": float(slope_valid.quantile(0.50)) if len(slope_valid) else float("nan"),
        "p90": float(slope_valid.quantile(0.90)) if len(slope_valid) else float("nan"),
    }

    result = {
        "asset": args.asset,
        "btc": args.btc,
        "start": str(pd.to_datetime(df["date"].iloc[0]).date()),
        "end": str(pd.to_datetime(df["date"].iloc[-1]).date()),
        "beta_window_anchor": args.beta_window,
        "rows_total": int(len(work)),
        "rows_evaluated": int(len(eval_df)),
        "overall": overall,
        "slope_stats": slope_stats,
        "regime_metrics": regime_metric_table(eval_df),
        "period_metrics": period_metric_table(eval_df),
    }

    out_series = base / "dynamic_beta_model_series.csv"
    out_json = base / "dynamic_beta_model_summary.json"
    out_md = base / "dynamic_beta_model_report.md"

    work.to_csv(out_series, index=False)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Chart: true vs predicted beta.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eval_df["date"], eval_df["beta_true"], color="#1f5c99", linewidth=1.3, label="beta_true (rolling40)")
    ax.plot(eval_df["date"], eval_df["beta_pred"], color="#cc6b2c", linewidth=1.0, alpha=0.9, label="beta_pred (dynamic model)")
    ax.set_title("Dynamic Beta Model vs Rolling-40 Beta")
    ax.set_xlabel("Date")
    ax.set_ylabel("Beta")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(base / "dynamic_beta_model_beta_chart.png", dpi=160)
    plt.close(fig)

    # Chart: learned slope over time.
    fig2, ax2 = plt.subplots(figsize=(12, 4.5))
    slope_plot = eval_df.dropna(subset=["slope_coef_time"])
    ax2.plot(slope_plot["date"], slope_plot["slope_coef_time"], color="#4b8f29", linewidth=1.2)
    ax2.axhline(0.0, color="#444", linestyle="--", linewidth=0.8)
    ax2.set_title("Estimated Time-Slope Coefficient in Dynamic Beta Model")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Slope Coef")
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(base / "dynamic_beta_model_slope_chart.png", dpi=160)
    plt.close(fig2)

    lines = []
    lines.append("# Dynamic Beta Model (Rolling-40 Anchor)")
    lines.append("")
    lines.append(f"- Pair: {args.asset} vs {args.btc}")
    lines.append(f"- Period: {result['start']} to {result['end']}")
    lines.append(f"- Anchor rolling beta window: {args.beta_window}")
    lines.append(f"- Evaluated rows: {result['rows_evaluated']}")
    lines.append("")
    lines.append("## Overall")
    lines.append(
        f"- Corr(asset,BTC)={overall['corr_asset_btc']:.4f}, Corr(hedged_true,BTC)={overall['corr_truehedge_btc']:.4f}, Corr(hedged_model,BTC)={overall['corr_modelhedge_btc']:.4f}"
    )
    lines.append(
        f"- Corr reduction true={overall['truehedge_corr_reduction']:.4f}, model={overall['modelhedge_corr_reduction']:.4f}"
    )
    lines.append(f"- Beta MAE={overall['beta_mae']:.4f}, RMSE={overall['beta_rmse']:.4f}")
    lines.append("")
    lines.append("## Time-Slope Coefficient Stats")
    lines.append(
        f"- mean={slope_stats['mean']:.6f}, std={slope_stats['std']:.6f}, p10={slope_stats['p10']:.6f}, p50={slope_stats['p50']:.6f}, p90={slope_stats['p90']:.6f}"
    )
    lines.append("")
    lines.append("## Regime Metrics")
    for reg, m in result["regime_metrics"].items():
        lines.append(
            f"- {reg}: bars={m['bars']}, corr_true={m['corr_truehedge_btc']:.4f}, corr_model={m['corr_modelhedge_btc']:.4f}, beta_mae={m['beta_mae']:.4f}"
        )

    lines.append("")
    lines.append("- Artifacts: dynamic_beta_model_series.csv, dynamic_beta_model_summary.json, dynamic_beta_model_beta_chart.png, dynamic_beta_model_slope_chart.png")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="MSTR")
    parser.add_argument("--btc", default="BTC-USD")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-03-23")

    parser.add_argument("--beta-window", type=int, default=40)
    parser.add_argument("--min-train-bars", type=int, default=252)
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--pred-clip-low-q", type=float, default=0.02)
    parser.add_argument("--pred-clip-high-q", type=float, default=0.98)

    parser.add_argument("--vol-window", type=int, default=40)
    parser.add_argument("--quantile-lookback", type=int, default=252)
    parser.add_argument("--min-history", type=int, default=84)

    parser.add_argument("--z-window", type=int, default=30)
    parser.add_argument("--fee-rate", type=float, default=0.00045)
    args = parser.parse_args()

    run(args)
