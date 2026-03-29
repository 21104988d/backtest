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


def ridge_fit(X, y, alpha):
    if len(X) == 0:
        return None
    xtx = X.T @ X
    reg = np.eye(xtx.shape[0])
    reg[0, 0] = 0.0
    w = np.linalg.pinv(xtx + alpha * reg) @ (X.T @ y)
    return w


def build_position_signal(spread, z_window, z_entry, z_exit):
    z = np.full(len(spread), np.nan)
    for i in range(z_window - 1, len(spread)):
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
    return pos


def build_net_returns(pos, hedged_ret, fee_rate):
    gross = np.zeros(len(pos), dtype=float)
    for i in range(1, len(pos)):
        gross[i] = pos[i - 1] * hedged_ret[i]

    net = np.zeros(len(pos), dtype=float)
    for i in range(len(pos)):
        turnover = 0.0
        if i >= 1:
            prev = pos[i - 2] if i >= 2 else 0
            curr = pos[i - 1]
            turnover = abs(curr - prev)
        net[i] = gross[i] - turnover * fee_rate
    return net


def metrics_from_returns(ret, bars_per_year=365.0):
    s = pd.Series(ret).fillna(0.0)
    if len(s) == 0:
        return {
            "total_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "active_ratio": float("nan"),
            "mean_daily": float("nan"),
            "vol_daily": float("nan"),
            "bars": 0,
        }

    eq = (1.0 + s).cumprod()
    dd = (eq / eq.cummax()) - 1.0
    sd = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    sharpe = float("nan")
    if sd and not math.isnan(sd) and sd > 0:
        sharpe = float((s.mean() / sd) * math.sqrt(bars_per_year))

    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()) if len(dd) else float("nan"),
        "active_ratio": float((s != 0.0).mean()),
        "mean_daily": float(s.mean()),
        "vol_daily": sd,
        "bars": int(len(s)),
    }


def build_dynamic_beta(beta_true, rolling_vol, vol_regime, min_train_bars, ridge_alpha, clip_low_q, clip_high_q):
    work = pd.DataFrame({"beta_true": beta_true, "rolling_vol": rolling_vol, "vol_regime": vol_regime})
    work["beta_l1"] = work["beta_true"].shift(1)
    work["beta_diff_5"] = work["beta_true"] - work["beta_true"].shift(5)
    work["beta_mean_20"] = work["beta_true"].rolling(20).mean()
    work["beta_std_20"] = work["beta_true"].rolling(20).std(ddof=1)
    work["time_idx"] = np.arange(len(work), dtype=float) / max(1, len(work) - 1)
    work["reg_high"] = (work["vol_regime"] == "high").astype(float)
    work["reg_mid"] = (work["vol_regime"] == "mid").astype(float)
    work["reg_low"] = (work["vol_regime"] == "low").astype(float)
    work["intercept"] = 1.0

    cols = [
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

    beta_pred = np.full(len(work), np.nan)
    for t in range(min_train_bars + 1, len(work)):
        y_idx = np.arange(1, t)
        x_idx = y_idx - 1
        X = work.loc[x_idx, cols].copy()
        y = work.loc[y_idx, "beta_true"].copy()
        train = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True).rename("y")], axis=1).dropna()
        if len(train) < min_train_bars:
            continue

        w = ridge_fit(train[cols].to_numpy(dtype=float), train["y"].to_numpy(dtype=float), alpha=ridge_alpha)
        if w is None:
            continue

        x_pred = work.loc[t - 1, cols]
        if x_pred.isna().any():
            continue

        p = float(np.dot(x_pred.to_numpy(dtype=float), w))
        lo = float(np.nanquantile(train["y"].to_numpy(dtype=float), clip_low_q))
        hi = float(np.nanquantile(train["y"].to_numpy(dtype=float), clip_high_q))
        beta_pred[t] = min(max(p, lo), hi)

    return beta_pred


def summarize_by_regime(df, ret_col):
    out = {}
    for reg in ["high", "mid", "low", "unknown"]:
        x = df[df["vol_regime"] == reg]
        out[reg] = metrics_from_returns(x[ret_col].values)
    return out


def summarize_by_period(df, ret_col):
    periods = {
        "full": (pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")),
        "pre_2024": (pd.Timestamp("1900-01-01"), pd.Timestamp("2024-01-01")),
        "y2024_2025": (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01")),
        "y2026_plus": (pd.Timestamp("2026-01-01"), pd.Timestamp("2100-01-01")),
    }
    out = {}
    for name, (st, en) in periods.items():
        x = df[(df["date"] >= st) & (df["date"] < en)]
        out[name] = metrics_from_returns(x[ret_col].values)
    return out


def run(args):
    base = Path(__file__).resolve().parent

    asset = core.fetch_yahoo_close(args.asset, args.start, args.end)
    btc = core.fetch_yahoo_close(args.btc, args.start, args.end)
    if asset.empty or btc.empty:
        raise RuntimeError("Failed to fetch Yahoo data")

    px = (
        asset.merge(btc, on="date", how="inner", suffixes=("_asset", "_btc"))
        .sort_values("date")
        .reset_index(drop=True)
    )

    a_close = px["close_asset"].to_numpy()
    b_close = px["close_btc"].to_numpy()
    a_ret = (a_close[1:] / a_close[:-1]) - 1.0
    b_ret = (b_close[1:] / b_close[:-1]) - 1.0
    ret_dates = pd.to_datetime(px["date"].iloc[1:].to_numpy())

    log_a = np.log(a_close)
    log_b = np.log(b_close)
    a0, b0 = ols_alpha_beta(log_a, log_b)
    spread = log_a - (a0 + b0 * log_b)

    pos = build_position_signal(spread, args.z_window, args.z_entry, args.z_exit)

    beta_roll40 = rolling_beta_past_only(a_ret, b_ret, args.beta_window)
    hedged_roll40 = np.where(np.isnan(beta_roll40), 0.0, a_ret - beta_roll40 * b_ret)

    reg = core.assign_regimes_leak_safe(
        pd.DataFrame({"date": ret_dates, "hedged_ret": hedged_roll40, "strategy_ret_net": np.zeros(len(hedged_roll40))}),
        vol_window=args.vol_window,
        quantile_lookback=args.quantile_lookback,
        min_history=args.min_history,
    )

    beta_ml = build_dynamic_beta(
        beta_true=beta_roll40,
        rolling_vol=reg["rolling_vol"].to_numpy(),
        vol_regime=reg["vol_regime"].to_numpy(),
        min_train_bars=args.min_train_bars,
        ridge_alpha=args.ridge_alpha,
        clip_low_q=args.pred_clip_low_q,
        clip_high_q=args.pred_clip_high_q,
    )
    beta_ml = np.where(np.isnan(beta_ml), beta_roll40, beta_ml)

    hedged_ml = np.where(np.isnan(beta_ml), 0.0, a_ret - beta_ml * b_ret)
    ret_roll40 = build_net_returns(pos, hedged_roll40, args.fee_rate)
    ret_ml = build_net_returns(pos, hedged_ml, args.fee_rate)

    df = pd.DataFrame(
        {
            "date": ret_dates,
            "vol_regime": reg["vol_regime"].to_numpy(),
            "ret_roll40": ret_roll40,
            "ret_mlbeta": ret_ml,
            "beta_roll40": beta_roll40,
            "beta_ml": beta_ml,
        }
    )

    oos_mask = df["date"] >= pd.Timestamp(args.wf_start_date)
    df_oos = df[oos_mask].copy()

    regime_share_all = df["vol_regime"].value_counts(normalize=True, dropna=False).to_dict()
    regime_share_oos = df_oos["vol_regime"].value_counts(normalize=True, dropna=False).to_dict()

    summary = {
        "asset": args.asset,
        "btc": args.btc,
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
        "wf_start_date": args.wf_start_date,
        "config": {
            "beta_window": args.beta_window,
            "z_entry": args.z_entry,
            "z_exit": args.z_exit,
            "z_window": args.z_window,
            "fee_rate": args.fee_rate,
        },
        "overall": {
            "all": {
                "roll40": metrics_from_returns(df["ret_roll40"].values),
                "mlbeta": metrics_from_returns(df["ret_mlbeta"].values),
            },
            "oos": {
                "roll40": metrics_from_returns(df_oos["ret_roll40"].values),
                "mlbeta": metrics_from_returns(df_oos["ret_mlbeta"].values),
            },
        },
        "regime_share_all": regime_share_all,
        "regime_share_oos": regime_share_oos,
        "by_regime_all": {
            "roll40": summarize_by_regime(df, "ret_roll40"),
            "mlbeta": summarize_by_regime(df, "ret_mlbeta"),
        },
        "by_regime_oos": {
            "roll40": summarize_by_regime(df_oos, "ret_roll40"),
            "mlbeta": summarize_by_regime(df_oos, "ret_mlbeta"),
        },
        "by_period": {
            "roll40": summarize_by_period(df, "ret_roll40"),
            "mlbeta": summarize_by_period(df, "ret_mlbeta"),
        },
    }

    out_json = base / "regime_performance_mlbeta_summary.json"
    out_md = base / "regime_performance_mlbeta_report.md"
    out_csv = base / "regime_performance_mlbeta_series.csv"

    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Charts for OOS regime-level comparison.
    regs = ["high", "mid", "low", "unknown"]
    roll_ret = [summary["by_regime_oos"]["roll40"][r]["total_return"] for r in regs]
    ml_ret = [summary["by_regime_oos"]["mlbeta"][r]["total_return"] for r in regs]

    x = np.arange(len(regs))
    width = 0.38

    fig1, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(x - width / 2, roll_ret, width=width, label="roll40", color="#4c78a8")
    ax1.bar(x + width / 2, ml_ret, width=width, label="mlbeta", color="#f58518")
    ax1.axhline(0.0, color="#333", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regs)
    ax1.set_ylabel("Total Return")
    ax1.set_title("OOS Total Return by Regime")
    ax1.grid(axis="y", alpha=0.2)
    ax1.legend(loc="best")
    fig1.tight_layout()
    fig1.savefig(base / "regime_performance_mlbeta_oos_return_by_regime.png", dpi=160)
    plt.close(fig1)

    roll_sh = [summary["by_regime_oos"]["roll40"][r]["sharpe"] for r in regs]
    ml_sh = [summary["by_regime_oos"]["mlbeta"][r]["sharpe"] for r in regs]

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    ax2.bar(x - width / 2, roll_sh, width=width, label="roll40", color="#4c78a8")
    ax2.bar(x + width / 2, ml_sh, width=width, label="mlbeta", color="#f58518")
    ax2.axhline(0.0, color="#333", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(regs)
    ax2.set_ylabel("Sharpe")
    ax2.set_title("OOS Sharpe by Regime")
    ax2.grid(axis="y", alpha=0.2)
    ax2.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig(base / "regime_performance_mlbeta_oos_sharpe_by_regime.png", dpi=160)
    plt.close(fig2)

    lines = []
    lines.append("# Regime Performance: ML Beta vs Rolling40")
    lines.append("")
    lines.append(f"- Pair: {args.asset} vs {args.btc}")
    lines.append(f"- Period: {summary['start']} to {summary['end']}")
    lines.append(f"- OOS view starts at: {args.wf_start_date}")
    lines.append("")

    oa_r = summary["overall"]["all"]["roll40"]
    oa_m = summary["overall"]["all"]["mlbeta"]
    oo_r = summary["overall"]["oos"]["roll40"]
    oo_m = summary["overall"]["oos"]["mlbeta"]

    lines.append("## Overall")
    lines.append(
        f"- All bars roll40: Return={oa_r['total_return']:.2%}, Sharpe={oa_r['sharpe']:.4f}, MDD={oa_r['max_drawdown']:.2%}"
    )
    lines.append(
        f"- All bars mlbeta: Return={oa_m['total_return']:.2%}, Sharpe={oa_m['sharpe']:.4f}, MDD={oa_m['max_drawdown']:.2%}"
    )
    lines.append(
        f"- OOS bars roll40: Return={oo_r['total_return']:.2%}, Sharpe={oo_r['sharpe']:.4f}, MDD={oo_r['max_drawdown']:.2%}"
    )
    lines.append(
        f"- OOS bars mlbeta: Return={oo_m['total_return']:.2%}, Sharpe={oo_m['sharpe']:.4f}, MDD={oo_m['max_drawdown']:.2%}"
    )
    lines.append("")

    lines.append("## OOS Regime Breakdown")
    for r in regs:
        mr = summary["by_regime_oos"]["roll40"][r]
        mm = summary["by_regime_oos"]["mlbeta"][r]
        lines.append(
            f"- {r}: bars={mr['bars']} | roll40 ret={mr['total_return']:.2%}, sh={mr['sharpe']:.3f} | mlbeta ret={mm['total_return']:.2%}, sh={mm['sharpe']:.3f}"
        )

    lines.append("")
    lines.append(f"- Series CSV: {out_csv.name}")
    lines.append(f"- JSON: {out_json.name}")
    lines.append("- Chart: regime_performance_mlbeta_oos_return_by_regime.png")
    lines.append("- Chart: regime_performance_mlbeta_oos_sharpe_by_regime.png")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="MSTR")
    parser.add_argument("--btc", default="BTC-USD")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-03-23")
    parser.add_argument("--wf-start-date", default="2024-01-01")

    parser.add_argument("--beta-window", type=int, default=40)
    parser.add_argument("--z-window", type=int, default=30)
    parser.add_argument("--z-entry", type=float, default=2.25)
    parser.add_argument("--z-exit", type=float, default=0.5)
    parser.add_argument("--fee-rate", type=float, default=0.00045)

    parser.add_argument("--vol-window", type=int, default=40)
    parser.add_argument("--quantile-lookback", type=int, default=252)
    parser.add_argument("--min-history", type=int, default=84)

    parser.add_argument("--min-train-bars", type=int, default=252)
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--pred-clip-low-q", type=float, default=0.02)
    parser.add_argument("--pred-clip-high-q", type=float, default=0.98)

    run(parser.parse_args())
