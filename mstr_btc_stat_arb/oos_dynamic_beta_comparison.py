import argparse
import bisect
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import optimize_oos_20_regime as core


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def std_sample(vals):
    n = len(vals)
    if n < 2:
        return float("nan")
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))


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


def max_drawdown(eq):
    if len(eq) == 0:
        return float("nan")
    peak = eq[0]
    mdd = 0.0
    for v in eq:
        peak = max(peak, v)
        dd = (v / peak) - 1.0 if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
    return mdd


def metrics_from_returns(ret, bars_per_year=365.0):
    s = pd.Series(ret).fillna(0.0)
    if len(s) == 0:
        return {
            "total_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "active_ratio": float("nan"),
        }

    eq = []
    v = 1.0
    for r in s:
        v *= 1.0 + float(r)
        eq.append(v)

    sigma = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    sharpe = float("nan")
    if sigma and not math.isnan(sigma) and sigma > 0:
        sharpe = float((s.mean() / sigma) * math.sqrt(bars_per_year))

    return {
        "total_return": float(eq[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(max_drawdown(eq)),
        "active_ratio": float((s != 0.0).mean()),
    }


def month_end_day(y, m):
    if m in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if m in (4, 6, 9, 11):
        return 30
    leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
    return 29 if leap else 28


def add_months(d, months):
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, month_end_day(y, m))
    return date(y, m, day)


def build_oos_folds(ret_dates, wf_start_date, oos_months, min_train_bars, min_oos_bars, embargo_bars):
    wf_start_idx = bisect.bisect_left(ret_dates, wf_start_date)
    folds = []
    start_idx = wf_start_idx

    while start_idx < len(ret_dates):
        train_end = start_idx - embargo_bars
        if train_end < min_train_bars:
            start_idx += 1
            continue

        start_dt = datetime.strptime(ret_dates[start_idx], "%Y-%m-%d").date()
        end_dt = add_months(start_dt, oos_months)
        end_idx = bisect.bisect_left(ret_dates, end_dt.isoformat())
        if end_idx <= start_idx:
            start_idx += 1
            continue

        oos_len = end_idx - start_idx
        if oos_len < min_oos_bars:
            break

        folds.append({"train_end": train_end, "oos_start": start_idx, "oos_end": end_idx})
        start_idx = end_idx

    return folds


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


def build_dynamic_beta_prediction(beta_true, rolling_vol, vol_regime, min_train_bars, ridge_alpha, clip_low_q, clip_high_q):
    work = pd.DataFrame(
        {
            "beta_true": beta_true,
            "rolling_vol": rolling_vol,
            "vol_regime": vol_regime,
        }
    )
    work["beta_l1"] = work["beta_true"].shift(1)
    work["beta_diff_5"] = work["beta_true"] - work["beta_true"].shift(5)
    work["beta_mean_20"] = work["beta_true"].rolling(20).mean()
    work["beta_std_20"] = work["beta_true"].rolling(20).std(ddof=1)
    work["time_idx"] = np.arange(len(work), dtype=float) / max(1, len(work) - 1)
    work["reg_high"] = (work["vol_regime"] == "high").astype(float)
    work["reg_mid"] = (work["vol_regime"] == "mid").astype(float)
    work["reg_low"] = (work["vol_regime"] == "low").astype(float)
    work["intercept"] = 1.0

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

    beta_pred = np.full(len(work), np.nan)
    for t in range(min_train_bars + 1, len(work)):
        y_idx = np.arange(1, t)
        x_idx = y_idx - 1
        X = work.loc[x_idx, feature_cols].copy()
        y = work.loc[y_idx, "beta_true"].copy()

        train = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True).rename("y")], axis=1).dropna()
        if len(train) < min_train_bars:
            continue

        X_train = train[feature_cols].to_numpy(dtype=float)
        y_train = train["y"].to_numpy(dtype=float)
        w = ridge_fit(X_train, y_train, alpha=ridge_alpha)
        if w is None:
            continue

        x_pred = work.loc[t - 1, feature_cols]
        if x_pred.isna().any():
            continue

        p = float(np.dot(x_pred.to_numpy(dtype=float), w))
        lo = float(np.nanquantile(y_train, clip_low_q))
        hi = float(np.nanquantile(y_train, clip_high_q))
        beta_pred[t] = min(max(p, lo), hi)

    return beta_pred


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
    if len(df) < args.beta_window + args.min_train_bars + args.min_oos_bars + 50:
        raise RuntimeError("Not enough aligned rows for OOS evaluation")

    a_close = df["close_asset"].to_numpy()
    b_close = df["close_btc"].to_numpy()
    dates = pd.to_datetime(df["date"])
    ret_dates = [d.strftime("%Y-%m-%d") for d in dates.iloc[1:]]

    a_ret = (a_close[1:] / a_close[:-1]) - 1.0
    b_ret = (b_close[1:] / b_close[:-1]) - 1.0

    log_a = np.log(a_close)
    log_b = np.log(b_close)
    a0, b0 = ols_alpha_beta(log_a, log_b)
    spread = log_a - (a0 + b0 * log_b)

    pos = build_position_signal(spread, args.z_window, args.z_entry, args.z_exit)

    beta_true = rolling_beta_past_only(a_ret, b_ret, args.beta_window)
    hedged_true = np.where(np.isnan(beta_true), 0.0, a_ret - beta_true * b_ret)

    # Regime features for beta model are built from rolling40 hedged stream.
    reg = core.assign_regimes_leak_safe(
        pd.DataFrame(
            {
                "date": dates.iloc[1:].values,
                "hedged_ret": hedged_true,
                "strategy_ret_net": np.zeros(len(hedged_true)),
            }
        ),
        vol_window=args.vol_window,
        quantile_lookback=args.quantile_lookback,
        min_history=args.min_history,
    )

    beta_pred = build_dynamic_beta_prediction(
        beta_true=beta_true,
        rolling_vol=reg["rolling_vol"].to_numpy(),
        vol_regime=reg["vol_regime"].to_numpy(),
        min_train_bars=args.min_train_bars,
        ridge_alpha=args.ridge_alpha,
        clip_low_q=args.pred_clip_low_q,
        clip_high_q=args.pred_clip_high_q,
    )

    beta_model = np.where(np.isnan(beta_pred), beta_true, beta_pred)
    hedged_model = np.where(np.isnan(beta_model), 0.0, a_ret - beta_model * b_ret)

    ret_true = build_net_returns(pos, hedged_true, args.fee_rate)
    ret_model = build_net_returns(pos, hedged_model, args.fee_rate)

    folds = build_oos_folds(
        ret_dates=ret_dates,
        wf_start_date=args.wf_start_date,
        oos_months=args.oos_months,
        min_train_bars=args.min_train_bars,
        min_oos_bars=args.min_oos_bars,
        embargo_bars=args.embargo_bars,
    )
    if not folds:
        raise RuntimeError("No valid OOS folds generated")

    rows = []
    all_true = []
    all_model = []
    for i, f in enumerate(folds, start=1):
        o0, o1 = f["oos_start"], f["oos_end"]
        tr = ret_true[o0:o1]
        mr = ret_model[o0:o1]

        tm = metrics_from_returns(tr)
        mm = metrics_from_returns(mr)

        all_true.extend(tr)
        all_model.extend(mr)

        rows.append(
            {
                "fold": i,
                "train_end_date": ret_dates[f["train_end"] - 1],
                "oos_start_date": ret_dates[o0],
                "oos_end_date": ret_dates[o1 - 1],
                "oos_bars": o1 - o0,
                "roll40_oos_return": tm["total_return"],
                "roll40_oos_sharpe": tm["sharpe"],
                "roll40_oos_mdd": tm["max_drawdown"],
                "mlbeta_oos_return": mm["total_return"],
                "mlbeta_oos_sharpe": mm["sharpe"],
                "mlbeta_oos_mdd": mm["max_drawdown"],
                "return_delta_ml_minus_roll40": mm["total_return"] - tm["total_return"],
                "sharpe_delta_ml_minus_roll40": (mm["sharpe"] - tm["sharpe"]) if (not math.isnan(mm["sharpe"]) and not math.isnan(tm["sharpe"])) else float("nan"),
            }
        )

    fold_df = pd.DataFrame(rows)
    agg_true = metrics_from_returns(all_true)
    agg_model = metrics_from_returns(all_model)

    previous_nested = None
    prev_path = base / "nested_wf_regularized_summary.json"
    if prev_path.exists():
        try:
            previous_nested = json.loads(prev_path.read_text(encoding="utf-8"))
        except Exception:
            previous_nested = None

    summary = {
        "asset": args.asset,
        "btc": args.btc,
        "period_start": ret_dates[0],
        "period_end": ret_dates[-1],
        "fold_count": int(len(fold_df)),
        "config": {
            "beta_window": args.beta_window,
            "z_window": args.z_window,
            "z_entry": args.z_entry,
            "z_exit": args.z_exit,
            "fee_rate": args.fee_rate,
            "wf_start_date": args.wf_start_date,
            "oos_months": args.oos_months,
            "embargo_bars": args.embargo_bars,
        },
        "aggregate": {
            "roll40": agg_true,
            "ml_beta": agg_model,
            "delta_ml_minus_roll40": {
                "total_return": agg_model["total_return"] - agg_true["total_return"],
                "sharpe": (agg_model["sharpe"] - agg_true["sharpe"]) if (not math.isnan(agg_model["sharpe"]) and not math.isnan(agg_true["sharpe"])) else float("nan"),
                "max_drawdown": agg_model["max_drawdown"] - agg_true["max_drawdown"],
            },
        },
        "ml_better_fold_ratio": float((fold_df["mlbeta_oos_return"] > fold_df["roll40_oos_return"]).mean()),
        "previous_nested_summary": previous_nested,
    }

    out_csv = base / "oos_dynamic_beta_vs_roll40_folds.csv"
    out_json = base / "oos_dynamic_beta_vs_roll40_summary.json"
    out_md = base / "oos_dynamic_beta_vs_roll40_report.md"

    fold_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# OOS Comparison: ML Predicted Beta vs Rolling-40 Beta")
    lines.append("")
    lines.append("Method:")
    lines.append("- Same signal logic (spread z-score entries/exits)")
    lines.append("- Compare hedge engine only: rolling40 beta vs ML-predicted dynamic beta")
    lines.append("- Same purged walk-forward OOS folds")
    lines.append("")
    lines.append(f"- Fold count: {summary['fold_count']}")
    lines.append(
        f"- Rolling40 OOS: Return={agg_true['total_return']:.2%}, Sharpe={agg_true['sharpe']:.4f}, MDD={agg_true['max_drawdown']:.2%}"
    )
    lines.append(
        f"- ML beta OOS: Return={agg_model['total_return']:.2%}, Sharpe={agg_model['sharpe']:.4f}, MDD={agg_model['max_drawdown']:.2%}"
    )
    lines.append(
        f"- Delta (ML - Rolling40): Return={summary['aggregate']['delta_ml_minus_roll40']['total_return']:.2%}, Sharpe={summary['aggregate']['delta_ml_minus_roll40']['sharpe']:.4f}, MDD={summary['aggregate']['delta_ml_minus_roll40']['max_drawdown']:.2%}"
    )
    lines.append(f"- ML better fold ratio: {summary['ml_better_fold_ratio']:.2%}")
    lines.append("")

    if previous_nested and isinstance(previous_nested, dict):
        agg_prev = previous_nested.get("summary", {}).get("aggregate", {})
        ng = agg_prev.get("nested_global_oos")
        nv = agg_prev.get("naive_is_best_oos")
        if ng:
            lines.append(
                f"- Previous nested global OOS: Return={ng.get('total_return', float('nan')):.2%}, Sharpe={ng.get('sharpe_365', float('nan')):.4f}, MDD={ng.get('max_drawdown', float('nan')):.2%}"
            )
        if nv:
            lines.append(
                f"- Previous naive IS-best OOS: Return={nv.get('total_return', float('nan')):.2%}, Sharpe={nv.get('sharpe_365', float('nan')):.4f}, MDD={nv.get('max_drawdown', float('nan')):.2%}"
            )

    lines.append("")
    lines.append(f"- Fold CSV: {out_csv.name}")
    lines.append(f"- JSON summary: {out_json.name}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"summary": summary, "csv": out_csv.name, "json": out_json.name, "md": out_md.name}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="MSTR")
    parser.add_argument("--btc", default="BTC-USD")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-03-23")

    parser.add_argument("--beta-window", type=int, default=40)
    parser.add_argument("--z-window", type=int, default=30)
    parser.add_argument("--z-entry", type=float, default=2.25)
    parser.add_argument("--z-exit", type=float, default=0.5)
    parser.add_argument("--fee-rate", type=float, default=0.00045)

    parser.add_argument("--vol-window", type=int, default=40)
    parser.add_argument("--quantile-lookback", type=int, default=252)
    parser.add_argument("--min-history", type=int, default=84)

    parser.add_argument("--min-train-bars", type=int, default=252)
    parser.add_argument("--wf-start-date", default="2024-01-01")
    parser.add_argument("--oos-months", type=int, default=3)
    parser.add_argument("--min-oos-bars", type=int, default=35)
    parser.add_argument("--embargo-bars", type=int, default=10)

    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--pred-clip-low-q", type=float, default=0.02)
    parser.add_argument("--pred-clip-high-q", type=float, default=0.98)

    run(parser.parse_args())
