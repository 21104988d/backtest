import argparse
import bisect
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import optimize_oos_20_regime as core


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


def metrics_from_returns(ret, bars_per_year=365.0):
    s = pd.Series(ret).fillna(0.0)
    if len(s) == 0:
        return {
            "total_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "active_ratio": float("nan"),
        }

    eq = (1.0 + s).cumprod()
    dd = (eq / eq.cummax()) - 1.0
    sigma = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    sharpe = float("nan")
    if sigma and not math.isnan(sigma) and sigma > 0:
        sharpe = float((s.mean() / sigma) * math.sqrt(bars_per_year))

    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()) if len(dd) else float("nan"),
        "active_ratio": float((s != 0.0).mean()),
    }


def build_outer_folds(dates, wf_start_date, oos_months, min_train_bars, min_oos_bars, embargo_bars):
    wf_start_idx = bisect.bisect_left(dates, wf_start_date)
    folds = []
    start_idx = wf_start_idx

    while start_idx < len(dates):
        train_end = start_idx - embargo_bars
        if train_end < min_train_bars:
            start_idx += 1
            continue

        start_dt = datetime.strptime(dates[start_idx], "%Y-%m-%d").date()
        end_dt = add_months(start_dt, oos_months)
        end_idx = bisect.bisect_left(dates, end_dt.isoformat())
        if end_idx <= start_idx:
            start_idx += 1
            continue

        oos_len = end_idx - start_idx
        if oos_len < min_oos_bars:
            break

        folds.append({"train_end": train_end, "oos_start": start_idx, "oos_end": end_idx})
        start_idx = end_idx

    return folds


def build_inner_folds(dates, train_end_idx, inner_oos_months, min_inner_train_bars, min_oos_bars, embargo_bars):
    folds = []
    start_idx = min_inner_train_bars + embargo_bars

    while start_idx < train_end_idx:
        inner_train_end = start_idx - embargo_bars
        if inner_train_end < min_inner_train_bars:
            start_idx += 1
            continue

        start_dt = datetime.strptime(dates[start_idx], "%Y-%m-%d").date()
        end_dt = add_months(start_dt, inner_oos_months)
        end_idx = bisect.bisect_left(dates, end_dt.isoformat())
        end_idx = min(end_idx, train_end_idx)
        if end_idx <= start_idx:
            start_idx += 1
            continue

        oos_len = end_idx - start_idx
        if oos_len < min_oos_bars:
            break

        folds.append({"train_end": inner_train_end, "oos_start": start_idx, "oos_end": end_idx})
        start_idx = end_idx

    return folds


def objective(inner_stats, args):
    if inner_stats is None or np.isnan(inner_stats["mean_sharpe"]):
        return -1e18

    score = inner_stats["mean_sharpe"]
    score -= args.penalty_sharpe_std * inner_stats["std_sharpe"]

    if inner_stats["median_return"] < 0:
        score -= args.penalty_negative_median * abs(inner_stats["median_return"])
    if inner_stats["mean_mdd"] < 0:
        score -= args.penalty_mdd * abs(inner_stats["mean_mdd"])
    if inner_stats["nonneg_fold_ratio"] < args.min_nonneg_inner_ratio:
        score -= args.penalty_nonneg_ratio * (args.min_nonneg_inner_ratio - inner_stats["nonneg_fold_ratio"])
    return score


def evaluate_candidate_on_inner(ret_stream, inner_folds):
    sharpes = []
    rets = []
    mdds = []
    for f in inner_folds:
        r = ret_stream[f["oos_start"] : f["oos_end"]]
        m = metrics_from_returns(r)
        if not np.isnan(m["sharpe"]):
            sharpes.append(m["sharpe"])
        rets.append(m["total_return"])
        mdds.append(m["max_drawdown"])

    if not rets:
        return None

    return {
        "mean_sharpe": float(np.nanmean(sharpes)) if sharpes else float("nan"),
        "std_sharpe": float(np.nanstd(sharpes, ddof=1)) if len(sharpes) >= 2 else 0.0,
        "median_return": float(np.nanmedian(rets)),
        "mean_mdd": float(np.nanmean(mdds)),
        "nonneg_fold_ratio": float(np.mean(np.array(rets) >= 0.0)),
    }


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


def run(args):
    base = Path(__file__).resolve().parent

    z_entries = [float(x.strip()) for x in args.z_entries.split(",") if x.strip()]
    z_exits = [float(x.strip()) for x in args.z_exits.split(",") if x.strip()]
    candidates = []
    for ze in z_entries:
        for zx in z_exits:
            if zx >= ze:
                continue
            candidates.append((ze, zx))

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
        raise RuntimeError("Not enough aligned rows")

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

    beta_roll40 = rolling_beta_past_only(a_ret, b_ret, args.beta_window)
    hedged_roll40 = np.where(np.isnan(beta_roll40), 0.0, a_ret - beta_roll40 * b_ret)

    reg = core.assign_regimes_leak_safe(
        pd.DataFrame({"date": dates.iloc[1:].values, "hedged_ret": hedged_roll40, "strategy_ret_net": np.zeros(len(hedged_roll40))}),
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

    # Precompute return streams by candidate and hedge mode.
    streams_roll40 = {}
    streams_ml = {}
    for ze, zx in candidates:
        pos = build_position_signal(spread, args.z_window, ze, zx)
        streams_roll40[(ze, zx)] = build_net_returns(pos, hedged_roll40, args.fee_rate)
        streams_ml[(ze, zx)] = build_net_returns(pos, hedged_ml, args.fee_rate)

    outer_folds = build_outer_folds(
        dates=ret_dates,
        wf_start_date=args.wf_start_date,
        oos_months=args.oos_months,
        min_train_bars=args.min_train_bars,
        min_oos_bars=args.min_oos_bars,
        embargo_bars=args.embargo_bars,
    )
    if not outer_folds:
        raise RuntimeError("No valid outer folds")

    rows = []
    all_roll40 = []
    all_ml = []

    for i, of in enumerate(outer_folds, start=1):
        train_end = of["train_end"]
        o0 = of["oos_start"]
        o1 = of["oos_end"]

        inner_folds = build_inner_folds(
            dates=ret_dates,
            train_end_idx=train_end,
            inner_oos_months=args.inner_oos_months,
            min_inner_train_bars=args.min_inner_train_bars,
            min_oos_bars=args.min_oos_bars,
            embargo_bars=args.embargo_bars,
        )
        if not inner_folds:
            continue

        best_roll = None
        best_ml = None

        for c in candidates:
            st_r = evaluate_candidate_on_inner(streams_roll40[c], inner_folds)
            sc_r = objective(st_r, args)
            if (best_roll is None) or (sc_r > best_roll["score"]):
                best_roll = {"candidate": c, "score": sc_r, "inner": st_r}

            st_m = evaluate_candidate_on_inner(streams_ml[c], inner_folds)
            sc_m = objective(st_m, args)
            if (best_ml is None) or (sc_m > best_ml["score"]):
                best_ml = {"candidate": c, "score": sc_m, "inner": st_m}

        if best_roll is None or best_ml is None:
            continue

        r_oos = streams_roll40[best_roll["candidate"]][o0:o1]
        m_oos = streams_ml[best_ml["candidate"]][o0:o1]
        rm = metrics_from_returns(r_oos)
        mm = metrics_from_returns(m_oos)

        all_roll40.extend(r_oos)
        all_ml.extend(m_oos)

        rows.append(
            {
                "fold": i,
                "train_end_date": ret_dates[train_end - 1],
                "oos_start_date": ret_dates[o0],
                "oos_end_date": ret_dates[o1 - 1],
                "oos_bars": o1 - o0,
                "roll40_ze": best_roll["candidate"][0],
                "roll40_zx": best_roll["candidate"][1],
                "roll40_inner_score": best_roll["score"],
                "ml_ze": best_ml["candidate"][0],
                "ml_zx": best_ml["candidate"][1],
                "ml_inner_score": best_ml["score"],
                "roll40_oos_return": rm["total_return"],
                "roll40_oos_sharpe": rm["sharpe"],
                "roll40_oos_mdd": rm["max_drawdown"],
                "ml_oos_return": mm["total_return"],
                "ml_oos_sharpe": mm["sharpe"],
                "ml_oos_mdd": mm["max_drawdown"],
                "delta_ret_ml_minus_roll40": mm["total_return"] - rm["total_return"],
                "delta_sharpe_ml_minus_roll40": (mm["sharpe"] - rm["sharpe"]) if (not math.isnan(mm["sharpe"]) and not math.isnan(rm["sharpe"])) else float("nan"),
            }
        )

    if not rows:
        raise RuntimeError("No folds evaluated")

    fold_df = pd.DataFrame(rows)
    agg_r = metrics_from_returns(all_roll40)
    agg_m = metrics_from_returns(all_ml)

    prev = None
    prev_path = base / "nested_wf_regularized_summary.json"
    if prev_path.exists():
        try:
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
        except Exception:
            prev = None

    summary = {
        "asset": args.asset,
        "btc": args.btc,
        "fold_count": int(len(fold_df)),
        "config": {
            "beta_window": args.beta_window,
            "wf_start_date": args.wf_start_date,
            "oos_months": args.oos_months,
            "inner_oos_months": args.inner_oos_months,
            "embargo_bars": args.embargo_bars,
            "candidate_count": len(candidates),
        },
        "aggregate": {
            "roll40_nested": agg_r,
            "mlbeta_nested": agg_m,
            "delta_ml_minus_roll40": {
                "total_return": agg_m["total_return"] - agg_r["total_return"],
                "sharpe": (agg_m["sharpe"] - agg_r["sharpe"]) if (not math.isnan(agg_m["sharpe"]) and not math.isnan(agg_r["sharpe"])) else float("nan"),
                "max_drawdown": agg_m["max_drawdown"] - agg_r["max_drawdown"],
            },
        },
        "ml_better_fold_ratio": float((fold_df["ml_oos_return"] > fold_df["roll40_oos_return"]).mean()),
        "previous_nested_summary": prev,
    }

    out_csv = base / "nested_wf_mlbeta_folds.csv"
    out_json = base / "nested_wf_mlbeta_summary.json"
    out_md = base / "nested_wf_mlbeta_report.md"

    fold_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Nested WF: ML Beta vs Rolling-40")
    lines.append("")
    lines.append("- Nested selection on inner folds for each hedge mode separately")
    lines.append("- OOS evaluation on purged outer folds")
    lines.append("")
    lines.append(
        f"- Rolling40 nested OOS: Return={agg_r['total_return']:.2%}, Sharpe={agg_r['sharpe']:.4f}, MDD={agg_r['max_drawdown']:.2%}"
    )
    lines.append(
        f"- ML-beta nested OOS: Return={agg_m['total_return']:.2%}, Sharpe={agg_m['sharpe']:.4f}, MDD={agg_m['max_drawdown']:.2%}"
    )
    d = summary["aggregate"]["delta_ml_minus_roll40"]
    lines.append(f"- Delta (ML - Rolling40): Return={d['total_return']:.2%}, Sharpe={d['sharpe']:.4f}, MDD={d['max_drawdown']:.2%}")
    lines.append(f"- ML better fold ratio: {summary['ml_better_fold_ratio']:.2%}")

    if prev and isinstance(prev, dict):
        ng = prev.get("summary", {}).get("aggregate", {}).get("nested_global_oos")
        if ng:
            lines.append(
                f"- Previous nested_global_oos: Return={ng.get('total_return', float('nan')):.2%}, Sharpe={ng.get('sharpe_365', float('nan')):.4f}, MDD={ng.get('max_drawdown', float('nan')):.2%}"
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
    parser.add_argument("--z-entries", default="1.5,1.75,2.0,2.25")
    parser.add_argument("--z-exits", default="0.5,0.75,1.0")
    parser.add_argument("--fee-rate", type=float, default=0.00045)

    parser.add_argument("--vol-window", type=int, default=40)
    parser.add_argument("--quantile-lookback", type=int, default=252)
    parser.add_argument("--min-history", type=int, default=84)

    parser.add_argument("--wf-start-date", default="2024-01-01")
    parser.add_argument("--oos-months", type=int, default=3)
    parser.add_argument("--inner-oos-months", type=int, default=2)
    parser.add_argument("--embargo-bars", type=int, default=10)
    parser.add_argument("--min-train-bars", type=int, default=252)
    parser.add_argument("--min-inner-train-bars", type=int, default=180)
    parser.add_argument("--min-oos-bars", type=int, default=35)

    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--pred-clip-low-q", type=float, default=0.02)
    parser.add_argument("--pred-clip-high-q", type=float, default=0.98)

    parser.add_argument("--penalty-sharpe-std", type=float, default=0.8)
    parser.add_argument("--penalty-negative-median", type=float, default=8.0)
    parser.add_argument("--penalty-mdd", type=float, default=1.2)
    parser.add_argument("--min-nonneg-inner-ratio", type=float, default=0.7)
    parser.add_argument("--penalty-nonneg-ratio", type=float, default=3.0)

    run(parser.parse_args())
