import argparse
import bisect
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import optimize_oos_20_regime as core


def parse_int_list(v):
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def parse_float_list(v):
    return [float(x.strip()) for x in v.split(",") if x.strip()]


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


def sharpe_365(returns):
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0 or np.isnan(sd):
        return float("nan")
    return float((r.mean() / sd) * math.sqrt(365.0))


def total_return(returns):
    r = pd.Series(returns).fillna(0.0)
    if len(r) == 0:
        return float("nan")
    return float((1.0 + r).prod() - 1.0)


def max_drawdown(returns):
    r = pd.Series(returns).fillna(0.0)
    if len(r) == 0:
        return float("nan")
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax()) - 1.0
    return float(dd.min())


def active_ratio(returns):
    s = pd.Series(returns).fillna(0.0)
    if len(s) == 0:
        return float("nan")
    return float((s != 0.0).mean())


def metric_pack(returns):
    return {
        "total_return": total_return(returns),
        "sharpe_365": sharpe_365(returns),
        "max_drawdown": max_drawdown(returns),
        "active_ratio": active_ratio(returns),
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

        folds.append(
            {
                "train_end": train_end,
                "oos_start": start_idx,
                "oos_end": end_idx,
            }
        )
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

        folds.append(
            {
                "train_end": inner_train_end,
                "oos_start": start_idx,
                "oos_end": end_idx,
            }
        )
        start_idx = end_idx

    return folds


def evaluate_candidate_on_inner(candidate_col, df, inner_folds):
    oos_sharpes = []
    oos_returns = []
    oos_mdds = []
    oos_actives = []

    for f in inner_folds:
        r = df[candidate_col].iloc[f["oos_start"] : f["oos_end"]].fillna(0.0).values
        m = metric_pack(r)
        if not np.isnan(m["sharpe_365"]):
            oos_sharpes.append(m["sharpe_365"])
        oos_returns.append(m["total_return"])
        oos_mdds.append(m["max_drawdown"])
        oos_actives.append(m["active_ratio"])

    if not oos_returns:
        return None

    mean_sharpe = float(np.nanmean(oos_sharpes)) if oos_sharpes else float("nan")
    std_sharpe = float(np.nanstd(oos_sharpes, ddof=1)) if len(oos_sharpes) >= 2 else 0.0
    median_return = float(np.nanmedian(oos_returns))
    mean_return = float(np.nanmean(oos_returns))
    mean_mdd = float(np.nanmean(oos_mdds))
    mean_active = float(np.nanmean(oos_actives))
    nonneg_ratio = float(np.mean(np.array(oos_returns) >= 0.0))

    return {
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "median_return": median_return,
        "mean_return": mean_return,
        "mean_mdd": mean_mdd,
        "mean_active": mean_active,
        "nonneg_fold_ratio": nonneg_ratio,
    }


def objective(inner_stats, args):
    if inner_stats is None or np.isnan(inner_stats["mean_sharpe"]):
        return -1e18

    score = inner_stats["mean_sharpe"]
    score -= args.penalty_sharpe_std * inner_stats["std_sharpe"]

    if inner_stats["median_return"] < 0:
        score -= args.penalty_negative_median * abs(inner_stats["median_return"])

    if inner_stats["mean_mdd"] < 0:
        score -= args.penalty_mdd * abs(inner_stats["mean_mdd"])

    if inner_stats["mean_active"] < args.min_active_ratio:
        score -= args.penalty_low_active * (args.min_active_ratio - inner_stats["mean_active"])

    if inner_stats["nonneg_fold_ratio"] < args.min_nonneg_inner_ratio:
        score -= args.penalty_nonneg_ratio * (args.min_nonneg_inner_ratio - inner_stats["nonneg_fold_ratio"])

    return score


def constrained_regime_map(df, train_end, candidates, global_cfg, args):
    param_map = {}
    fit_rows = []
    regimes = ["high", "mid", "low"]

    g_bw, g_ze, g_zx = global_cfg["beta_window"], global_cfg["z_entry"], global_cfg["z_exit"]
    g_col = global_cfg["col"]

    for reg in regimes:
        mask = (df["vol_regime"] == reg).values
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[:train_end] = True
        reg_train = train_mask & mask
        bars = int(reg_train.sum())

        if bars < args.min_regime_bars:
            param_map[reg] = {"beta_window": g_bw, "z_entry": g_ze, "z_exit": g_zx, "col": g_col}
            fit_rows.append(
                {
                    "regime": reg,
                    "bars_is": bars,
                    "use_global": True,
                    "beta_window": g_bw,
                    "z_entry": g_ze,
                    "z_exit": g_zx,
                    "is_sharpe": float("nan"),
                }
            )
            continue

        global_stats = metric_pack(df.loc[reg_train, g_col].values)
        global_reg_score = global_stats["sharpe_365"]

        best = None
        for c in candidates:
            bw, ze, zx, col = c["beta_window"], c["z_entry"], c["z_exit"], c["col"]

            if abs(ze - g_ze) > args.max_regime_z_drift:
                continue
            if abs(zx - g_zx) > args.max_regime_exit_drift:
                continue
            if abs(bw - g_bw) > args.max_regime_bw_drift:
                continue

            m = metric_pack(df.loc[reg_train, col].values)
            s = m["sharpe_365"]
            if np.isnan(s):
                continue

            if (best is None) or (s > best["sharpe"]):
                best = {
                    "beta_window": bw,
                    "z_entry": ze,
                    "z_exit": zx,
                    "col": col,
                    "sharpe": s,
                }

        use_global = True
        chosen = {"beta_window": g_bw, "z_entry": g_ze, "z_exit": g_zx, "col": g_col}

        if best is not None:
            if np.isnan(global_reg_score):
                improve = 0.0
            else:
                improve = best["sharpe"] - global_reg_score

            if improve >= args.min_regime_sharpe_improve:
                use_global = False
                chosen = {k: best[k] for k in ["beta_window", "z_entry", "z_exit", "col"]}

        param_map[reg] = chosen
        fit_rows.append(
            {
                "regime": reg,
                "bars_is": bars,
                "use_global": use_global,
                "beta_window": chosen["beta_window"],
                "z_entry": chosen["z_entry"],
                "z_exit": chosen["z_exit"],
                "is_sharpe": (best["sharpe"] if (best is not None) else float("nan")),
                "global_regime_sharpe": global_reg_score,
            }
        )

    return param_map, fit_rows


def mapped_returns(df, param_map):
    out = np.zeros(len(df), dtype=float)
    for reg, cfg in param_map.items():
        col = cfg["col"]
        mask = (df["vol_regime"] == reg).values
        out[mask] = df.loc[mask, col].fillna(0.0).values
    return out


def evaluate_stream(returns, start_idx, end_idx):
    oos = pd.Series(returns).iloc[start_idx:end_idx].fillna(0.0).values
    return metric_pack(oos)


def pass_fail_gates(fold_df, args):
    med_ret = float(fold_df["map_oos_return"].median())
    nonneg = float((fold_df["map_oos_return"] >= 0.0).mean())
    worst_mdd = float(fold_df["map_oos_mdd"].min())
    mean_sharpe = float(fold_df["map_oos_sharpe"].mean())

    return {
        "median_oos_return": med_ret,
        "nonneg_fold_ratio": nonneg,
        "worst_fold_mdd": worst_mdd,
        "mean_oos_sharpe": mean_sharpe,
        "gate_median_return_pass": med_ret >= args.gate_min_median_oos_return,
        "gate_nonneg_ratio_pass": nonneg >= args.gate_min_nonneg_ratio,
        "gate_mdd_pass": worst_mdd >= args.gate_min_worst_mdd,
        "overall_pass": (
            med_ret >= args.gate_min_median_oos_return
            and nonneg >= args.gate_min_nonneg_ratio
            and worst_mdd >= args.gate_min_worst_mdd
        ),
    }


def run(args):
    base = Path(__file__).resolve().parent

    beta_windows = parse_int_list(args.beta_windows)
    z_entries = parse_float_list(args.z_entries)
    z_exits = parse_float_list(args.z_exits)

    candidates = []
    for bw in beta_windows:
        for ze in z_entries:
            for zx in z_exits:
                if zx >= ze:
                    continue
                col = f"ret_bw{bw}_ze{ze}_zx{zx}"
                candidates.append({"beta_window": bw, "z_entry": ze, "z_exit": zx, "col": col})

    asset = core.fetch_yahoo_close(args.asset, args.start, args.end)
    btc = core.fetch_yahoo_close(args.btc, args.start, args.end)
    if asset.empty or btc.empty:
        raise RuntimeError("Failed to fetch Yahoo data")

    base_df = core.build_strategy_df(
        asset_close=asset,
        btc_close=btc,
        beta_window=args.regime_base_beta_window,
        z_window=args.z_window,
        z_entry=args.regime_base_z_entry,
        z_exit=args.regime_base_z_exit,
        fee_rate=args.fee_rate,
    )
    if base_df.empty:
        raise RuntimeError("Base strategy dataframe is empty")

    reg_df = core.assign_regimes_leak_safe(
        base_df,
        vol_window=args.vol_window,
        quantile_lookback=args.quantile_lookback,
        min_history=args.min_history,
    )

    frame = reg_df[["date", "vol_regime"]].copy()

    # Precompute all candidate return streams once to keep nested CV fast and consistent.
    for c in candidates:
        s = core.build_strategy_df(
            asset_close=asset,
            btc_close=btc,
            beta_window=c["beta_window"],
            z_window=args.z_window,
            z_entry=c["z_entry"],
            z_exit=c["z_exit"],
            fee_rate=args.fee_rate,
        )
        if s.empty:
            continue
        tmp = s[["date", "strategy_ret_net"]].rename(columns={"strategy_ret_net": c["col"]})
        frame = frame.merge(tmp, on="date", how="left")

    dates = [d.strftime("%Y-%m-%d") for d in frame["date"]]
    outer_folds = build_outer_folds(
        dates=dates,
        wf_start_date=args.wf_start_date,
        oos_months=args.oos_months,
        min_train_bars=args.min_train_bars,
        min_oos_bars=args.min_oos_bars,
        embargo_bars=args.embargo_bars,
    )
    if not outer_folds:
        raise RuntimeError("No valid outer folds generated")

    rows = []
    all_map_oos = []
    all_global_oos = []
    all_naive_oos = []

    for i, of in enumerate(outer_folds, start=1):
        train_end = of["train_end"]
        oos_start = of["oos_start"]
        oos_end = of["oos_end"]

        inner_folds = build_inner_folds(
            dates=dates,
            train_end_idx=train_end,
            inner_oos_months=args.inner_oos_months,
            min_inner_train_bars=args.min_inner_train_bars,
            min_oos_bars=args.min_oos_bars,
            embargo_bars=args.embargo_bars,
        )
        if not inner_folds:
            continue

        # Nested selection: optimize stability-penalized inner OOS objective.
        best_nested = None
        for c in candidates:
            if c["col"] not in frame.columns:
                continue
            inner_stats = evaluate_candidate_on_inner(c["col"], frame, inner_folds)
            score = objective(inner_stats, args)
            if (best_nested is None) or (score > best_nested["score"]):
                best_nested = {
                    **c,
                    "score": score,
                    "inner": inner_stats,
                }

        if best_nested is None:
            continue

        # Naive comparator: best IS Sharpe only (single-split style).
        best_naive = None
        for c in candidates:
            if c["col"] not in frame.columns:
                continue
            train_ret = frame[c["col"]].iloc[:train_end].fillna(0.0).values
            m = metric_pack(train_ret)
            s = m["sharpe_365"]
            if np.isnan(s):
                continue
            if (best_naive is None) or (s > best_naive["is_sharpe"]):
                best_naive = {**c, "is_sharpe": s}

        if best_naive is None:
            continue

        # Regularized regime map around nested global candidate.
        param_map, fit_rows = constrained_regime_map(
            df=frame,
            train_end=train_end,
            candidates=candidates,
            global_cfg=best_nested,
            args=args,
        )

        map_ret_full = mapped_returns(frame, param_map)
        map_oos = map_ret_full[oos_start:oos_end]
        global_oos = frame[best_nested["col"]].iloc[oos_start:oos_end].fillna(0.0).values
        naive_oos = frame[best_naive["col"]].iloc[oos_start:oos_end].fillna(0.0).values

        map_m = metric_pack(map_oos)
        global_m = metric_pack(global_oos)
        naive_m = metric_pack(naive_oos)

        all_map_oos.extend(map_oos)
        all_global_oos.extend(global_oos)
        all_naive_oos.extend(naive_oos)

        rows.append(
            {
                "fold": i,
                "train_end_date": dates[train_end - 1],
                "oos_start_date": dates[oos_start],
                "oos_end_date": dates[oos_end - 1],
                "train_bars": train_end,
                "oos_bars": (oos_end - oos_start),
                "inner_fold_count": len(inner_folds),
                "nested_bw": best_nested["beta_window"],
                "nested_ze": best_nested["z_entry"],
                "nested_zx": best_nested["z_exit"],
                "nested_inner_score": best_nested["score"],
                "nested_inner_mean_sharpe": best_nested["inner"]["mean_sharpe"],
                "nested_inner_sharpe_std": best_nested["inner"]["std_sharpe"],
                "nested_inner_nonneg": best_nested["inner"]["nonneg_fold_ratio"],
                "naive_bw": best_naive["beta_window"],
                "naive_ze": best_naive["z_entry"],
                "naive_zx": best_naive["z_exit"],
                "naive_is_sharpe": best_naive["is_sharpe"],
                "global_oos_return": global_m["total_return"],
                "global_oos_sharpe": global_m["sharpe_365"],
                "global_oos_mdd": global_m["max_drawdown"],
                "naive_oos_return": naive_m["total_return"],
                "naive_oos_sharpe": naive_m["sharpe_365"],
                "naive_oos_mdd": naive_m["max_drawdown"],
                "map_oos_return": map_m["total_return"],
                "map_oos_sharpe": map_m["sharpe_365"],
                "map_oos_mdd": map_m["max_drawdown"],
                "map_regs_using_global": int(sum(1 for v in fit_rows if v["use_global"])),
            }
        )

    if not rows:
        raise RuntimeError("No folds evaluated after nested selection")

    fold_df = pd.DataFrame(rows)

    agg = {
        "nested_global_oos": metric_pack(all_global_oos),
        "regularized_map_oos": metric_pack(all_map_oos),
        "naive_is_best_oos": metric_pack(all_naive_oos),
    }

    gates = pass_fail_gates(fold_df, args)

    summary = {
        "config": {
            "asset": args.asset,
            "btc": args.btc,
            "start": args.start,
            "end": args.end,
            "wf_start_date": args.wf_start_date,
            "oos_months": args.oos_months,
            "inner_oos_months": args.inner_oos_months,
            "embargo_bars": args.embargo_bars,
            "min_train_bars": args.min_train_bars,
            "min_inner_train_bars": args.min_inner_train_bars,
            "min_oos_bars": args.min_oos_bars,
            "z_window": args.z_window,
            "fee_rate": args.fee_rate,
            "candidate_count": len(candidates),
            "regularization": {
                "max_regime_z_drift": args.max_regime_z_drift,
                "max_regime_exit_drift": args.max_regime_exit_drift,
                "max_regime_bw_drift": args.max_regime_bw_drift,
                "min_regime_sharpe_improve": args.min_regime_sharpe_improve,
                "min_regime_bars": args.min_regime_bars,
            },
            "objective_penalties": {
                "penalty_sharpe_std": args.penalty_sharpe_std,
                "penalty_negative_median": args.penalty_negative_median,
                "penalty_mdd": args.penalty_mdd,
                "penalty_low_active": args.penalty_low_active,
                "penalty_nonneg_ratio": args.penalty_nonneg_ratio,
            },
        },
        "fold_count": int(len(fold_df)),
        "aggregate": agg,
        "gates": gates,
    }

    out_csv = base / "nested_wf_regularized_folds.csv"
    out_json = base / "nested_wf_regularized_summary.json"
    out_md = base / "nested_wf_regularized_report.md"

    fold_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Nested Walk-Forward Robustness Report")
    lines.append("")
    lines.append("Method:")
    lines.append("- Purged (embargoed) outer walk-forward folds for final OOS evaluation")
    lines.append("- Nested inner folds to select global parameters with stability-penalized objective")
    lines.append("- Regularized regime map constrained near selected global params")
    lines.append("")
    lines.append(f"- Fold count: {summary['fold_count']}")
    lines.append(f"- Embargo bars: {args.embargo_bars}")
    lines.append(f"- Candidate count: {len(candidates)}")
    lines.append("")
    lines.append("## Aggregate OOS Metrics")

    for k, v in agg.items():
        lines.append(
            f"- {k}: Return={v['total_return']:.2%}, Sharpe={v['sharpe_365']:.4f}, MDD={v['max_drawdown']:.2%}, Active={v['active_ratio']:.2%}"
        )

    lines.append("")
    lines.append("## Robustness Gates (regularized map)")
    lines.append(f"- Median OOS return: {gates['median_oos_return']:.2%} (pass={gates['gate_median_return_pass']})")
    lines.append(f"- Non-negative fold ratio: {gates['nonneg_fold_ratio']:.2%} (pass={gates['gate_nonneg_ratio_pass']})")
    lines.append(f"- Worst fold MDD: {gates['worst_fold_mdd']:.2%} (pass={gates['gate_mdd_pass']})")
    lines.append(f"- Overall gate pass: {gates['overall_pass']}")
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
    parser.add_argument("--wf-start-date", default="2024-01-01")
    parser.add_argument("--oos-months", type=int, default=3)
    parser.add_argument("--inner-oos-months", type=int, default=2)
    parser.add_argument("--embargo-bars", type=int, default=10)
    parser.add_argument("--min-train-bars", type=int, default=252)
    parser.add_argument("--min-inner-train-bars", type=int, default=180)
    parser.add_argument("--min-oos-bars", type=int, default=35)
    parser.add_argument("--z-window", type=int, default=30)
    parser.add_argument("--fee-rate", type=float, default=0.00045)

    parser.add_argument("--beta-windows", default="60,90,120")
    parser.add_argument("--z-entries", default="1.75,2.0,2.25")
    parser.add_argument("--z-exits", default="0.5,0.75,1.0")

    parser.add_argument("--regime-base-beta-window", type=int, default=90)
    parser.add_argument("--regime-base-z-entry", type=float, default=1.75)
    parser.add_argument("--regime-base-z-exit", type=float, default=1.0)
    parser.add_argument("--vol-window", type=int, default=40)
    parser.add_argument("--quantile-lookback", type=int, default=252)
    parser.add_argument("--min-history", type=int, default=84)

    parser.add_argument("--max-regime-z-drift", type=float, default=0.25)
    parser.add_argument("--max-regime-exit-drift", type=float, default=0.25)
    parser.add_argument("--max-regime-bw-drift", type=int, default=30)
    parser.add_argument("--min-regime-sharpe-improve", type=float, default=0.15)
    parser.add_argument("--min-regime-bars", type=int, default=40)

    parser.add_argument("--penalty-sharpe-std", type=float, default=0.40)
    parser.add_argument("--penalty-negative-median", type=float, default=4.0)
    parser.add_argument("--penalty-mdd", type=float, default=1.2)
    parser.add_argument("--min-active-ratio", type=float, default=0.05)
    parser.add_argument("--penalty-low-active", type=float, default=2.0)
    parser.add_argument("--min-nonneg-inner-ratio", type=float, default=0.60)
    parser.add_argument("--penalty-nonneg-ratio", type=float, default=1.0)

    parser.add_argument("--gate-min-median-oos-return", type=float, default=0.0)
    parser.add_argument("--gate-min-nonneg-ratio", type=float, default=0.70)
    parser.add_argument("--gate-min-worst-mdd", type=float, default=-0.35)

    run(parser.parse_args())
