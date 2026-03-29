import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import optimize_oos_20_regime as core


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


def parse_int_list(v):
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def parse_float_list(v):
    return [float(x.strip()) for x in v.split(",") if x.strip()]


def metric_pack(returns):
    s = pd.Series(returns).fillna(0.0)
    return {
        "total_return": total_return(s),
        "sharpe_365": sharpe_365(s),
        "max_drawdown": max_drawdown(s),
        "active_ratio": float((s != 0.0).mean()),
    }


def build_returns_with_map(df, param_map):
    out = np.zeros(len(df), dtype=float)

    for reg, conf in param_map.items():
        reg_mask = df["vol_regime"] == reg
        if reg_mask.sum() == 0:
            continue

        bw = int(conf["beta_window"])
        ze = float(conf["z_entry"])
        zx = float(conf["z_exit"])

        # This strategy return stream is precomputed for each parameter tuple.
        ret_col = f"ret_bw{bw}_ze{ze}_zx{zx}"
        if ret_col not in df.columns:
            continue
        out[reg_mask.values] = df.loc[reg_mask, ret_col].fillna(0.0).values

    return out


def add_short_long_vol_state(df, short_window, long_window, expansion_thr, compression_thr):
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["short_vol"] = out["hedged_ret"].rolling(short_window).std(ddof=1)
    out["long_vol"] = out["hedged_ret"].rolling(long_window).std(ddof=1)
    out["vol_ratio_sl"] = out["short_vol"] / out["long_vol"].replace(0.0, np.nan)

    state = np.array(["unknown"] * len(out), dtype=object)
    valid = out["vol_ratio_sl"].notna()
    state[(valid & (out["vol_ratio_sl"] >= expansion_thr)).values] = "expansion"
    state[(valid & (out["vol_ratio_sl"] <= compression_thr)).values] = "compression"
    state[(valid & (out["vol_ratio_sl"] > compression_thr) & (out["vol_ratio_sl"] < expansion_thr)).values] = "neutral"
    out["vol_state_sl"] = state
    return out


def run(args):
    base = Path(__file__).resolve().parent

    beta_windows = parse_int_list(args.beta_windows)
    z_entries = parse_float_list(args.z_entries)
    z_exits = parse_float_list(args.z_exits)

    asset = core.fetch_yahoo_close(args.asset, args.start, args.end)
    btc = core.fetch_yahoo_close(args.btc, args.start, args.end)
    if asset.empty or btc.empty:
        raise RuntimeError("Failed to fetch Yahoo data")

    # Precompute regime labels once using fixed regime-definition parameters.
    # We use one base strategy to derive hedged_ret for volatility regime labeling.
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
    reg_df = add_short_long_vol_state(
        reg_df,
        short_window=args.short_vol_window,
        long_window=args.long_vol_window,
        expansion_thr=args.expansion_threshold,
        compression_thr=args.compression_threshold,
    )

    split_idx = int(len(reg_df) * args.oos_ratio)
    split_idx = max(50, min(len(reg_df) - 50, split_idx))
    split_date = pd.Timestamp(reg_df.iloc[split_idx]["date"])

    # Precompute return columns for each strategy parameter tuple over same date index.
    date_index = reg_df[["date", "vol_regime"]].copy()
    for bw in beta_windows:
        for ze in z_entries:
            for zx in z_exits:
                if zx >= ze:
                    continue
                s = core.build_strategy_df(
                    asset_close=asset,
                    btc_close=btc,
                    beta_window=bw,
                    z_window=args.z_window,
                    z_entry=ze,
                    z_exit=zx,
                    fee_rate=args.fee_rate,
                )
                if s.empty:
                    continue
                col = f"ret_bw{bw}_ze{ze}_zx{zx}"
                tmp = s[["date", "strategy_ret_net"]].rename(columns={"strategy_ret_net": col})
                date_index = date_index.merge(tmp, on="date", how="left")

    # Unknown regime explanation stats.
    unknown_rows = date_index[date_index["vol_regime"] == "unknown"]

    is_mask = date_index["date"] < split_date
    oos_mask = date_index["date"] >= split_date

    regimes = ["high", "mid", "low"]
    if args.include_unknown_in_map:
        regimes.append("unknown")
    param_map = {}
    regime_fit_rows = []

    # Fit best parameters per regime on IS only.
    for reg in regimes:
        reg_is = date_index[is_mask & (date_index["vol_regime"] == reg)]
        if len(reg_is) < args.min_regime_bars:
            continue

        best = None
        best_key = None

        for bw in beta_windows:
            for ze in z_entries:
                for zx in z_exits:
                    if zx >= ze:
                        continue
                    col = f"ret_bw{bw}_ze{ze}_zx{zx}"
                    if col not in reg_is.columns:
                        continue
                    m = metric_pack(reg_is[col])
                    score = (m["sharpe_365"], m["total_return"])
                    if best is None or score > (best["sharpe_365"], best["total_return"]):
                        best = m
                        best_key = {"beta_window": bw, "z_entry": ze, "z_exit": zx}

        if best_key is not None:
            param_map[reg] = best_key
            regime_fit_rows.append(
                {
                    "regime": reg,
                    "bars_is": int(len(reg_is)),
                    "beta_window": best_key["beta_window"],
                    "z_entry": best_key["z_entry"],
                    "z_exit": best_key["z_exit"],
                    "is_sharpe": best["sharpe_365"],
                    "is_return": best["total_return"],
                    "is_mdd": best["max_drawdown"],
                }
            )

    if not param_map:
        raise RuntimeError("No regime had enough IS bars for parameter fit")

    # Build mapped portfolio returns (frozen map from IS) and evaluate OOS.
    mapped_ret = build_returns_with_map(date_index, param_map)

    # Benchmark: single best global config on IS (across all IS bars)
    global_best = None
    global_cfg = None
    for bw in beta_windows:
        for ze in z_entries:
            for zx in z_exits:
                if zx >= ze:
                    continue
                col = f"ret_bw{bw}_ze{ze}_zx{zx}"
                if col not in date_index.columns:
                    continue
                m = metric_pack(date_index.loc[is_mask, col])
                score = (m["sharpe_365"], m["total_return"])
                if global_best is None or score > (global_best["sharpe_365"], global_best["total_return"]):
                    global_best = m
                    global_cfg = {"beta_window": bw, "z_entry": ze, "z_exit": zx, "col": col}

    if global_cfg is None:
        raise RuntimeError("No global parameter config found")

    global_ret = date_index[global_cfg["col"]].fillna(0.0).values

    res = {
        "split_date": split_date.strftime("%Y-%m-%d"),
        "oos_ratio": args.oos_ratio,
        "regime_definition": {
            "vol_window": args.vol_window,
            "quantile_lookback": args.quantile_lookback,
            "min_history": args.min_history,
            "unknown_definition": "bars with insufficient past rolling-vol history for q33/q66 thresholds",
            "unknown_policy": "mapped" if args.include_unknown_in_map else "no_trade",
            "short_long_vol": {
                "short_window": args.short_vol_window,
                "long_window": args.long_vol_window,
                "expansion_threshold": args.expansion_threshold,
                "compression_threshold": args.compression_threshold,
            },
        },
        "unknown_stats": {
            "total_unknown_bars": int(len(unknown_rows)),
            "unknown_is_bars": int((unknown_rows["date"] < split_date).sum()),
            "unknown_oos_bars": int((unknown_rows["date"] >= split_date).sum()),
        },
        "regime_share_is": date_index.loc[is_mask, "vol_regime"].value_counts(normalize=True, dropna=False).to_dict(),
        "regime_share_oos": date_index.loc[oos_mask, "vol_regime"].value_counts(normalize=True, dropna=False).to_dict(),
        "vol_state_share_is": reg_df.loc[is_mask, "vol_state_sl"].value_counts(normalize=True, dropna=False).to_dict(),
        "vol_state_share_oos": reg_df.loc[oos_mask, "vol_state_sl"].value_counts(normalize=True, dropna=False).to_dict(),
        "param_map": param_map,
        "regime_fit_rows": regime_fit_rows,
        "global_best_is": global_cfg,
        "metrics": {
            "mapped_is": metric_pack(mapped_ret[is_mask.values]),
            "mapped_oos": metric_pack(mapped_ret[oos_mask.values]),
            "global_is": metric_pack(global_ret[is_mask.values]),
            "global_oos": metric_pack(global_ret[oos_mask.values]),
        },
    }

    out_json = base / "regime_param_map_oos20_summary.json"
    out_md = base / "regime_param_map_oos20_report.md"
    out_series = base / "regime_param_map_oos20_series.csv"

    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")

    series_df = date_index[["date", "vol_regime"]].copy()
    series_df["mapped_ret"] = mapped_ret
    series_df["global_ret"] = global_ret
    series_df["is_flag"] = is_mask.values
    series_df.to_csv(out_series, index=False)

    lines = []
    lines.append("# Regime-Specific Parameter Map (IS Fit, OOS Test)")
    lines.append("")
    lines.append(f"- Split date (80/20 chronological): {res['split_date']}")
    lines.append("- Workflow: fit params per regime on IS only, freeze map, test on OOS")
    lines.append("")
    lines.append("## Unknown Regime")
    lines.append(f"- Definition: {res['regime_definition']['unknown_definition']}")
    lines.append(f"- Unknown bars total: {res['unknown_stats']['total_unknown_bars']}")
    lines.append(f"- Unknown bars IS: {res['unknown_stats']['unknown_is_bars']}")
    lines.append(f"- Unknown bars OOS: {res['unknown_stats']['unknown_oos_bars']}")
    lines.append(f"- Unknown policy in mapping: {res['regime_definition']['unknown_policy']}")
    lines.append("")
    lines.append("## Regime Shares")
    lines.append(f"- IS: {res['regime_share_is']}")
    lines.append(f"- OOS: {res['regime_share_oos']}")
    lines.append("")
    lines.append("## Short-vs-Long Volatility State Shares")
    lines.append(f"- IS: {res['vol_state_share_is']}")
    lines.append(f"- OOS: {res['vol_state_share_oos']}")
    lines.append("")
    lines.append("## IS-Fitted Parameter Map")
    for r in regime_fit_rows:
        lines.append(
            f"- {r['regime']}: bw={r['beta_window']}, z_entry={r['z_entry']}, z_exit={r['z_exit']} | "
            f"IS Sharpe={r['is_sharpe']:.4f}, IS Return={r['is_return']:.2%}, IS MDD={r['is_mdd']:.2%}, bars={r['bars_is']}"
        )
    lines.append("")
    lines.append("## Performance Comparison")
    mi = res["metrics"]["mapped_is"]
    mo = res["metrics"]["mapped_oos"]
    gi = res["metrics"]["global_is"]
    go = res["metrics"]["global_oos"]
    lines.append(
        f"- Mapped IS: Return={mi['total_return']:.2%}, Sharpe={mi['sharpe_365']:.4f}, MDD={mi['max_drawdown']:.2%}"
    )
    lines.append(
        f"- Mapped OOS: Return={mo['total_return']:.2%}, Sharpe={mo['sharpe_365']:.4f}, MDD={mo['max_drawdown']:.2%}"
    )
    lines.append(
        f"- Global IS: Return={gi['total_return']:.2%}, Sharpe={gi['sharpe_365']:.4f}, MDD={gi['max_drawdown']:.2%}"
    )
    lines.append(
        f"- Global OOS: Return={go['total_return']:.2%}, Sharpe={go['sharpe_365']:.4f}, MDD={go['max_drawdown']:.2%}"
    )
    lines.append("")
    lines.append(f"- Series CSV: {out_series.name}")
    lines.append(f"- JSON: {out_json.name}")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="MSTR")
    parser.add_argument("--btc", default="BTC-USD")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-03-23")
    parser.add_argument("--oos-ratio", type=float, default=0.8)
    parser.add_argument("--z-window", type=int, default=30)
    parser.add_argument("--fee-rate", type=float, default=0.00045)
    parser.add_argument("--beta-windows", default="60,90,120")
    parser.add_argument("--z-entries", default="1.75,2.0,2.25")
    parser.add_argument("--z-exits", default="0.5,0.75,1.0")
    parser.add_argument("--vol-window", type=int, default=40)
    parser.add_argument("--quantile-lookback", type=int, default=252)
    parser.add_argument("--min-history", type=int, default=84)
    parser.add_argument("--regime-base-beta-window", type=int, default=90)
    parser.add_argument("--regime-base-z-entry", type=float, default=1.75)
    parser.add_argument("--regime-base-z-exit", type=float, default=1.0)
    parser.add_argument("--min-regime-bars", type=int, default=30)
    parser.add_argument("--include-unknown-in-map", action="store_true")
    parser.add_argument("--short-vol-window", type=int, default=20)
    parser.add_argument("--long-vol-window", type=int, default=120)
    parser.add_argument("--expansion-threshold", type=float, default=1.2)
    parser.add_argument("--compression-threshold", type=float, default=0.8)
    args = parser.parse_args()
    run(args)
