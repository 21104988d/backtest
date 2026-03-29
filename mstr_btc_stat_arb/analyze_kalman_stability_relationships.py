import json
from pathlib import Path

import numpy as np
import pandas as pd

import regime_conditional_rule_backtest as core


def corr(a, b):
    z = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(z) < 20:
        return np.nan
    return float(z["a"].corr(z["b"]))


def mean(v):
    s = pd.Series(v).dropna()
    return float(s.mean()) if len(s) else np.nan


def sample_stats(d):
    st = d[d["beta_stable"] == 1]
    un = d[d["beta_stable"] == 0]

    return {
        "n": int(len(d)),
        "corr_stable_abs_z": corr(d["beta_stable"], d["zscore"].abs()),
        "corr_stable_abs_spread": corr(d["beta_stable"], d["spread"].abs()),
        "corr_stable_spread": corr(d["beta_stable"], d["spread"]),
        "corr_stable_dspread_next": corr(d["beta_stable"], d["dspread_next"]),
        "corr_stable_mr_score_next": corr(d["beta_stable"], d["mr_score_next"]),
        "corr_stable_mr_hit_next": corr(d["beta_stable"], d["mr_hit_next"]),
        "corr_stable_ret_base_next": corr(d["beta_stable"], d["ret_base_next"]),
        "corr_stable_ret_kstable_next": corr(d["beta_stable"], d["ret_kstable_next"]),
        "stable_mean_abs_z": mean(st["zscore"].abs()),
        "unstable_mean_abs_z": mean(un["zscore"].abs()),
        "stable_mr_hit": mean(st["mr_hit_next"]),
        "unstable_mr_hit": mean(un["mr_hit_next"]),
        "stable_mr_score": mean(st["mr_score_next"]),
        "unstable_mr_score": mean(un["mr_score_next"]),
        "stable_ret_base_next": mean(st["ret_base_next"]),
        "unstable_ret_base_next": mean(un["ret_base_next"]),
        "stable_ret_kstable_next": mean(st["ret_kstable_next"]),
        "unstable_ret_kstable_next": mean(un["ret_kstable_next"]),
    }


def main():
    base = Path(__file__).resolve().parent

    k = pd.read_csv(base / "kalman_beta_stability_filter_series.csv")
    k["date"] = pd.to_datetime(k["date"])

    s = pd.read_csv(base / "regime_portfolio_full_history_series.csv")
    s["date"] = pd.to_datetime(s["date"])

    ac = s["asset_close"].to_numpy(dtype=float)
    bc = s["btc_close"].to_numpy(dtype=float)
    a0, b0 = core.ols_alpha_beta(np.log(ac), np.log(bc))
    spread = np.log(ac) - (a0 + b0 * np.log(bc))
    dspread = np.diff(spread)

    x = k.merge(s[["date", "asset_close", "btc_close", "a_ret", "b_ret"]], on="date", how="inner")
    x = x.sort_values("date").reset_index(drop=True)

    # Return-indexed alignment with explicit common length guard.
    n = min(len(x), len(dspread), len(spread) - 1)
    x = x.iloc[:n].copy()
    x["spread"] = spread[1 : n + 1]
    x["dspread"] = dspread[:n]
    x["dspread_next"] = x["dspread"].shift(-1)

    # Positive -> spread moves toward mean next bar.
    x["mr_score_next"] = -np.sign(x["zscore"]) * x["dspread_next"]
    x["mr_hit_next"] = (x["mr_score_next"] > 0).astype(float)

    x["ret_base_next"] = x["ret_baseline"].shift(-1)
    x["ret_kstable_next"] = x["ret_kalman_stable"].shift(-1)

    samples = {
        "FULL": x["date"] >= pd.Timestamp("1900-01-01"),
        "OOS_2024_PLUS": x["date"] >= pd.Timestamp("2024-01-01"),
        "OOS_SPLIT_2025_05_16": x["date"] >= pd.Timestamp("2025-05-16"),
        "RECENT_2026": x["date"] >= pd.Timestamp("2026-01-01"),
    }

    result = {"samples": {}, "regime_mr_hit_delta": {}}
    for name, mask in samples.items():
        result["samples"][name] = sample_stats(x[mask].copy())

    for rg in ["high", "mid", "low", "unknown"]:
        d = x[x["vol_regime"] == rg].copy()
        if len(d) < 30:
            continue
        st = d[d["beta_stable"] == 1]
        un = d[d["beta_stable"] == 0]
        st_hit = mean(st["mr_hit_next"])
        un_hit = mean(un["mr_hit_next"])
        result["regime_mr_hit_delta"][rg] = {
            "n": int(len(d)),
            "stable_n": int(len(st)),
            "unstable_n": int(len(un)),
            "stable_mr_hit": st_hit,
            "unstable_mr_hit": un_hit,
            "delta": st_hit - un_hit if not np.isnan(st_hit) and not np.isnan(un_hit) else np.nan,
        }

    out_json = base / "kalman_stability_relationships_summary.json"
    out_md = base / "kalman_stability_relationships_report.md"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = ["# Kalman Stability Relationship Checks", ""]
    for name, v in result["samples"].items():
        lines.append(f"## {name}")
        lines.append(f"- n={v['n']}")
        lines.append(f"- corr(stable, |z|)={v['corr_stable_abs_z']:.6f}")
        lines.append(f"- corr(stable, |spread|)={v['corr_stable_abs_spread']:.6f}")
        lines.append(f"- corr(stable, mr_score_next)={v['corr_stable_mr_score_next']:.6f}")
        lines.append(f"- corr(stable, mr_hit_next)={v['corr_stable_mr_hit_next']:.6f}")
        lines.append(f"- corr(stable, ret_base_next)={v['corr_stable_ret_base_next']:.6f}")
        lines.append(f"- stable vs unstable |z|={v['stable_mean_abs_z']:.6f} vs {v['unstable_mean_abs_z']:.6f}")
        lines.append(f"- stable vs unstable MR-hit={v['stable_mr_hit']:.4f} vs {v['unstable_mr_hit']:.4f}")
        lines.append(f"- stable vs unstable next baseline ret={v['stable_ret_base_next']:.8f} vs {v['unstable_ret_base_next']:.8f}")
        lines.append("")

    lines.append("## Regime MR-Hit Delta")
    for rg, v in result["regime_mr_hit_delta"].items():
        lines.append(
            f"- {rg}: n={v['n']}, stable_mr_hit={v['stable_mr_hit']:.4f}, unstable_mr_hit={v['unstable_mr_hit']:.4f}, delta={v['delta']:.4f}"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
