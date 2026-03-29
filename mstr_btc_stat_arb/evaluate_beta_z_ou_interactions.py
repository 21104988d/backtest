import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def ols_alpha_beta(y, x):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
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


def rolling_zscore(x, window):
    x = pd.Series(x, dtype=float)
    m = x.rolling(window).mean()
    s = x.rolling(window).std(ddof=1)
    z = (x - m) / s
    return z.to_numpy()


def fit_ou_ar1(series):
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    if len(x) < 30:
        return np.nan
    x0 = x[:-1]
    x1 = x[1:]
    mx0 = x0.mean()
    mx1 = x1.mean()
    sxx = ((x0 - mx0) ** 2).sum()
    if sxx <= 0:
        return np.nan
    sxy = ((x0 - mx0) * (x1 - mx1)).sum()
    b = float(sxy / sxx)
    if 0.0 < b < 1.0:
        return float(-math.log(b))
    return np.nan


def rolling_ou_kappa(series, lookback):
    s = pd.Series(series, dtype=float).to_numpy()
    out = np.full(len(s), np.nan)
    for i in range(lookback, len(s)):
        out[i] = fit_ou_ar1(s[i - lookback : i])
    return out


def safe_corr(x, y, min_n=15):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < min_n:
        return np.nan, int(len(d))
    return float(d["x"].corr(d["y"])), int(len(d))


def main():
    base = Path(__file__).resolve().parent

    cfg = json.loads((base / "oos20_regime_parameter_sweep_summary.json").read_text(encoding="utf-8"))["best"]
    beta_window = int(cfg["beta_window"])
    z_window = 30

    s = pd.read_csv(base / "regime_portfolio_full_history_series.csv")
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").reset_index(drop=True)

    a_ret = s["a_ret"].to_numpy(dtype=float)
    b_ret = s["b_ret"].to_numpy(dtype=float)
    a_close = s["asset_close"].to_numpy(dtype=float)
    b_close = s["btc_close"].to_numpy(dtype=float)

    rb = rolling_beta_past_only(a_ret, b_ret, beta_window)
    rb_lag = pd.Series(rb).shift(1).to_numpy(dtype=float)
    d_beta = rb - rb_lag

    # term contributed by beta update between t-1 and t in hedged return definition
    # hedged_t(curr) - hedged_t(prev_beta) = -(delta_beta_t) * b_ret_t
    beta_adjustment = -d_beta * b_ret

    # spread and zscore are static-log-beta style, consistent with optimize_oos_20_regime
    log_a = np.log(a_close)
    log_b = np.log(b_close)
    a0, b0 = ols_alpha_beta(log_a, log_b)
    spread = log_a - (a0 + b0 * log_b)
    z = rolling_zscore(spread, z_window)

    ou_kappa = rolling_ou_kappa(spread, 120)

    df = pd.DataFrame(
        {
            "date": s["date"],
            "vol_regime": s["vol_regime"],
            "ret": s["ret_baseline"],
            "a_ret": a_ret,
            "b_ret": b_ret,
            "rolling_beta": rb,
            "delta_beta": d_beta,
            "beta_adjustment": beta_adjustment,
            "spread": spread,
            "zscore": z,
            "ou_kappa": ou_kappa,
        }
    )

    # Forward returns and mean-reversion targets
    for h in [1, 5, 10]:
        fwd = pd.Series(df["ret"]).shift(-1)
        if h > 1:
            for i in range(2, h + 1):
                fwd = fwd + pd.Series(df["ret"]).shift(-i)
        df[f"fwd_ret_{h}d"] = fwd

    # Positive if z moves toward zero next day
    df["z_reversion_1d"] = np.abs(df["zscore"]) - np.abs(pd.Series(df["zscore"]).shift(-1))

    # Correlation grid by regime and sample
    features = ["rolling_beta", "delta_beta", "beta_adjustment", "zscore", "ou_kappa"]
    targets = ["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_10d", "z_reversion_1d"]

    rows = []
    for sample_name, sample_mask in [
        ("FULL", df["date"] >= pd.Timestamp("1900-01-01")),
        ("OOS_2024+", df["date"] >= pd.Timestamp("2024-01-01")),
    ]:
        sub = df[sample_mask].copy()
        for rg in ["high", "mid", "low"]:
            g = sub[sub["vol_regime"] == rg].copy()
            if len(g) == 0:
                continue
            for f in features:
                for t in targets:
                    c, n = safe_corr(g[f], g[t])
                    rows.append(
                        {
                            "sample": sample_name,
                            "regime": rg,
                            "feature": f,
                            "target": t,
                            "corr": c,
                            "n": n,
                        }
                    )

    corr_tbl = pd.DataFrame(rows)

    # Structural relationship checks requested by user:
    # beta vs price returns, spread/zscore, and OU.
    structural_rows = []
    structural_features = ["rolling_beta", "delta_beta", "beta_adjustment"]
    structural_targets = ["a_ret", "b_ret", "spread", "zscore", "ou_kappa", "z_reversion_1d"]
    for sample_name, sample_mask in [
        ("FULL", df["date"] >= pd.Timestamp("1900-01-01")),
        ("OOS_2024+", df["date"] >= pd.Timestamp("2024-01-01")),
    ]:
        sub = df[sample_mask].copy()
        for rg in ["high", "mid", "low"]:
            g = sub[sub["vol_regime"] == rg].copy()
            if len(g) == 0:
                continue
            for f in structural_features:
                for t in structural_targets:
                    c, n = safe_corr(g[f], g[t])
                    structural_rows.append(
                        {
                            "sample": sample_name,
                            "regime": rg,
                            "feature": f,
                            "target": t,
                            "corr": c,
                            "n": n,
                        }
                    )
    structural_tbl = pd.DataFrame(structural_rows)

    # Quantile spread test for each feature against forward returns
    bucket_rows = []
    for sample_name, sample_mask in [
        ("FULL", df["date"] >= pd.Timestamp("1900-01-01")),
        ("OOS_2024+", df["date"] >= pd.Timestamp("2024-01-01")),
    ]:
        sub = df[sample_mask].copy()
        for rg in ["high", "mid", "low"]:
            g = sub[sub["vol_regime"] == rg].copy()
            if len(g) < 30:
                continue
            for f in features:
                tmp = g[[f, "fwd_ret_1d", "fwd_ret_5d"]].dropna(subset=[f]).copy()
                if len(tmp) < 30:
                    continue
                try:
                    tmp["q"] = pd.qcut(tmp[f], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
                except Exception:
                    continue
                for h in [1, 5]:
                    col = f"fwd_ret_{h}d"
                    q1 = tmp.loc[tmp["q"] == "Q1_low", col].dropna()
                    q4 = tmp.loc[tmp["q"] == "Q4_high", col].dropna()
                    bucket_rows.append(
                        {
                            "sample": sample_name,
                            "regime": rg,
                            "feature": f,
                            "horizon": f"{h}d",
                            "q1_mean": float(q1.mean()) if len(q1) else np.nan,
                            "q4_mean": float(q4.mean()) if len(q4) else np.nan,
                            "q4_minus_q1": float(q4.mean() - q1.mean()) if len(q1) and len(q4) else np.nan,
                            "n": int(min(len(q1), len(q4))),
                        }
                    )

    bucket_tbl = pd.DataFrame(bucket_rows)

    # Build specific checks for user's mechanism question
    mechanism = []
    for sample_name, sample_mask in [
        ("FULL", df["date"] >= pd.Timestamp("1900-01-01")),
        ("OOS_2024+", df["date"] >= pd.Timestamp("2024-01-01")),
    ]:
        sub = df[sample_mask].copy()
        for rg in ["high", "mid", "low"]:
            g = sub[sub["vol_regime"] == rg].copy()
            if len(g) == 0:
                continue
            # If beta changes are driving "reversion", these should be strong:
            c1, n1 = safe_corr(g["beta_adjustment"], g["z_reversion_1d"])
            c2, n2 = safe_corr(np.abs(g["delta_beta"]), g["z_reversion_1d"])
            c3, n3 = safe_corr(g["beta_adjustment"], g["fwd_ret_1d"])
            mechanism.append(
                {
                    "sample": sample_name,
                    "regime": rg,
                    "corr_betaAdj_vs_zReversion": c1,
                    "corr_absDeltaBeta_vs_zReversion": c2,
                    "corr_betaAdj_vs_fwdRet1d": c3,
                    "n": min(n1, n2, n3),
                }
            )
    mechanism_tbl = pd.DataFrame(mechanism)

    corr_path = base / "beta_ou_z_relationships_correlations.csv"
    bucket_path = base / "beta_ou_z_relationships_buckets.csv"
    mech_path = base / "beta_ou_z_mechanism_checks.csv"
    structural_path = base / "beta_ou_z_structural_relationships.csv"

    corr_tbl.to_csv(corr_path, index=False)
    bucket_tbl.to_csv(bucket_path, index=False)
    mechanism_tbl.to_csv(mech_path, index=False)
    structural_tbl.to_csv(structural_path, index=False)

    # Compact summary for easy read
    summary = {
        "config": {
            "beta_window": beta_window,
            "z_window": z_window,
            "ou_lookback": 120,
        },
        "top_abs_corr_to_fwd5d_oos": [],
    }

    oos_f5 = corr_tbl[(corr_tbl["sample"] == "OOS_2024+") & (corr_tbl["target"] == "fwd_ret_5d")].copy()
    if not oos_f5.empty:
        oos_f5["abs_corr"] = oos_f5["corr"].abs()
        top = oos_f5.sort_values("abs_corr", ascending=False).head(12)
        summary["top_abs_corr_to_fwd5d_oos"] = top.to_dict(orient="records")

    summary_path = base / "beta_ou_z_relationships_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Beta, Spread-Z, OU Interaction Diagnostics")
    lines.append("")
    lines.append("Question tested:")
    lines.append("- Is mean reversion in zscore mainly coming from beta changes, and therefore less monetizable in traded returns?")
    lines.append("")
    lines.append("Produced files:")
    lines.append("- beta_ou_z_relationships_correlations.csv")
    lines.append("- beta_ou_z_relationships_buckets.csv")
    lines.append("- beta_ou_z_mechanism_checks.csv")
    lines.append("- beta_ou_z_structural_relationships.csv")
    lines.append("- beta_ou_z_relationships_summary.json")
    lines.append("")

    lines.append("## Mechanism Check Snapshot")
    for _, r in mechanism_tbl.iterrows():
        lines.append(
            f"- {r['sample']} | {r['regime']} | corr(betaAdj, zReversion1d)={r['corr_betaAdj_vs_zReversion']:.4f} | "
            f"corr(|deltaBeta|, zReversion1d)={r['corr_absDeltaBeta_vs_zReversion']:.4f} | "
            f"corr(betaAdj, fwdRet1d)={r['corr_betaAdj_vs_fwdRet1d']:.4f}"
        )

    (base / "beta_ou_z_relationships_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Saved:")
    print(corr_path.name)
    print(bucket_path.name)
    print(mech_path.name)
    print(structural_path.name)
    print(summary_path.name)
    print("\nMechanism checks:")
    print(mechanism_tbl.to_string(index=False))


if __name__ == "__main__":
    main()
