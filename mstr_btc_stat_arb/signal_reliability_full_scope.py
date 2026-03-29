import json
from pathlib import Path

import numpy as np
import pandas as pd

import relationship_horizon_alpha_test as hz


def permutation_pvalue(x, y, n_perm=5000, seed=42):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 30:
        return np.nan, int(len(d)), np.nan

    rng = np.random.default_rng(seed)
    xv = d["x"].to_numpy()
    yv = d["y"].to_numpy()
    obs = float(np.corrcoef(xv, yv)[0, 1])

    perm = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(yv)
        perm[i] = np.corrcoef(xv, yp)[0, 1]

    p_two = float((np.abs(perm) >= abs(obs)).mean())
    return obs, int(len(d)), p_two


def bootstrap_mean_ci(x, n_boot=5000, seed=123):
    s = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(s) < 20:
        return np.nan, np.nan, np.nan, int(len(s))

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = rng.choice(s, size=len(s), replace=True).mean()

    mean = float(s.mean())
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return mean, lo, hi, int(len(s))


def main():
    base = Path(__file__).resolve().parent

    df = hz.build_df(base)
    horizons = [1, 3, 5, 10, 15, 20]
    df = hz.make_forward_returns(df, horizons)

    cfg = json.loads((base / "oos20_regime_parameter_sweep_summary.json").read_text(encoding="utf-8"))["best"]
    z_entry = float(cfg["z_entry"])

    scopes = {
        "FULL": df["date"] >= pd.Timestamp("1900-01-01"),
        "OOS_2024_PLUS": df["date"] >= pd.Timestamp("2024-01-01"),
    }

    corr_rows = []
    edge_rows = []

    for scope_name, scope_mask in scopes.items():
        d = df[scope_mask].copy()

        for rg in ["high", "mid", "low"]:
            g = d[d["vol_regime"] == rg].copy()
            if len(g) == 0:
                continue

            for h in horizons:
                obs, n, p = permutation_pvalue(g["zscore"], g[f"fwd_ret_{h}d"], n_perm=4000, seed=42 + h)
                corr_rows.append(
                    {
                        "scope": scope_name,
                        "regime": rg,
                        "horizon": h,
                        "corr": obs,
                        "abs_corr": float(abs(obs)) if not np.isnan(obs) else np.nan,
                        "r2": float(obs * obs) if not np.isnan(obs) else np.nan,
                        "pvalue_perm_two_sided": p,
                        "n": n,
                    }
                )

            # Entry-conditioned edge: contrarian signed forward 5d and 10d
            sig = np.sign(g["zscore"])
            entry = g["zscore"].abs() >= z_entry

            for h in [5, 10]:
                edge = (-sig[entry] * g.loc[entry, f"fwd_ret_{h}d"]).dropna()
                mean, lo, hi, n_edge = bootstrap_mean_ci(edge, n_boot=4000, seed=100 + h)
                edge_rows.append(
                    {
                        "scope": scope_name,
                        "regime": rg,
                        "horizon": h,
                        "entry_count": n_edge,
                        "edge_mean": mean,
                        "edge_ci95_lo": lo,
                        "edge_ci95_hi": hi,
                        "edge_significant_nonzero": bool((not np.isnan(lo)) and (lo > 0 or hi < 0)),
                    }
                )

    corr_tbl = pd.DataFrame(corr_rows)
    edge_tbl = pd.DataFrame(edge_rows)

    corr_tbl.to_csv(base / "signal_reliability_fullscope_corr.csv", index=False)
    edge_tbl.to_csv(base / "signal_reliability_fullscope_edge.csv", index=False)

    # Compact summary: best horizon by abs corr and significant rows
    summary = {
        "best_abs_corr_per_scope_regime": [],
        "significant_corr_rows_p_lt_0_05": [],
        "significant_edge_rows_ci_excludes_0": [],
    }

    for scope in ["FULL", "OOS_2024_PLUS"]:
        for rg in ["high", "mid", "low"]:
            x = corr_tbl[(corr_tbl["scope"] == scope) & (corr_tbl["regime"] == rg)].copy()
            if len(x) == 0:
                continue
            x = x.sort_values("abs_corr", ascending=False)
            summary["best_abs_corr_per_scope_regime"].append(x.iloc[0].to_dict())

    sig_corr = corr_tbl[corr_tbl["pvalue_perm_two_sided"] < 0.05].copy()
    summary["significant_corr_rows_p_lt_0_05"] = sig_corr.to_dict(orient="records")

    sig_edge = edge_tbl[edge_tbl["edge_significant_nonzero"]].copy()
    summary["significant_edge_rows_ci_excludes_0"] = sig_edge.to_dict(orient="records")

    (base / "signal_reliability_fullscope_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Signal Reliability Full-Scope Test")
    lines.append("")
    lines.append("Includes FULL and OOS_2024_PLUS, by regime and horizon.")
    lines.append("")

    lines.append("## Best abs-correlation by scope/regime")
    for r in summary["best_abs_corr_per_scope_regime"]:
        lines.append(
            f"- {r['scope']} | {r['regime']} | h={int(r['horizon'])}d | corr={r['corr']:.4f} | p={r['pvalue_perm_two_sided']:.4f} | n={int(r['n'])}"
        )

    lines.append("")
    lines.append(f"- Significant correlation rows (p<0.05): {len(summary['significant_corr_rows_p_lt_0_05'])}")
    lines.append(f"- Significant entry-edge rows (CI excludes 0): {len(summary['significant_edge_rows_ci_excludes_0'])}")
    lines.append("")
    lines.append("Files:")
    lines.append("- signal_reliability_fullscope_corr.csv")
    lines.append("- signal_reliability_fullscope_edge.csv")
    lines.append("- signal_reliability_fullscope_summary.json")

    (base / "signal_reliability_fullscope_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
