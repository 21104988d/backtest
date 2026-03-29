import json
import numpy as np
import pandas as pd


def safe_corr(a, b):
    z = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(z) < 8:
        return np.nan, len(z)
    return float(z["a"].corr(z["b"])), len(z)


def main():
    s = pd.read_csv("regime_portfolio_full_history_series.csv")
    ou = pd.read_csv("ou_beta_regime_diagnostics_series.csv")

    s["date"] = pd.to_datetime(s["date"])
    ou["date"] = pd.to_datetime(ou["date"])

    df = (
        s[["date", "vol_regime", "ret_baseline"]]
        .merge(ou[["date", "ou_kappa"]], on="date", how="inner")
        .dropna(subset=["ou_kappa"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    for h in [1, 5, 10]:
        fwd = pd.Series(df["ret_baseline"]).shift(-1)
        if h > 1:
            for i in range(2, h + 1):
                fwd = fwd + pd.Series(df["ret_baseline"]).shift(-i)
        df[f"fwd_ret_{h}d"] = fwd

    rows = []
    for sample, mask in [
        ("FULL", df["date"] >= pd.Timestamp("1900-01-01")),
        ("OOS_2024+", df["date"] >= pd.Timestamp("2024-01-01")),
    ]:
        d = df[mask].copy()
        for rg in ["high", "mid", "low"]:
            g = d[d["vol_regime"] == rg].copy()
            if len(g) == 0:
                continue

            c1, _ = safe_corr(g["ou_kappa"], g["fwd_ret_1d"])
            c5, _ = safe_corr(g["ou_kappa"], g["fwd_ret_5d"])
            c10, _ = safe_corr(g["ou_kappa"], g["fwd_ret_10d"])

            tmp = g[["ou_kappa", "fwd_ret_1d", "fwd_ret_5d", "fwd_ret_10d"]].dropna(subset=["ou_kappa"]).copy()
            tmp["q"] = pd.qcut(tmp["ou_kappa"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")

            vals = {}
            for h in [1, 5, 10]:
                col = f"fwd_ret_{h}d"
                q1 = tmp.loc[tmp["q"] == "Q1_low", col].dropna()
                q4 = tmp.loc[tmp["q"] == "Q4_high", col].dropna()
                vals[f"q4_minus_q1_{h}d"] = float(q4.mean() - q1.mean()) if len(q1) and len(q4) else np.nan

            score = 0
            score += int(pd.notna(c1) and c1 > 0)
            score += int(pd.notna(c5) and c5 > 0)
            score += int(pd.notna(vals["q4_minus_q1_1d"]) and vals["q4_minus_q1_1d"] > 0)
            score += int(pd.notna(vals["q4_minus_q1_5d"]) and vals["q4_minus_q1_5d"] > 0)
            verdict = "supported" if score >= 3 else ("mixed" if score == 2 else "not_supported")

            rows.append(
                {
                    "sample": sample,
                    "regime": rg,
                    "n": int(len(g)),
                    "corr_fwd_1d": c1,
                    "corr_fwd_5d": c5,
                    "corr_fwd_10d": c10,
                    "q4_minus_q1_1d": vals["q4_minus_q1_1d"],
                    "q4_minus_q1_5d": vals["q4_minus_q1_5d"],
                    "q4_minus_q1_10d": vals["q4_minus_q1_10d"],
                    "verdict": verdict,
                }
            )

    res = pd.DataFrame(rows)
    res.to_csv("ou_kappa_higher_return_by_regime.csv", index=False)

    overall = {
        "hypothesis": "higher_ou_kappa_higher_return",
        "verdict_counts": {k: int(v) for k, v in res["verdict"].value_counts(dropna=False).to_dict().items()},
    }
    with open("ou_kappa_higher_return_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    lines = [
        "# Higher OU Kappa -> Higher Return (Regime Test)",
        "",
        "Using local saved series only.",
        "",
    ]
    for _, r in res.iterrows():
        lines.append(
            f"- {r['sample']} | {r['regime']} | corr1d={r['corr_fwd_1d']:.4f} | corr5d={r['corr_fwd_5d']:.4f} | q4-q1 1d={r['q4_minus_q1_1d']:.5f} | q4-q1 5d={r['q4_minus_q1_5d']:.5f} | verdict={r['verdict']}"
        )
    with open("ou_kappa_higher_return_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(res.to_string(index=False))
    print("\n" + json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
