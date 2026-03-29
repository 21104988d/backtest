import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(ret):
    r = pd.Series(ret).fillna(0.0)
    if len(r) == 0:
        return {
            "ret": np.nan,
            "sharpe": np.nan,
            "mdd": np.nan,
            "n": 0,
            "active_ratio": np.nan,
        }
    eq = (1 + r).cumprod()
    total = float(eq.iloc[-1] - 1)
    sd = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    sharpe = float((r.mean() / sd) * np.sqrt(365)) if (sd and not np.isnan(sd) and sd > 0) else np.nan
    mdd = float((eq / eq.cummax() - 1).min())
    active = float((r != 0).mean())
    return {
        "ret": total,
        "sharpe": sharpe,
        "mdd": mdd,
        "n": int(len(r)),
        "active_ratio": active,
    }


def main():
    base = Path(__file__).resolve().parent

    df = pd.read_csv(base / "best_variant_trading_rule_series.csv")
    df["date"] = pd.to_datetime(df["date"])

    cfg = json.loads((base / "oos20_regime_parameter_sweep_summary.json").read_text(encoding="utf-8"))["best"]
    split_date = pd.Timestamp(cfg["split_date"])

    samples = [
        ("FULL", df["date"] >= pd.Timestamp("1900-01-01")),
        ("OOS_2024_PLUS", df["date"] >= pd.Timestamp("2024-01-01")),
        ("OOS_SPLIT_DATE", df["date"] >= split_date),
    ]
    regimes = ["high", "mid", "low", "unknown"]

    rows = []
    for sample_name, sample_mask in samples:
        d = df[sample_mask].copy()

        # pooled rows
        mb_all = metrics(d["ret_baseline"])
        mv_all = metrics(d["ret_best_variant"])
        rows.append(
            {
                "sample": sample_name,
                "regime": "ALL",
                "baseline_ret": mb_all["ret"],
                "baseline_sharpe": mb_all["sharpe"],
                "baseline_mdd": mb_all["mdd"],
                "baseline_active": mb_all["active_ratio"],
                "variant_ret": mv_all["ret"],
                "variant_sharpe": mv_all["sharpe"],
                "variant_mdd": mv_all["mdd"],
                "variant_active": mv_all["active_ratio"],
                "delta_ret": mv_all["ret"] - mb_all["ret"],
                "delta_sharpe": (mv_all["sharpe"] if not np.isnan(mv_all["sharpe"]) else 0.0)
                - (mb_all["sharpe"] if not np.isnan(mb_all["sharpe"]) else 0.0),
                "bars": int(len(d)),
            }
        )

        for rg in regimes:
            g = d[d["vol_regime"] == rg].copy()
            if len(g) == 0:
                continue
            mb = metrics(g["ret_baseline"])
            mv = metrics(g["ret_best_variant"])
            rows.append(
                {
                    "sample": sample_name,
                    "regime": rg,
                    "baseline_ret": mb["ret"],
                    "baseline_sharpe": mb["sharpe"],
                    "baseline_mdd": mb["mdd"],
                    "baseline_active": mb["active_ratio"],
                    "variant_ret": mv["ret"],
                    "variant_sharpe": mv["sharpe"],
                    "variant_mdd": mv["mdd"],
                    "variant_active": mv["active_ratio"],
                    "delta_ret": mv["ret"] - mb["ret"],
                    "delta_sharpe": (mv["sharpe"] if not np.isnan(mv["sharpe"]) else 0.0)
                    - (mb["sharpe"] if not np.isnan(mb["sharpe"]) else 0.0),
                    "bars": int(len(g)),
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["sample", "regime"]).reset_index(drop=True)
    out.to_csv(base / "best_variant_regime_performance.csv", index=False)

    # concise summary for quick read
    summary = {
        "split_date": str(split_date.date()),
        "top_improvements_oos_2024_plus_by_delta_sharpe": [],
        "top_improvements_oos_split_date_by_delta_sharpe": [],
    }

    o1 = out[(out["sample"] == "OOS_2024_PLUS") & (out["regime"] != "ALL")].copy()
    if not o1.empty:
        s1 = o1.sort_values("delta_sharpe", ascending=False).head(4)
        summary["top_improvements_oos_2024_plus_by_delta_sharpe"] = s1.to_dict(orient="records")

    o2 = out[(out["sample"] == "OOS_SPLIT_DATE") & (out["regime"] != "ALL")].copy()
    if not o2.empty:
        s2 = o2.sort_values("delta_sharpe", ascending=False).head(4)
        summary["top_improvements_oos_split_date_by_delta_sharpe"] = s2.to_dict(orient="records")

    (base / "best_variant_regime_performance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Best Variant Performance By Regime")
    lines.append("")
    lines.append("Columns: baseline vs best variant, plus delta.")
    lines.append("")
    for sample_name in ["FULL", "OOS_2024_PLUS", "OOS_SPLIT_DATE"]:
        lines.append(f"## {sample_name}")
        part = out[out["sample"] == sample_name]
        for _, r in part.iterrows():
            lines.append(
                f"- {r['regime']}: bars={int(r['bars'])}, "
                f"baseline(ret={r['baseline_ret']:.2%}, sh={r['baseline_sharpe']:.4f}), "
                f"variant(ret={r['variant_ret']:.2%}, sh={r['variant_sharpe']:.4f}), "
                f"delta(ret={r['delta_ret']:.2%}, sh={r['delta_sharpe']:.4f})"
            )
        lines.append("")

    (base / "best_variant_regime_performance_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
