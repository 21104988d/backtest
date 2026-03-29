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


def summarize_triplet(df_slice):
    mb = metrics(df_slice["ret_baseline"])
    mv = metrics(df_slice["ret_best_variant"])
    mh = metrics(df_slice["ret_hybrid"])

    return {
        "baseline": mb,
        "variant": mv,
        "hybrid": mh,
        "hybrid_minus_baseline_ret": mh["ret"] - mb["ret"],
        "hybrid_minus_baseline_sharpe": (mh["sharpe"] if not np.isnan(mh["sharpe"]) else 0.0)
        - (mb["sharpe"] if not np.isnan(mb["sharpe"]) else 0.0),
        "hybrid_minus_variant_ret": mh["ret"] - mv["ret"],
        "hybrid_minus_variant_sharpe": (mh["sharpe"] if not np.isnan(mh["sharpe"]) else 0.0)
        - (mv["sharpe"] if not np.isnan(mv["sharpe"]) else 0.0),
    }


def main():
    base = Path(__file__).resolve().parent

    df = pd.read_csv(base / "best_variant_trading_rule_series.csv")
    df["date"] = pd.to_datetime(df["date"])

    cfg = json.loads((base / "oos20_regime_parameter_sweep_summary.json").read_text(encoding="utf-8"))["best"]
    split_date = pd.Timestamp(cfg["split_date"])

    # Hybrid rule requested by user:
    # - high regime: baseline
    # - low regime: variant
    # - mid/unknown: baseline (default fallback)
    df["ret_hybrid"] = np.where(df["vol_regime"] == "low", df["ret_best_variant"], df["ret_baseline"])

    # Save time series for plotting or further analysis.
    df[["date", "vol_regime", "ret_baseline", "ret_best_variant", "ret_hybrid"]].to_csv(
        base / "hybrid_high_baseline_low_variant_series.csv", index=False
    )

    sample_masks = {
        "FULL": df["date"] >= pd.Timestamp("1900-01-01"),
        "IS_PRE_2024": df["date"] < pd.Timestamp("2024-01-01"),
        "IS_SPLIT_DATE": df["date"] < split_date,
        "OOS_2024_PLUS": df["date"] >= pd.Timestamp("2024-01-01"),
        "OOS_SPLIT_DATE": df["date"] >= split_date,
    }

    regimes = ["ALL", "high", "mid", "low", "unknown"]
    rows = []

    summary = {"split_date": str(split_date.date()), "hybrid_rule": "high->baseline, low->variant, mid/unknown->baseline", "samples": {}}

    for sample_name, sample_mask in sample_masks.items():
        d = df[sample_mask].copy()
        summary["samples"][sample_name] = {}

        for rg in regimes:
            if rg == "ALL":
                x = d
            else:
                x = d[d["vol_regime"] == rg]
            if len(x) == 0:
                continue

            s = summarize_triplet(x)
            summary["samples"][sample_name][rg] = s

            rows.append(
                {
                    "sample": sample_name,
                    "regime": rg,
                    "bars": int(len(x)),
                    "baseline_ret": s["baseline"]["ret"],
                    "baseline_sharpe": s["baseline"]["sharpe"],
                    "baseline_mdd": s["baseline"]["mdd"],
                    "variant_ret": s["variant"]["ret"],
                    "variant_sharpe": s["variant"]["sharpe"],
                    "variant_mdd": s["variant"]["mdd"],
                    "hybrid_ret": s["hybrid"]["ret"],
                    "hybrid_sharpe": s["hybrid"]["sharpe"],
                    "hybrid_mdd": s["hybrid"]["mdd"],
                    "hybrid_minus_baseline_ret": s["hybrid_minus_baseline_ret"],
                    "hybrid_minus_baseline_sharpe": s["hybrid_minus_baseline_sharpe"],
                    "hybrid_minus_variant_ret": s["hybrid_minus_variant_ret"],
                    "hybrid_minus_variant_sharpe": s["hybrid_minus_variant_sharpe"],
                }
            )

    table = pd.DataFrame(rows).sort_values(["sample", "regime"]).reset_index(drop=True)
    table.to_csv(base / "hybrid_high_baseline_low_variant_breakdown.csv", index=False)

    (base / "hybrid_high_baseline_low_variant_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Hybrid Rule Evaluation")
    lines.append("")
    lines.append("Hybrid rule:")
    lines.append("- high regime: baseline")
    lines.append("- low regime: variant")
    lines.append("- mid/unknown regime: baseline")
    lines.append("")

    for sample_name in ["FULL", "IS_PRE_2024", "IS_SPLIT_DATE", "OOS_2024_PLUS", "OOS_SPLIT_DATE"]:
        lines.append(f"## {sample_name}")
        part = table[table["sample"] == sample_name]
        for _, r in part.iterrows():
            lines.append(
                f"- {r['regime']}: bars={int(r['bars'])}, "
                f"baseline(ret={r['baseline_ret']:.2%}, sh={r['baseline_sharpe']:.4f}), "
                f"variant(ret={r['variant_ret']:.2%}, sh={r['variant_sharpe']:.4f}), "
                f"hybrid(ret={r['hybrid_ret']:.2%}, sh={r['hybrid_sharpe']:.4f}), "
                f"hybrid-baseline(ret={r['hybrid_minus_baseline_ret']:.2%}, sh={r['hybrid_minus_baseline_sharpe']:.4f})"
            )
        lines.append("")

    (base / "hybrid_high_baseline_low_variant_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
