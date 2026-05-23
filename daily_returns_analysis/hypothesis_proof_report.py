#!/usr/bin/env python3
"""Build a consolidated hypothesis-proof package for N=5 fixed 1.68."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIXED_PARAMS = (1.5, 1.68)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hypothesis proof artifacts")
    parser.add_argument(
        "--overlap-prefix",
        default="hypothesis_168_overlap",
        help="CSV prefix for overlap run (without _windows/_stress suffix)",
    )
    parser.add_argument(
        "--nonoverlap-prefix",
        default="hypothesis_168_nonoverlap",
        help="CSV prefix for non-overlap run (without _windows/_stress suffix)",
    )
    parser.add_argument(
        "--output-prefix",
        default="hypothesis_proof",
        help="Output file prefix for summary/report/visual",
    )
    return parser.parse_args()


def load_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def load_bundle(prefix: str) -> Dict[str, Optional[pd.DataFrame]]:
    p = Path(prefix)
    return {
        "summary": pd.read_csv(p.with_suffix(".csv")),
        "windows": pd.read_csv(Path(f"{prefix}_windows.csv")),
        "stress": load_csv_if_exists(Path(f"{prefix}_stress.csv")),
        "robust": load_csv_if_exists(Path(f"{prefix}_robust_band.csv")),
    }


def get_fixed_rows(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary[(summary["sl_mode"] == "fixed") & (summary["sl_param"].isin(FIXED_PARAMS))].copy()
    return out.set_index("sl_param", drop=False)


def pairwise_window_stats(windows: pd.DataFrame) -> Dict[str, float]:
    fixed = windows[(windows["sl_mode"] == "fixed") & (windows["sl_param"].isin(FIXED_PARAMS))].copy()
    pivot = fixed.pivot_table(index="window_id", columns="sl_param", values="test_total_return_pct")
    if not all(param in pivot.columns for param in FIXED_PARAMS):
        return {
            "window_count": 0.0,
            "beat_rate_168_vs_150_pct": np.nan,
            "median_delta_ret_pct": np.nan,
            "mean_delta_ret_pct": np.nan,
            "worst_delta_ret_pct": np.nan,
        }

    delta = pivot[1.68] - pivot[1.5]
    return {
        "window_count": float(len(delta)),
        "beat_rate_168_vs_150_pct": float((delta > 0).mean() * 100.0),
        "median_delta_ret_pct": float(delta.median()),
        "mean_delta_ret_pct": float(delta.mean()),
        "worst_delta_ret_pct": float(delta.min()),
    }


def candidate_window_distribution(windows: pd.DataFrame, sl_param: float) -> Dict[str, float]:
    rows = windows[(windows["sl_mode"] == "fixed") & (windows["sl_param"] == sl_param)].copy()
    if rows.empty:
        return {
            "window_count": 0.0,
            "positive_window_ratio_pct": np.nan,
            "median_test_return_pct": np.nan,
            "mean_test_return_pct": np.nan,
            "worst_test_return_pct": np.nan,
        }

    tr = rows["test_total_return_pct"]
    return {
        "window_count": float(len(rows)),
        "positive_window_ratio_pct": float((tr > 0).mean() * 100.0),
        "median_test_return_pct": float(tr.median()),
        "mean_test_return_pct": float(tr.mean()),
        "worst_test_return_pct": float(tr.min()),
    }


def append_metric(records, evidence_set: str, candidate: str, metric: str, value) -> None:
    if isinstance(value, (np.floating, np.integer)):
        value = float(value)
    records.append(
        {
            "evidence_set": evidence_set,
            "candidate": candidate,
            "metric": metric,
            "value": value,
        }
    )


def build_summary_records(evaluations: Dict[str, Dict[str, Optional[pd.DataFrame]]]) -> pd.DataFrame:
    records = []

    main_metrics = [
        "windows_evaluated",
        "window_pass_rate_pct",
        "test_positive_ratio_pct",
        "median_test_return_pct",
        "median_test_sharpe",
        "median_test_max_drawdown_pct",
        "median_test_cvar5_pct",
        "test_return_iqr_pct",
        "median_generalization_gap_abs_pct",
        "objective_score",
        "selected_by_objective",
        "in_robust_band",
    ]

    for evidence_set, bundle in evaluations.items():
        summary = bundle["summary"]
        windows = bundle["windows"]
        stress = bundle["stress"]
        robust = bundle["robust"]

        fixed_rows = get_fixed_rows(summary)

        robust_lookup = set()
        if robust is not None and not robust.empty:
            robust_lookup = {
                (int(r["n"]), str(r["sl_mode"]), float(r["sl_param"]))
                for _, r in robust.iterrows()
            }

        for sl in FIXED_PARAMS:
            if sl not in fixed_rows.index:
                continue
            row = fixed_rows.loc[sl]
            candidate = f"fixed_{sl:.2f}".rstrip("0").rstrip(".")

            row_copy = row.to_dict()
            in_robust = (int(row_copy["n"]), str(row_copy["sl_mode"]), float(row_copy["sl_param"])) in robust_lookup
            row_copy["in_robust_band"] = in_robust

            for metric in main_metrics:
                append_metric(records, evidence_set, candidate, metric, row_copy.get(metric, np.nan))

            win_dist = candidate_window_distribution(windows, sl)
            for metric_name, metric_val in win_dist.items():
                append_metric(records, evidence_set, candidate, f"window_dist_{metric_name}", metric_val)

        if all(sl in fixed_rows.index for sl in FIXED_PARAMS):
            delta_candidate = "delta_1.68_minus_1.5"
            for metric in [
                "window_pass_rate_pct",
                "test_positive_ratio_pct",
                "median_test_return_pct",
                "median_test_sharpe",
                "median_test_max_drawdown_pct",
                "median_test_cvar5_pct",
                "test_return_iqr_pct",
                "median_generalization_gap_abs_pct",
                "objective_score",
            ]:
                delta_val = float(fixed_rows.loc[1.68, metric]) - float(fixed_rows.loc[1.5, metric])
                append_metric(records, evidence_set, delta_candidate, metric, delta_val)

            pair_stats = pairwise_window_stats(windows)
            for metric_name, metric_val in pair_stats.items():
                append_metric(records, evidence_set, delta_candidate, metric_name, metric_val)

        if stress is not None and not stress.empty:
            for _, row in stress.iterrows():
                scenario = str(row["scenario"])
                candidate = f"stress_{scenario}"
                for metric in [
                    "test_return_pct",
                    "test_daily_win_rate_pct",
                    "test_position_win_rate_pct",
                    "test_max_drawdown_pct",
                    "portfolio_positive_ratio_pct",
                    "portfolio_median_return_pct",
                    "portfolio_worst_return_pct",
                ]:
                    append_metric(records, evidence_set, candidate, metric, float(row[metric]))

    return pd.DataFrame(records)


def get_metric(df: pd.DataFrame, evidence_set: str, candidate: str, metric: str) -> float:
    row = df[
        (df["evidence_set"] == evidence_set)
        & (df["candidate"] == candidate)
        & (df["metric"] == metric)
    ]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["value"])


def build_verdict(summary_records: pd.DataFrame) -> Dict[str, object]:
    core_checks = {}
    core_checks["selected_nonoverlap_is_168"] = bool(
        get_metric(summary_records, "nonoverlap", "fixed_1.68", "selected_by_objective") > 0.5
    )
    core_checks["pass_rate_nonoverlap_ge_80"] = (
        get_metric(summary_records, "nonoverlap", "fixed_1.68", "window_pass_rate_pct") >= 80.0
    )
    core_checks["positive_windows_nonoverlap_ge_90"] = (
        get_metric(summary_records, "nonoverlap", "fixed_1.68", "window_dist_positive_window_ratio_pct") >= 90.0
    )
    core_checks["worst_window_nonoverlap_positive"] = (
        get_metric(summary_records, "nonoverlap", "fixed_1.68", "window_dist_worst_test_return_pct") > 0.0
    )
    core_checks["stress_high_nonoverlap_positive"] = (
        get_metric(summary_records, "nonoverlap", "stress_high", "test_return_pct") > 0.0
    )

    relative_checks = {}
    relative_checks["beat_rate_nonoverlap_ge_50"] = (
        get_metric(summary_records, "nonoverlap", "delta_1.68_minus_1.5", "beat_rate_168_vs_150_pct")
        >= 50.0
    )
    relative_checks["median_delta_nonoverlap_ge_0"] = (
        get_metric(summary_records, "nonoverlap", "delta_1.68_minus_1.5", "median_delta_ret_pct")
        >= 0.0
    )

    checks = {**core_checks, **relative_checks}

    core_pass_count = int(sum(1 for passed in core_checks.values() if passed))
    core_total = len(core_checks)
    relative_pass_count = int(sum(1 for passed in relative_checks.values() if passed))
    relative_total = len(relative_checks)

    pass_count = int(sum(1 for passed in checks.values() if passed))
    total = len(checks)

    if core_pass_count == core_total and relative_pass_count == relative_total:
        conclusion = "SUPPORTED_STRONG"
    elif core_pass_count == core_total:
        conclusion = "SUPPORTED_WITH_CAVEAT"
    elif core_pass_count >= core_total - 1:
        conclusion = "MOSTLY_SUPPORTED"
    else:
        conclusion = "NOT_SUPPORTED"

    return {
        "core_checks": core_checks,
        "relative_checks": relative_checks,
        "checks": checks,
        "pass_count": pass_count,
        "total": total,
        "core_pass_count": core_pass_count,
        "core_total": core_total,
        "relative_pass_count": relative_pass_count,
        "relative_total": relative_total,
        "conclusion": conclusion,
    }


def render_visual(evaluations: Dict[str, Dict[str, Optional[pd.DataFrame]]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: pass-rate comparison.
    ax = axes[0, 0]
    labels = ["overlap", "nonoverlap"]
    x = np.arange(len(labels))
    width = 0.35

    pass_150 = []
    pass_168 = []
    for label in labels:
        fixed = get_fixed_rows(evaluations[label]["summary"])
        pass_150.append(float(fixed.loc[1.5, "window_pass_rate_pct"]))
        pass_168.append(float(fixed.loc[1.68, "window_pass_rate_pct"]))

    ax.bar(x - width / 2, pass_150, width=width, label="1.50%", color="#1f77b4")
    ax.bar(x + width / 2, pass_168, width=width, label="1.68%", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Window pass rate %")
    ax.set_title("Gate Pass Rate")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25, axis="y")
    ax.legend()

    # Panel 2: median return comparison.
    ax = axes[0, 1]
    med_150 = []
    med_168 = []
    for label in labels:
        fixed = get_fixed_rows(evaluations[label]["summary"])
        med_150.append(float(fixed.loc[1.5, "median_test_return_pct"]))
        med_168.append(float(fixed.loc[1.68, "median_test_return_pct"]))

    ax.bar(x - width / 2, med_150, width=width, label="1.50%", color="#1f77b4")
    ax.bar(x + width / 2, med_168, width=width, label="1.68%", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median test return %")
    ax.set_title("Median OOS Return")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()

    # Panel 3: per-window return delta (1.68 - 1.5).
    ax = axes[1, 0]
    for label, color in [("overlap", "#2ca02c"), ("nonoverlap", "#d62728")]:
        windows = evaluations[label]["windows"]
        fixed = windows[(windows["sl_mode"] == "fixed") & (windows["sl_param"].isin(FIXED_PARAMS))].copy()
        pivot = fixed.pivot_table(index="window_id", columns="sl_param", values="test_total_return_pct")
        delta = pivot[1.68] - pivot[1.5]
        ax.plot(delta.index, delta.values, marker="o", label=label, color=color)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Per-Window Delta: 1.68% - 1.5%")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Delta test return %")
    ax.grid(alpha=0.25)
    ax.legend()

    # Panel 4: stress test return for selected 1.68.
    ax = axes[1, 1]
    scenarios = ["low", "base", "high"]
    x2 = np.arange(len(scenarios))

    for label, color in [("overlap", "#9467bd"), ("nonoverlap", "#8c564b")]:
        stress = evaluations[label]["stress"]
        vals = []
        if stress is None:
            vals = [np.nan] * len(scenarios)
        else:
            lookup = stress.set_index("scenario")["test_return_pct"]
            vals = [float(lookup.get(s, np.nan)) for s in scenarios]
        ax.plot(x2, vals, marker="o", linewidth=2, label=label, color=color)

    ax.set_xticks(x2)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Test return %")
    ax.set_title("Stress Cost Sensitivity (1.68%)")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle("Hypothesis Proof Dashboard: N=5 Fixed 1.68", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_report(
    summary_records: pd.DataFrame,
    verdict: Dict[str, object],
    output_path: Path,
    visual_path: Path,
) -> None:
    checks = verdict["checks"]
    core_checks = verdict["core_checks"]
    relative_checks = verdict["relative_checks"]

    def fmt(v: float) -> str:
        if pd.isna(v):
            return "nan"
        return f"{v:.2f}"

    o_pass = get_metric(summary_records, "overlap", "fixed_1.68", "window_pass_rate_pct")
    n_pass = get_metric(summary_records, "nonoverlap", "fixed_1.68", "window_pass_rate_pct")
    o_ret = get_metric(summary_records, "overlap", "fixed_1.68", "median_test_return_pct")
    n_ret = get_metric(summary_records, "nonoverlap", "fixed_1.68", "median_test_return_pct")
    n_worst = get_metric(summary_records, "nonoverlap", "fixed_1.68", "window_dist_worst_test_return_pct")

    d_ov = get_metric(summary_records, "overlap", "delta_1.68_minus_1.5", "median_test_return_pct")
    d_no = get_metric(summary_records, "nonoverlap", "delta_1.68_minus_1.5", "median_test_return_pct")
    beat_ov = get_metric(summary_records, "overlap", "delta_1.68_minus_1.5", "beat_rate_168_vs_150_pct")
    beat_no = get_metric(summary_records, "nonoverlap", "delta_1.68_minus_1.5", "beat_rate_168_vs_150_pct")

    high_stress_no = get_metric(summary_records, "nonoverlap", "stress_high", "test_return_pct")

    lines = [
        "# Hypothesis Proof Report (N=5, fixed 1.68)",
        "",
        "## Verdict",
        (
            f"- Conclusion: **{verdict['conclusion']}** "
            f"(core {verdict['core_pass_count']}/{verdict['core_total']}, "
            f"relative {verdict['relative_pass_count']}/{verdict['relative_total']}, "
            f"overall {verdict['pass_count']}/{verdict['total']})"
        ),
        "- Interpretation: this does not prove certainty, but indicates whether forward consistency is strong enough for deployment confidence.",
        "",
        "## Core Evidence",
        f"- Overlap (33 windows): pass rate {fmt(o_pass)}%, median test return {fmt(o_ret)}%.",
        f"- Non-overlap (11 windows): pass rate {fmt(n_pass)}%, median test return {fmt(n_ret)}%, worst window {fmt(n_worst)}%.",
        f"- 1.68 vs 1.5 median-return delta: overlap {fmt(d_ov)}%, non-overlap {fmt(d_no)}%.",
        f"- 1.68 beat-rate vs 1.5 by window: overlap {fmt(beat_ov)}%, non-overlap {fmt(beat_no)}%.",
        f"- Non-overlap high-cost stress test return: {fmt(high_stress_no)}%.",
        "- Caveat: the strategy can pass gates and still lose head-to-head in many windows; monitor this in live tracking.",
        "",
        "## Gate Checklist",
        "- Core checks:",
        f"- Selected in non-overlap objective: {core_checks['selected_nonoverlap_is_168']}",
        f"- Non-overlap pass rate >= 80%: {core_checks['pass_rate_nonoverlap_ge_80']}",
        f"- Non-overlap positive-window ratio >= 90%: {core_checks['positive_windows_nonoverlap_ge_90']}",
        f"- Non-overlap worst window return > 0%: {core_checks['worst_window_nonoverlap_positive']}",
        f"- Non-overlap high-cost stress return > 0%: {core_checks['stress_high_nonoverlap_positive']}",
        "- Relative checks vs 1.5:",
        f"- Beat-rate >= 50%: {relative_checks['beat_rate_nonoverlap_ge_50']}",
        f"- Median return delta >= 0%: {relative_checks['median_delta_nonoverlap_ge_0']}",
        "",
        "## Output Artifacts",
        f"- Summary CSV: {Path(output_path.parent, output_path.stem.replace('_report','_summary') + '.csv')}",
        f"- Dashboard: {visual_path}",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    evaluations = {
        "overlap": load_bundle(args.overlap_prefix),
        "nonoverlap": load_bundle(args.nonoverlap_prefix),
    }

    summary_records = build_summary_records(evaluations)

    output_prefix = Path(args.output_prefix)
    summary_csv = Path(f"{output_prefix}_summary.csv")
    report_md = Path(f"{output_prefix}_report.md")
    visual_png = Path(f"{output_prefix}_visual.png")

    summary_records.to_csv(summary_csv, index=False)

    verdict = build_verdict(summary_records)
    render_visual(evaluations, visual_png)
    write_markdown_report(summary_records, verdict, report_md, visual_png)

    print(f"saved_summary {summary_csv}")
    print(f"saved_report {report_md}")
    print(f"saved_visual {visual_png}")
    print(f"verdict {verdict['conclusion']} ({verdict['pass_count']}/{verdict['total']})")


if __name__ == "__main__":
    main()
