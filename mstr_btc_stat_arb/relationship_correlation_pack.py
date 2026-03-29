import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ols_alpha_beta(y, x):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx = x.mean()
    my = y.mean()
    sxx = ((x - mx) ** 2).sum()
    if sxx <= 0:
        return float(my), 0.0
    sxy = ((x - mx) * (y - my)).sum()
    b = sxy / sxx
    a = my - b * mx
    return float(a), float(b)


def rolling_beta_past_only(a_ret, b_ret, window):
    out = np.full(len(a_ret), np.nan)
    for i in range(window, len(a_ret)):
        _, b = ols_alpha_beta(a_ret[i - window : i], b_ret[i - window : i])
        out[i] = b
    return out


def rolling_z(spread, window):
    s = pd.Series(spread, dtype=float)
    return ((s - s.rolling(window).mean()) / s.rolling(window).std(ddof=1)).to_numpy()


def fit_ou_kappa_from_window(x):
    x = pd.Series(x).dropna().to_numpy(dtype=float)
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
    if 0 < b < 1:
        return float(-np.log(b))
    return np.nan


def rolling_ou_kappa(series, lookback):
    s = pd.Series(series, dtype=float).to_numpy()
    out = np.full(len(s), np.nan)
    for i in range(lookback, len(s)):
        out[i] = fit_ou_kappa_from_window(s[i - lookback : i])
    return out


def safe_corr(x, y, min_n=20):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < min_n:
        return np.nan, int(len(d))
    return float(d["x"].corr(d["y"])), int(len(d))


def build_dataset(base):
    cfg = json.loads((base / "oos20_regime_parameter_sweep_summary.json").read_text(encoding="utf-8"))["best"]
    beta_window = int(cfg["beta_window"])
    z_window = int(cfg["z_window"])

    s = pd.read_csv(base / "regime_portfolio_full_history_series.csv")
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").reset_index(drop=True)

    a_ret = s["a_ret"].to_numpy(dtype=float)
    b_ret = s["b_ret"].to_numpy(dtype=float)
    a_close = s["asset_close"].to_numpy(dtype=float)
    b_close = s["btc_close"].to_numpy(dtype=float)

    rb = rolling_beta_past_only(a_ret, b_ret, beta_window)

    a0, b0 = ols_alpha_beta(np.log(a_close), np.log(b_close))
    spread = np.log(a_close) - (a0 + b0 * np.log(b_close))
    z = rolling_z(spread, z_window)
    ou_kappa = rolling_ou_kappa(spread, 120)

    out = pd.DataFrame(
        {
            "date": s["date"],
            "vol_regime": s["vol_regime"].astype(str),
            "ret": s["ret_baseline"].astype(float),
            "rolling_beta": rb,
            "zscore": z,
            "ou_kappa": ou_kappa,
        }
    )

    for h in [1, 5]:
        fwd = pd.Series(out["ret"]).shift(-1)
        if h > 1:
            for i in range(2, h + 1):
                fwd = fwd + pd.Series(out["ret"]).shift(-i)
        out[f"fwd_ret_{h}d"] = fwd

    return out


def plot_heatmap_oos(corr_tbl, out_path):
    regs = ["high", "mid", "low"]
    feats = ["zscore", "rolling_beta", "ou_kappa"]

    mat = np.full((len(feats), len(regs)), np.nan)
    for i, f in enumerate(feats):
        for j, r in enumerate(regs):
            x = corr_tbl[(corr_tbl["feature"] == f) & (corr_tbl["regime"] == r)]
            if len(x):
                mat[i, j] = x["corr_fwd_5d"].iloc[0]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(regs)))
    ax.set_xticklabels(regs)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.set_title("OOS 2024+: Corr(feature, fwd 5d return)")

    for i in range(len(feats)):
        for j in range(len(regs)):
            v = mat[i, j]
            txt = "nan" if np.isnan(v) else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_z_scatter_oos(df_oos, out_path):
    regs = ["high", "mid", "low"]
    colors = {"high": "#d62728", "mid": "#1f77b4", "low": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharey=True)
    for ax, rg in zip(axes, regs):
        g = df_oos[df_oos["vol_regime"] == rg][["zscore", "fwd_ret_5d"]].dropna()
        ax.scatter(g["zscore"], g["fwd_ret_5d"], s=12, alpha=0.35, color=colors[rg])
        if len(g) >= 20:
            p = np.polyfit(g["zscore"].to_numpy(), g["fwd_ret_5d"].to_numpy(), 1)
            xs = np.linspace(g["zscore"].min(), g["zscore"].max(), 80)
            ys = p[0] * xs + p[1]
            ax.plot(xs, ys, color="black", linewidth=1.2)
        ax.axhline(0, color="#999", linewidth=0.8)
        ax.axvline(0, color="#999", linewidth=0.8)
        ax.set_title(f"{rg} regime")
        ax.set_xlabel("zscore")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("forward 5d return")
    fig.suptitle("OOS 2024+: zscore vs forward 5d return by regime", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_z_quantile_curve(df_oos, out_path):
    regs = ["high", "mid", "low"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    for rg, color in [("high", "#d62728"), ("mid", "#1f77b4"), ("low", "#2ca02c")]:
        g = df_oos[df_oos["vol_regime"] == rg][["zscore", "fwd_ret_5d"]].dropna()
        if len(g) < 30:
            continue
        g = g.copy()
        g["q"] = pd.qcut(g["zscore"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        q = g.groupby("q", observed=False)["fwd_ret_5d"].mean()
        x = q.index.astype(int).to_numpy()
        y = q.to_numpy()
        ax.plot(x, y, marker="o", linewidth=1.6, color=color, label=rg)

    ax.axhline(0, color="#999", linewidth=0.8)
    ax.set_title("OOS 2024+: Mean forward 5d return by zscore quintile")
    ax.set_xlabel("zscore quintile (1=lowest z, 5=highest z)")
    ax.set_ylabel("mean forward 5d return")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    base = Path(__file__).resolve().parent
    df = build_dataset(base)

    df_oos = df[df["date"] >= pd.Timestamp("2024-01-01")].copy()

    rows = []
    for sample_name, d in [("FULL", df), ("OOS_2024_PLUS", df_oos)]:
        for rg in ["high", "mid", "low"]:
            g = d[d["vol_regime"] == rg].copy()
            if len(g) == 0:
                continue
            c_z1, n_z1 = safe_corr(g["zscore"], g["fwd_ret_1d"])
            c_z5, n_z5 = safe_corr(g["zscore"], g["fwd_ret_5d"])
            c_b5, n_b5 = safe_corr(g["rolling_beta"], g["fwd_ret_5d"])
            c_o5, n_o5 = safe_corr(g["ou_kappa"], g["fwd_ret_5d"])
            rows.append(
                {
                    "sample": sample_name,
                    "regime": rg,
                    "corr_z_fwd_1d": c_z1,
                    "corr_z_fwd_5d": c_z5,
                    "corr_beta_fwd_5d": c_b5,
                    "corr_ou_fwd_5d": c_o5,
                    "n": int(len(g)),
                }
            )

    corr_table = pd.DataFrame(rows)
    corr_table.to_csv(base / "relationship_correlation_numbers.csv", index=False)

    # Long form for plotting heatmap
    heat_rows = []
    for _, r in corr_table[corr_table["sample"] == "OOS_2024_PLUS"].iterrows():
        heat_rows.append({"regime": r["regime"], "feature": "zscore", "corr_fwd_5d": r["corr_z_fwd_5d"]})
        heat_rows.append({"regime": r["regime"], "feature": "rolling_beta", "corr_fwd_5d": r["corr_beta_fwd_5d"]})
        heat_rows.append({"regime": r["regime"], "feature": "ou_kappa", "corr_fwd_5d": r["corr_ou_fwd_5d"]})
    heat_tbl = pd.DataFrame(heat_rows)

    plot_heatmap_oos(heat_tbl, base / "relationship_corr_heatmap_oos.png")
    plot_z_scatter_oos(df_oos, base / "relationship_zscore_scatter_oos.png")
    plot_z_quantile_curve(df_oos, base / "relationship_zscore_quantile_curve_oos.png")

    # Report
    lines = []
    lines.append("# Correlation And Relationship Pack")
    lines.append("")
    lines.append("Main question:")
    lines.append("- How much does zscore direction relate to future return, and how does this vary by regime?")
    lines.append("")
    lines.append("## OOS 2024+ Key Numbers")
    oos = corr_table[corr_table["sample"] == "OOS_2024_PLUS"].copy()
    for _, r in oos.iterrows():
        lines.append(
            f"- {r['regime']}: corr(z, fwd1d)={r['corr_z_fwd_1d']:.4f}, corr(z, fwd5d)={r['corr_z_fwd_5d']:.4f}, "
            f"corr(beta, fwd5d)={r['corr_beta_fwd_5d']:.4f}, corr(ou, fwd5d)={r['corr_ou_fwd_5d']:.4f}, n={int(r['n'])}"
        )

    lines.append("")
    lines.append("## Files")
    lines.append("- relationship_correlation_numbers.csv")
    lines.append("- relationship_corr_heatmap_oos.png")
    lines.append("- relationship_zscore_scatter_oos.png")
    lines.append("- relationship_zscore_quantile_curve_oos.png")

    (base / "relationship_correlation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "oos_2024_plus": oos.to_dict(orient="records"),
        "artifacts": {
            "numbers": "relationship_correlation_numbers.csv",
            "heatmap": "relationship_corr_heatmap_oos.png",
            "scatter": "relationship_zscore_scatter_oos.png",
            "quantile_curve": "relationship_zscore_quantile_curve_oos.png",
            "report": "relationship_correlation_report.md",
        },
    }
    (base / "relationship_correlation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
