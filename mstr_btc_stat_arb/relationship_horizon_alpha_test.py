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


def safe_corr(x, y, min_n=30):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < min_n:
        return np.nan, int(len(d))
    return float(d["x"].corr(d["y"])), int(len(d))


def build_df(base):
    cfg = json.loads((base / "oos20_regime_parameter_sweep_summary.json").read_text(encoding="utf-8"))["best"]
    s = pd.read_csv(base / "regime_portfolio_full_history_series.csv")
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").reset_index(drop=True)

    a_ret = s["a_ret"].to_numpy(dtype=float)
    b_ret = s["b_ret"].to_numpy(dtype=float)
    a_close = s["asset_close"].to_numpy(dtype=float)
    b_close = s["btc_close"].to_numpy(dtype=float)

    rb = rolling_beta_past_only(a_ret, b_ret, int(cfg["beta_window"]))

    a0, b0 = ols_alpha_beta(np.log(a_close), np.log(b_close))
    spread = np.log(a_close) - (a0 + b0 * np.log(b_close))
    z = rolling_z(spread, int(cfg["z_window"]))

    out = pd.DataFrame(
        {
            "date": s["date"],
            "vol_regime": s["vol_regime"].astype(str),
            "ret": s["ret_baseline"].astype(float),
            "zscore": z,
            "a_ret": s["a_ret"].astype(float),
            "b_ret": s["b_ret"].astype(float),
            "rolling_beta": rb,
        }
    )
    return out


def make_forward_returns(df, horizons):
    out = df.copy()
    for h in horizons:
        fwd = pd.Series(out["ret"]).shift(-1)
        if h > 1:
            for i in range(2, h + 1):
                fwd = fwd + pd.Series(out["ret"]).shift(-i)
        out[f"fwd_ret_{h}d"] = fwd
    return out


def plot_corr_vs_horizon(oos_tbl, out_path):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = {"high": "#d62728", "mid": "#1f77b4", "low": "#2ca02c"}

    for rg in ["high", "mid", "low"]:
        g = oos_tbl[oos_tbl["regime"] == rg].sort_values("horizon")
        if len(g) == 0:
            continue
        ax.plot(g["horizon"], g["corr_z_fwd"], marker="o", linewidth=1.8, label=rg, color=colors[rg])

    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_title("OOS 2024+: Corr(zscore, forward return) vs horizon")
    ax.set_xlabel("horizon (days)")
    ax.set_ylabel("correlation")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_r2_vs_horizon(oos_tbl, out_path):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = {"high": "#d62728", "mid": "#1f77b4", "low": "#2ca02c"}

    for rg in ["high", "mid", "low"]:
        g = oos_tbl[oos_tbl["regime"] == rg].sort_values("horizon")
        if len(g) == 0:
            continue
        ax.plot(g["horizon"], g["r2_z_fwd"], marker="o", linewidth=1.8, label=rg, color=colors[rg])

    ax.set_title("OOS 2024+: R2 from zscore-forward correlation vs horizon")
    ax.set_xlabel("horizon (days)")
    ax.set_ylabel("R2")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    base = Path(__file__).resolve().parent
    horizons = [1, 3, 5, 10, 15, 20]

    df = build_df(base)
    df = make_forward_returns(df, horizons)

    samples = {
        "FULL": df["date"] >= pd.Timestamp("1900-01-01"),
        "OOS_2024_PLUS": df["date"] >= pd.Timestamp("2024-01-01"),
    }

    rows = []
    for sname, smask in samples.items():
        d = df[smask].copy()
        for rg in ["high", "mid", "low"]:
            g = d[d["vol_regime"] == rg].copy()
            if len(g) == 0:
                continue
            for h in horizons:
                c, n = safe_corr(g["zscore"], g[f"fwd_ret_{h}d"])
                rows.append(
                    {
                        "sample": sname,
                        "regime": rg,
                        "horizon": h,
                        "corr_z_fwd": c,
                        "r2_z_fwd": float(c * c) if not np.isnan(c) else np.nan,
                        "n": n,
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(base / "relationship_horizon_correlation_numbers.csv", index=False)

    oos = out[out["sample"] == "OOS_2024_PLUS"].copy()
    plot_corr_vs_horizon(oos, base / "relationship_corr_vs_horizon_oos.png")
    plot_r2_vs_horizon(oos, base / "relationship_r2_vs_horizon_oos.png")

    # Compare raw MSTR-BTC return correlation for context
    rows2 = []
    d2 = df[df["date"] >= pd.Timestamp("2024-01-01")].copy()
    for rg in ["high", "mid", "low"]:
        g = d2[d2["vol_regime"] == rg]
        if len(g) == 0:
            continue
        rows2.append(
            {
                "regime": rg,
                "corr_mstr_btc_ret": float(g["a_ret"].corr(g["b_ret"])),
                "bars": int(len(g)),
            }
        )
    context_tbl = pd.DataFrame(rows2)
    context_tbl.to_csv(base / "relationship_horizon_context_mstr_btc_corr.csv", index=False)

    lines = []
    lines.append("# Horizon Alpha Test")
    lines.append("")
    lines.append("Question:")
    lines.append("- Is 5-day horizon too short, and does zscore->return relation strengthen at 10-20 days?")
    lines.append("")
    lines.append("## OOS 2024+ snapshots")
    for rg in ["high", "mid", "low"]:
        g = oos[oos["regime"] == rg].sort_values("horizon")
        if len(g) == 0:
            continue
        at5 = g[g["horizon"] == 5]
        at10 = g[g["horizon"] == 10]
        at20 = g[g["horizon"] == 20]
        c5 = float(at5["corr_z_fwd"].iloc[0]) if len(at5) else np.nan
        c10 = float(at10["corr_z_fwd"].iloc[0]) if len(at10) else np.nan
        c20 = float(at20["corr_z_fwd"].iloc[0]) if len(at20) else np.nan
        lines.append(f"- {rg}: corr@5d={c5:.4f}, corr@10d={c10:.4f}, corr@20d={c20:.4f}")

    lines.append("")
    lines.append("## Files")
    lines.append("- relationship_horizon_correlation_numbers.csv")
    lines.append("- relationship_corr_vs_horizon_oos.png")
    lines.append("- relationship_r2_vs_horizon_oos.png")
    lines.append("- relationship_horizon_context_mstr_btc_corr.csv")

    (base / "relationship_horizon_alpha_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "oos_2024_plus": oos.to_dict(orient="records"),
        "files": {
            "numbers": "relationship_horizon_correlation_numbers.csv",
            "corr_plot": "relationship_corr_vs_horizon_oos.png",
            "r2_plot": "relationship_r2_vs_horizon_oos.png",
            "context": "relationship_horizon_context_mstr_btc_corr.csv",
            "report": "relationship_horizon_alpha_report.md",
        },
    }
    (base / "relationship_horizon_alpha_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
