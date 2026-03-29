import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_hl_daily(base: Path, filename: str, out_col: str):
    p = base / "data" / filename
    if not p.exists():
        raise RuntimeError(f"Missing file: {p}")
    df = pd.read_csv(p)
    if "time" not in df.columns or "close" not in df.columns:
        raise RuntimeError(f"Unexpected columns in {p}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["time"], utc=True, errors="coerce").dt.floor("D"),
            out_col: pd.to_numeric(df["close"], errors="coerce"),
        }
    ).dropna()
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out


def ols_alpha_beta(y, x):
    x = np.asarray(x)
    y = np.asarray(y)
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


def ridge_fit(X, y, alpha):
    if len(X) == 0:
        return None
    xtx = X.T @ X
    reg = np.eye(xtx.shape[0])
    reg[0, 0] = 0.0
    return np.linalg.pinv(xtx + alpha * reg) @ (X.T @ y)


def build_dynamic_beta(beta_true, min_train_bars=60, ridge_alpha=5.0, clip_low_q=0.02, clip_high_q=0.98):
    work = pd.DataFrame({"beta_true": beta_true})
    work["beta_l1"] = work["beta_true"].shift(1)
    work["beta_diff_5"] = work["beta_true"] - work["beta_true"].shift(5)
    work["beta_mean_20"] = work["beta_true"].rolling(20).mean()
    work["beta_std_20"] = work["beta_true"].rolling(20).std(ddof=1)
    work["time_idx"] = np.arange(len(work), dtype=float) / max(1, len(work) - 1)
    work["intercept"] = 1.0

    cols = ["intercept", "time_idx", "beta_l1", "beta_diff_5", "beta_mean_20", "beta_std_20"]
    pred = np.full(len(work), np.nan)

    for t in range(min_train_bars + 1, len(work)):
        y_idx = np.arange(1, t)
        x_idx = y_idx - 1
        X = work.loc[x_idx, cols].copy()
        y = work.loc[y_idx, "beta_true"].copy()
        tr = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True).rename("y")], axis=1).dropna()
        if len(tr) < min_train_bars:
            continue

        w = ridge_fit(tr[cols].to_numpy(dtype=float), tr["y"].to_numpy(dtype=float), alpha=ridge_alpha)
        if w is None:
            continue

        xp = work.loc[t - 1, cols]
        if xp.isna().any():
            continue
        p = float(np.dot(xp.to_numpy(dtype=float), w))
        lo = float(np.nanquantile(tr["y"].to_numpy(dtype=float), clip_low_q))
        hi = float(np.nanquantile(tr["y"].to_numpy(dtype=float), clip_high_q))
        pred[t] = min(max(p, lo), hi)

    return pred


def build_positions(spread, z_window, z_entry, z_exit):
    z = np.full(len(spread), np.nan)
    for i in range(z_window - 1, len(spread)):
        w = spread[i - z_window + 1 : i + 1]
        s = float(np.std(w, ddof=1)) if len(w) > 1 else np.nan
        if s and not np.isnan(s) and s > 0:
            z[i] = (spread[i] - float(np.mean(w))) / s

    z_ret = z[1:]
    pos = np.zeros(len(z_ret), dtype=int)
    cur = 0
    for i, zi in enumerate(z_ret):
        if np.isnan(zi):
            pos[i] = cur
            continue
        if cur == 0:
            if zi >= z_entry:
                cur = -1
            elif zi <= -z_entry:
                cur = 1
        else:
            if abs(zi) <= z_exit:
                cur = 0
        pos[i] = cur
    return pos


def build_net_returns(pos, hedged_ret, fee_rate):
    gross = np.zeros(len(pos), dtype=float)
    for i in range(1, len(pos)):
        gross[i] = pos[i - 1] * hedged_ret[i]

    net = np.zeros(len(pos), dtype=float)
    for i in range(len(pos)):
        turnover = 0.0
        if i >= 1:
            prev = pos[i - 2] if i >= 2 else 0
            curr = pos[i - 1]
            turnover = abs(curr - prev)
        net[i] = gross[i] - turnover * fee_rate
    return net


def metrics_from_returns(ret, bars_per_year=365.0):
    s = pd.Series(ret).fillna(0.0)
    if len(s) == 0:
        return {
            "total_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "active_ratio": float("nan"),
            "bars": 0,
        }
    eq = (1.0 + s).cumprod()
    dd = (eq / eq.cummax()) - 1.0
    sd = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    sh = float("nan")
    if sd and not math.isnan(sd) and sd > 0:
        sh = float((s.mean() / sd) * math.sqrt(bars_per_year))
    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "sharpe": sh,
        "max_drawdown": float(dd.min()) if len(dd) else float("nan"),
        "active_ratio": float((s != 0.0).mean()),
        "bars": int(len(s)),
    }


def run_portfolio(df, beta_window, z_entry, z_exit, use_ml_beta=False):
    a_close = df["mstr_close"].to_numpy()
    b_close = df["btc_close"].to_numpy()

    a_ret = (a_close[1:] / a_close[:-1]) - 1.0
    b_ret = (b_close[1:] / b_close[:-1]) - 1.0

    log_a = np.log(a_close)
    log_b = np.log(b_close)
    a0, b0 = ols_alpha_beta(log_a, log_b)
    spread = log_a - (a0 + b0 * log_b)

    rb = rolling_beta_past_only(a_ret, b_ret, beta_window)
    beta = rb.copy()
    if use_ml_beta:
        bpred = build_dynamic_beta(rb, min_train_bars=max(60, beta_window), ridge_alpha=5.0)
        beta = np.where(np.isnan(bpred), rb, bpred)

    hedged = np.where(np.isnan(beta), 0.0, a_ret - beta * b_ret)
    pos = build_positions(spread, z_window=30, z_entry=z_entry, z_exit=z_exit)
    ret = build_net_returns(pos, hedged, fee_rate=0.00045)

    out = pd.DataFrame({
        "date": df["date"].iloc[1:].values,
        "ret": ret,
        "position": pos,
    })
    return out, metrics_from_returns(ret)


def main():
    base = Path(__file__).resolve().parent.parent
    out_dir = Path(__file__).resolve().parent

    mstr = load_hl_daily(base, "xyz_mstr_1d.csv", "mstr_close")
    btc = load_hl_daily(base, "btc_1d.csv", "btc_close")
    df = mstr.merge(btc, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # Portfolio 1: Original HL daily strategy params.
    p1_series, p1_m = run_portfolio(df, beta_window=60, z_entry=2.0, z_exit=0.5, use_ml_beta=False)

    # Portfolio 2: Yahoo-derived params, traded on HL daily.
    p2_series, p2_m = run_portfolio(df, beta_window=90, z_entry=1.75, z_exit=1.0, use_ml_beta=False)

    # Portfolio 3: Yahoo-derived params + ML beta on HL daily.
    p3_series, p3_m = run_portfolio(df, beta_window=90, z_entry=1.75, z_exit=1.0, use_ml_beta=True)

    all_df = p1_series[["date"]].copy()
    all_df = all_df.merge(p1_series[["date", "ret"]].rename(columns={"ret": "ret_p1_hl_original"}), on="date", how="left")
    all_df = all_df.merge(p2_series[["date", "ret"]].rename(columns={"ret": "ret_p2_yahoo_params"}), on="date", how="left")
    all_df = all_df.merge(p3_series[["date", "ret"]].rename(columns={"ret": "ret_p3_yahoo_params_mlbeta"}), on="date", how="left")

    for c in ["ret_p1_hl_original", "ret_p2_yahoo_params", "ret_p3_yahoo_params_mlbeta"]:
        all_df[c] = all_df[c].fillna(0.0)

    def eq(s):
        return (1.0 + s).cumprod()

    all_df["eq_p1_hl_original"] = eq(all_df["ret_p1_hl_original"])
    all_df["eq_p2_yahoo_params"] = eq(all_df["ret_p2_yahoo_params"])
    all_df["eq_p3_yahoo_params_mlbeta"] = eq(all_df["ret_p3_yahoo_params_mlbeta"])

    summary = {
        "period": {
            "start": str(pd.to_datetime(all_df["date"].iloc[0]).date()),
            "end": str(pd.to_datetime(all_df["date"].iloc[-1]).date()),
            "bars": int(len(all_df)),
            "data_source": "Hyperliquid daily cache",
        },
        "portfolio_1_hl_original": {
            "params": {"beta_window": 60, "z_entry": 2.0, "z_exit": 0.5, "ml_beta": False},
            "metrics": p1_m,
        },
        "portfolio_2_yahoo_params_on_hl": {
            "params": {"beta_window": 90, "z_entry": 1.75, "z_exit": 1.0, "ml_beta": False},
            "metrics": p2_m,
        },
        "portfolio_3_yahoo_params_ml_beta_on_hl": {
            "params": {"beta_window": 90, "z_entry": 1.75, "z_exit": 1.0, "ml_beta": True},
            "metrics": p3_m,
        },
    }

    out_json = out_dir / "hl_daily_3portfolio_comparison_summary.json"
    out_csv = out_dir / "hl_daily_3portfolio_comparison_series.csv"
    out_md = out_dir / "hl_daily_3portfolio_comparison_report.md"
    out_png = out_dir / "hl_daily_3portfolio_comparison_equity.png"

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    all_df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(all_df["date"], all_df["eq_p1_hl_original"], label="P1 HL original", color="#1f77b4", linewidth=1.8)
    ax.plot(all_df["date"], all_df["eq_p2_yahoo_params"], label="P2 Yahoo params on HL", color="#ff7f0e", linewidth=1.8)
    ax.plot(all_df["date"], all_df["eq_p3_yahoo_params_mlbeta"], label="P3 Yahoo params + ML beta on HL", color="#2ca02c", linewidth=1.8)
    ax.axhline(1.0, color="#333", linewidth=0.8)
    ax.set_title("Hyperliquid Daily: 3-Portfolio Equity Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    def line(name, m):
        return f"- {name}: Return={m['total_return']:.2%}, Sharpe={m['sharpe']:.4f}, MDD={m['max_drawdown']:.2%}, Active={m['active_ratio']:.2%}, Bars={m['bars']}"

    lines = []
    lines.append("# Hyperliquid Daily 3-Portfolio Comparison")
    lines.append("")
    lines.append(f"- Period: {summary['period']['start']} to {summary['period']['end']}")
    lines.append("- Portfolios:")
    lines.append("  1) Original HL strategy params")
    lines.append("  2) Yahoo-derived params traded on HL")
    lines.append("  3) Yahoo-derived params + ML beta traded on HL")
    lines.append("")
    lines.append("## Results")
    lines.append(line("P1 HL original", p1_m))
    lines.append(line("P2 Yahoo params on HL", p2_m))
    lines.append(line("P3 Yahoo params + ML beta on HL", p3_m))
    lines.append("")
    lines.append(f"- Series CSV: {out_csv.name}")
    lines.append(f"- Equity chart: {out_png.name}")
    lines.append(f"- JSON: {out_json.name}")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
