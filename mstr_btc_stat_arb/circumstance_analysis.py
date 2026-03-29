import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


def mean(vals):
    return float(np.mean(vals)) if len(vals) else float("nan")


def std_sample(vals):
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals, ddof=1))


def corr(x, y):
    if len(x) < 2:
        return float("nan")
    c = np.corrcoef(x, y)
    return float(c[0, 1])


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
    out = [np.nan] * len(asset_ret)
    for i in range(window, len(asset_ret)):
        _, b = ols_alpha_beta(asset_ret[i - window : i], btc_ret[i - window : i])
        out[i] = b
    return np.asarray(out, dtype=float)


def fit_ou_ar1(series):
    if len(series) < 30:
        return np.nan, np.nan, np.nan, np.nan
    x0 = np.asarray(series[:-1], dtype=float)
    x1 = np.asarray(series[1:], dtype=float)
    mx0 = x0.mean()
    mx1 = x1.mean()
    sxx = ((x0 - mx0) ** 2).sum()
    if sxx <= 0:
        return np.nan, np.nan, np.nan, np.nan

    sxy = ((x0 - mx0) * (x1 - mx1)).sum()
    b = sxy / sxx
    a = mx1 - b * mx0
    resid = x1 - (a + b * x0)
    sigma = std_sample(resid)

    kappa = np.nan
    half_life = np.nan
    if 0.0 < b < 1.0:
        kappa = -math.log(b)
        if kappa > 0:
            half_life = math.log(2.0) / kappa
    return float(kappa), float(half_life), float(sigma), float(b)


def max_drawdown_from_returns(rets):
    eq = (1.0 + pd.Series(rets).fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def sharpe_365(rets):
    r = pd.Series(rets).dropna()
    if len(r) < 2:
        return float("nan")
    sigma = r.std(ddof=1)
    if sigma <= 0 or np.isnan(sigma):
        return float("nan")
    return float((r.mean() / sigma) * math.sqrt(365.0))


def compute_rolling_ou_metrics(series, lookback):
    kappa = np.full(len(series), np.nan)
    half_life = np.full(len(series), np.nan)
    sigma = np.full(len(series), np.nan)
    for i in range(lookback, len(series) + 1):
        win = series[i - lookback : i]
        k, h, s, _ = fit_ou_ar1(win)
        kappa[i - 1] = k
        half_life[i - 1] = h
        sigma[i - 1] = s
    return kappa, half_life, sigma


def load_hyperliquid_4h_signals(base):
    p = base / "xyz:mstr_4h_rolling_beta_signals.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["source"] = "hyperliquid_4h"
    return df


def fetch_yahoo_close(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    close = df["Close"]
    # yfinance can return a 2D Close block under some pandas/yfinance combos.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = pd.DataFrame({"date": pd.to_datetime(df.index), "close": close.astype(float).to_numpy()})
    return out


def build_yahoo_strategy_df(asset_close, btc_close, beta_window, z_window, z_entry, z_exit, fee_rate):
    a = asset_close.copy()
    b = btc_close.copy()
    df = a.merge(b, on="date", how="inner", suffixes=("_asset", "_btc")).sort_values("date").reset_index(drop=True)
    if len(df) < beta_window + z_window + 5:
        return pd.DataFrame()

    a_close = df["close_asset"].values
    b_close = df["close_btc"].values

    a_ret = np.zeros(len(df) - 1)
    b_ret = np.zeros(len(df) - 1)
    a_ret[:] = (a_close[1:] / a_close[:-1]) - 1.0
    b_ret[:] = (b_close[1:] / b_close[:-1]) - 1.0

    log_a = np.log(a_close)
    log_b = np.log(b_close)
    a0, b0 = ols_alpha_beta(log_a, log_b)
    spread = log_a - (a0 + b0 * log_b)

    rb = rolling_beta_past_only(a_ret, b_ret, beta_window)
    hedged_ret = np.zeros(len(a_ret))
    valid_beta = ~np.isnan(rb)
    hedged_ret[valid_beta] = a_ret[valid_beta] - rb[valid_beta] * b_ret[valid_beta]

    z = np.full(len(df), np.nan)
    for i in range(z_window - 1, len(df)):
        w = spread[i - z_window + 1 : i + 1]
        m = float(np.mean(w))
        s = float(np.std(w, ddof=1)) if len(w) > 1 else np.nan
        if s and not np.isnan(s) and s > 0:
            z[i] = (spread[i] - m) / s

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

    gross = np.zeros(len(z_ret))
    for i in range(1, len(z_ret)):
        gross[i] = pos[i - 1] * hedged_ret[i]

    fee = np.zeros(len(z_ret))
    net = np.zeros(len(z_ret))
    for i in range(len(z_ret)):
        turnover = 0.0
        if i >= 1:
            prev = pos[i - 2] if i >= 2 else 0
            curr = pos[i - 1]
            turnover = abs(curr - prev)
        fee[i] = turnover * fee_rate
        net[i] = gross[i] - fee[i]

    out = pd.DataFrame(
        {
            "date": df["date"].iloc[1:].values,
            "asset_close": a_close[1:],
            "btc_close": b_close[1:],
            "a_ret": a_ret,
            "b_ret": b_ret,
            "spread": spread[1:],
            "z": z_ret,
            "rolling_beta": rb,
            "position": pos,
            "hedged_ret": hedged_ret,
            "fee_cost": fee,
            "strategy_ret_net": net,
        }
    )
    out["source"] = "yahoo_daily"
    out["beta_window"] = beta_window
    out["z_entry"] = z_entry
    out["z_exit"] = z_exit
    return out


def add_features(df, ou_lookback=252, feature_window=30):
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["rolling_vol"] = out["hedged_ret"].rolling(feature_window).std(ddof=1)
    out["rolling_corr"] = out["a_ret"].rolling(feature_window).corr(out["b_ret"])

    eq = (1.0 + out["strategy_ret_net"].fillna(0.0)).cumprod()
    peak = eq.cummax()
    out["drawdown_depth"] = (eq / peak) - 1.0

    signz = np.sign(out["z"].fillna(0.0))
    crossings = (signz != signz.shift(1)).astype(float)
    out["signal_density"] = crossings.rolling(feature_window).sum()

    mr_series = out["hedged_ret"].fillna(0.0).cumsum().values
    kappa, half_life, ou_sigma = compute_rolling_ou_metrics(mr_series, ou_lookback)
    out["ou_kappa"] = kappa
    out["ou_half_life"] = half_life
    out["ou_sigma"] = ou_sigma

    return out


def regime_labels(df):
    out = df.copy()

    out["ou_regime"] = pd.cut(
        out["ou_half_life"],
        bins=[-np.inf, 20, 80, np.inf],
        labels=["fast", "medium", "slow"],
    )

    valid_vol = out["rolling_vol"].dropna()
    q1, q2 = (valid_vol.quantile(0.33), valid_vol.quantile(0.66)) if len(valid_vol) else (np.nan, np.nan)
    out["vol_regime"] = pd.cut(
        out["rolling_vol"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["low", "mid", "high"],
    )

    out["corr_regime"] = pd.cut(
        out["rolling_corr"],
        bins=[-np.inf, 0.5, 0.75, np.inf],
        labels=["low", "mid", "high"],
    )

    out["dd_regime"] = pd.cut(
        out["drawdown_depth"],
        bins=[-np.inf, -0.2, -0.05, np.inf],
        labels=["stress", "elevated", "calm"],
    )
    return out


def summarize_by_regime(df, group_cols, ret_col="strategy_ret_net"):
    grp = (
        df.dropna(subset=group_cols)
        .groupby(group_cols, dropna=True)[ret_col]
        .agg(["count", "mean", "sum", lambda s: (s > 0).mean()])
        .reset_index()
    )
    grp = grp.rename(columns={"mean": "avg_ret", "sum": "total_ret", "<lambda_0>": "hit_rate"})
    return grp


def load_fold_csv(base, horizon):
    p = base / f"wf_mstr_20240101_{horizon}_ou_sharpe_folds.csv"
    df = pd.read_csv(p)
    df["oos_start"] = pd.to_datetime(df["oos_start"])
    df["oos_end"] = pd.to_datetime(df["oos_end"])
    df["train_end"] = pd.to_datetime(df["train_end"])
    df["horizon"] = horizon
    return df


def apply_fold_guardrails(sim_df, fold_row, ou_min_hl, ou_max_hl, ou_min_kappa):
    oos = sim_df[(sim_df["date"] >= fold_row.oos_start) & (sim_df["date"] <= fold_row.oos_end)].copy()
    train = sim_df[sim_df["date"] <= fold_row.train_end].copy()

    if len(oos) == 0:
        return pd.DataFrame()

    vol_thr = train["rolling_vol"].quantile(0.75) if train["rolling_vol"].notna().any() else np.nan

    cond_ou = (
        (oos["ou_half_life"] >= ou_min_hl)
        & (oos["ou_half_life"] <= ou_max_hl)
        & (oos["ou_kappa"] >= ou_min_kappa)
    )
    cond_corr = oos["rolling_corr"] >= 0.6
    cond_vol = oos["rolling_vol"] <= vol_thr if not np.isnan(vol_thr) else pd.Series(False, index=oos.index)

    # Adaptive sizing components.
    hl_span = max(ou_max_hl - ou_min_hl, 1e-9)
    hl_score = ((oos["ou_half_life"] - ou_min_hl) / hl_span).clip(lower=0.0, upper=1.0).fillna(0.0)
    corr_score = ((oos["rolling_corr"] - 0.4) / 0.35).clip(lower=0.0, upper=1.0).fillna(0.0)
    if np.isnan(vol_thr):
        vol_score = pd.Series(1.0, index=oos.index)
    else:
        vol_score = (vol_thr / oos["rolling_vol"].replace(0.0, np.nan)).clip(lower=0.3, upper=1.5).fillna(0.3)

    oos["ret_baseline"] = oos["strategy_ret_net"]
    oos["ret_guard_ou"] = np.where(cond_ou, oos["strategy_ret_net"], 0.0)
    oos["ret_guard_ou_corr"] = np.where(cond_ou & cond_corr, oos["strategy_ret_net"], 0.0)
    oos["ret_guard_ou_corr_vol"] = np.where(cond_ou & cond_corr & cond_vol, oos["strategy_ret_net"], 0.0)

    # Soft sizing alternatives to reduce regime risk while preserving participation.
    oos["ret_size_ou_soft"] = oos["strategy_ret_net"] * np.where(cond_ou, 1.0, 0.35)
    oos["ret_size_ou_corr_soft"] = oos["strategy_ret_net"] * np.where(cond_ou & cond_corr, 1.0, 0.45)
    size_adaptive = 0.15 + 0.85 * (hl_score * corr_score)
    size_adaptive = size_adaptive * vol_score
    size_adaptive = size_adaptive.clip(lower=0.10, upper=1.10)
    oos["ret_size_adaptive"] = oos["strategy_ret_net"] * size_adaptive
    oos["guard_ou_active"] = cond_ou
    oos["guard_ou_corr_active"] = cond_ou & cond_corr
    oos["guard_ou_corr_vol_active"] = cond_ou & cond_corr & cond_vol
    oos["horizon"] = fold_row.horizon
    oos["fold"] = int(fold_row.fold)
    oos["vol_thr_train_q75"] = vol_thr
    return oos


def aggregate_guardrail_metrics(df, ret_col):
    r = df[ret_col].fillna(0.0)
    total_ret = float((1.0 + r).prod() - 1.0)
    sh = sharpe_365(r.values)
    mdd = max_drawdown_from_returns(r.values)
    return {"total_return": total_ret, "sharpe": sh, "max_drawdown": mdd}


def render_heatmap(pivot_df, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = pivot_df.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_xticklabels([str(c) for c in pivot_df.columns])
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_yticklabels([str(i) for i in pivot_df.index])
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def render_horizon_comparison(metrics_by_horizon, out_path):
    horizons = ["1m", "3m", "6m"]
    methods = [
        "ret_baseline",
        "ret_guard_ou",
        "ret_guard_ou_corr",
        "ret_guard_ou_corr_vol",
        "ret_size_ou_soft",
        "ret_size_ou_corr_soft",
        "ret_size_adaptive",
    ]
    labels = ["baseline", "ou", "ou+corr", "ou+corr+vol", "size_ou", "size_ou+corr", "size_adaptive"]

    vals = np.zeros((len(methods), len(horizons)))
    for i, m in enumerate(methods):
        for j, h in enumerate(horizons):
            vals[i, j] = metrics_by_horizon[h][m]["total_return"]

    x = np.arange(len(horizons))
    width = 0.11

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, lbl in enumerate(labels):
        ax.bar(x + (i - 3) * width, vals[i], width=width, label=lbl)

    ax.axhline(0.0, color="#333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_title("OOS Return By Horizon: Baseline vs Guardrails")
    ax.set_xlabel("OOS Horizon")
    ax.set_ylabel("Total OOS Return")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def render_equity_overlay(df_h, out_path, title):
    methods = [
        ("ret_baseline", "baseline"),
        ("ret_guard_ou", "ou"),
        ("ret_guard_ou_corr", "ou+corr"),
        ("ret_guard_ou_corr_vol", "ou+corr+vol"),
        ("ret_size_adaptive", "size_adaptive"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, lbl in methods:
        eq = (1.0 + df_h[col].fillna(0.0)).cumprod()
        ax.plot(df_h["date"], eq, label=lbl, linewidth=1.5)
    ax.axhline(1.0, color="#444", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run(args):
    base = Path(__file__).resolve().parent

    # Phase 1: feature engineering for Hyperliquid 4H
    h4_raw = load_hyperliquid_4h_signals(base)
    h4 = h4_raw.rename(
        columns={
            "asset_close": "asset_close",
            "btc_close": "btc_close",
            "strategy_ret_net": "strategy_ret_net",
        }
    )
    h4["a_ret"] = h4["asset_close"].pct_change().fillna(0.0)
    h4["b_ret"] = h4["btc_close"].pct_change().fillna(0.0)
    h4 = add_features(h4, ou_lookback=args.ou_lookback_bars, feature_window=args.feature_window)
    h4 = regime_labels(h4)

    # Phase 1: feature engineering for Yahoo daily using fold-selected params
    asset = fetch_yahoo_close(args.asset, args.start, args.end)
    btc = fetch_yahoo_close(args.btc, args.start, args.end)
    if asset.empty or btc.empty:
        raise RuntimeError("Failed to fetch Yahoo data for circumstance analysis")

    folds = {h: load_fold_csv(base, h) for h in ["1m", "3m", "6m"]}

    unique_params = set()
    for h, f in folds.items():
        for _, r in f.iterrows():
            unique_params.add((int(r.beta_window), float(r.z_entry), float(r.z_exit)))

    sim_cache = {}
    for bw, ze, zx in sorted(unique_params):
        sim = build_yahoo_strategy_df(
            asset_close=asset,
            btc_close=btc,
            beta_window=bw,
            z_window=args.z_window,
            z_entry=ze,
            z_exit=zx,
            fee_rate=args.fee_rate,
        )
        sim = add_features(sim, ou_lookback=args.ou_lookback_bars, feature_window=args.feature_window)
        sim = regime_labels(sim)
        sim_cache[(bw, ze, zx)] = sim

    # Phase 2/3/4: leakage-safe OOS attribution and guardrail tests by horizon
    oos_frames = []
    for h, f in folds.items():
        for _, fr in f.iterrows():
            key = (int(fr.beta_window), float(fr.z_entry), float(fr.z_exit))
            sim_df = sim_cache[key]
            oos = apply_fold_guardrails(
                sim_df=sim_df,
                fold_row=fr,
                ou_min_hl=args.ou_min_half_life,
                ou_max_hl=args.ou_max_half_life,
                ou_min_kappa=args.ou_min_kappa,
            )
            if not oos.empty:
                oos_frames.append(oos)

    if not oos_frames:
        raise RuntimeError("No OOS rows produced for analysis")

    yahoo_oos = pd.concat(oos_frames, ignore_index=True).sort_values(["horizon", "date"]).reset_index(drop=True)

    # Regime attributions
    yahoo_regime = summarize_by_regime(yahoo_oos, ["horizon", "ou_regime", "vol_regime"], ret_col="ret_baseline")
    h4_regime = summarize_by_regime(h4, ["ou_regime", "vol_regime"], ret_col="strategy_ret_net")

    # Guardrail metrics by horizon
    metrics_by_horizon = {}
    for h in ["1m", "3m", "6m"]:
        d = yahoo_oos[yahoo_oos["horizon"] == h].copy().sort_values("date")
        metrics_by_horizon[h] = {
            "ret_baseline": aggregate_guardrail_metrics(d, "ret_baseline"),
            "ret_guard_ou": aggregate_guardrail_metrics(d, "ret_guard_ou"),
            "ret_guard_ou_corr": aggregate_guardrail_metrics(d, "ret_guard_ou_corr"),
            "ret_guard_ou_corr_vol": aggregate_guardrail_metrics(d, "ret_guard_ou_corr_vol"),
            "ret_size_ou_soft": aggregate_guardrail_metrics(d, "ret_size_ou_soft"),
            "ret_size_ou_corr_soft": aggregate_guardrail_metrics(d, "ret_size_ou_corr_soft"),
            "ret_size_adaptive": aggregate_guardrail_metrics(d, "ret_size_adaptive"),
        }

    # Determine robust guardrails (improve in >=2/3 horizons on Sharpe or MDD)
    guard_names = [
        "ret_guard_ou",
        "ret_guard_ou_corr",
        "ret_guard_ou_corr_vol",
        "ret_size_ou_soft",
        "ret_size_ou_corr_soft",
        "ret_size_adaptive",
    ]
    robustness = []
    for g in guard_names:
        improved = 0
        details = []
        for h in ["1m", "3m", "6m"]:
            b = metrics_by_horizon[h]["ret_baseline"]
            m = metrics_by_horizon[h][g]
            better = False
            if (not np.isnan(m["sharpe"])) and (np.isnan(b["sharpe"]) or m["sharpe"] > b["sharpe"]):
                better = True
            if (not np.isnan(m["max_drawdown"])) and (np.isnan(b["max_drawdown"]) or m["max_drawdown"] > b["max_drawdown"]):
                better = True
            if better:
                improved += 1
            details.append({"horizon": h, "baseline": b, "guardrail": m, "better": better})
        robustness.append({"guardrail": g, "improved_horizons": improved, "details": details})

    # Phase 6: charts
    heat_in = yahoo_regime[yahoo_regime["horizon"] == "3m"].pivot_table(
        index="ou_regime", columns="vol_regime", values="avg_ret", aggfunc="mean"
    )
    heat_in = heat_in.reindex(index=["fast", "medium", "slow"], columns=["low", "mid", "high"])

    chart_heat = base / "circumstance_yahoo_3m_ou_vol_heatmap.png"
    chart_bar = base / "circumstance_oos_horizon_guardrail_returns.png"
    chart_eq_1m = base / "circumstance_equity_overlay_1m.png"
    chart_eq_3m = base / "circumstance_equity_overlay_3m.png"
    chart_eq_6m = base / "circumstance_equity_overlay_6m.png"

    render_heatmap(heat_in, chart_heat, "Yahoo OOS (3m): Avg Return by OU x Vol Regime")
    render_horizon_comparison(metrics_by_horizon, chart_bar)
    render_equity_overlay(
        yahoo_oos[yahoo_oos["horizon"] == "1m"].sort_values("date"),
        chart_eq_1m,
        "Yahoo OOS 1m: Baseline vs Guardrails",
    )
    render_equity_overlay(
        yahoo_oos[yahoo_oos["horizon"] == "3m"].sort_values("date"),
        chart_eq_3m,
        "Yahoo OOS 3m: Baseline vs Guardrails",
    )
    render_equity_overlay(
        yahoo_oos[yahoo_oos["horizon"] == "6m"].sort_values("date"),
        chart_eq_6m,
        "Yahoo OOS 6m: Baseline vs Guardrails",
    )

    # Outputs
    out_yahoo = base / "circumstance_yahoo_oos_diagnostics.csv"
    out_h4 = base / "circumstance_hyperliquid_4h_diagnostics.csv"
    out_yahoo_regime = base / "circumstance_yahoo_regime_summary.csv"
    out_h4_regime = base / "circumstance_hyperliquid_regime_summary.csv"

    yahoo_oos.to_csv(out_yahoo, index=False)
    h4.to_csv(out_h4, index=False)
    yahoo_regime.to_csv(out_yahoo_regime, index=False)
    h4_regime.to_csv(out_h4_regime, index=False)

    best_guard = max(robustness, key=lambda x: x["improved_horizons"])

    report = []
    report.append("# MSTR Strategy Circumstance Analysis")
    report.append("")
    report.append("## Scope")
    report.append("- Sources: Yahoo daily (walk-forward OOS) and Hyperliquid 4H.")
    report.append("- Objective: identify circumstances where strategy profits vs loses.")
    report.append("- Leakage control: all fold-level guardrail decisions use train-only information before OOS window.")
    report.append("")
    report.append("## Baseline Context")
    for h in ["1m", "3m", "6m"]:
        b = metrics_by_horizon[h]["ret_baseline"]
        report.append(
            f"- Yahoo OOS {h}: Return={b['total_return']:.2%}, Sharpe={b['sharpe']:.4f}, MaxDD={b['max_drawdown']:.2%}"
        )

    h4_ret = h4["strategy_ret_net"].fillna(0.0)
    report.append(
        f"- Hyperliquid 4H baseline: Return={(1+h4_ret).prod()-1:.2%}, Sharpe={sharpe_365(h4_ret.values):.4f}, MaxDD={max_drawdown_from_returns(h4_ret.values):.2%}"
    )
    report.append("")

    report.append("## Profitable Circumstances (Yahoo OOS)")
    top_rows = yahoo_regime.sort_values("avg_ret", ascending=False).head(10)
    for _, r in top_rows.iterrows():
        report.append(
            f"- horizon={r['horizon']}, ou={r['ou_regime']}, vol={r['vol_regime']}: avg_ret={r['avg_ret']:.5f}, hit_rate={r['hit_rate']:.2%}, count={int(r['count'])}"
        )

    report.append("")
    report.append("## Losing Circumstances (Yahoo OOS)")
    bot_rows = yahoo_regime.sort_values("avg_ret", ascending=True).head(10)
    for _, r in bot_rows.iterrows():
        report.append(
            f"- horizon={r['horizon']}, ou={r['ou_regime']}, vol={r['vol_regime']}: avg_ret={r['avg_ret']:.5f}, hit_rate={r['hit_rate']:.2%}, count={int(r['count'])}"
        )

    report.append("")
    report.append("## Guardrail Counterfactuals")
    for h in ["1m", "3m", "6m"]:
        report.append(f"- Horizon {h}:")
        for g in [
            "ret_guard_ou",
            "ret_guard_ou_corr",
            "ret_guard_ou_corr_vol",
            "ret_size_ou_soft",
            "ret_size_ou_corr_soft",
            "ret_size_adaptive",
        ]:
            m = metrics_by_horizon[h][g]
            report.append(
                f"  - {g}: Return={m['total_return']:.2%}, Sharpe={m['sharpe']:.4f}, MaxDD={m['max_drawdown']:.2%}"
            )

    report.append("")
    report.append("## Robustness Rule")
    report.append("- Acceptance criterion: improve at least 2 of 3 horizons on OOS Sharpe or OOS drawdown.")
    for r in robustness:
        report.append(f"- {r['guardrail']}: improved_horizons={r['improved_horizons']}/3")
    report.append(f"- Best candidate by criterion: {best_guard['guardrail']}")

    report.append("")
    report.append("## Artifacts")
    report.append(f"- Yahoo diagnostics: {out_yahoo.name}")
    report.append(f"- Hyperliquid diagnostics: {out_h4.name}")
    report.append(f"- Yahoo regime summary: {out_yahoo_regime.name}")
    report.append(f"- Hyperliquid regime summary: {out_h4_regime.name}")
    report.append(f"- Chart: {chart_heat.name}")
    report.append(f"- Chart: {chart_bar.name}")
    report.append(f"- Chart: {chart_eq_1m.name}")
    report.append(f"- Chart: {chart_eq_3m.name}")
    report.append(f"- Chart: {chart_eq_6m.name}")

    report_path = base / "circumstance_analysis_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    summary = {
        "report": report_path.name,
        "yahoo_diagnostics": out_yahoo.name,
        "h4_diagnostics": out_h4.name,
        "yahoo_regime_summary": out_yahoo_regime.name,
        "h4_regime_summary": out_h4_regime.name,
        "charts": [
            chart_heat.name,
            chart_bar.name,
            chart_eq_1m.name,
            chart_eq_3m.name,
            chart_eq_6m.name,
        ],
        "best_guardrail": best_guard,
        "metrics_by_horizon": metrics_by_horizon,
    }

    summary_path = base / "circumstance_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="MSTR")
    parser.add_argument("--btc", default="BTC-USD")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-03-23")
    parser.add_argument("--z-window", type=int, default=30)
    parser.add_argument("--fee-rate", type=float, default=0.00045)
    parser.add_argument("--feature-window", type=int, default=30)
    parser.add_argument("--ou-lookback-bars", type=int, default=252)
    parser.add_argument("--ou-min-half-life", type=float, default=4.0)
    parser.add_argument("--ou-max-half-life", type=float, default=90.0)
    parser.add_argument("--ou-min-kappa", type=float, default=0.005)
    args = parser.parse_args()
    run(args)
