import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


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


def sharpe_365(returns):
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0 or np.isnan(sd):
        return float("nan")
    return float((r.mean() / sd) * math.sqrt(365.0))


def max_drawdown(returns):
    r = pd.Series(returns).fillna(0.0)
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def total_return(returns):
    r = pd.Series(returns).fillna(0.0)
    if len(r) == 0:
        return float("nan")
    return float((1.0 + r).prod() - 1.0)


def fetch_yahoo_close(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = pd.DataFrame({"date": pd.to_datetime(df.index), "close": close.astype(float).to_numpy()})
    return out


def build_strategy_df(asset_close, btc_close, beta_window, z_window, z_entry, z_exit, fee_rate):
    df = (
        asset_close.merge(btc_close, on="date", how="inner", suffixes=("_asset", "_btc"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(df) < beta_window + z_window + 10:
        return pd.DataFrame()

    a_close = df["close_asset"].values
    b_close = df["close_btc"].values

    a_ret = (a_close[1:] / a_close[:-1]) - 1.0
    b_ret = (b_close[1:] / b_close[:-1]) - 1.0

    log_a = np.log(a_close)
    log_b = np.log(b_close)
    a0, b0 = ols_alpha_beta(log_a, log_b)
    spread = log_a - (a0 + b0 * log_b)

    rb = rolling_beta_past_only(a_ret, b_ret, beta_window)
    hedged_ret = np.zeros(len(a_ret))
    mask = ~np.isnan(rb)
    hedged_ret[mask] = a_ret[mask] - rb[mask] * b_ret[mask]

    z = np.full(len(df), np.nan)
    for i in range(z_window - 1, len(df)):
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

    gross = np.zeros(len(pos))
    for i in range(1, len(pos)):
        gross[i] = pos[i - 1] * hedged_ret[i]

    fee = np.zeros(len(pos))
    net = np.zeros(len(pos))
    for i in range(len(pos)):
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
            "hedged_ret": hedged_ret,
            "strategy_ret_net": net,
        }
    )
    return out


def assign_regimes_leak_safe(df, vol_window=30, quantile_lookback=252, min_history=126):
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["rolling_vol"] = out["hedged_ret"].rolling(vol_window).std(ddof=1)

    q33 = np.full(len(out), np.nan)
    q66 = np.full(len(out), np.nan)
    rv = out["rolling_vol"].values
    for i in range(len(out)):
        if i < min_history:
            continue
        s = max(0, i - quantile_lookback)
        hist = rv[s:i]
        hist = hist[~np.isnan(hist)]
        if len(hist) < min_history:
            continue
        q33[i] = float(np.quantile(hist, 0.33))
        q66[i] = float(np.quantile(hist, 0.66))

    reg = np.array(["unknown"] * len(out), dtype=object)
    high = (out["rolling_vol"] >= q66) & ~np.isnan(q66)
    low = (out["rolling_vol"] <= q33) & ~np.isnan(q33)
    mid = (~high) & (~low) & ~np.isnan(q33) & ~np.isnan(q66)
    reg[high] = "high"
    reg[low] = "low"
    reg[mid] = "mid"
    out["vol_regime"] = reg
    return out


def portfolio_returns(df, mode):
    base = df["strategy_ret_net"].fillna(0.0).to_numpy()
    reg = df["vol_regime"].to_numpy()

    if mode == "baseline":
        return base
    if mode == "high_only":
        return np.where(reg == "high", base, 0.0)
    if mode == "low_only":
        return np.where(reg == "low", base, 0.0)
    if mode == "high_low_only":
        return np.where((reg == "high") | (reg == "low"), base, 0.0)
    if mode == "regime_weighted":
        return np.where(reg == "high", base, np.where(reg == "mid", 0.5 * base, np.where(reg == "low", 0.2 * base, 0.0)))
    raise ValueError(f"unknown mode: {mode}")


def metric_pack(returns):
    return {
        "total_return": total_return(returns),
        "sharpe_365": sharpe_365(returns),
        "max_drawdown": max_drawdown(returns),
        "active_ratio": float((pd.Series(returns).fillna(0.0) != 0.0).mean()),
    }


def parse_int_list(v):
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def parse_float_list(v):
    return [float(x.strip()) for x in v.split(",") if x.strip()]


def run(args):
    base = Path(__file__).resolve().parent

    beta_windows = parse_int_list(args.beta_windows)
    z_entries = parse_float_list(args.z_entries)
    z_exits = parse_float_list(args.z_exits)
    vol_windows = parse_int_list(args.vol_windows)
    q_lookbacks = parse_int_list(args.quantile_lookbacks)
    min_histories = parse_int_list(args.min_histories)
    modes = [m.strip() for m in args.portfolio_modes.split(",") if m.strip()]

    asset = fetch_yahoo_close(args.asset, args.start, args.end)
    btc = fetch_yahoo_close(args.btc, args.start, args.end)
    if asset.empty or btc.empty:
        raise RuntimeError("Failed to fetch Yahoo data")

    rows = []

    for bw in beta_windows:
        for ze in z_entries:
            for zx in z_exits:
                if zx >= ze:
                    continue

                strat = build_strategy_df(
                    asset_close=asset,
                    btc_close=btc,
                    beta_window=bw,
                    z_window=args.z_window,
                    z_entry=ze,
                    z_exit=zx,
                    fee_rate=args.fee_rate,
                )
                if strat.empty:
                    continue

                split_idx = int(len(strat) * args.oos_ratio)
                split_idx = max(50, min(len(strat) - 50, split_idx))
                split_date = pd.Timestamp(strat.iloc[split_idx]["date"])

                for vw in vol_windows:
                    for qlb in q_lookbacks:
                        for mh in min_histories:
                            reg_df = assign_regimes_leak_safe(
                                strat,
                                vol_window=vw,
                                quantile_lookback=qlb,
                                min_history=mh,
                            )

                            is_mask = reg_df["date"] < split_date
                            oos_mask = reg_df["date"] >= split_date

                            for mode in modes:
                                is_ret = portfolio_returns(reg_df[is_mask], mode)
                                oos_ret = portfolio_returns(reg_df[oos_mask], mode)
                                full_ret = portfolio_returns(reg_df, mode)

                                is_m = metric_pack(is_ret)
                                oos_m = metric_pack(oos_ret)
                                full_m = metric_pack(full_ret)

                                rows.append(
                                    {
                                        "beta_window": bw,
                                        "z_entry": ze,
                                        "z_exit": zx,
                                        "z_window": args.z_window,
                                        "vol_window": vw,
                                        "quantile_lookback": qlb,
                                        "min_history": mh,
                                        "mode": mode,
                                        "oos_ratio": args.oos_ratio,
                                        "split_date": split_date.strftime("%Y-%m-%d"),
                                        "is_return": is_m["total_return"],
                                        "is_sharpe": is_m["sharpe_365"],
                                        "is_mdd": is_m["max_drawdown"],
                                        "oos_return": oos_m["total_return"],
                                        "oos_sharpe": oos_m["sharpe_365"],
                                        "oos_mdd": oos_m["max_drawdown"],
                                        "full_return": full_m["total_return"],
                                        "full_sharpe": full_m["sharpe_365"],
                                        "full_mdd": full_m["max_drawdown"],
                                        "oos_active_ratio": oos_m["active_ratio"],
                                    }
                                )

    if not rows:
        raise RuntimeError("No optimization results generated")

    df = pd.DataFrame(rows)
    df = df.sort_values(["oos_sharpe", "oos_return"], ascending=[False, False]).reset_index(drop=True)

    top_n = max(1, args.top_n)
    top = df.head(top_n)

    out_csv = base / "oos20_regime_parameter_sweep.csv"
    out_json = base / "oos20_regime_parameter_sweep_summary.json"
    out_md = base / "oos20_regime_parameter_sweep.md"

    df.to_csv(out_csv, index=False)

    summary = {
        "asset": args.asset,
        "btc": args.btc,
        "start": args.start,
        "end": args.end,
        "oos_ratio": args.oos_ratio,
        "top_n": top_n,
        "search_space": {
            "beta_windows": beta_windows,
            "z_entries": z_entries,
            "z_exits": z_exits,
            "vol_windows": vol_windows,
            "quantile_lookbacks": q_lookbacks,
            "min_histories": min_histories,
            "portfolio_modes": modes,
        },
        "best": top.iloc[0].to_dict(),
        "top": top.to_dict(orient="records"),
        "csv": out_csv.name,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# OOS 20% Regime Parameter Sweep")
    lines.append("")
    lines.append(f"- Period: {args.start} to {args.end}")
    lines.append(f"- Split: first {int(args.oos_ratio*100)}% IS, last {int((1-args.oos_ratio)*100)}% OOS (chronological)")
    lines.append(f"- Rows evaluated: {len(df)}")
    lines.append("")
    lines.append("## Top Configurations By OOS Sharpe")
    for i, r in top.iterrows():
        lines.append(
            f"{i+1}. mode={r['mode']}, bw={int(r['beta_window'])}, z_entry={r['z_entry']}, z_exit={r['z_exit']}, "
            f"vol_window={int(r['vol_window'])}, qlb={int(r['quantile_lookback'])}, min_hist={int(r['min_history'])} | "
            f"OOS Sharpe={r['oos_sharpe']:.4f}, OOS Return={r['oos_return']:.2%}, OOS MDD={r['oos_mdd']:.2%} | "
            f"IS Sharpe={r['is_sharpe']:.4f}, IS Return={r['is_return']:.2%}"
        )

    lines.append("")
    lines.append(f"- Full CSV: {out_csv.name}")
    lines.append(f"- JSON summary: {out_json.name}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="MSTR")
    parser.add_argument("--btc", default="BTC-USD")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-03-23")
    parser.add_argument("--oos-ratio", type=float, default=0.8)
    parser.add_argument("--z-window", type=int, default=30)
    parser.add_argument("--fee-rate", type=float, default=0.00045)
    parser.add_argument("--beta-windows", default="60,90,120")
    parser.add_argument("--z-entries", default="1.75,2.0,2.25")
    parser.add_argument("--z-exits", default="0.5,0.75,1.0")
    parser.add_argument("--vol-windows", default="20,30,40")
    parser.add_argument("--quantile-lookbacks", default="126,252")
    parser.add_argument("--min-histories", default="84,126")
    parser.add_argument("--portfolio-modes", default="baseline,high_only,low_only,high_low_only,regime_weighted")
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()
    run(args)
