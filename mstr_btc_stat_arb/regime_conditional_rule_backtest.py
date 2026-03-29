import json
from pathlib import Path

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
    rb = np.full(len(a_ret), np.nan)
    for i in range(window, len(a_ret)):
        _, beta = ols_alpha_beta(a_ret[i - window : i], b_ret[i - window : i])
        rb[i] = beta
    return rb


def rolling_z(spread, window):
    s = pd.Series(spread, dtype=float)
    m = s.rolling(window).mean()
    v = s.rolling(window).std(ddof=1)
    return ((s - m) / v).to_numpy()


def metrics(ret):
    r = pd.Series(ret).fillna(0.0)
    if len(r) == 0:
        return {"ret": np.nan, "sharpe": np.nan, "mdd": np.nan, "n": 0}
    eq = (1 + r).cumprod()
    total = float(eq.iloc[-1] - 1)
    sd = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    sharpe = float((r.mean() / sd) * np.sqrt(365)) if (sd and not np.isnan(sd) and sd > 0) else np.nan
    mdd = float((eq / eq.cummax() - 1).min())
    return {"ret": total, "sharpe": sharpe, "mdd": mdd, "n": int(len(r))}


def simulate_baseline(hedged_ret, z, z_entry, z_exit, fee_rate):
    n = len(hedged_ret)
    pos = np.zeros(n, dtype=int)
    cur = 0
    for i in range(n):
        zi = z[i]
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

    # Position on bar i earns return on bar i+1.
    gross = np.zeros(n)
    gross[1:] = pos[:-1] * hedged_ret[1:]

    turnover = np.zeros(n)
    turnover[1:] = np.abs(pos[1:] - pos[:-1])
    fee = turnover * fee_rate
    net = gross - fee
    return pos, net


def choose_regime_direction(df_train, z_entry):
    # direction map: +1 => momentum, -1 => contrarian, 0 => no-trade
    out = {}
    for rg in ["high", "mid", "low"]:
        d = df_train[df_train["vol_regime"] == rg].copy()
        d = d[d["signal_side"] != 0]
        if len(d) < 20:
            out[rg] = 0
            continue

        pnl_contrarian = (-d["signal_side"] * d["hedged_ret_next"]).mean()
        pnl_momentum = (d["signal_side"] * d["hedged_ret_next"]).mean()

        best = max(pnl_contrarian, pnl_momentum)
        if pd.isna(best) or best <= 0:
            out[rg] = 0
        elif pnl_momentum > pnl_contrarian:
            out[rg] = 1
        else:
            out[rg] = -1
    return out


def simulate_regime_conditional(
    hedged_ret,
    z,
    vol_regime,
    delta_beta_abs,
    z_entry,
    z_exit,
    fee_rate,
    dir_map,
    beta_stability_threshold,
):
    n = len(hedged_ret)
    pos = np.zeros(n, dtype=int)
    cur = 0

    for i in range(n):
        zi = z[i]
        rg = vol_regime[i]
        stable = (not np.isnan(delta_beta_abs[i])) and (delta_beta_abs[i] <= beta_stability_threshold)

        if np.isnan(zi):
            pos[i] = cur
            continue

        if cur == 0:
            if abs(zi) >= z_entry and stable:
                side = 1 if zi > 0 else (-1 if zi < 0 else 0)
                direction = dir_map.get(rg, 0)
                if direction != 0 and side != 0:
                    cur = direction * side
        else:
            if abs(zi) <= z_exit:
                cur = 0

        pos[i] = cur

    gross = np.zeros(n)
    gross[1:] = pos[:-1] * hedged_ret[1:]

    turnover = np.zeros(n)
    turnover[1:] = np.abs(pos[1:] - pos[:-1])
    fee = turnover * fee_rate
    net = gross - fee
    return pos, net


def main():
    base = Path(__file__).resolve().parent

    cfg = json.loads((base / "oos20_regime_parameter_sweep_summary.json").read_text(encoding="utf-8"))["best"]
    beta_window = int(cfg["beta_window"])
    z_entry = float(cfg["z_entry"])
    z_exit = float(cfg["z_exit"])
    z_window = int(cfg["z_window"])
    fee_rate = 0.00045

    split_date = pd.Timestamp(cfg["split_date"])
    train_cut = pd.Timestamp("2024-01-01")

    s = pd.read_csv(base / "regime_portfolio_full_history_series.csv")
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").reset_index(drop=True)

    a_ret = s["a_ret"].to_numpy(dtype=float)
    b_ret = s["b_ret"].to_numpy(dtype=float)
    a_close = s["asset_close"].to_numpy(dtype=float)
    b_close = s["btc_close"].to_numpy(dtype=float)
    reg = s["vol_regime"].astype(str).to_numpy()

    rb = rolling_beta_past_only(a_ret, b_ret, beta_window)
    hedged_ret = np.where(np.isnan(rb), 0.0, a_ret - rb * b_ret)

    log_a = np.log(a_close)
    log_b = np.log(b_close)
    a0, b0 = ols_alpha_beta(log_a, log_b)
    spread = log_a - (a0 + b0 * log_b)
    z = rolling_z(spread, z_window)

    delta_beta_abs = np.abs(pd.Series(rb).diff().to_numpy(dtype=float))

    work = pd.DataFrame(
        {
            "date": s["date"],
            "vol_regime": reg,
            "hedged_ret": hedged_ret,
            "zscore": z,
            "delta_beta_abs": delta_beta_abs,
        }
    )
    work["signal_side"] = np.where(work["zscore"] >= z_entry, 1, np.where(work["zscore"] <= -z_entry, -1, 0))
    work["hedged_ret_next"] = work["hedged_ret"].shift(-1)

    train = work[work["date"] < train_cut].copy()

    # Learn direction map once from train sample.
    dir_map = choose_regime_direction(train, z_entry)

    # Tune beta-stability threshold on train sample only.
    candidates = [0.6, 0.7, 0.8, 0.9, 1.0]
    best_q = None
    best_train_sharpe = -1e9
    best_train_ret = -1e9

    for q in candidates:
        th = np.nanquantile(train["delta_beta_abs"].dropna(), q)
        _, net = simulate_regime_conditional(
            hedged_ret,
            z,
            reg,
            delta_beta_abs,
            z_entry,
            z_exit,
            fee_rate,
            dir_map,
            th,
        )
        m = metrics(pd.Series(net)[s["date"] < train_cut])
        sh = m["sharpe"] if not np.isnan(m["sharpe"]) else -1e9
        rt = m["ret"] if not np.isnan(m["ret"]) else -1e9
        if (sh > best_train_sharpe) or (sh == best_train_sharpe and rt > best_train_ret):
            best_train_sharpe = sh
            best_train_ret = rt
            best_q = q

    best_th = np.nanquantile(train["delta_beta_abs"].dropna(), best_q)

    bpos, bret = simulate_baseline(hedged_ret, z, z_entry, z_exit, fee_rate)
    rpos, rret = simulate_regime_conditional(
        hedged_ret,
        z,
        reg,
        delta_beta_abs,
        z_entry,
        z_exit,
        fee_rate,
        dir_map,
        best_th,
    )

    out = pd.DataFrame(
        {
            "date": s["date"],
            "vol_regime": reg,
            "zscore": z,
            "delta_beta_abs": delta_beta_abs,
            "pos_baseline": bpos,
            "ret_baseline_rebuilt": bret,
            "pos_regime_rule": rpos,
            "ret_regime_rule": rret,
        }
    )

    def block(mask):
        a = metrics(out.loc[mask, "ret_baseline_rebuilt"])
        b = metrics(out.loc[mask, "ret_regime_rule"])
        return {
            "baseline": a,
            "regime_rule": b,
            "delta_ret": float(b["ret"] - a["ret"]),
            "delta_sharpe": float((b["sharpe"] if not np.isnan(b["sharpe"]) else 0.0) - (a["sharpe"] if not np.isnan(a["sharpe"]) else 0.0)),
        }

    summary = {
        "params": {
            "beta_window": beta_window,
            "z_window": z_window,
            "z_entry": z_entry,
            "z_exit": z_exit,
            "fee_rate": fee_rate,
            "split_date": str(split_date.date()),
            "train_cut": str(train_cut.date()),
        },
        "learned": {
            "direction_map": dir_map,
            "beta_stability_quantile": best_q,
            "beta_stability_threshold": float(best_th),
        },
        "results": {
            "full": block(out["date"] >= pd.Timestamp("1900-01-01")),
            "oos_2024_plus": block(out["date"] >= train_cut),
            "oos_split_date": block(out["date"] >= split_date),
        },
    }

    out_csv = base / "regime_conditional_rule_series.csv"
    out_json = base / "regime_conditional_rule_summary.json"
    out_md = base / "regime_conditional_rule_report.md"

    out.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Regime-Conditional Rule Backtest")
    lines.append("")
    lines.append("Learned rule:")
    lines.append(f"- Direction map: {dir_map}")
    lines.append(f"- Beta stability quantile (train only): {best_q}")
    lines.append(f"- Beta stability threshold: {best_th:.6f}")
    lines.append("")
    for k, v in summary["results"].items():
        lines.append(f"## {k}")
        lines.append(
            f"- Baseline: ret={v['baseline']['ret']:.2%}, sharpe={v['baseline']['sharpe']:.4f}, mdd={v['baseline']['mdd']:.2%}"
        )
        lines.append(
            f"- Regime rule: ret={v['regime_rule']['ret']:.2%}, sharpe={v['regime_rule']['sharpe']:.4f}, mdd={v['regime_rule']['mdd']:.2%}"
        )
        lines.append(f"- Delta: ret={v['delta_ret']:.2%}, sharpe={v['delta_sharpe']:.4f}")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
