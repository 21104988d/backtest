import json
from pathlib import Path

import numpy as np
import pandas as pd

import regime_conditional_rule_backtest as core


def metrics(ret):
    r = pd.Series(ret).fillna(0.0)
    if len(r) == 0:
        return {"ret": np.nan, "sharpe": np.nan, "mdd": np.nan, "n": 0, "active": np.nan}
    eq = (1.0 + r).cumprod()
    total = float(eq.iloc[-1] - 1.0)
    sd = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    sharpe = float((r.mean() / sd) * np.sqrt(365.0)) if (sd and not np.isnan(sd) and sd > 0) else np.nan
    mdd = float((eq / eq.cummax() - 1.0).min())
    active = float((r != 0.0).mean())
    return {"ret": total, "sharpe": sharpe, "mdd": mdd, "n": int(len(r)), "active": active}


def simulate_with_entry_filter(hedged_ret, z, z_entry, z_exit, fee_rate, stable_mask):
    n = len(hedged_ret)
    pos = np.zeros(n, dtype=int)
    cur = 0

    for i in range(n):
        zi = z[i]
        if np.isnan(zi):
            pos[i] = cur
            continue

        if cur == 0:
            if stable_mask[i] and zi >= z_entry:
                cur = -1
            elif stable_mask[i] and zi <= -z_entry:
                cur = 1
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


def kalman_beta_series(y, x, q, r, init_beta=1.0, init_p=1.0):
    n = len(y)
    beta_used = np.full(n, np.nan)
    beta_post = np.full(n, np.nan)
    p_prior_sqrt = np.full(n, np.nan)
    innov_z = np.full(n, np.nan)

    beta = float(init_beta)
    p = float(init_p)

    for i in range(n):
        # Prior state (available before observing y_i).
        beta_prior = beta
        p_prior = p + q

        beta_used[i] = beta_prior
        p_prior_sqrt[i] = np.sqrt(max(p_prior, 0.0))

        xi = float(x[i])
        yi = float(y[i])
        nu = yi - beta_prior * xi
        f = (xi * xi) * p_prior + r

        if f > 1e-12:
            k = (p_prior * xi) / f
            beta = beta_prior + k * nu
            p = (1.0 - k * xi) * p_prior
            innov_z[i] = abs(nu) / np.sqrt(f)
        else:
            beta = beta_prior
            p = p_prior
            innov_z[i] = np.nan

        p = max(p, 1e-12)
        beta_post[i] = beta

    beta_vel = np.abs(pd.Series(beta_used).diff().to_numpy(dtype=float))
    return {
        "beta_used": beta_used,
        "beta_post": beta_post,
        "beta_uncertainty": p_prior_sqrt,
        "innovation_z": innov_z,
        "beta_velocity": beta_vel,
    }


def stable_mask_from_quantile(beta_velocity, beta_uncertainty, innovation_z, train_mask, qtile):
    tr_vel = pd.Series(beta_velocity)[train_mask].dropna()
    tr_unc = pd.Series(beta_uncertainty)[train_mask].dropna()
    tr_inz = pd.Series(innovation_z)[train_mask].dropna()

    if len(tr_vel) < 30 or len(tr_unc) < 30 or len(tr_inz) < 30:
        return np.zeros(len(beta_velocity), dtype=bool), {"vel": np.nan, "unc": np.nan, "inz": np.nan}

    th_vel = float(np.quantile(tr_vel, qtile))
    th_unc = float(np.quantile(tr_unc, qtile))
    th_inz = float(np.quantile(tr_inz, qtile))

    stable = (
        (pd.Series(beta_velocity).to_numpy(dtype=float) <= th_vel)
        & (pd.Series(beta_uncertainty).to_numpy(dtype=float) <= th_unc)
        & (pd.Series(innovation_z).to_numpy(dtype=float) <= th_inz)
    )
    stable &= ~np.isnan(beta_velocity) & ~np.isnan(beta_uncertainty) & ~np.isnan(innovation_z)

    return stable, {"vel": th_vel, "unc": th_unc, "inz": th_inz}


def build_rows_for_sample(sample_name, mask, baseline_ret, kalman_ret, kalman_stable_ret):
    mb = metrics(pd.Series(baseline_ret)[mask])
    mk = metrics(pd.Series(kalman_ret)[mask])
    ms = metrics(pd.Series(kalman_stable_ret)[mask])
    return {
        "sample": sample_name,
        "baseline_ret": mb["ret"],
        "baseline_sharpe": mb["sharpe"],
        "baseline_mdd": mb["mdd"],
        "baseline_active": mb["active"],
        "kalman_ret": mk["ret"],
        "kalman_sharpe": mk["sharpe"],
        "kalman_mdd": mk["mdd"],
        "kalman_active": mk["active"],
        "kalman_stable_ret": ms["ret"],
        "kalman_stable_sharpe": ms["sharpe"],
        "kalman_stable_mdd": ms["mdd"],
        "kalman_stable_active": ms["active"],
        "delta_kalman_vs_base_ret": mk["ret"] - mb["ret"],
        "delta_kalman_vs_base_sharpe": (mk["sharpe"] if not np.isnan(mk["sharpe"]) else 0.0)
        - (mb["sharpe"] if not np.isnan(mb["sharpe"]) else 0.0),
        "delta_stable_vs_base_ret": ms["ret"] - mb["ret"],
        "delta_stable_vs_base_sharpe": (ms["sharpe"] if not np.isnan(ms["sharpe"]) else 0.0)
        - (mb["sharpe"] if not np.isnan(mb["sharpe"]) else 0.0),
    }


def main():
    base = Path(__file__).resolve().parent

    cfg = json.loads((base / "oos20_regime_parameter_sweep_summary.json").read_text(encoding="utf-8"))["best"]
    beta_window = int(cfg["beta_window"])
    z_window = int(cfg["z_window"])
    z_entry = float(cfg["z_entry"])
    z_exit = float(cfg["z_exit"])
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

    rb = core.rolling_beta_past_only(a_ret, b_ret, beta_window)
    hedged_roll = np.where(np.isnan(rb), 0.0, a_ret - rb * b_ret)

    a0, b0 = core.ols_alpha_beta(np.log(a_close), np.log(b_close))
    spread = np.log(a_close) - (a0 + b0 * np.log(b_close))
    z = core.rolling_z(spread, z_window)

    train_mask = s["date"] < train_cut

    beta_diff_train = pd.Series(rb).diff()[train_mask].dropna()
    resid_train = pd.Series(a_ret - pd.Series(rb).ffill().fillna(1.0).to_numpy() * b_ret)[train_mask].dropna()

    q_base = float(beta_diff_train.var(ddof=1)) if len(beta_diff_train) > 2 else 1e-4
    r_base = float(resid_train.var(ddof=1)) if len(resid_train) > 2 else 1e-4
    q_base = max(q_base, 1e-8)
    r_base = max(r_base, 1e-8)

    q_multipliers = [0.5, 1.0, 2.0, 4.0]
    r_multipliers = [0.5, 1.0, 2.0, 4.0]
    stable_quantiles = [0.60, 0.70, 0.80, 0.90]

    best = None
    best_score = -1e18

    for qm in q_multipliers:
        for rm in r_multipliers:
            q = q_base * qm
            r = r_base * rm
            k = kalman_beta_series(a_ret, b_ret, q=q, r=r, init_beta=1.0, init_p=1.0)

            hedged_kalman = a_ret - k["beta_used"] * b_ret
            _, ret_kalman = core.simulate_baseline(hedged_kalman, z, z_entry, z_exit, fee_rate)

            for sq in stable_quantiles:
                stable, th = stable_mask_from_quantile(
                    beta_velocity=k["beta_velocity"],
                    beta_uncertainty=k["beta_uncertainty"],
                    innovation_z=k["innovation_z"],
                    train_mask=train_mask,
                    qtile=sq,
                )
                _, ret_stable = simulate_with_entry_filter(hedged_kalman, z, z_entry, z_exit, fee_rate, stable)
                train_m = metrics(pd.Series(ret_stable)[train_mask])
                sh = train_m["sharpe"] if not np.isnan(train_m["sharpe"]) else -1e9
                rt = train_m["ret"] if not np.isnan(train_m["ret"]) else -1e9
                score = sh * 10.0 + rt
                if score > best_score:
                    best_score = score
                    best = {
                        "q": q,
                        "r": r,
                        "q_multiplier": qm,
                        "r_multiplier": rm,
                        "stable_quantile": sq,
                        "thresholds": th,
                        "kalman": k,
                        "ret_kalman": ret_kalman,
                        "ret_stable": ret_stable,
                        "stable": stable,
                    }

    if best is None:
        raise RuntimeError("Failed to find valid Kalman stability configuration")

    # Baseline with rolling beta from current production rule.
    pos_baseline, ret_baseline = core.simulate_baseline(hedged_roll, z, z_entry, z_exit, fee_rate)

    # Kalman versions with best tuned parameters.
    k = best["kalman"]
    hedged_kalman = a_ret - k["beta_used"] * b_ret
    pos_kalman, ret_kalman = core.simulate_baseline(hedged_kalman, z, z_entry, z_exit, fee_rate)
    pos_kalman_stable, ret_kalman_stable = simulate_with_entry_filter(
        hedged_kalman,
        z,
        z_entry,
        z_exit,
        fee_rate,
        best["stable"],
    )

    series = pd.DataFrame(
        {
            "date": s["date"],
            "vol_regime": s["vol_regime"],
            "zscore": z,
            "beta_roll": rb,
            "beta_kalman_used": k["beta_used"],
            "beta_kalman_post": k["beta_post"],
            "beta_uncertainty": k["beta_uncertainty"],
            "innovation_z": k["innovation_z"],
            "beta_velocity": k["beta_velocity"],
            "beta_stable": best["stable"].astype(int),
            "pos_baseline": pos_baseline,
            "ret_baseline": ret_baseline,
            "pos_kalman": pos_kalman,
            "ret_kalman": ret_kalman,
            "pos_kalman_stable": pos_kalman_stable,
            "ret_kalman_stable": ret_kalman_stable,
        }
    )

    masks = {
        "FULL": s["date"] >= pd.Timestamp("1900-01-01"),
        "IS_PRE_2024": s["date"] < pd.Timestamp("2024-01-01"),
        "IS_SPLIT_DATE": s["date"] < split_date,
        "OOS_2024_PLUS": s["date"] >= pd.Timestamp("2024-01-01"),
        "OOS_SPLIT_DATE": s["date"] >= split_date,
        "RECENT_2026": s["date"] >= pd.Timestamp("2026-01-01"),
    }

    result_rows = []
    for name, mask in masks.items():
        result_rows.append(
            build_rows_for_sample(
                sample_name=name,
                mask=mask,
                baseline_ret=ret_baseline,
                kalman_ret=ret_kalman,
                kalman_stable_ret=ret_kalman_stable,
            )
        )

    result_df = pd.DataFrame(result_rows)

    stable_rate_full = float(pd.Series(best["stable"]).mean())
    stable_rate_by_regime = {}
    for rg in ["high", "mid", "low", "unknown"]:
        m = s["vol_regime"].astype(str) == rg
        stable_rate_by_regime[rg] = float(pd.Series(best["stable"])[m].mean()) if m.any() else np.nan

    summary = {
        "method": "Kalman beta as stability filter",
        "config": {
            "beta_window_baseline": beta_window,
            "z_window": z_window,
            "z_entry": z_entry,
            "z_exit": z_exit,
            "fee_rate": fee_rate,
            "train_cut": str(train_cut.date()),
            "split_date": str(split_date.date()),
            "kalman_q": best["q"],
            "kalman_r": best["r"],
            "kalman_q_multiplier": best["q_multiplier"],
            "kalman_r_multiplier": best["r_multiplier"],
            "stable_quantile": best["stable_quantile"],
            "stability_thresholds": best["thresholds"],
        },
        "stable_rate_full": stable_rate_full,
        "stable_rate_by_regime": stable_rate_by_regime,
        "results": {r["sample"]: r for r in result_rows},
    }

    out_series = base / "kalman_beta_stability_filter_series.csv"
    out_results = base / "kalman_beta_stability_filter_results.csv"
    out_summary = base / "kalman_beta_stability_filter_summary.json"
    out_report = base / "kalman_beta_stability_filter_report.md"

    series.to_csv(out_series, index=False)
    result_df.to_csv(out_results, index=False)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Kalman Beta Stability Filter Evaluation",
        "",
        "Compared methods:",
        "- baseline: rolling-beta hedge + default z-score entries",
        "- kalman: Kalman beta hedge + default z-score entries",
        "- kalman_stable: Kalman beta hedge + only enter when beta stability condition passes",
        "",
        "Selected parameters (trained on IS_PRE_2024):",
        f"- kalman_q={best['q']:.8g}, kalman_r={best['r']:.8g}",
        f"- stability_quantile={best['stable_quantile']:.2f}",
        f"- thresholds: velocity<={best['thresholds']['vel']:.6g}, uncertainty<={best['thresholds']['unc']:.6g}, innovation_z<={best['thresholds']['inz']:.6g}",
        f"- stable rate full sample={stable_rate_full:.2%}",
        "",
    ]

    for row in result_rows:
        lines.append(
            f"- {row['sample']}: baseline(ret={row['baseline_ret']:.2%}, sh={row['baseline_sharpe']:.4f}, mdd={row['baseline_mdd']:.2%}) | "
            f"kalman(ret={row['kalman_ret']:.2%}, sh={row['kalman_sharpe']:.4f}, mdd={row['kalman_mdd']:.2%}) | "
            f"kalman_stable(ret={row['kalman_stable_ret']:.2%}, sh={row['kalman_stable_sharpe']:.4f}, mdd={row['kalman_stable_mdd']:.2%})"
        )

    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
