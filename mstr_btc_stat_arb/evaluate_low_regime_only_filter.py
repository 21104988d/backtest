import json
from pathlib import Path

import numpy as np
import pandas as pd

import regime_conditional_rule_backtest as core


def metrics(ret):
    r = pd.Series(ret).fillna(0.0)
    if len(r) == 0:
        return {"ret": np.nan, "sharpe": np.nan, "mdd": np.nan, "n": 0, "active": np.nan}
    eq = (1 + r).cumprod()
    total = float(eq.iloc[-1] - 1)
    sd = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    sharpe = float((r.mean() / sd) * np.sqrt(365)) if (sd and not np.isnan(sd) and sd > 0) else np.nan
    mdd = float((eq / eq.cummax() - 1).min())
    active = float((r != 0).mean())
    return {"ret": total, "sharpe": sharpe, "mdd": mdd, "n": int(len(r)), "active": active}


def simulate_low_only_strict(hedged_ret, z, vol_regime, z_entry, z_exit, fee_rate):
    n = len(hedged_ret)
    pos = np.zeros(n, dtype=int)
    cur = 0

    for i in range(n):
        zi = z[i]
        rg = vol_regime[i]

        # strict filter: no position allowed outside low regime
        if rg != "low":
            cur = 0
            pos[i] = 0
            continue

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

    s = pd.read_csv(base / "regime_portfolio_full_history_series.csv")
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").reset_index(drop=True)

    beta_window = int(cfg["beta_window"])
    z_window = int(cfg["z_window"])
    z_entry = float(cfg["z_entry"])
    z_exit = float(cfg["z_exit"])
    fee_rate = 0.00045
    split_date = pd.Timestamp(cfg["split_date"])

    a_ret = s["a_ret"].to_numpy(dtype=float)
    b_ret = s["b_ret"].to_numpy(dtype=float)
    a_close = s["asset_close"].to_numpy(dtype=float)
    b_close = s["btc_close"].to_numpy(dtype=float)
    reg = s["vol_regime"].astype(str).to_numpy()

    rb = core.rolling_beta_past_only(a_ret, b_ret, beta_window)
    hedged_ret = np.where(np.isnan(rb), 0.0, a_ret - rb * b_ret)
    a0, b0 = core.ols_alpha_beta(np.log(a_close), np.log(b_close))
    spread = np.log(a_close) - (a0 + b0 * np.log(b_close))
    z = core.rolling_z(spread, z_window)

    p_base, r_base = core.simulate_baseline(hedged_ret, z, z_entry, z_exit, fee_rate)
    p_low, r_low = simulate_low_only_strict(hedged_ret, z, reg, z_entry, z_exit, fee_rate)

    out = pd.DataFrame(
        {
            "date": s["date"],
            "vol_regime": reg,
            "zscore": z,
            "pos_baseline": p_base,
            "ret_baseline": r_base,
            "pos_low_only_strict": p_low,
            "ret_low_only_strict": r_low,
        }
    )
    out.to_csv(base / "low_only_strict_filter_series.csv", index=False)

    masks = {
        "FULL": s["date"] >= pd.Timestamp("1900-01-01"),
        "IS_PRE_2024": s["date"] < pd.Timestamp("2024-01-01"),
        "IS_SPLIT_DATE": s["date"] < split_date,
        "OOS_2024_PLUS": s["date"] >= pd.Timestamp("2024-01-01"),
        "OOS_SPLIT_DATE": s["date"] >= split_date,
    }

    rows = []
    summary = {"rule": "Trade only in low regime; force flat in non-low regimes", "results": {}}

    for name, mask in masks.items():
        mb = metrics(pd.Series(r_base)[mask])
        ml = metrics(pd.Series(r_low)[mask])
        summary["results"][name] = {
            "baseline": mb,
            "low_only_strict": ml,
            "delta_ret": ml["ret"] - mb["ret"],
            "delta_sharpe": (ml["sharpe"] if not np.isnan(ml["sharpe"]) else 0.0)
            - (mb["sharpe"] if not np.isnan(mb["sharpe"]) else 0.0),
        }

        rows.append(
            {
                "sample": name,
                "baseline_ret": mb["ret"],
                "baseline_sharpe": mb["sharpe"],
                "baseline_mdd": mb["mdd"],
                "baseline_active": mb["active"],
                "low_only_ret": ml["ret"],
                "low_only_sharpe": ml["sharpe"],
                "low_only_mdd": ml["mdd"],
                "low_only_active": ml["active"],
                "delta_ret": ml["ret"] - mb["ret"],
                "delta_sharpe": (ml["sharpe"] if not np.isnan(ml["sharpe"]) else 0.0)
                - (mb["sharpe"] if not np.isnan(mb["sharpe"]) else 0.0),
            }
        )

    pd.DataFrame(rows).to_csv(base / "low_only_strict_filter_results.csv", index=False)
    (base / "low_only_strict_filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Low-Regime-Only Strict Filter Results",
        "",
        "Rule:",
        "- Trade only when vol_regime == low",
        "- Force flat position for high/mid/unknown bars",
        "",
    ]
    for r in rows:
        lines.append(
            f"- {r['sample']}: baseline(ret={r['baseline_ret']:.2%}, sh={r['baseline_sharpe']:.4f}, mdd={r['baseline_mdd']:.2%}) | "
            f"low_only(ret={r['low_only_ret']:.2%}, sh={r['low_only_sharpe']:.4f}, mdd={r['low_only_mdd']:.2%}) | "
            f"delta(ret={r['delta_ret']:.2%}, sh={r['delta_sharpe']:.4f})"
        )
    (base / "low_only_strict_filter_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
