import json
from pathlib import Path

import numpy as np
import pandas as pd

import regime_conditional_rule_backtest as core


def block_metrics(series, mask):
    return core.metrics(pd.Series(series)[mask])


def main():
    base = Path(__file__).resolve().parent

    cfg = json.loads((base / "oos20_regime_parameter_sweep_summary.json").read_text(encoding="utf-8"))["best"]
    beta_window = int(cfg["beta_window"])
    z_window = int(cfg["z_window"])
    z_entry = float(cfg["z_entry"])
    z_exit = float(cfg["z_exit"])
    fee_rate = 0.00045

    split_date = pd.Timestamp(cfg["split_date"])

    # Best variant from regime rule sweep.
    direction_map = {"high": -1, "mid": 1, "low": -1}
    beta_stability_quantile = 1.0

    s = pd.read_csv(base / "regime_portfolio_full_history_series.csv")
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").reset_index(drop=True)

    a_ret = s["a_ret"].to_numpy(dtype=float)
    b_ret = s["b_ret"].to_numpy(dtype=float)
    a_close = s["asset_close"].to_numpy(dtype=float)
    b_close = s["btc_close"].to_numpy(dtype=float)
    vol_regime = s["vol_regime"].astype(str).to_numpy()

    rb = core.rolling_beta_past_only(a_ret, b_ret, beta_window)
    hedged_ret = np.where(np.isnan(rb), 0.0, a_ret - rb * b_ret)

    a0, b0 = core.ols_alpha_beta(np.log(a_close), np.log(b_close))
    spread = np.log(a_close) - (a0 + b0 * np.log(b_close))
    z = core.rolling_z(spread, z_window)

    delta_beta_abs = np.abs(pd.Series(rb).diff().to_numpy(dtype=float))
    th = float(np.nanquantile(pd.Series(delta_beta_abs).dropna(), beta_stability_quantile))

    pos_baseline, ret_baseline = core.simulate_baseline(hedged_ret, z, z_entry, z_exit, fee_rate)
    pos_best, ret_best = core.simulate_regime_conditional(
        hedged_ret,
        z,
        vol_regime,
        delta_beta_abs,
        z_entry,
        z_exit,
        fee_rate,
        direction_map,
        th,
    )

    out = pd.DataFrame(
        {
            "date": s["date"],
            "vol_regime": vol_regime,
            "zscore": z,
            "pos_baseline": pos_baseline,
            "ret_baseline": ret_baseline,
            "pos_best_variant": pos_best,
            "ret_best_variant": ret_best,
        }
    )

    summary = {
        "rule": {
            "beta_window": beta_window,
            "z_window": z_window,
            "z_entry": z_entry,
            "z_exit": z_exit,
            "fee_rate": fee_rate,
            "direction_map": direction_map,
            "beta_stability_quantile": beta_stability_quantile,
            "beta_stability_threshold": th,
        },
        "results": {
            "full": {
                "baseline": block_metrics(ret_baseline, s["date"] >= pd.Timestamp("1900-01-01")),
                "best_variant": block_metrics(ret_best, s["date"] >= pd.Timestamp("1900-01-01")),
            },
            "oos_2024_plus": {
                "baseline": block_metrics(ret_baseline, s["date"] >= pd.Timestamp("2024-01-01")),
                "best_variant": block_metrics(ret_best, s["date"] >= pd.Timestamp("2024-01-01")),
            },
            "oos_split_date": {
                "baseline": block_metrics(ret_baseline, s["date"] >= split_date),
                "best_variant": block_metrics(ret_best, s["date"] >= split_date),
            },
        },
    }

    (base / "best_variant_trading_rule_series.csv").write_text(out.to_csv(index=False), encoding="utf-8")
    (base / "best_variant_trading_rule_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
