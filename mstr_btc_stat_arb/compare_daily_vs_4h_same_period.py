import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

import optimize_oos_20_regime as core


def load_4h_metrics(base: Path):
    p = base / "xyz:mstr_4h_rolling_beta_metrics.json"
    if not p.exists():
        raise RuntimeError("Missing 4H metrics file: xyz:mstr_4h_rolling_beta_metrics.json")
    return json.loads(p.read_text(encoding="utf-8"))


def run_daily_same_period(start_date: str, end_date: str):
    # Yahoo end date is exclusive; include the end day by adding one day.
    end_exclusive = (pd.Timestamp(end_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    # Add warmup history so rolling beta/z can initialize before evaluation start.
    warmup_start = (pd.Timestamp(start_date) - timedelta(days=220)).strftime("%Y-%m-%d")

    asset = core.fetch_yahoo_close("MSTR", warmup_start, end_exclusive)
    btc = core.fetch_yahoo_close("BTC-USD", warmup_start, end_exclusive)
    if asset.empty or btc.empty:
        raise RuntimeError("Failed to fetch Yahoo daily data for same-period comparison")

    # Use the same core strategy parameters as the 4H baseline run.
    strat = core.build_strategy_df(
        asset_close=asset,
        btc_close=btc,
        beta_window=60,
        z_window=30,
        z_entry=2.0,
        z_exit=0.5,
        fee_rate=0.00045,
    )
    if strat.empty:
        raise RuntimeError("Daily strategy dataframe is empty for the selected period")

    eval_mask = (strat["date"] >= pd.Timestamp(start_date)) & (strat["date"] <= pd.Timestamp(end_date))
    strat = strat[eval_mask].copy()
    if strat.empty:
        raise RuntimeError("No daily strategy rows remained in target same-period window")

    s = strat["strategy_ret_net"].fillna(0.0)
    eq = (1.0 + s).cumprod()
    dd = (eq / eq.cummax()) - 1.0

    sd = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    sharpe = float("nan")
    if sd and sd > 0:
        sharpe = float((s.mean() / sd) * (365.0**0.5))

    return {
        "source": "Yahoo daily",
        "start": str(pd.to_datetime(strat["date"].iloc[0]).date()),
        "end": str(pd.to_datetime(strat["date"].iloc[-1]).date()),
        "rows": int(len(strat)),
        "params": {
            "beta_window": 60,
            "z_window": 30,
            "z_entry": 2.0,
            "z_exit": 0.5,
            "fee_rate": 0.00045,
        },
        "strategy_total_return_net": float(eq.iloc[-1] - 1.0),
        "strategy_sharpe_365_net": sharpe,
        "max_drawdown_net": float(dd.min()) if len(dd) else float("nan"),
        "active_ratio": float((s != 0.0).mean()),
    }


def run_daily_same_period_alt(start_date: str, end_date: str):
    # Optional calibration pair from 4H z-threshold tuning for sensitivity check.
    end_exclusive = (pd.Timestamp(end_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    warmup_start = (pd.Timestamp(start_date) - timedelta(days=220)).strftime("%Y-%m-%d")

    asset = core.fetch_yahoo_close("MSTR", warmup_start, end_exclusive)
    btc = core.fetch_yahoo_close("BTC-USD", warmup_start, end_exclusive)
    if asset.empty or btc.empty:
        raise RuntimeError("Failed to fetch Yahoo daily data for same-period comparison")

    strat = core.build_strategy_df(
        asset_close=asset,
        btc_close=btc,
        beta_window=60,
        z_window=30,
        z_entry=1.75,
        z_exit=0.5,
        fee_rate=0.00045,
    )
    if strat.empty:
        raise RuntimeError("Daily alt strategy dataframe is empty for the selected period")

    eval_mask = (strat["date"] >= pd.Timestamp(start_date)) & (strat["date"] <= pd.Timestamp(end_date))
    strat = strat[eval_mask].copy()
    if strat.empty:
        raise RuntimeError("No daily alt strategy rows remained in target same-period window")

    s = strat["strategy_ret_net"].fillna(0.0)
    eq = (1.0 + s).cumprod()
    dd = (eq / eq.cummax()) - 1.0

    sd = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    sharpe = float("nan")
    if sd and sd > 0:
        sharpe = float((s.mean() / sd) * (365.0**0.5))

    return {
        "source": "Yahoo daily",
        "start": str(pd.to_datetime(strat["date"].iloc[0]).date()),
        "end": str(pd.to_datetime(strat["date"].iloc[-1]).date()),
        "rows": int(len(strat)),
        "params": {
            "beta_window": 60,
            "z_window": 30,
            "z_entry": 1.75,
            "z_exit": 0.5,
            "fee_rate": 0.00045,
        },
        "strategy_total_return_net": float(eq.iloc[-1] - 1.0),
        "strategy_sharpe_365_net": sharpe,
        "max_drawdown_net": float(dd.min()) if len(dd) else float("nan"),
        "active_ratio": float((s != 0.0).mean()),
    }


def main():
    base = Path(__file__).resolve().parent

    m4h = load_4h_metrics(base)
    start = pd.Timestamp(m4h["start_date"]).strftime("%Y-%m-%d")
    end = pd.Timestamp(m4h["end_date"]).strftime("%Y-%m-%d")

    daily_same = run_daily_same_period(start, end)
    daily_alt = run_daily_same_period_alt(start, end)

    comparison = {
        "period": {
            "start": start,
            "end": end,
            "same_as_4h_report": True,
        },
        "four_hour_reference": {
            "source": "Hyperliquid 4H",
            "rows": int(m4h.get("rows_aligned", 0)),
            "params": {
                "beta_window": int(m4h.get("rolling_beta_window", 60)),
                "z_window": 30,
                "z_entry": 2.0,
                "z_exit": 0.5,
                "fee_rate": float(m4h.get("taker_fee_rate", 0.00045)),
            },
            "strategy_total_return_net": float(m4h.get("strategy_total_return_net", float("nan"))),
            "strategy_sharpe_365_net": float(m4h.get("strategy_sharpe_365_net", float("nan"))),
            "max_drawdown_net": float(m4h.get("max_drawdown_net", float("nan"))),
        },
        "daily_same_params": daily_same,
        "daily_alt_params": daily_alt,
    }

    out_json = base / "daily_vs_4h_same_period_summary.json"
    out_md = base / "daily_vs_4h_same_period_report.md"
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    r4 = comparison["four_hour_reference"]
    d1 = comparison["daily_same_params"]
    d2 = comparison["daily_alt_params"]

    lines = []
    lines.append("# Daily vs 4H Same-Period Strategy Check")
    lines.append("")
    lines.append(f"- Period: {start} to {end}")
    lines.append("- Goal: test whether the same strategy behavior is unique to 4H")
    lines.append("")
    lines.append("## 4H Reference (Hyperliquid)")
    lines.append(
        f"- Net Return={r4['strategy_total_return_net']:.2%}, Sharpe={r4['strategy_sharpe_365_net']:.4f}, MDD={r4['max_drawdown_net']:.2%}"
    )
    lines.append(
        f"- Params: bw={r4['params']['beta_window']}, z_window={r4['params']['z_window']}, z_entry={r4['params']['z_entry']}, z_exit={r4['params']['z_exit']}, fee={r4['params']['fee_rate']}"
    )
    lines.append("")
    lines.append("## Daily (Yahoo) Same Params")
    lines.append(
        f"- Net Return={d1['strategy_total_return_net']:.2%}, Sharpe={d1['strategy_sharpe_365_net']:.4f}, MDD={d1['max_drawdown_net']:.2%}, Active={d1['active_ratio']:.2%}"
    )
    lines.append(
        f"- Params: bw={d1['params']['beta_window']}, z_window={d1['params']['z_window']}, z_entry={d1['params']['z_entry']}, z_exit={d1['params']['z_exit']}, fee={d1['params']['fee_rate']}"
    )
    lines.append("")
    lines.append("## Daily (Yahoo) Alt Thresholds")
    lines.append(
        f"- Net Return={d2['strategy_total_return_net']:.2%}, Sharpe={d2['strategy_sharpe_365_net']:.4f}, MDD={d2['max_drawdown_net']:.2%}, Active={d2['active_ratio']:.2%}"
    )
    lines.append(
        f"- Params: bw={d2['params']['beta_window']}, z_window={d2['params']['z_window']}, z_entry={d2['params']['z_entry']}, z_exit={d2['params']['z_exit']}, fee={d2['params']['fee_rate']}"
    )
    lines.append("")
    lines.append(f"- JSON: {out_json.name}")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
