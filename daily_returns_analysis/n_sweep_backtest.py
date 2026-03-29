"""Leakage-aware N sweep for mean-reversion daily strategy.

This script evaluates N from config.N_SWEEP_VALUES using:
1) Train window: all days except trailing validation window.
2) Validation window: trailing config.VALIDATION_DAYS.

Selection should focus on validation robustness, not train-only best return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from config import (
    DAILY_OHLC_FILE,
    DYNAMIC_SL_MULTIPLIER,
    INITIAL_CAPITAL,
    MAX_SL_PCT,
    MIN_ASSETS_PER_DAY,
    MIN_SL_PCT,
    N_SWEEP_VALUES,
    POSITION_SIZE_FIXED,
    ROUND_TRIP_FEE_PCT,
    VALIDATION_DAYS,
)


@dataclass
class SweepMetrics:
    n: int
    period: str
    trading_days: int
    total_return_pct: float
    annualized_return_pct: float
    win_rate_pct: float
    sharpe: float
    max_drawdown_pct: float


def calculate_position_return(row: pd.Series, is_long: bool, sl_pct: float) -> float | None:
    open_price = float(row["open"])
    high_price = float(row["high"])
    low_price = float(row["low"])
    close_price = float(row["close"])

    if is_long:
        stop_loss_price = open_price * (1 - sl_pct / 100)
        if low_price <= stop_loss_price:
            return -sl_pct
        return (close_price / open_price - 1) * 100

    stop_loss_price = open_price * (1 + sl_pct / 100)
    if high_price >= stop_loss_price:
        return -sl_pct
    return -(close_price / open_price - 1) * 100


def run_backtest(ohlc_df: pd.DataFrame, all_dates: Sequence, active_trade_dates: set, n: int) -> Dict[str, float]:
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    pnl_history: List[float] = []
    win_days = 0

    for idx in range(1, len(all_dates)):
        signal_date = all_dates[idx - 1]
        trade_date = all_dates[idx]
        if signal_date not in active_trade_dates or trade_date not in active_trade_dates:
            continue

        signal_data = ohlc_df[ohlc_df["date"] == signal_date]
        required_assets = max(MIN_ASSETS_PER_DAY, n * 2)
        if len(signal_data) < required_assets:
            continue

        top_n = signal_data.nlargest(n, "daily_return")[["coin", "daily_return"]].values.tolist()
        bottom_n = signal_data.nsmallest(n, "daily_return")[["coin", "daily_return"]].values.tolist()

        trade_data = ohlc_df[ohlc_df["date"] == trade_date]

        daily_pnl = 0.0

        for coin, prev_ret in top_n:
            row = trade_data[trade_data["coin"] == coin]
            if len(row) == 0:
                continue
            sl_pct = max(MIN_SL_PCT, min(MAX_SL_PCT, abs(prev_ret) * DYNAMIC_SL_MULTIPLIER))
            ret_pct = calculate_position_return(row.iloc[0], is_long=False, sl_pct=sl_pct)
            if ret_pct is None:
                continue
            daily_pnl += ((ret_pct - ROUND_TRIP_FEE_PCT) / 100) * POSITION_SIZE_FIXED

        for coin, prev_ret in bottom_n:
            row = trade_data[trade_data["coin"] == coin]
            if len(row) == 0:
                continue
            sl_pct = max(MIN_SL_PCT, min(MAX_SL_PCT, abs(prev_ret) * DYNAMIC_SL_MULTIPLIER))
            ret_pct = calculate_position_return(row.iloc[0], is_long=True, sl_pct=sl_pct)
            if ret_pct is None:
                continue
            daily_pnl += ((ret_pct - ROUND_TRIP_FEE_PCT) / 100) * POSITION_SIZE_FIXED

        pnl_history.append(daily_pnl)
        if daily_pnl > 0:
            win_days += 1

        equity += daily_pnl
        equity_curve.append(equity)

    if not pnl_history:
        return {
            "trading_days": 0,
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "win_rate_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
        }

    pnl_series = pd.Series(pnl_history)
    equity_series = pd.Series(equity_curve)
    total_return_pct = (equity_series.iloc[-1] / INITIAL_CAPITAL - 1) * 100

    years = max(len(pnl_series) / 365, 1 / 365)
    annualized_return_pct = ((equity_series.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100

    daily_ret = equity_series.pct_change().dropna()
    daily_mean = daily_ret.mean()
    daily_std = daily_ret.std()
    sharpe = (daily_mean / daily_std) * np.sqrt(365) if daily_std > 0 else 0.0

    rolling_peak = equity_series.cummax()
    drawdown = (equity_series - rolling_peak) / rolling_peak * 100
    max_drawdown_pct = float(drawdown.min())

    win_rate_pct = win_days / len(pnl_series) * 100

    return {
        "trading_days": int(len(pnl_series)),
        "total_return_pct": float(total_return_pct),
        "annualized_return_pct": float(annualized_return_pct),
        "win_rate_pct": float(win_rate_pct),
        "sharpe": float(sharpe),
        "max_drawdown_pct": max_drawdown_pct,
    }


def select_robust_n(results_df: pd.DataFrame) -> pd.DataFrame:
    ranked = results_df.copy()
    dd_abs = ranked["validation_max_drawdown_pct"].abs().replace(0, 1e-9)
    ranked["validation_return_over_dd"] = ranked["validation_total_return_pct"] / dd_abs
    ranked["generalization_gap"] = (
        ranked["train_total_return_pct"] - ranked["validation_total_return_pct"]
    ).abs()
    ranked["robust_score"] = ranked["validation_return_over_dd"] - ranked["generalization_gap"]
    return ranked.sort_values(["robust_score", "validation_total_return_pct"], ascending=False)


def main() -> None:
    print("=" * 80)
    print("LEAKAGE-AWARE N SWEEP (MEAN REVERSION)")
    print("=" * 80)

    ohlc_df = pd.read_csv(DAILY_OHLC_FILE)
    ohlc_df["date"] = pd.to_datetime(ohlc_df["date"]).dt.date
    ohlc_df = ohlc_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["daily_return"])

    all_dates = sorted(ohlc_df["date"].unique())
    if len(all_dates) <= VALIDATION_DAYS + 1:
        raise ValueError(
            f"Not enough days for split. Have {len(all_dates)} days, need > {VALIDATION_DAYS + 1}."
        )

    validation_dates = set(all_dates[-VALIDATION_DAYS:])
    train_dates = set(all_dates[:-VALIDATION_DAYS])

    print(f"Total days: {len(all_dates)}")
    print(f"Train days: {len(train_dates)}")
    print(f"Validation days: {len(validation_dates)}")
    print(f"N values: {N_SWEEP_VALUES}")

    rows = []
    for n in N_SWEEP_VALUES:
        train_metrics = run_backtest(ohlc_df, all_dates, train_dates, n)
        validation_metrics = run_backtest(ohlc_df, all_dates, validation_dates, n)

        rows.append(
            {
                "n": n,
                "train_days": train_metrics["trading_days"],
                "train_total_return_pct": train_metrics["total_return_pct"],
                "train_annualized_return_pct": train_metrics["annualized_return_pct"],
                "train_win_rate_pct": train_metrics["win_rate_pct"],
                "train_sharpe": train_metrics["sharpe"],
                "train_max_drawdown_pct": train_metrics["max_drawdown_pct"],
                "validation_days": validation_metrics["trading_days"],
                "validation_total_return_pct": validation_metrics["total_return_pct"],
                "validation_annualized_return_pct": validation_metrics["annualized_return_pct"],
                "validation_win_rate_pct": validation_metrics["win_rate_pct"],
                "validation_sharpe": validation_metrics["sharpe"],
                "validation_max_drawdown_pct": validation_metrics["max_drawdown_pct"],
            }
        )

    results = pd.DataFrame(rows)
    ranked = select_robust_n(results)

    output_file = DAILY_OHLC_FILE.parent / "n_sweep_results.csv"
    ranked.to_csv(output_file, index=False)

    display_cols = [
        "n",
        "train_total_return_pct",
        "validation_total_return_pct",
        "validation_max_drawdown_pct",
        "validation_return_over_dd",
        "generalization_gap",
        "robust_score",
    ]

    print("\nTop 10 robust N values")
    print(ranked[display_cols].head(10).round(3).to_string(index=False))
    best = ranked.iloc[0]
    print(
        f"\nSuggested baseline N: {int(best['n'])} "
        f"(robust_score={best['robust_score']:.3f}, "
        f"validation_return={best['validation_total_return_pct']:.2f}%)"
    )
    print(f"Saved full results: {output_file}")


if __name__ == "__main__":
    main()
