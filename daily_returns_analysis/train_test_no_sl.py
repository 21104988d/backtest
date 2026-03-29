#!/usr/bin/env python3
"""Train/test parameter search for mean reversion without stop loss.

Workflow:
1) Split data chronologically into train and test groups.
2) Sweep N on train only.
3) Select best N from train objective.
4) Evaluate selected N on test to assess overfitting.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from config import (
    DAILY_OHLC_FILE,
    INITIAL_CAPITAL,
    MIN_ASSETS_PER_DAY,
    N_SWEEP_VALUES,
    POSITION_SIZE_FIXED,
    ROUND_TRIP_FEE_PCT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test N search without stop loss.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Chronological fraction of dates used for train set (default 0.70).",
    )
    parser.add_argument(
        "--output",
        default="no_sl_train_test_results.csv",
        help="Output CSV for N sweep metrics.",
    )
    return parser.parse_args()


def backtest_no_sl(
    ohlc_df: pd.DataFrame,
    all_dates: Sequence,
    active_dates: set,
    n: int,
) -> Dict[str, float]:
    equity = INITIAL_CAPITAL
    pnl_history: List[float] = []
    win_days = 0

    for idx in range(1, len(all_dates)):
        signal_date = all_dates[idx - 1]
        trade_date = all_dates[idx]
        if signal_date not in active_dates or trade_date not in active_dates:
            continue

        signal_day = ohlc_df[ohlc_df["date"] == signal_date]
        required_assets = max(MIN_ASSETS_PER_DAY, n * 2)
        if len(signal_day) < required_assets:
            continue

        top_n = signal_day.nlargest(n, "daily_return")[["coin"]]["coin"].tolist()
        bottom_n = signal_day.nsmallest(n, "daily_return")[["coin"]]["coin"].tolist()
        trade_day = ohlc_df[ohlc_df["date"] == trade_date]

        daily_pnl = 0.0

        # Mean reversion without stop loss:
        # short top performers, long bottom performers, open->close return only.
        for coin in top_n:
            row = trade_day[trade_day["coin"] == coin]
            if len(row) == 0:
                continue
            ret_pct = -(row.iloc[0]["close"] / row.iloc[0]["open"] - 1) * 100
            daily_pnl += ((ret_pct - ROUND_TRIP_FEE_PCT) / 100) * POSITION_SIZE_FIXED

        for coin in bottom_n:
            row = trade_day[trade_day["coin"] == coin]
            if len(row) == 0:
                continue
            ret_pct = (row.iloc[0]["close"] / row.iloc[0]["open"] - 1) * 100
            daily_pnl += ((ret_pct - ROUND_TRIP_FEE_PCT) / 100) * POSITION_SIZE_FIXED

        pnl_history.append(daily_pnl)
        if daily_pnl > 0:
            win_days += 1
        equity += daily_pnl

    if not pnl_history:
        return {
            "trading_days": 0,
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "win_rate_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "return_over_dd": 0.0,
            "ending_equity": INITIAL_CAPITAL,
        }

    pnl_series = pd.Series(pnl_history)
    equity_series = INITIAL_CAPITAL + pnl_series.cumsum()

    ending_equity = float(equity_series.iloc[-1])
    total_return_pct = (ending_equity / INITIAL_CAPITAL - 1) * 100
    years = max(len(pnl_series) / 365, 1 / 365)
    if ending_equity > 0:
        annualized_return_pct = ((ending_equity / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    else:
        annualized_return_pct = float("nan")

    daily_ret = equity_series.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() > 0 else 0.0

    rolling_peak = equity_series.cummax()
    drawdown = (equity_series - rolling_peak) / rolling_peak * 100
    max_drawdown_pct = float(drawdown.min())
    dd_abs = abs(max_drawdown_pct) if abs(max_drawdown_pct) > 0 else 1e-9

    return {
        "trading_days": int(len(pnl_series)),
        "total_return_pct": float(total_return_pct),
        "annualized_return_pct": float(annualized_return_pct),
        "win_rate_pct": float(win_days / len(pnl_series) * 100),
        "sharpe": float(sharpe),
        "max_drawdown_pct": max_drawdown_pct,
        "return_over_dd": float(total_return_pct / dd_abs),
        "ending_equity": ending_equity,
    }


def main() -> None:
    args = parse_args()
    if not (0.5 <= args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in [0.5, 1.0)")

    ohlc_df = pd.read_csv(DAILY_OHLC_FILE)
    ohlc_df["date"] = pd.to_datetime(ohlc_df["date"]).dt.date
    ohlc_df = ohlc_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["daily_return", "open", "close"])

    all_dates = sorted(ohlc_df["date"].unique())
    split_idx = int(len(all_dates) * args.train_ratio)
    train_dates = set(all_dates[:split_idx])
    test_dates = set(all_dates[split_idx:])

    if len(train_dates) < 30 or len(test_dates) < 30:
        raise ValueError("Not enough data after split. Need at least 30 dates in both train and test.")

    print("=" * 80)
    print("TRAIN/TEST PARAMETER SEARCH (NO STOP LOSS)")
    print("=" * 80)
    print(f"Total dates: {len(all_dates)}")
    print(f"Train dates: {len(train_dates)}")
    print(f"Test dates:  {len(test_dates)}")
    print(f"N sweep: {N_SWEEP_VALUES}")

    rows = []
    for n in N_SWEEP_VALUES:
        train = backtest_no_sl(ohlc_df, all_dates, train_dates, n)
        test = backtest_no_sl(ohlc_df, all_dates, test_dates, n)
        rows.append(
            {
                "n": n,
                "train_days": train["trading_days"],
                "train_total_return_pct": train["total_return_pct"],
                "train_annualized_return_pct": train["annualized_return_pct"],
                "train_sharpe": train["sharpe"],
                "train_max_drawdown_pct": train["max_drawdown_pct"],
                "train_return_over_dd": train["return_over_dd"],
                "test_days": test["trading_days"],
                "test_total_return_pct": test["total_return_pct"],
                "test_annualized_return_pct": test["annualized_return_pct"],
                "test_sharpe": test["sharpe"],
                "test_max_drawdown_pct": test["max_drawdown_pct"],
                "test_return_over_dd": test["return_over_dd"],
            }
        )

    results = pd.DataFrame(rows)

    # Select parameter only on train data.
    ranked_train = results.sort_values(["train_return_over_dd", "train_total_return_pct"], ascending=False).reset_index(drop=True)
    best_n = int(ranked_train.iloc[0]["n"])

    chosen = results[results["n"] == best_n].iloc[0].to_dict()
    chosen["selected_by_train"] = True
    chosen["generalization_gap_return_pct"] = chosen["train_total_return_pct"] - chosen["test_total_return_pct"]
    chosen["generalization_gap_sharpe"] = chosen["train_sharpe"] - chosen["test_sharpe"]

    output_df = results.copy()
    output_df["selected_by_train"] = output_df["n"] == best_n
    output_df["generalization_gap_return_pct"] = output_df["train_total_return_pct"] - output_df["test_total_return_pct"]
    output_df["generalization_gap_sharpe"] = output_df["train_sharpe"] - output_df["test_sharpe"]

    output_df.to_csv(output_path := DAILY_OHLC_FILE.parent / args.output, index=False)

    print("\nTrain ranking (top 10 by return/drawdown)")
    print(
        ranked_train[[
            "n",
            "train_total_return_pct",
            "train_max_drawdown_pct",
            "train_return_over_dd",
            "test_total_return_pct",
            "test_max_drawdown_pct",
        ]]
        .head(10)
        .round(3)
        .to_string(index=False)
    )

    print("\nSelected parameter from TRAIN")
    print(
        f"Best N = {best_n} | "
        f"Train return {chosen['train_total_return_pct']:.2f}% | "
        f"Test return {chosen['test_total_return_pct']:.2f}% | "
        f"Gap {chosen['generalization_gap_return_pct']:.2f}%"
    )

    overfit_flag = chosen["test_total_return_pct"] < 0 and chosen["train_total_return_pct"] > 0
    print(f"Overfitting warning: {'YES' if overfit_flag else 'NO'}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
