#!/usr/bin/env python3
"""Train/test parameter search across stop-loss families.

Adds deployment-focused constraints:
1) Train-only selection to prevent test leakage.
2) Win-rate floors and day participation gates.
3) Execution realism via spread/slippage penalties.
4) Portfolio-subset robustness checks on selected parameter set.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    DAILY_OHLC_FILE,
    DYNAMIC_MULTIPLIERS,
    FIXED_SL_PCTS,
    INITIAL_CAPITAL,
    MAX_SL_PCT,
    MIN_ASSETS_PER_DAY,
    MIN_SL_PCT,
    N_SWEEP_VALUES,
    POSITION_SIZE_FIXED,
    ROUND_TRIP_FEE_PCT,
    SLIPPAGE_BPS_PER_SIDE,
    SPREAD_BPS_PER_SIDE,
    STOP_EXTRA_SLIPPAGE_BPS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test N + SL family search.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Chronological train ratio (default 0.70)")
    parser.add_argument("--min-train-winrate", type=float, default=45.0, help="Minimum train daily win rate %")
    parser.add_argument(
        "--min-train-position-winrate",
        type=float,
        default=45.0,
        help="Minimum train position win rate %",
    )
    parser.add_argument(
        "--min-train-day-participation",
        type=float,
        default=75.0,
        help="Minimum train day participation %",
    )
    parser.add_argument("--portfolio-runs", type=int, default=12, help="Number of random portfolio subsets for robustness")
    parser.add_argument("--portfolio-fraction", type=float, default=0.60, help="Fraction of coins per portfolio subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for portfolio subset generation")
    parser.add_argument(
        "--spread-bps-per-side",
        type=float,
        default=SPREAD_BPS_PER_SIDE,
        help="Bid/ask spread cost per side (bps)",
    )
    parser.add_argument(
        "--slippage-bps-per-side",
        type=float,
        default=SLIPPAGE_BPS_PER_SIDE,
        help="Execution slippage per side (bps)",
    )
    parser.add_argument(
        "--stop-extra-slippage-bps",
        type=float,
        default=STOP_EXTRA_SLIPPAGE_BPS,
        help="Additional stop-hit slippage (bps)",
    )
    parser.add_argument(
        "--stress-matrix",
        action="store_true",
        help="Run low/base/high execution-cost stress matrix for selected parameters.",
    )
    parser.add_argument("--output", default="train_test_sl_search_results.csv", help="Output CSV filename")
    return parser.parse_args()


def bps_to_pct(value_bps: float) -> float:
    return value_bps / 100.0


def apply_sl_return(row: pd.Series, is_long: bool, sl_pct: float | None) -> Tuple[float, bool]:
    open_price = float(row["open"])
    close_price = float(row["close"])

    if sl_pct is None:
        if is_long:
            return (close_price / open_price - 1) * 100, False
        return -(close_price / open_price - 1) * 100, False

    high_price = float(row["high"])
    low_price = float(row["low"])

    if is_long:
        stop_price = open_price * (1 - sl_pct / 100)
        if low_price <= stop_price:
            return -sl_pct, True
        return (close_price / open_price - 1) * 100, False

    stop_price = open_price * (1 + sl_pct / 100)
    if high_price >= stop_price:
        return -sl_pct, True
    return -(close_price / open_price - 1) * 100, False


def resolve_sl_pct(mode: str, param: float, prev_day_return_pct: float) -> float | None:
    if mode == "none":
        return None
    if mode == "fixed":
        return float(param)
    if mode == "dynamic":
        raw = abs(prev_day_return_pct) * float(param)
        return max(MIN_SL_PCT, min(MAX_SL_PCT, raw))
    raise ValueError(f"Unsupported mode: {mode}")


def summarize_steadiness(pnl_df: pd.DataFrame) -> Tuple[float, float]:
    if pnl_df.empty:
        return 0.0, 0.0

    monthly = pnl_df.copy()
    monthly["month"] = monthly["trade_date"].dt.to_period("M")
    monthly_pnl = monthly.groupby("month")["daily_pnl"].sum()
    if len(monthly_pnl) == 0:
        return 0.0, 0.0

    positive_month_ratio = float((monthly_pnl > 0).mean() * 100)
    worst_month_pnl = float(monthly_pnl.min())
    return positive_month_ratio, worst_month_pnl


def backtest_param_set(
    ohlc_df: pd.DataFrame,
    all_dates: Sequence,
    active_dates: set,
    n: int,
    mode: str,
    mode_param: float,
    spread_bps_per_side: float,
    slippage_bps_per_side: float,
    stop_extra_slippage_bps: float,
    allowed_coins: Set[str] | None = None,
) -> Dict[str, float]:
    pnl_history: List[float] = []
    trade_dates_used: List[pd.Timestamp] = []
    win_days = 0
    total_positions = 0
    winning_positions = 0

    base_execution_cost_pct = 2 * (bps_to_pct(spread_bps_per_side) + bps_to_pct(slippage_bps_per_side))
    stop_extra_cost_pct = bps_to_pct(stop_extra_slippage_bps)

    for idx in range(1, len(all_dates)):
        signal_date = all_dates[idx - 1]
        trade_date = all_dates[idx]
        if signal_date not in active_dates or trade_date not in active_dates:
            continue

        signal_day = ohlc_df[ohlc_df["date"] == signal_date]
        trade_day = ohlc_df[ohlc_df["date"] == trade_date]
        if allowed_coins is not None:
            signal_day = signal_day[signal_day["coin"].isin(allowed_coins)]
            trade_day = trade_day[trade_day["coin"].isin(allowed_coins)]

        required_assets = max(MIN_ASSETS_PER_DAY, n * 2)
        if len(signal_day) < required_assets:
            continue

        top_n = signal_day.nlargest(n, "daily_return")[["coin", "daily_return"]].values.tolist()
        bottom_n = signal_day.nsmallest(n, "daily_return")[["coin", "daily_return"]].values.tolist()

        daily_pnl = 0.0

        for coin, prev_ret in top_n:
            row = trade_day[trade_day["coin"] == coin]
            if len(row) == 0:
                continue
            sl_pct = resolve_sl_pct(mode, mode_param, prev_ret)
            gross_ret_pct, stopped = apply_sl_return(row.iloc[0], is_long=False, sl_pct=sl_pct)
            extra_cost_pct = stop_extra_cost_pct if stopped else 0.0
            net_ret_pct = gross_ret_pct - ROUND_TRIP_FEE_PCT - base_execution_cost_pct - extra_cost_pct
            daily_pnl += (net_ret_pct / 100) * POSITION_SIZE_FIXED
            total_positions += 1
            if net_ret_pct > 0:
                winning_positions += 1

        for coin, prev_ret in bottom_n:
            row = trade_day[trade_day["coin"] == coin]
            if len(row) == 0:
                continue
            sl_pct = resolve_sl_pct(mode, mode_param, prev_ret)
            gross_ret_pct, stopped = apply_sl_return(row.iloc[0], is_long=True, sl_pct=sl_pct)
            extra_cost_pct = stop_extra_cost_pct if stopped else 0.0
            net_ret_pct = gross_ret_pct - ROUND_TRIP_FEE_PCT - base_execution_cost_pct - extra_cost_pct
            daily_pnl += (net_ret_pct / 100) * POSITION_SIZE_FIXED
            total_positions += 1
            if net_ret_pct > 0:
                winning_positions += 1

        pnl_history.append(daily_pnl)
        trade_dates_used.append(pd.Timestamp(trade_date))
        if daily_pnl > 0:
            win_days += 1

    if not pnl_history:
        return {
            "trading_days": 0,
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "daily_win_rate_pct": 0.0,
            "position_win_rate_pct": 0.0,
            "day_participation_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "return_over_dd": 0.0,
            "positive_month_ratio_pct": 0.0,
            "worst_month_pnl": 0.0,
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

    pnl_df = pd.DataFrame({"trade_date": trade_dates_used, "daily_pnl": pnl_history})
    positive_month_ratio_pct, worst_month_pnl = summarize_steadiness(pnl_df)

    expected_days = max(len(active_dates) - 1, 1)
    day_participation_pct = len(pnl_series) / expected_days * 100
    position_win_rate_pct = (winning_positions / total_positions * 100) if total_positions > 0 else 0.0

    return {
        "trading_days": int(len(pnl_series)),
        "total_return_pct": float(total_return_pct),
        "annualized_return_pct": float(annualized_return_pct),
        "daily_win_rate_pct": float(win_days / len(pnl_series) * 100),
        "position_win_rate_pct": float(position_win_rate_pct),
        "day_participation_pct": float(day_participation_pct),
        "sharpe": float(sharpe),
        "max_drawdown_pct": max_drawdown_pct,
        "return_over_dd": float(total_return_pct / dd_abs),
        "positive_month_ratio_pct": positive_month_ratio_pct,
        "worst_month_pnl": worst_month_pnl,
        "ending_equity": ending_equity,
    }


def evaluate_portfolio_robustness(
    ohlc_df: pd.DataFrame,
    all_dates: Sequence,
    test_dates: set,
    n: int,
    mode: str,
    param: float,
    runs: int,
    fraction: float,
    seed: int,
    spread_bps_per_side: float,
    slippage_bps_per_side: float,
    stop_extra_slippage_bps: float,
) -> Dict[str, float]:
    coins = sorted(ohlc_df["coin"].unique().tolist())
    if not coins:
        return {
            "portfolio_positive_ratio_pct": 0.0,
            "portfolio_median_return_pct": 0.0,
            "portfolio_worst_return_pct": 0.0,
            "portfolio_median_win_rate_pct": 0.0,
        }

    subset_size = max(10, int(len(coins) * fraction))
    rng = np.random.default_rng(seed)
    subset_returns = []
    subset_winrates = []

    for _ in range(runs):
        if subset_size >= len(coins):
            selected = set(coins)
        else:
            selected = set(rng.choice(coins, size=subset_size, replace=False).tolist())

        metrics = backtest_param_set(
            ohlc_df=ohlc_df,
            all_dates=all_dates,
            active_dates=test_dates,
            n=n,
            mode=mode,
            mode_param=param,
            spread_bps_per_side=spread_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
            stop_extra_slippage_bps=stop_extra_slippage_bps,
            allowed_coins=selected,
        )
        subset_returns.append(metrics["total_return_pct"])
        subset_winrates.append(metrics["daily_win_rate_pct"])

    returns_series = pd.Series(subset_returns)
    winrate_series = pd.Series(subset_winrates)
    return {
        "portfolio_positive_ratio_pct": float((returns_series > 0).mean() * 100),
        "portfolio_median_return_pct": float(returns_series.median()),
        "portfolio_worst_return_pct": float(returns_series.min()),
        "portfolio_median_win_rate_pct": float(winrate_series.median()),
    }


def run_stress_matrix(
    ohlc_df: pd.DataFrame,
    all_dates: Sequence,
    train_dates: set,
    test_dates: set,
    n: int,
    mode: str,
    param: float,
    args: argparse.Namespace,
) -> pd.DataFrame:
    scenarios = [
        ("low", 0.5),
        ("base", 1.0),
        ("high", 2.0),
    ]

    rows = []
    for name, mult in scenarios:
        spread = args.spread_bps_per_side * mult
        slip = args.slippage_bps_per_side * mult
        stop_slip = args.stop_extra_slippage_bps * mult

        train = backtest_param_set(
            ohlc_df=ohlc_df,
            all_dates=all_dates,
            active_dates=train_dates,
            n=n,
            mode=mode,
            mode_param=param,
            spread_bps_per_side=spread,
            slippage_bps_per_side=slip,
            stop_extra_slippage_bps=stop_slip,
        )
        test = backtest_param_set(
            ohlc_df=ohlc_df,
            all_dates=all_dates,
            active_dates=test_dates,
            n=n,
            mode=mode,
            mode_param=param,
            spread_bps_per_side=spread,
            slippage_bps_per_side=slip,
            stop_extra_slippage_bps=stop_slip,
        )
        robustness = evaluate_portfolio_robustness(
            ohlc_df=ohlc_df,
            all_dates=all_dates,
            test_dates=test_dates,
            n=n,
            mode=mode,
            param=param,
            runs=args.portfolio_runs,
            fraction=args.portfolio_fraction,
            seed=args.seed,
            spread_bps_per_side=spread,
            slippage_bps_per_side=slip,
            stop_extra_slippage_bps=stop_slip,
        )

        rows.append(
            {
                "scenario": name,
                "cost_multiplier": mult,
                "spread_bps_per_side": spread,
                "slippage_bps_per_side": slip,
                "stop_extra_slippage_bps": stop_slip,
                "train_return_pct": train["total_return_pct"],
                "test_return_pct": test["total_return_pct"],
                "train_daily_win_rate_pct": train["daily_win_rate_pct"],
                "test_daily_win_rate_pct": test["daily_win_rate_pct"],
                "train_position_win_rate_pct": train["position_win_rate_pct"],
                "test_position_win_rate_pct": test["position_win_rate_pct"],
                "test_max_drawdown_pct": test["max_drawdown_pct"],
                "portfolio_positive_ratio_pct": robustness["portfolio_positive_ratio_pct"],
                "portfolio_median_return_pct": robustness["portfolio_median_return_pct"],
                "portfolio_worst_return_pct": robustness["portfolio_worst_return_pct"],
            }
        )

    return pd.DataFrame(rows)


def plot_stress_matrix(stress_df: pd.DataFrame, output_png: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Execution Cost Stress Matrix", fontsize=14, fontweight="bold")

    x = np.arange(len(stress_df))
    labels = stress_df["scenario"].tolist()

    axes[0, 0].bar(x - 0.15, stress_df["train_return_pct"], width=0.3, label="Train")
    axes[0, 0].bar(x + 0.15, stress_df["test_return_pct"], width=0.3, label="Test")
    axes[0, 0].set_title("Return by Scenario (%)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].axhline(0, color="black", linewidth=1)
    axes[0, 0].legend()

    axes[0, 1].bar(x - 0.15, stress_df["train_daily_win_rate_pct"], width=0.3, label="Train Day Win")
    axes[0, 1].bar(x + 0.15, stress_df["test_daily_win_rate_pct"], width=0.3, label="Test Day Win")
    axes[0, 1].set_title("Daily Win Rate by Scenario (%)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].legend()

    axes[1, 0].bar(x, stress_df["portfolio_median_return_pct"], width=0.5, label="Portfolio Median Return")
    axes[1, 0].set_title("Portfolio Median Return (Test, %)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].axhline(0, color="black", linewidth=1)

    axes[1, 1].bar(x, stress_df["portfolio_worst_return_pct"], width=0.5, label="Portfolio Worst Return")
    axes[1, 1].set_title("Portfolio Worst Return (Test, %)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].axhline(0, color="black", linewidth=1)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not (0.5 <= args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in [0.5, 1.0)")
    if not (0 < args.portfolio_fraction <= 1.0):
        raise ValueError("--portfolio-fraction must be in (0, 1]")

    ohlc_df = pd.read_csv(DAILY_OHLC_FILE)
    ohlc_df["date"] = pd.to_datetime(ohlc_df["date"]).dt.date
    ohlc_df = ohlc_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["daily_return", "open", "high", "low", "close"]
    )

    all_dates = sorted(ohlc_df["date"].unique())
    split_idx = int(len(all_dates) * args.train_ratio)
    train_dates = set(all_dates[:split_idx])
    test_dates = set(all_dates[split_idx:])

    if len(train_dates) < 30 or len(test_dates) < 30:
        raise ValueError("Not enough dates after split; need at least 30 in train and test")

    print("=" * 80)
    print("TRAIN/TEST SEARCH: FIXED VS DYNAMIC STOP LOSS")
    print("=" * 80)
    print(f"Total dates: {len(all_dates)}")
    print(f"Train dates: {len(train_dates)}")
    print(f"Test dates:  {len(test_dates)}")
    print(f"N sweep: {N_SWEEP_VALUES}")
    print(f"Fixed SL candidates: {FIXED_SL_PCTS}")
    print(f"Dynamic multipliers: {DYNAMIC_MULTIPLIERS}")
    print(f"Train daily win-rate floor: {args.min_train_winrate}%")
    print(f"Train position win-rate floor: {args.min_train_position_winrate}%")
    print(f"Train day participation floor: {args.min_train_day_participation}%")
    print(
        "Execution costs: "
        f"spread={args.spread_bps_per_side}bps/side, "
        f"slippage={args.slippage_bps_per_side}bps/side, "
        f"stop_extra={args.stop_extra_slippage_bps}bps"
    )

    candidates = [("none", 0.0)]
    candidates.extend(("fixed", sl) for sl in FIXED_SL_PCTS)
    candidates.extend(("dynamic", m) for m in DYNAMIC_MULTIPLIERS)

    rows = []
    for n in N_SWEEP_VALUES:
        for mode, param in candidates:
            train = backtest_param_set(
                ohlc_df,
                all_dates,
                train_dates,
                n,
                mode,
                param,
                spread_bps_per_side=args.spread_bps_per_side,
                slippage_bps_per_side=args.slippage_bps_per_side,
                stop_extra_slippage_bps=args.stop_extra_slippage_bps,
            )
            test = backtest_param_set(
                ohlc_df,
                all_dates,
                test_dates,
                n,
                mode,
                param,
                spread_bps_per_side=args.spread_bps_per_side,
                slippage_bps_per_side=args.slippage_bps_per_side,
                stop_extra_slippage_bps=args.stop_extra_slippage_bps,
            )
            rows.append(
                {
                    "n": n,
                    "sl_mode": mode,
                    "sl_param": param,
                    "train_days": train["trading_days"],
                    "train_total_return_pct": train["total_return_pct"],
                    "train_sharpe": train["sharpe"],
                    "train_max_drawdown_pct": train["max_drawdown_pct"],
                    "train_return_over_dd": train["return_over_dd"],
                    "train_daily_win_rate_pct": train["daily_win_rate_pct"],
                    "train_position_win_rate_pct": train["position_win_rate_pct"],
                    "train_day_participation_pct": train["day_participation_pct"],
                    "train_positive_month_ratio_pct": train["positive_month_ratio_pct"],
                    "train_worst_month_pnl": train["worst_month_pnl"],
                    "test_days": test["trading_days"],
                    "test_total_return_pct": test["total_return_pct"],
                    "test_sharpe": test["sharpe"],
                    "test_max_drawdown_pct": test["max_drawdown_pct"],
                    "test_return_over_dd": test["return_over_dd"],
                    "test_daily_win_rate_pct": test["daily_win_rate_pct"],
                    "test_position_win_rate_pct": test["position_win_rate_pct"],
                    "test_day_participation_pct": test["day_participation_pct"],
                    "test_positive_month_ratio_pct": test["positive_month_ratio_pct"],
                    "test_worst_month_pnl": test["worst_month_pnl"],
                }
            )

    results = pd.DataFrame(rows)
    results["generalization_gap_return_pct"] = results["train_total_return_pct"] - results["test_total_return_pct"]
    results["generalization_gap_sharpe"] = results["train_sharpe"] - results["test_sharpe"]

    eligible = results[
        (results["train_daily_win_rate_pct"] >= args.min_train_winrate)
        & (results["train_position_win_rate_pct"] >= args.min_train_position_winrate)
        & (results["train_day_participation_pct"] >= args.min_train_day_participation)
    ].copy()

    if eligible.empty:
        print("\nNo candidates pass train-side quality floors; falling back to full set.")
        eligible = results.copy()

    # Train-only selection to avoid test leakage.
    eligible["steady_score"] = (
        eligible["train_return_over_dd"]
        + 0.05 * eligible["train_positive_month_ratio_pct"]
        + 0.03 * eligible["train_position_win_rate_pct"]
    )

    ranked = eligible.sort_values(
        ["steady_score", "train_return_over_dd", "train_total_return_pct"],
        ascending=False,
    ).reset_index(drop=True)
    best = ranked.iloc[0]

    robustness = evaluate_portfolio_robustness(
        ohlc_df=ohlc_df,
        all_dates=all_dates,
        test_dates=test_dates,
        n=int(best["n"]),
        mode=str(best["sl_mode"]),
        param=float(best["sl_param"]),
        runs=args.portfolio_runs,
        fraction=args.portfolio_fraction,
        seed=args.seed,
        spread_bps_per_side=args.spread_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
        stop_extra_slippage_bps=args.stop_extra_slippage_bps,
    )

    output = results.copy()
    output["selected_by_train"] = (
        (output["n"] == int(best["n"]))
        & (output["sl_mode"] == str(best["sl_mode"]))
        & (output["sl_param"] == float(best["sl_param"]))
    )
    output["selected_steady_score"] = np.where(output["selected_by_train"], float(best["steady_score"]), np.nan)
    output["portfolio_positive_ratio_pct"] = np.where(output["selected_by_train"], robustness["portfolio_positive_ratio_pct"], np.nan)
    output["portfolio_median_return_pct"] = np.where(output["selected_by_train"], robustness["portfolio_median_return_pct"], np.nan)
    output["portfolio_worst_return_pct"] = np.where(output["selected_by_train"], robustness["portfolio_worst_return_pct"], np.nan)
    output["portfolio_median_win_rate_pct"] = np.where(output["selected_by_train"], robustness["portfolio_median_win_rate_pct"], np.nan)

    output_path = DAILY_OHLC_FILE.parent / args.output
    output.to_csv(output_path, index=False)

    print("\nTop 12 train-selected candidates")
    print(
        ranked[
            [
                "n",
                "sl_mode",
                "sl_param",
                "train_daily_win_rate_pct",
                "train_position_win_rate_pct",
                "test_daily_win_rate_pct",
                "test_position_win_rate_pct",
                "train_total_return_pct",
                "test_total_return_pct",
                "steady_score",
            ]
        ]
        .head(12)
        .round(3)
        .to_string(index=False)
    )

    print("\nSelected parameter set")
    print(
        f"N={int(best['n'])}, mode={best['sl_mode']}, param={best['sl_param']:.3f} | "
        f"train_win(day/pos)={best['train_daily_win_rate_pct']:.2f}%/{best['train_position_win_rate_pct']:.2f}% | "
        f"test_win(day/pos)={best['test_daily_win_rate_pct']:.2f}%/{best['test_position_win_rate_pct']:.2f}% | "
        f"train_ret={best['train_total_return_pct']:.2f}% | test_ret={best['test_total_return_pct']:.2f}%"
    )
    print(
        "Portfolio robustness (test subsets): "
        f"positive={robustness['portfolio_positive_ratio_pct']:.1f}% | "
        f"median_ret={robustness['portfolio_median_return_pct']:.2f}% | "
        f"worst_ret={robustness['portfolio_worst_return_pct']:.2f}% | "
        f"median_win={robustness['portfolio_median_win_rate_pct']:.2f}%"
    )

    overfit_flag = best["train_total_return_pct"] > 0 and best["test_total_return_pct"] < 0
    print(f"Overfitting warning: {'YES' if overfit_flag else 'NO'}")
    print(f"Saved: {output_path}")

    if args.stress_matrix:
        stress_df = run_stress_matrix(
            ohlc_df=ohlc_df,
            all_dates=all_dates,
            train_dates=train_dates,
            test_dates=test_dates,
            n=int(best["n"]),
            mode=str(best["sl_mode"]),
            param=float(best["sl_param"]),
            args=args,
        )

        stress_csv = str(output_path).replace(".csv", "_stress.csv")
        stress_png = str(output_path).replace(".csv", "_stress.png")
        stress_df.to_csv(stress_csv, index=False)
        plot_stress_matrix(stress_df, stress_png)

        print("\nStress matrix summary")
        print(stress_df.round(3).to_string(index=False))
        print(f"Saved stress CSV: {stress_csv}")
        print(f"Saved stress chart: {stress_png}")


if __name__ == "__main__":
    main()
