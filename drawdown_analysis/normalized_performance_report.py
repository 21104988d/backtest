from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

INITIAL_CAPITAL = 100_000.0


def annualized_sharpe(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return 0.0
    std = float(daily_returns.std())
    if std <= 1e-12:
        return 0.0
    mean = float(daily_returns.mean())
    return (mean / std) * math.sqrt(365.0)


def max_drawdown_pct(equity_series: pd.Series) -> float:
    running_max = equity_series.cummax()
    dd = (equity_series / running_max) - 1.0
    return float(dd.min() * 100.0)


def compute_metrics_for_run(run_dir: Path, run_name: str) -> dict:
    equity_path = run_dir / "deribit_calendar_equity.csv"
    trades_path = run_dir / "deribit_calendar_trades.csv"

    equity_df = pd.read_csv(equity_path)
    trades_df = pd.read_csv(trades_path) if trades_path.exists() and trades_path.stat().st_size > 0 else pd.DataFrame()

    if equity_df.empty:
        return {
            "run": run_name,
            "initial_capital": INITIAL_CAPITAL,
            "ending_capital": INITIAL_CAPITAL,
            "pnl_usd": 0.0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_365": 0.0,
            "cycle_open_count": 0,
            "cycle_close_count": 0,
            "win_rate_pct": 0.0,
        }

    account_value = INITIAL_CAPITAL + equity_df["equity_usd"].astype(float)
    daily_ret = account_value.pct_change().fillna(0.0)

    ending_capital = float(account_value.iloc[-1])
    pnl = ending_capital - INITIAL_CAPITAL
    ret_pct = (pnl / INITIAL_CAPITAL) * 100.0

    open_count = 0
    close_count = 0
    recovered_count = 0
    if not trades_df.empty and "event" in trades_df.columns:
        open_count = int((trades_df["event"] == "cycle_open").sum())
        close_count = int((trades_df["event"] == "cycle_close").sum())
        if "reason" in trades_df.columns:
            recovered_count = int(
                ((trades_df["event"] == "cycle_close") & (trades_df["reason"] == "recovered")).sum()
            )

    win_rate = (recovered_count / close_count * 100.0) if close_count > 0 else 0.0

    return {
        "run": run_name,
        "initial_capital": INITIAL_CAPITAL,
        "ending_capital": ending_capital,
        "pnl_usd": pnl,
        "return_pct": ret_pct,
        "max_drawdown_pct": max_drawdown_pct(account_value),
        "sharpe_365": annualized_sharpe(daily_ret),
        "cycle_open_count": open_count,
        "cycle_close_count": close_count,
        "win_rate_pct": win_rate,
    }


def to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    matrix_root = Path(__file__).resolve().parent / "matrix_results"
    summary_csv = matrix_root / "validation_matrix_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing {summary_csv}")

    summary_df = pd.read_csv(summary_csv)

    # Only run for period where we have acquired option data and strategy activity.
    # We infer this as windows where any mode has cycle opens > 0.
    active_windows = sorted(summary_df.loc[summary_df["cycle_open_count"] > 0, "window"].unique())
    if not active_windows:
        raise RuntimeError("No active windows found with opened option cycles.")

    target_df = summary_df[summary_df["window"].isin(active_windows)].copy()

    rows = []
    for _, row in target_df.iterrows():
        run_name = str(row["run"])
        run_dir = matrix_root / run_name
        rows.append(compute_metrics_for_run(run_dir, run_name))

    perf_df = pd.DataFrame(rows).sort_values("run")

    out_csv = matrix_root / "normalized_performance_summary.csv"
    out_md = matrix_root / "normalized_performance_summary.md"

    perf_df.to_csv(out_csv, index=False)

    display_cols = [
        "run",
        "initial_capital",
        "ending_capital",
        "pnl_usd",
        "return_pct",
        "max_drawdown_pct",
        "sharpe_365",
        "cycle_open_count",
        "cycle_close_count",
        "win_rate_pct",
    ]
    md = to_markdown(perf_df[display_cols].round(4))
    out_md.write_text(md + "\n", encoding="utf-8")

    print("Active windows:", active_windows)
    print("Saved:", out_csv)
    print("Saved:", out_md)
    print("\n" + md)


if __name__ == "__main__":
    main()
