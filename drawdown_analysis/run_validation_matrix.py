from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from deribit_option_calendar_backtest import run_backtest


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def summarize_run(output_dir: Path) -> dict:
    equity_path = output_dir / "deribit_calendar_equity.csv"
    trades_path = output_dir / "deribit_calendar_trades.csv"

    try:
        equity_df = pd.read_csv(equity_path) if equity_path.exists() else pd.DataFrame()
    except EmptyDataError:
        equity_df = pd.DataFrame()

    try:
        trades_df = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    except EmptyDataError:
        trades_df = pd.DataFrame()

    cycle_open_count = 0
    cycle_close_count = 0
    add_short_count = 0
    final_equity = 0.0
    max_active_cycles = 0

    if not trades_df.empty and "event" in trades_df.columns:
        cycle_open_count = int((trades_df["event"] == "cycle_open").sum())
        cycle_close_count = int((trades_df["event"] == "cycle_close").sum())
        add_short_count = int((trades_df["event"] == "add_short_call").sum())

    if not equity_df.empty:
        final_equity = float(equity_df.iloc[-1]["equity_usd"])
        if "active_cycles" in equity_df.columns:
            max_active_cycles = int(equity_df["active_cycles"].max())

    return {
        "cycle_open_count": cycle_open_count,
        "cycle_close_count": cycle_close_count,
        "add_short_count": add_short_count,
        "final_equity_usd": final_equity,
        "max_active_cycles": max_active_cycles,
        "equity_rows": int(len(equity_df)),
        "trade_rows": int(len(trades_df)),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    matrix_root = root / "matrix_results"
    matrix_root.mkdir(parents=True, exist_ok=True)

    windows = [
        ("2024H1", "2024-01-01", "2024-06-30"),
        ("2024H2", "2024-07-01", "2024-12-31"),
        ("2025Y", "2025-01-01", "2025-12-31"),
    ]
    modes = [
        ("strict", True),
        ("nearest", False),
    ]

    records = []
    for window_name, start_date, end_date in windows:
        for mode_name, strict in modes:
            run_name = f"{window_name}_{mode_name}"
            out_dir = matrix_root / run_name
            out_dir.mkdir(parents=True, exist_ok=True)

            summary = run_backtest(
                currency="BTC",
                start_date=start_date,
                end_date=end_date,
                strict_strike=strict,
                output_dir=out_dir,
            )
            extra = summarize_run(out_dir)
            record = {
                "run": run_name,
                "window": window_name,
                "mode": mode_name,
                "start": start_date,
                "end": end_date,
                **summary,
                **extra,
            }
            records.append(record)
            print(f"Completed {run_name}: trades={record['cycle_open_count']} final_equity={record['final_equity_usd']:.2f}")

    results_df = pd.DataFrame(records)
    csv_path = matrix_root / "validation_matrix_summary.csv"
    md_path = matrix_root / "validation_matrix_summary.md"

    results_df.to_csv(csv_path, index=False)

    display_cols = [
        "run",
        "start",
        "end",
        "mode",
        "cycle_open_count",
        "cycle_close_count",
        "add_short_count",
        "max_active_cycles",
        "final_equity_usd",
        "equity_rows",
        "trade_rows",
    ]
    md_table = dataframe_to_markdown(results_df[display_cols])
    md_path.write_text(md_table + "\n", encoding="utf-8")

    print("\nSummary CSV:", csv_path)
    print("Summary MD:", md_path)
    print("\n" + md_table)


if __name__ == "__main__":
    main()
