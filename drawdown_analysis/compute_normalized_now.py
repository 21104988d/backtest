import math
from pathlib import Path
import pandas as pd

INITIAL_CAPITAL = 100_000.0
root = Path(__file__).resolve().parent / "matrix_results"
summary_df = pd.read_csv(root / "validation_matrix_summary.csv")
active_windows = sorted(summary_df.loc[summary_df["cycle_open_count"] > 0, "window"].unique())

def annualized_sharpe(s):
    s = s.dropna()
    if len(s) < 2:
        return 0.0
    std = float(s.std())
    if std <= 1e-12:
        return 0.0
    return float(s.mean()) / std * math.sqrt(365.0)

def max_dd(series):
    running_max = series.cummax()
    dd = (series / running_max) - 1.0
    return float(dd.min() * 100.0)

rows = []
for _, row in summary_df[summary_df["window"].isin(active_windows)].iterrows():
    run = row["run"]
    run_dir = root / run
    eq = pd.read_csv(run_dir / "deribit_calendar_equity.csv")
    trades_path = run_dir / "deribit_calendar_trades.csv"
    if trades_path.exists() and trades_path.stat().st_size > 0:
        tr = pd.read_csv(trades_path)
    else:
        tr = pd.DataFrame()

    account = INITIAL_CAPITAL + eq["equity_usd"].astype(float)
    pnl = float(account.iloc[-1] - INITIAL_CAPITAL)
    ret_pct = pnl / INITIAL_CAPITAL * 100.0

    opens = int((tr["event"] == "cycle_open").sum()) if not tr.empty else 0
    closes = int((tr["event"] == "cycle_close").sum()) if not tr.empty else 0
    recovered = int(((tr["event"] == "cycle_close") & (tr["reason"] == "recovered")).sum()) if (not tr.empty and "reason" in tr.columns) else 0
    win_rate = recovered / closes * 100.0 if closes > 0 else 0.0

    rows.append({
        "run": run,
        "initial_capital": INITIAL_CAPITAL,
        "ending_capital": float(account.iloc[-1]),
        "pnl_usd": pnl,
        "return_pct": ret_pct,
        "max_drawdown_pct": max_dd(account),
        "sharpe_365": annualized_sharpe(account.pct_change().fillna(0.0)),
        "cycle_open_count": opens,
        "cycle_close_count": closes,
        "win_rate_pct": win_rate,
    })

out = pd.DataFrame(rows).sort_values("run")
out_csv = root / "normalized_performance_summary.csv"
out_md = root / "normalized_performance_summary.md"
out.to_csv(out_csv, index=False)

cols = [
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

lines = [
    "| " + " | ".join(cols) + " |",
    "| " + " | ".join(["---"] * len(cols)) + " |",
]
for _, r in out[cols].iterrows():
    lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("active_windows:", active_windows)
print("saved", out_csv)
print("saved", out_md)
print(out[cols].to_string(index=False))
