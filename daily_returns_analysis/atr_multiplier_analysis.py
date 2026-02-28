"""
ATR Multiplier Analysis for Dynamic Stop Loss
==============================================

This script answers: "How do we determine the multiplier?"

For a dynamic stop loss we set:
    stop_loss_distance = ATR(N) * multiplier

where ATR (Average True Range) measures how much an asset typically moves
in a day.  The multiplier controls how many "typical day moves" of room we
give each trade before cutting it.

Workflow
--------
1. Compute per-coin 14-day ATR (rolling, using the period BEFORE each trade).
2. Run the mean-reversion backtest with stop_loss = ATR * multiplier for
   multipliers in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0].
3. Compare Sharpe ratio, total return, max drawdown, and stop-hit rate.
4. Print an interpretation guide so you can pick the right multiplier.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OHLC_FILE  = os.path.join(SCRIPT_DIR, "daily_ohlc.csv")

# ── Strategy parameters (must match mean_reversion_backtest.py) ───────────────
N                = 3       # top/bottom N coins to trade each day
INITIAL_CAPITAL  = 1_000
POSITION_SIZE    = 100     # fixed $100 per position
POSITION_FRACTION = 1 / 6  # for dynamic equity sizing
TRADING_FEE      = 0.045  # 0.045% per leg (taker fee)
ROUND_TRIP_FEE   = TRADING_FEE * 2

# ── ATR configuration ─────────────────────────────────────────────────────────
ATR_PERIOD = 14            # look-back window for ATR calculation

# ── Multipliers to evaluate ───────────────────────────────────────────────────
MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("ATR MULTIPLIER ANALYSIS FOR DYNAMIC STOP LOSS")
print("=" * 80)

# ── 1.  Load daily OHLC ───────────────────────────────────────────────────────
print("\nLoading daily OHLC data …")
ohlc_df = pd.read_csv(OHLC_FILE)
ohlc_df["date"] = pd.to_datetime(ohlc_df["date"]).dt.date
ohlc_df = ohlc_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["daily_return"])
ohlc_df = ohlc_df.sort_values(["coin", "date"])

dates = sorted(ohlc_df["date"].unique())
print(f"  {len(ohlc_df):,} records  |  {len(dates)} days  |  "
      f"{ohlc_df['coin'].nunique()} coins")
print(f"  Date range: {dates[0]}  →  {dates[-1]}")

# ── 2.  Compute ATR per coin ──────────────────────────────────────────────────
#   True Range (TR) = max(high-low, |high-prev_close|, |low-prev_close|)
#   ATR = rolling mean of TR over ATR_PERIOD days
#   We express ATR as a *percentage of the open* so it is scale-free and
#   directly comparable across coins and price levels.
print(f"\nComputing {ATR_PERIOD}-day ATR per coin …")

atr_records = []
for coin, grp in ohlc_df.groupby("coin"):
    grp = grp.sort_values("date").copy()
    grp["prev_close"] = grp["close"].shift(1)
    grp["tr"] = np.maximum(
        grp["high"] - grp["low"],
        np.maximum(
            (grp["high"] - grp["prev_close"]).abs(),
            (grp["low"]  - grp["prev_close"]).abs(),
        ),
    )
    # ATR as % of open so the multiplier is in the same units as our SL %
    grp["atr_pct"] = (grp["tr"] / grp["open"] * 100).rolling(ATR_PERIOD).mean()
    atr_records.append(grp[["date", "coin", "atr_pct"]])

atr_df = pd.concat(atr_records, ignore_index=True)
# Merge ATR back into the OHLC table
ohlc_df = ohlc_df.merge(atr_df, on=["date", "coin"], how="left")

# Quick sanity check
valid_atr = ohlc_df["atr_pct"].dropna()
print(f"  ATR(14) statistics (% of open):")
print(f"    Mean  = {valid_atr.mean():.2f}%")
print(f"    Median= {valid_atr.median():.2f}%")
print(f"    10th  = {valid_atr.quantile(0.10):.2f}%")
print(f"    90th  = {valid_atr.quantile(0.90):.2f}%")

# ── 3.  Back-test helper ──────────────────────────────────────────────────────

def calc_position_return(coin, is_long, trade_data, sl_pct):
    """Return (pct_return, was_stopped).  sl_pct=None ⇒ no stop loss."""
    row = trade_data.loc[trade_data["coin"] == coin]
    if row.empty:
        return None, False
    row = row.iloc[0]
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    if sl_pct is None:
        return ((c / o - 1) * 100 if is_long else -(c / o - 1) * 100), False
    if is_long:
        if l <= o * (1 - sl_pct / 100):
            return -sl_pct, True
        return (c / o - 1) * 100, False
    else:
        if h >= o * (1 + sl_pct / 100):
            return -sl_pct, True
        return -(c / o - 1) * 100, False


def run_backtest(multiplier):
    """
    Run mean-reversion backtest where each position's stop loss equals
    that coin's ATR(14) on the trade day multiplied by `multiplier`.

    Returns a dict with summary statistics.
    """
    equity_fixed   = INITIAL_CAPITAL
    equity_dynamic = INITIAL_CAPITAL
    results, all_returns = [], []
    total_stops, total_trades = 0, 0

    for i in range(1, len(dates)):
        signal_date = dates[i - 1]
        trade_date  = dates[i]

        signal_data = ohlc_df[ohlc_df["date"] == signal_date]
        if len(signal_data) < N * 2:
            continue

        top_coins    = signal_data.nlargest(N, "daily_return")["coin"].tolist()
        bottom_coins = signal_data.nsmallest(N, "daily_return")["coin"].tolist()

        trade_data = ohlc_df[ohlc_df["date"] == trade_date]

        day_returns, day_stops = [], 0

        # Mean Reversion: SHORT top, LONG bottom
        for coin, is_long in (
            [(c, False) for c in top_coins] + [(c, True) for c in bottom_coins]
        ):
            # Per-coin dynamic stop loss
            atr_row = trade_data.loc[trade_data["coin"] == coin, "atr_pct"]
            if atr_row.empty or pd.isna(atr_row.iloc[0]):
                sl_pct = None          # no ATR available ⇒ no stop
            else:
                sl_pct = atr_row.iloc[0] * multiplier

            ret, stopped = calc_position_return(coin, is_long, trade_data, sl_pct)
            if ret is None:
                continue
            net_ret = ret - ROUND_TRIP_FEE
            day_returns.append(net_ret)
            if stopped:
                day_stops += 1
            total_trades += 1

        total_stops += day_stops

        pnl_fixed   = sum(r / 100 * POSITION_SIZE          for r in day_returns)
        pnl_dynamic = sum(r / 100 * equity_dynamic * POSITION_FRACTION
                         for r in day_returns)
        equity_fixed   += pnl_fixed
        equity_dynamic += pnl_dynamic

        all_returns.extend(day_returns)
        results.append({
            "trade_date":     trade_date,
            "pnl_fixed":      pnl_fixed,
            "equity_fixed":   equity_fixed,
            "equity_dynamic": equity_dynamic,
        })

    if not results:
        return None

    df = pd.DataFrame(results)
    ret_series = pd.Series(all_returns)

    # Sharpe (annualised, 0 risk-free rate)
    daily_ret_pct = df["pnl_fixed"] / (POSITION_SIZE * N * 2) * 100
    sharpe = (
        (daily_ret_pct.mean() * 365) / (daily_ret_pct.std() * np.sqrt(365))
        if daily_ret_pct.std() > 0 else 0
    )
    # Max drawdown
    eq = df["equity_fixed"]
    max_dd = ((eq - eq.expanding().max()) / eq.expanding().max() * 100).min()

    total_return = (df["equity_fixed"].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate     = (ret_series > 0).mean() * 100
    stop_rate    = total_stops / total_trades * 100 if total_trades else 0
    risk_adj     = total_return / abs(max_dd) if max_dd != 0 else 0

    avg_sl = valid_atr.median() * multiplier   # rough average stop in %

    return {
        "multiplier":    multiplier,
        "avg_sl_pct":    avg_sl,
        "total_return":  total_return,
        "sharpe":        sharpe,
        "max_dd":        max_dd,
        "win_rate":      win_rate,
        "stop_rate":     stop_rate,
        "risk_adj":      risk_adj,
        "equity_curve":  df["equity_fixed"].values,
        "trade_dates":   pd.to_datetime(df["trade_date"]),
    }


# ── 4.  Run for all multipliers ───────────────────────────────────────────────
print("\n" + "=" * 80)
print("RUNNING BACKTESTS …")
print("=" * 80)

all_results = []
for m in MULTIPLIERS:
    r = run_backtest(m)
    if r:
        all_results.append(r)
        print(
            f"  multiplier={m:.2f}  avg_SL≈{r['avg_sl_pct']:.2f}%  "
            f"return={r['total_return']:+.1f}%  sharpe={r['sharpe']:.3f}  "
            f"maxDD={r['max_dd']:.1f}%  stop_rate={r['stop_rate']:.1f}%"
        )

# ── 5.  Summary table ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
hdr = (f"{'Multiplier':>12}  {'Avg SL %':>9}  {'Return':>9}  "
       f"{'Sharpe':>8}  {'MaxDD':>8}  {'WinRate':>9}  {'StopRate':>9}")
print(hdr)
print("-" * len(hdr))
for r in all_results:
    print(
        f"  {r['multiplier']:>9.2f}x  {r['avg_sl_pct']:>8.2f}%  "
        f"{r['total_return']:>+8.1f}%  {r['sharpe']:>8.3f}  "
        f"{r['max_dd']:>+7.1f}%  {r['win_rate']:>8.1f}%  "
        f"{r['stop_rate']:>8.1f}%"
    )

# Best by each metric
best_sharpe = max(all_results, key=lambda x: x["sharpe"])
best_return = max(all_results, key=lambda x: x["total_return"])
best_dd     = max(all_results, key=lambda x: -x["max_dd"])
best_risk   = max(all_results, key=lambda x: x["risk_adj"])

print(f"\n  Best Sharpe:             multiplier = {best_sharpe['multiplier']:.2f}x  "
      f"(Sharpe = {best_sharpe['sharpe']:.3f})")
print(f"  Best Return:             multiplier = {best_return['multiplier']:.2f}x  "
      f"(Return = {best_return['total_return']:+.1f}%)")
print(f"  Smallest Drawdown:       multiplier = {best_dd['multiplier']:.2f}x  "
      f"(MaxDD = {best_dd['max_dd']:.1f}%)")
print(f"  Best Return/|MaxDD|:     multiplier = {best_risk['multiplier']:.2f}x  "
      f"(Ratio = {best_risk['risk_adj']:.2f})")

# ── 6.  Charts ────────────────────────────────────────────────────────────────
print("\nGenerating charts …")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f"ATR({ATR_PERIOD}) Multiplier Analysis — Mean Reversion (Short Winners / Long Losers)",
    fontsize=13, fontweight="bold",
)

mults  = [r["multiplier"]   for r in all_results]
sls    = [r["avg_sl_pct"]   for r in all_results]
returns= [r["total_return"] for r in all_results]
sharpes= [r["sharpe"]       for r in all_results]
dds    = [r["max_dd"]       for r in all_results]
stops  = [r["stop_rate"]    for r in all_results]
xlabels = [f"{m:.2f}x\n≈{s:.1f}%" for m, s in zip(mults, sls)]

# 1. Total Return
ax = axes[0, 0]
colors = ["green" if r >= 0 else "red" for r in returns]
ax.bar(xlabels, returns, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Total Return by Multiplier", fontweight="bold")
ax.set_xlabel("Multiplier  (avg stop loss %)")
ax.set_ylabel("Total Return (%)")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f%%"))
ax.grid(True, alpha=0.3, axis="y")

# 2. Sharpe Ratio
ax = axes[0, 1]
ax.plot(xlabels, sharpes, "bo-", markersize=8, linewidth=2)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
best_idx = sharpes.index(max(sharpes))
ax.scatter([xlabels[best_idx]], [sharpes[best_idx]], color="gold",
           s=150, zorder=5, label=f"Best: {mults[best_idx]:.2f}x")
ax.set_title("Sharpe Ratio by Multiplier", fontweight="bold")
ax.set_xlabel("Multiplier  (avg stop loss %)")
ax.set_ylabel("Sharpe Ratio")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3. Max Drawdown
ax = axes[0, 2]
ax.plot(xlabels, dds, "r^-", markersize=8, linewidth=2)
ax.fill_between(range(len(xlabels)), dds, 0, alpha=0.2, color="red")
ax.set_title("Max Drawdown by Multiplier", fontweight="bold")
ax.set_xlabel("Multiplier  (avg stop loss %)")
ax.set_ylabel("Max Drawdown (%)")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.grid(True, alpha=0.3)
ax.invert_yaxis()

# 4. Stop-Hit Rate
ax = axes[1, 0]
ax.bar(xlabels, stops, color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5)
ax.set_title("Stop-Loss Trigger Rate by Multiplier", fontweight="bold")
ax.set_xlabel("Multiplier  (avg stop loss %)")
ax.set_ylabel("% of Positions Stopped Out")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.grid(True, alpha=0.3, axis="y")

# 5. Equity curves
ax = axes[1, 1]
cmap = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
for r, c in zip(all_results, cmap):
    ax.plot(r["trade_dates"], r["equity_curve"],
            color=c, linewidth=1.2, label=f"{r['multiplier']:.2f}x")
ax.axhline(INITIAL_CAPITAL, color="gray", linestyle=":", alpha=0.6)
ax.set_title("Equity Curves by Multiplier (Fixed $100)", fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value ($)")
ax.tick_params(axis="x", rotation=30)
ax.legend(fontsize=7, ncol=2, loc="upper left")
ax.grid(True, alpha=0.3)

# 6. Return / |MaxDD| ratio (Calmar-like)
ax = axes[1, 2]
ratios = [r["risk_adj"] for r in all_results]
bar_colors = ["green" if v >= 0 else "red" for v in ratios]
ax.bar(xlabels, ratios, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Risk-Adjusted Return  (Return / |MaxDD|)", fontweight="bold")
ax.set_xlabel("Multiplier  (avg stop loss %)")
ax.set_ylabel("Return / |Max Drawdown|")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "atr_multiplier_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"  Chart saved → {out_path}")

# Save CSV
csv_records = [
    {
        "multiplier":   r["multiplier"],
        "avg_sl_pct":   round(r["avg_sl_pct"], 3),
        "total_return": round(r["total_return"], 2),
        "sharpe":       round(r["sharpe"], 4),
        "max_dd":       round(r["max_dd"], 2),
        "win_rate":     round(r["win_rate"], 2),
        "stop_rate":    round(r["stop_rate"], 2),
    }
    for r in all_results
]
csv_path = os.path.join(SCRIPT_DIR, "atr_multiplier_results.csv")
pd.DataFrame(csv_records).to_csv(csv_path, index=False)
print(f"  Results saved → {csv_path}")

# ── 7.  How to read the results ───────────────────────────────────────────────
print("\n" + "=" * 80)
print("HOW TO DETERMINE THE MULTIPLIER")
print("=" * 80)
print(f"""
CONCEPT
-------
  Dynamic stop loss = ATR({ATR_PERIOD}) × multiplier

  ATR measures the asset's *typical* daily range.  Multiplying it lets you
  express the stop in units of "how volatile this coin normally is today".

  Example (median ATR ≈ {valid_atr.median() if not valid_atr.empty else float('nan'):.1f}%):
    multiplier 1.0x  →  stop ≈ {valid_atr.median()*1.0 if not valid_atr.empty else float('nan'):.1f}%
    multiplier 1.5x  →  stop ≈ {valid_atr.median()*1.5 if not valid_atr.empty else float('nan'):.1f}%
    multiplier 2.0x  →  stop ≈ {valid_atr.median()*2.0 if not valid_atr.empty else float('nan'):.1f}%

DECISION GUIDE
--------------
  1. Highest Sharpe  → multiplier = {best_sharpe['multiplier']:.2f}x
     Best risk-adjusted return after accounting for daily volatility.
     Recommended as the primary selection criterion.

  2. Highest Return  → multiplier = {best_return['multiplier']:.2f}x
     Raw performance, but may come with larger drawdowns.

  3. Smallest MaxDD  → multiplier = {best_dd['multiplier']:.2f}x
     Most conservative; tolerable for risk-averse accounts.

  4. Return/|MaxDD|  → multiplier = {best_risk['multiplier']:.2f}x
     Calmar-like ratio balancing growth against worst loss.

PRACTICAL RULES OF THUMB
-------------------------
  • Multiplier < 1.0 : very tight stop — gets triggered frequently,
    hurts win-rate and profit factor (noise wipes you out).
  • Multiplier 1.0–1.5: balanced — stops real adverse moves while
    giving the trade breathing room.
  • Multiplier > 2.0 : loose stop — fewer triggers but larger losses
    when you are wrong; higher stop-loss cost per event.

  For volatile crypto pairs we generally recommend starting at 1.5x and
  tuning from there using the Sharpe ratio shown in the table above.

HOW TO RECALIBRATE FOR NEW ASSETS
----------------------------------
  1. Compute the coin's ATR(14) over at least 3 months of history.
  2. Run this script's sweep on that coin's sub-universe.
  3. Pick the multiplier that maximises Sharpe (or Return/|MaxDD|).
  4. Re-run every quarter as volatility regimes change.
""")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
