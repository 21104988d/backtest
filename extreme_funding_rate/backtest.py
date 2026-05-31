"""
Extreme Funding Rate Backtest — Bottom to Top Selection
========================================================

Strategy:
  • Each hour, sort all coins by funding_rate ascending
    (most negative = bottom → least negative = top).
  • Select the NUM_POSITIONS coins with the most extreme
    negative funding rates that fall within:
        MIN_FUNDING_THRESHOLD ≤ FR ≤ MAX_FUNDING_THRESHOLD
  • Go LONG on those coins.
    Rationale: extreme negative FR signals a heavily-shorted
    asset; mean-reversion predicts price will rise as shorts
    cover, and the long position also *collects* the negative
    funding payments (shorts pay longs).
  • Exit after HOLDING_PERIOD_HOURS (or earlier on
    stop-loss / take-profit if configured).
  • All parameters driven by .env via config.py.
"""
import gzip
from datetime import datetime

import numpy as np
import pandas as pd

from config import load_config, print_config

# =============================================================================
# CONFIG
# =============================================================================
config = load_config()

print("=" * 80)
print("EXTREME FUNDING RATE BACKTEST — BOTTOM TO TOP SELECTION")
print("=" * 80)
print(f"Run time: {datetime.now()}")
print()
print_config(config)

INITIAL_CAPITAL      = config["initial_capital"]
POSITION_SIZE        = config["position_size_fixed"]
TRANSACTION_COST     = config["transaction_cost"]
NUM_POSITIONS        = config["num_positions"]
HOLDING_PERIOD_HOURS = config["holding_period_hours"]
STOP_LOSS_PCT        = config["stop_loss_pct"]
TAKE_PROFIT_PCT      = config["take_profit_pct"]
MIN_FUNDING          = config["min_funding_threshold"]
MAX_FUNDING          = config["max_funding_threshold"]

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

with gzip.open("funding_history.csv.gz", "rt") as f:
    funding_raw = pd.read_csv(f)
with gzip.open("price_history.csv.gz", "rt") as f:
    prices_raw = pd.read_csv(f)

funding_raw["timestamp"] = pd.to_datetime(funding_raw["timestamp"], unit="ms", utc=True)
prices_raw["timestamp"]  = pd.to_datetime(prices_raw["timestamp"], utc=True)

funding_raw["hour"] = funding_raw["timestamp"].dt.floor("h")
prices_raw["hour"]  = prices_raw["timestamp"].dt.floor("h")

prices  = prices_raw.groupby(["hour", "coin"])["price"].last().reset_index()
funding = funding_raw.groupby(["hour", "coin"])["funding_rate"].last().reset_index()

data = funding.merge(prices, on=["hour", "coin"], how="inner")
data = data.sort_values(["hour", "coin"]).reset_index(drop=True)

price_lookup   = {(r["hour"], r["coin"]): r["price"]        for _, r in prices.iterrows()}
funding_lookup = {(r["hour"], r["coin"]): r["funding_rate"] for _, r in data.iterrows()}

print(f"  Data points:   {len(data):,}")
print(f"  Unique coins:  {data['coin'].nunique()}")
print(f"  Date range:    {data['hour'].min()} to {data['hour'].max()}")

# =============================================================================
# BACKTEST
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING BACKTEST")
print("=" * 80)

positions      = {}   # coin → {entry_hour, entry_price, current_price, entry_fr}
trades         = []
hourly_records = []

equity        = float(INITIAL_CAPITAL)
peak_equity   = float(INITIAL_CAPITAL)
max_drawdown  = 0.0
total_funding_pnl = 0.0
total_price_pnl   = 0.0
total_fees        = 0.0

hours = sorted(data["hour"].unique())
print(f"  Processing {len(hours):,} hours…")

for i, hour in enumerate(hours):
    hour_data = data[data["hour"] == hour]

    hour_price_pnl   = 0.0
    hour_funding_pnl = 0.0
    hour_fees        = 0.0

    # ── 1. UPDATE CURRENT PRICES ─────────────────────────────────────────────
    for coin in list(positions.keys()):
        current_price = price_lookup.get((hour, coin))
        if current_price is not None:
            positions[coin]["current_price"] = current_price

    # ── 2. COLLECT HOURLY FUNDING FOR OPEN POSITIONS ─────────────────────────
    for coin, pos in positions.items():
        fr = funding_lookup.get((hour, coin), 0.0)
        current_notional = POSITION_SIZE * pos["current_price"] / pos["entry_price"]

        # LONG receives funding when FR < 0 (shorts pay longs)
        funding_payment = -fr * current_notional
        hour_funding_pnl += funding_payment
        pos["funding_collected"] = pos.get("funding_collected", 0.0) + funding_payment

    total_funding_pnl += hour_funding_pnl

    # ── 3. CHECK EXITS ────────────────────────────────────────────────────────
    for coin in list(positions.keys()):
        pos         = positions[coin]
        entry_price = pos["entry_price"]
        cur_price   = pos["current_price"]
        hold_hours  = (hour - pos["entry_hour"]).total_seconds() / 3600

        # Unrealised % move (LONG)
        unrealised_pct = (cur_price - entry_price) / entry_price

        should_exit = False
        exit_reason = ""

        if hold_hours >= HOLDING_PERIOD_HOURS:
            should_exit = True
            exit_reason = "time"
        elif STOP_LOSS_PCT > 0 and unrealised_pct <= -STOP_LOSS_PCT:
            should_exit = True
            exit_reason = "stop_loss"
        elif TAKE_PROFIT_PCT > 0 and unrealised_pct >= TAKE_PROFIT_PCT:
            should_exit = True
            exit_reason = "take_profit"

        if should_exit:
            price_pnl  = (cur_price - entry_price) / entry_price * POSITION_SIZE
            exit_fee   = POSITION_SIZE * TRANSACTION_COST         # exit only
            all_fees   = pos.get("entry_fee", 0.0) + exit_fee    # entry + exit
            net_pnl    = price_pnl + pos.get("funding_collected", 0.0) - all_fees

            trades.append({
                "coin":        coin,
                "direction":   "LONG",
                "entry_hour":  pos["entry_hour"],
                "exit_hour":   hour,
                "hold_hours":  hold_hours,
                "entry_price": entry_price,
                "exit_price":  cur_price,
                "entry_fr":    pos["entry_fr"],
                "price_pnl":   price_pnl,
                "funding_pnl": pos.get("funding_collected", 0.0),
                "fees":        all_fees,
                "net_pnl":     net_pnl,
                "exit_reason": exit_reason,
            })

            hour_price_pnl += price_pnl
            hour_fees      += exit_fee
            del positions[coin]

    total_price_pnl += hour_price_pnl
    total_fees      += hour_fees

    # ── 4. SELECT NEW ENTRIES — BOTTOM TO TOP ────────────────────────────────
    # Filter: apply funding threshold range, exclude BTC and coins already held
    candidates = hour_data[
        (hour_data["coin"] != "BTC") &
        (~hour_data["coin"].isin(positions.keys())) &
        (hour_data["funding_rate"] >= MIN_FUNDING) &
        (hour_data["funding_rate"] <= MAX_FUNDING)
    ].copy()

    # Sort ascending → most negative FR first (bottom to top)
    candidates = candidates.sort_values("funding_rate", ascending=True)

    slots = NUM_POSITIONS - len(positions)
    if slots > 0 and not candidates.empty:
        # Take from the bottom (most extreme negative)
        for _, row in candidates.head(slots).iterrows():
            coin  = row["coin"]
            price = row["price"]
            fr    = row["funding_rate"]

            if price > 0:
                entry_fee = POSITION_SIZE * TRANSACTION_COST
                positions[coin] = {
                    "entry_hour":        hour,
                    "entry_price":       price,
                    "current_price":     price,
                    "entry_fr":          fr,
                    "funding_collected": 0.0,
                    "entry_fee":         entry_fee,
                }
                hour_fees  += entry_fee
                total_fees += entry_fee

    # ── 5. UPDATE EQUITY ──────────────────────────────────────────────────────
    hour_net = hour_price_pnl + hour_funding_pnl - hour_fees
    equity  += hour_net

    peak_equity  = max(peak_equity, equity)
    drawdown     = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0.0
    max_drawdown = max(max_drawdown, drawdown)

    n_long = len(positions)

    hourly_records.append({
        "hour":        hour,
        "equity":      equity,
        "n_long":      n_long,
        "price_pnl":   hour_price_pnl,
        "funding_pnl": hour_funding_pnl,
        "fees":        hour_fees,
        "net_pnl":     hour_net,
        "drawdown":    drawdown,
    })

    if (i + 1) % 5000 == 0:
        print(f"    {i+1:,}/{len(hours):,} hours  |  equity: ${equity:,.0f}  |  open: {n_long}")

# =============================================================================
# RESULTS
# =============================================================================
trades_df  = pd.DataFrame(trades)
hourly_df  = pd.DataFrame(hourly_records)

total_pnl = equity - INITIAL_CAPITAL
n_trades  = len(trades_df)
n_wins    = int((trades_df["net_pnl"] > 0).sum()) if n_trades > 0 else 0
win_rate  = 100.0 * n_wins / n_trades if n_trades > 0 else 0.0

hourly_returns = hourly_df["equity"].pct_change().dropna()
sharpe = (
    hourly_returns.mean() / hourly_returns.std() * np.sqrt(24 * 365)
    if hourly_returns.std() > 0 else 0.0
)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"""
  Starting Capital:     ${INITIAL_CAPITAL:,.2f}
  Final Equity:         ${equity:,.2f}
  Net PnL:              ${total_pnl:,.2f}  ({100 * total_pnl / INITIAL_CAPITAL:.1f}%)

  ── PnL Breakdown ──────────────────────────────────
  Price PnL:            ${total_price_pnl:,.2f}
  Funding PnL:          ${total_funding_pnl:,.2f}
  Total Fees:           ${total_fees:,.2f}
  ──────────────────────────────────────────────────
  NET:                  ${total_pnl:,.2f}

  ── Trade Statistics ───────────────────────────────
  Total Trades:         {n_trades}
  Winning Trades:       {n_wins}
  Win Rate:             {win_rate:.1f}%""")

if n_trades > 0:
    print(f"""  Avg PnL per Trade:    ${trades_df['net_pnl'].mean():.2f}
  Best Trade:           ${trades_df['net_pnl'].max():.2f}
  Worst Trade:          ${trades_df['net_pnl'].min():.2f}
  Avg Hold Time:        {trades_df['hold_hours'].mean():.1f} hours""")

    exit_breakdown = trades_df["exit_reason"].value_counts()
    print(f"\n  ── Exit Reasons ───────────────────────────────")
    for reason, count in exit_breakdown.items():
        print(f"  {reason:<20} {count:>6}  ({100*count/n_trades:.1f}%)")

print(f"""
  ── Risk Metrics ───────────────────────────────────
  Max Drawdown:         {max_drawdown:.1f}%
  Avg Open Positions:   {hourly_df['n_long'].mean():.1f}
  Max Open Positions:   {int(hourly_df['n_long'].max())}
  Sharpe Ratio:         {sharpe:.2f}
""")

# =============================================================================
# MONTHLY BREAKDOWN
# =============================================================================
if n_trades > 0:
    print("=" * 80)
    print("MONTHLY BREAKDOWN")
    print("=" * 80)

    trades_df["month"] = trades_df["exit_hour"].dt.to_period("M")
    monthly = trades_df.groupby("month").agg(
        trades=("net_pnl", "count"),
        win_rate=("net_pnl", lambda x: 100.0 * (x > 0).mean()),
        pnl=("net_pnl", "sum"),
    )

    print(f"\n{'Month':<12} {'Trades':>8} {'Win Rate':>10} {'PnL':>12}")
    print("-" * 46)
    for month, row in monthly.iterrows():
        print(f"{str(month):<12} {int(row['trades']):>8} {row['win_rate']:>9.1f}% ${row['pnl']:>10,.0f}")

# =============================================================================
# SAVE RESULTS
# =============================================================================
import os
os.makedirs("mean_reversion_results", exist_ok=True)

trades_df.to_csv("mean_reversion_results/bottom_top_trades.csv", index=False)
hourly_df.to_csv("mean_reversion_results/bottom_top_hourly.csv", index=False)

print("\n" + "=" * 80)
print("✅ Saved: mean_reversion_results/bottom_top_trades.csv")
print("✅ Saved: mean_reversion_results/bottom_top_hourly.csv")
print("=" * 80)
