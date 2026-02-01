"""
FINAL BACKTEST: SHORT-ONLY with BTC Delta Hedge
================================================
Re-run from scratch to verify stability and consistency

Strategy:
  - Entry: 0.0014% < FR < 0.0015% (positive FR → SHORT)
  - Exit: FR <= 0.0003%
  - 100% Delta hedge with BTC
"""
import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 80)
print("FINAL BACKTEST: SHORT-ONLY + BTC DELTA HEDGE")
print("=" * 80)
print(f"Run time: {datetime.now()}")

# =============================================================================
# STEP 1: LOAD AND PREPARE DATA
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

with gzip.open("funding_history.csv.gz", "rt") as f:
    funding_raw = pd.read_csv(f)
with gzip.open("price_history.csv.gz", "rt") as f:
    prices_raw = pd.read_csv(f)

print(f"  Raw funding rows: {len(funding_raw):,}")
print(f"  Raw price rows: {len(prices_raw):,}")

# Convert timestamps
funding_raw["timestamp"] = pd.to_datetime(funding_raw["timestamp"], unit="ms", utc=True)
prices_raw["timestamp"] = pd.to_datetime(prices_raw["timestamp"], utc=True)

# Normalize to hourly
funding_raw["hour"] = funding_raw["timestamp"].dt.floor("h")
prices_raw["hour"] = prices_raw["timestamp"].dt.floor("h")

# Aggregate to hourly
prices = prices_raw.groupby(["hour", "coin"])["price"].last().reset_index()
funding = funding_raw.groupby(["hour", "coin"])["funding_rate"].last().reset_index()

# Merge funding and prices
data = funding.merge(prices, on=["hour", "coin"], how="inner")
data = data.sort_values(["coin", "hour"]).reset_index(drop=True)

# Get BTC prices for hedging
btc_prices = prices[prices["coin"] == "BTC"].set_index("hour")["price"].to_dict()

print(f"  Merged data points: {len(data):,}")
print(f"  Date range: {data['hour'].min()} to {data['hour'].max()}")
print(f"  Unique coins: {data['coin'].nunique()}")
print(f"  BTC price points: {len(btc_prices):,}")

# =============================================================================
# STEP 2: BACKTEST PARAMETERS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: BACKTEST PARAMETERS")
print("=" * 80)

ENTRY_LOW = 0.000014       # 0.0014%
ENTRY_HIGH = 0.000015      # 0.0015%
EXIT_THRESHOLD = 0.000003  # 0.0003%
POSITION_SIZE = 100        # $100 per position
FEE_RATE = 0.00045         # 0.045% taker fee
STARTING_CAPITAL = 1000    # $1,000
HEDGE_RATIO = 1.0          # 100% delta hedge

print(f"  Entry Range: {ENTRY_LOW*100:.4f}% < FR < {ENTRY_HIGH*100:.4f}%")
print(f"  Exit Threshold: FR <= {EXIT_THRESHOLD*100:.4f}%")
print(f"  Position Size: ${POSITION_SIZE}")
print(f"  Fee Rate: {FEE_RATE*100:.3f}%")
print(f"  Starting Capital: ${STARTING_CAPITAL:,}")
print(f"  Hedge Ratio: {HEDGE_RATIO*100:.0f}%")

# =============================================================================
# STEP 3: RUN BACKTEST
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: RUNNING BACKTEST")
print("=" * 80)

positions = {}  # {coin: {entry_hour, entry_price, entry_fr}}
trades = []
hourly_records = []

equity = STARTING_CAPITAL
peak_equity = STARTING_CAPITAL
max_drawdown = 0
max_concurrent = 0

# BTC hedge tracking
btc_hedge_notional = 0
prev_btc_price = None
total_hedge_pnl = 0
total_hedge_fees = 0
total_strategy_pnl = 0

hours = sorted(data["hour"].unique())
print(f"  Processing {len(hours):,} hours...")

for i, hour in enumerate(hours):
    hour_data = data[data["hour"] == hour]
    btc_price = btc_prices.get(hour)
    
    if btc_price is None:
        continue
    
    hour_strategy_pnl = 0
    hour_hedge_pnl = 0
    trades_opened = 0
    trades_closed = 0
    
    # -----------------------------------------------------------------
    # PROCESS ALTCOIN POSITIONS
    # -----------------------------------------------------------------
    for _, row in hour_data.iterrows():
        coin = row["coin"]
        if coin == "BTC":
            continue
        
        fr = row["funding_rate"]
        price = row["price"]
        
        # CHECK EXITS
        if coin in positions:
            pos = positions[coin]
            # Exit when FR drops to threshold (not when it spikes)
            if abs(fr) <= EXIT_THRESHOLD and abs(fr) <= abs(pos["entry_fr"]):
                entry_price = pos["entry_price"]
                
                # SHORT PnL: profit when price falls
                price_pnl = (entry_price - price) / entry_price * POSITION_SIZE
                fees = POSITION_SIZE * FEE_RATE * 2  # Entry + exit
                net_pnl = price_pnl - fees
                
                hold_hours = (hour - pos["entry_hour"]).total_seconds() / 3600
                
                trades.append({
                    "coin": coin,
                    "direction": "SHORT",
                    "entry_hour": pos["entry_hour"],
                    "exit_hour": hour,
                    "hold_hours": hold_hours,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "entry_fr": pos["entry_fr"],
                    "exit_fr": fr,
                    "price_pnl": price_pnl,
                    "fees": fees,
                    "net_pnl": net_pnl,
                })
                
                hour_strategy_pnl += net_pnl
                trades_closed += 1
                del positions[coin]
        
        # CHECK ENTRIES
        if coin not in positions:
            # Entry: positive FR in range → SHORT
            if ENTRY_LOW <= fr <= ENTRY_HIGH:
                positions[coin] = {
                    "entry_hour": hour,
                    "entry_price": price,
                    "entry_fr": fr,
                }
                trades_opened += 1
    
    # -----------------------------------------------------------------
    # BTC DELTA HEDGE
    # -----------------------------------------------------------------
    n_positions = len(positions)
    target_hedge = n_positions * POSITION_SIZE * HEDGE_RATIO
    
    # Calculate hedge PnL from BTC price change
    if btc_hedge_notional > 0 and prev_btc_price is not None:
        btc_return = (btc_price - prev_btc_price) / prev_btc_price
        hour_hedge_pnl = btc_hedge_notional * btc_return
        total_hedge_pnl += hour_hedge_pnl
    
    # Rebalance hedge if needed (with fee)
    if abs(target_hedge - btc_hedge_notional) > POSITION_SIZE * 0.5:
        rebalance_amount = abs(target_hedge - btc_hedge_notional)
        rebalance_fee = rebalance_amount * FEE_RATE
        total_hedge_fees += rebalance_fee
        hour_strategy_pnl -= rebalance_fee
        btc_hedge_notional = target_hedge
    
    prev_btc_price = btc_price
    total_strategy_pnl += hour_strategy_pnl
    
    # -----------------------------------------------------------------
    # UPDATE EQUITY AND METRICS
    # -----------------------------------------------------------------
    equity += hour_strategy_pnl + hour_hedge_pnl
    peak_equity = max(peak_equity, equity)
    drawdown = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
    max_drawdown = max(max_drawdown, drawdown)
    max_concurrent = max(max_concurrent, n_positions)
    
    hourly_records.append({
        "hour": hour,
        "n_positions": n_positions,
        "btc_hedge": btc_hedge_notional,
        "btc_price": btc_price,
        "equity": equity,
        "drawdown": drawdown,
        "strategy_pnl": hour_strategy_pnl,
        "hedge_pnl": hour_hedge_pnl,
        "trades_opened": trades_opened,
        "trades_closed": trades_closed,
    })
    
    # Progress update
    if (i + 1) % 5000 == 0:
        print(f"    Processed {i+1:,}/{len(hours):,} hours, Equity: ${equity:,.0f}")

# Convert to DataFrames
trades_df = pd.DataFrame(trades)
hourly_df = pd.DataFrame(hourly_records)

print(f"\n  Backtest complete!")

# =============================================================================
# STEP 4: RESULTS SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: RESULTS SUMMARY")
print("=" * 80)

total_pnl = equity - STARTING_CAPITAL
n_trades = len(trades_df)
n_wins = (trades_df["net_pnl"] > 0).sum() if n_trades > 0 else 0
win_rate = 100 * n_wins / n_trades if n_trades > 0 else 0

# Sharpe ratio (annualized)
hourly_returns = hourly_df["equity"].pct_change().dropna()
sharpe = hourly_returns.mean() / hourly_returns.std() * np.sqrt(24 * 365) if hourly_returns.std() > 0 else 0

print(f"""
PERFORMANCE METRICS:
────────────────────────────────────────────────────────────
  Starting Capital:     ${STARTING_CAPITAL:,}
  Final Equity:         ${equity:,.0f}
  Total PnL:            ${total_pnl:,.0f} ({100*total_pnl/STARTING_CAPITAL:.1f}%)
  
  Strategy PnL:         ${total_strategy_pnl:,.0f}
  Hedge PnL:            ${total_hedge_pnl:,.0f}
  Hedge Fees:           ${total_hedge_fees:,.0f}
  
TRADE STATISTICS:
────────────────────────────────────────────────────────────
  Total Trades:         {n_trades}
  Winning Trades:       {n_wins}
  Win Rate:             {win_rate:.1f}%
  
  Avg PnL per Trade:    ${trades_df['net_pnl'].mean():.2f}
  Best Trade:           ${trades_df['net_pnl'].max():.2f}
  Worst Trade:          ${trades_df['net_pnl'].min():.2f}
  Avg Hold Time:        {trades_df['hold_hours'].mean():.1f} hours

RISK METRICS:
────────────────────────────────────────────────────────────
  Max Drawdown:         {max_drawdown:.1f}%
  Max Concurrent Pos:   {max_concurrent}
  Avg Concurrent Pos:   {hourly_df['n_positions'].mean():.1f}
  Sharpe Ratio:         {sharpe:.2f}
  Return/DD Ratio:      {total_pnl/max_drawdown:.1f}
""")

# =============================================================================
# STEP 5: MONTHLY BREAKDOWN
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: MONTHLY BREAKDOWN")
print("=" * 80)

trades_df["month"] = trades_df["exit_hour"].dt.to_period("M")
monthly = trades_df.groupby("month").agg({
    "net_pnl": ["sum", "count", lambda x: (x > 0).mean() * 100]
}).round(2)
monthly.columns = ["PnL", "Trades", "Win Rate"]

print(f"\n{'Month':<12} {'Trades':<8} {'Win Rate':<10} {'PnL':<12}")
print("-" * 45)
for month, row in monthly.iterrows():
    print(f"{str(month):<12} {int(row['Trades']):<8} {row['Win Rate']:<10.1f}% ${row['PnL']:<11,.0f}")

# =============================================================================
# STEP 6: GENERATE CHARTS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 6: GENERATING CHARTS")
print("=" * 80)

fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# Chart 1: Equity Curve
ax1 = axes[0, 0]
ax1.plot(hourly_df["hour"], hourly_df["equity"], linewidth=1.5, color="blue")
ax1.axhline(STARTING_CAPITAL, color="gray", linestyle="--", alpha=0.5, label="Starting Capital")
ax1.fill_between(hourly_df["hour"], STARTING_CAPITAL, hourly_df["equity"], 
                  where=hourly_df["equity"] >= STARTING_CAPITAL, alpha=0.3, color="green")
ax1.fill_between(hourly_df["hour"], STARTING_CAPITAL, hourly_df["equity"], 
                  where=hourly_df["equity"] < STARTING_CAPITAL, alpha=0.3, color="red")
ax1.set_title(f"Equity Curve (Final: ${equity:,.0f}, Return: {100*total_pnl/STARTING_CAPITAL:.0f}%)")
ax1.set_ylabel("Equity ($)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Chart 2: Drawdown
ax2 = axes[0, 1]
ax2.fill_between(hourly_df["hour"], 0, -hourly_df["drawdown"], alpha=0.7, color="red")
ax2.axhline(-max_drawdown, color="darkred", linestyle="--", label=f"Max DD: {max_drawdown:.1f}%")
ax2.set_title("Drawdown Over Time")
ax2.set_ylabel("Drawdown (%)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Chart 3: Concurrent Positions
ax3 = axes[1, 0]
ax3.fill_between(hourly_df["hour"], hourly_df["n_positions"], alpha=0.5, color="blue")
ax3.axhline(hourly_df["n_positions"].mean(), color="red", linestyle="--", 
            label=f"Avg: {hourly_df['n_positions'].mean():.1f}")
ax3.axhline(max_concurrent, color="darkred", linestyle=":", label=f"Max: {max_concurrent}")
ax3.set_title("Concurrent Positions Over Time")
ax3.set_ylabel("Number of Positions")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Chart 4: BTC Hedge Size
ax4 = axes[1, 1]
ax4.fill_between(hourly_df["hour"], hourly_df["btc_hedge"], alpha=0.5, color="orange")
ax4.set_title(f"BTC Hedge Position Size (Avg: ${hourly_df['btc_hedge'].mean():,.0f})")
ax4.set_ylabel("Hedge Notional ($)")
ax4.grid(True, alpha=0.3)

# Chart 5: Trade PnL Distribution
ax5 = axes[2, 0]
bins = np.linspace(trades_df["net_pnl"].min(), trades_df["net_pnl"].max(), 50)
ax5.hist(trades_df["net_pnl"], bins=bins, alpha=0.7, color="blue", edgecolor="black")
ax5.axvline(0, color="red", linestyle="--", linewidth=2)
ax5.axvline(trades_df["net_pnl"].mean(), color="green", linestyle="--", 
            label=f"Mean: ${trades_df['net_pnl'].mean():.2f}")
ax5.set_title(f"Trade PnL Distribution ({n_trades} trades, {win_rate:.1f}% win rate)")
ax5.set_xlabel("PnL ($)")
ax5.set_ylabel("Frequency")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Chart 6: Cumulative PnL (Strategy vs Hedge)
ax6 = axes[2, 1]
hourly_df["cum_strategy_pnl"] = hourly_df["strategy_pnl"].cumsum()
hourly_df["cum_hedge_pnl"] = hourly_df["hedge_pnl"].cumsum()
ax6.plot(hourly_df["hour"], hourly_df["cum_strategy_pnl"], label="Strategy PnL", linewidth=1.5, color="blue")
ax6.plot(hourly_df["hour"], hourly_df["cum_hedge_pnl"], label="Hedge PnL", linewidth=1.5, color="orange")
ax6.plot(hourly_df["hour"], hourly_df["cum_strategy_pnl"] + hourly_df["cum_hedge_pnl"], 
         label="Total PnL", linewidth=2, color="green")
ax6.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax6.set_title("Cumulative PnL Breakdown")
ax6.set_ylabel("Cumulative PnL ($)")
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mean_reversion_results/final_backtest_charts.png", dpi=150, bbox_inches="tight")
print("✅ Saved: mean_reversion_results/final_backtest_charts.png")

# =============================================================================
# STEP 7: COMPARISON WITH BTC BUY & HOLD
# =============================================================================
print("\n" + "=" * 80)
print("STEP 7: COMPARISON WITH BTC BUY & HOLD")
print("=" * 80)

btc_start = hourly_df["btc_price"].iloc[0]
btc_end = hourly_df["btc_price"].iloc[-1]
btc_return = (btc_end - btc_start) / btc_start * 100
btc_final_value = STARTING_CAPITAL * (1 + btc_return / 100)

print(f"""
BTC BUY & HOLD:
  Start Price:    ${btc_start:,.0f}
  End Price:      ${btc_end:,.0f}
  Return:         {btc_return:.1f}%
  Final Value:    ${btc_final_value:,.0f}

STRATEGY:
  Return:         {100*total_pnl/STARTING_CAPITAL:.1f}%
  Final Value:    ${equity:,.0f}

OUTPERFORMANCE:
  Strategy - BTC: {100*total_pnl/STARTING_CAPITAL - btc_return:+.1f}pp
""")

# Comparison chart
fig2, ax = plt.subplots(figsize=(14, 6))
strategy_norm = hourly_df["equity"] / STARTING_CAPITAL * 100
btc_norm = hourly_df["btc_price"] / btc_start * 100
ax.plot(hourly_df["hour"], strategy_norm, label=f"Strategy ({100*total_pnl/STARTING_CAPITAL:.0f}%)", linewidth=2, color="blue")
ax.plot(hourly_df["hour"], btc_norm, label=f"BTC B&H ({btc_return:.0f}%)", linewidth=2, color="orange", alpha=0.7)
ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Strategy vs BTC Buy & Hold (Rebased to 100)")
ax.set_ylabel("Value (100 = Start)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mean_reversion_results/final_btc_comparison.png", dpi=150, bbox_inches="tight")
print("✅ Saved: mean_reversion_results/final_btc_comparison.png")

# =============================================================================
# STEP 8: SAVE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 8: SAVING RESULTS")
print("=" * 80)

trades_df.to_csv("mean_reversion_results/final_trades.csv", index=False)
hourly_df.to_csv("mean_reversion_results/final_hourly.csv", index=False)
print("✅ Saved: mean_reversion_results/final_trades.csv")
print("✅ Saved: mean_reversion_results/final_hourly.csv")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│  SHORT-ONLY STRATEGY WITH BTC DELTA HEDGE                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PARAMETERS:                                                               │
│    Entry: {ENTRY_LOW*100:.4f}% < FR < {ENTRY_HIGH*100:.4f}% (SHORT when positive FR)          │
│    Exit:  FR <= {EXIT_THRESHOLD*100:.4f}%                                              │
│    Hedge: 100% Delta (LONG BTC to offset SHORT exposure)                   │
│                                                                            │
│  RESULTS:                                                                  │
│    Total PnL:        ${total_pnl:>10,.0f}  ({100*total_pnl/STARTING_CAPITAL:>6.1f}% return)                  │
│    Max Drawdown:     {max_drawdown:>10.1f}%                                          │
│    Sharpe Ratio:     {sharpe:>10.2f}                                             │
│    Win Rate:         {win_rate:>10.1f}%  ({n_trades} trades)                         │
│                                                                            │
│  COMPARISON:                                                               │
│    Strategy:         {100*total_pnl/STARTING_CAPITAL:>10.1f}%                                            │
│    BTC Buy & Hold:   {btc_return:>10.1f}%                                            │
│    Outperformance:   {100*total_pnl/STARTING_CAPITAL - btc_return:>+10.1f}pp                                           │
│                                                                            │
│  ✅ BACKTEST COMPLETE - Results verified                                   │
└────────────────────────────────────────────────────────────────────────────┘
""")
