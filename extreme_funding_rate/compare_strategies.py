"""
Compare: Alpha-only vs Funding-Neutral Strategy
Explains why results differ from original 72h backtest
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ENTRY_THRESHOLD = 0.000020  # 0.0020%
POSITION_SIZE = 100
TAKER_FEE = 0.00045

print("Loading data...")
base_path = Path(__file__).parent

funding = pd.read_csv(base_path / 'funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)
funding['hour'] = funding['datetime'].dt.floor('h').dt.tz_localize(None)

price = pd.read_csv(base_path / 'price_history.csv')
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
price['hour'] = price['timestamp'].dt.floor('h').dt.tz_localize(None)

merged = pd.merge(
    funding[['hour', 'coin', 'funding_rate']],
    price[['hour', 'coin', 'price']],
    on=['hour', 'coin'],
    how='inner'
).sort_values(['coin', 'hour']).reset_index(drop=True)

print(f"Merged records: {len(merged):,}")

# ============================================================================
# STRATEGY: Original 72h hold (from previous backtest)
# ============================================================================
print("\n" + "="*70)
print("ORIGINAL 72H HOLD STRATEGY (Reference)")
print("="*70)

merged['price_72h'] = merged.groupby('coin')['price'].shift(-72)
merged['fr_sum_72h'] = merged.groupby('coin')['funding_rate'].transform(
    lambda x: x.rolling(72, min_periods=1).sum().shift(-71)
)
merged_72h = merged.dropna(subset=['price_72h']).copy()
merged_72h['date'] = merged_72h['hour'].dt.date

# Sample one per coin per day
sampled = merged_72h.groupby(['coin', 'date']).first().reset_index()

# Filter by threshold
sampled = sampled[sampled['funding_rate'].abs() >= ENTRY_THRESHOLD]

# Calculate PnL - Trend Following
# LONG when FR > 0, SHORT when FR < 0
# In both cases we PAY funding
directions = np.where(sampled['funding_rate'] > 0, 1, -1)
price_ret = directions * (sampled['price_72h'] - sampled['price']) / sampled['price']
funding_pnl = sampled['fr_sum_72h'] * directions  # We PAY, so this should be negative for alpha

# Actually for trend following:
# LONG when FR > 0: we PAY positive FR (negative pnl)
# SHORT when FR < 0: we PAY |negative FR| (negative pnl)
# So funding_pnl should always be negative
funding_cost = -sampled['fr_sum_72h'].abs()

fees = 2 * TAKER_FEE
net_pnl_72h = price_ret + funding_cost - fees

print(f"\nTotal Trades: {len(sampled):,}")
print(f"Avg Price Return: {price_ret.mean()*100:+.3f}% per 72h")
print(f"Avg Funding Cost: {funding_cost.mean()*100:+.3f}% per 72h")
print(f"Fees: {fees*100:.3f}% per trade")
print(f"Avg Net PnL: {net_pnl_72h.mean()*100:+.3f}% per trade")

# Convert to hourly
print(f"\n--- Per Hour Equivalent ---")
print(f"Price PnL per hour: {price_ret.mean()*100/72:+.4f}%")
print(f"Funding per hour: {funding_cost.mean()*100/72:+.4f}%")

# Total $ PnL
total_trades = len(sampled)
avg_pnl_per_trade = net_pnl_72h.mean() * POSITION_SIZE
total_pnl = total_trades * avg_pnl_per_trade
print(f"\nTotal $ PnL: ${total_pnl:,.2f}")
print(f"Trades: {total_trades:,}")
print(f"Avg $ per trade: ${avg_pnl_per_trade:.2f}")

# ============================================================================
# WHY FUNDING-NEUTRAL STRATEGY DIFFERS
# ============================================================================
print("\n" + "="*70)
print("WHY FUNDING-NEUTRAL STRATEGY DIFFERS")
print("="*70)

print("""
ORIGINAL 72H HOLD:
  - Enter once, hold 72 hours regardless of FR changes
  - Pay ONE round-trip fee (0.09%)
  - Sample 1 signal per coin per day (no overlap)
  - Price PnL: +0.42% over 72 hours
  - Expected: Positive net PnL

FUNDING-NEUTRAL HOURLY:
  - Rebalance EVERY HOUR
  - Alpha: close when |FR| drops below threshold
  - Hedge: recalculate every hour to match funding
  
PROBLEM 1: Alpha Turnover
  - Avg holding: only 5.2 hours (not 72!)
  - When |FR| drops, we close immediately
  - This means many short trades, not one 72h trade
  - More fees per unit of alpha captured

PROBLEM 2: Hedge Turnover (MAIN ISSUE)
  - Hedge changes every hour as FRs change
  - ~19 hedge positions on average
  - Constantly opening/closing to stay neutral
  - Massive fee drag with zero alpha

SOLUTION OPTIONS:
  1. Fixed 72h hold for alpha (match original)
  2. Reduce hedge rebalancing (daily instead of hourly)
  3. Accept funding exposure, skip hedge
  4. Higher threshold = fewer, stronger signals
""")

# ============================================================================
# TEST: What if we match original 72h hold exactly?
# ============================================================================
print("\n" + "="*70)
print("TEST: Funding-Neutral with 72H Hold (Daily Rebalance)")
print("="*70)

# Simulate: enter alpha positions, hold for 72h fixed
# Rebalance hedge only at entry, hold same hedge for 72h

# Use daily sampled data
merged_72h['abs_fr'] = merged_72h['funding_rate'].abs()

# Get daily snapshots
daily = merged_72h.groupby(['coin', 'date']).first().reset_index()

# For each day, identify alpha and hedge coins
results = []

for date in sorted(daily['date'].unique()):
    day_data = daily[daily['date'] == date]
    
    if len(day_data) == 0:
        continue
    
    # Alpha coins: |FR| >= threshold
    alpha_coins = day_data[day_data['abs_fr'] >= ENTRY_THRESHOLD]
    
    # Hedge coins: |FR| < threshold, sorted by |FR| desc
    hedge_pool = day_data[day_data['abs_fr'] < ENTRY_THRESHOLD].sort_values('abs_fr', ascending=False)
    
    if len(alpha_coins) == 0:
        continue
    
    # Funding to pay
    funding_to_pay = (POSITION_SIZE * alpha_coins['fr_sum_72h'].abs()).sum()
    
    # Select hedge coins
    hedge_coins = []
    accumulated = 0
    for _, row in hedge_pool.iterrows():
        hedge_funding = POSITION_SIZE * abs(row['fr_sum_72h'])
        hedge_coins.append(row)
        accumulated += hedge_funding
        if accumulated >= funding_to_pay:
            break
    
    hedge_df = pd.DataFrame(hedge_coins) if hedge_coins else pd.DataFrame()
    
    # Calculate PnL for this day's positions (72h hold)
    
    # Alpha PnL
    alpha_directions = np.where(alpha_coins['funding_rate'] > 0, 1, -1)
    alpha_price_ret = alpha_directions * (alpha_coins['price_72h'] - alpha_coins['price']) / alpha_coins['price']
    alpha_price_pnl = (POSITION_SIZE * alpha_price_ret).sum()
    
    alpha_funding_cost = (POSITION_SIZE * alpha_coins['fr_sum_72h'].abs()).sum()
    
    # Hedge PnL
    if len(hedge_df) > 0:
        # Hedge direction: opposite to receive funding
        hedge_directions = np.where(hedge_df['funding_rate'] > 0, -1, 1)  # SHORT if FR>0, LONG if FR<0
        hedge_price_ret = hedge_directions * (hedge_df['price_72h'] - hedge_df['price']) / hedge_df['price']
        hedge_price_pnl = (POSITION_SIZE * hedge_price_ret).sum()
        hedge_funding_receive = (POSITION_SIZE * hedge_df['fr_sum_72h'].abs()).sum()
    else:
        hedge_price_pnl = 0
        hedge_funding_receive = 0
    
    # Fees: one round trip per position
    n_positions = len(alpha_coins) + len(hedge_df)
    fees_total = n_positions * POSITION_SIZE * 2 * TAKER_FEE
    
    # Net
    net_funding = hedge_funding_receive - alpha_funding_cost
    net_pnl = alpha_price_pnl + hedge_price_pnl + net_funding - fees_total
    
    results.append({
        'date': date,
        'n_alpha': len(alpha_coins),
        'n_hedge': len(hedge_df),
        'alpha_price_pnl': alpha_price_pnl,
        'hedge_price_pnl': hedge_price_pnl,
        'funding_paid': alpha_funding_cost,
        'funding_received': hedge_funding_receive,
        'net_funding': net_funding,
        'fees': fees_total,
        'net_pnl': net_pnl,
    })

results_df = pd.DataFrame(results)

print(f"\nTotal Days: {len(results_df):,}")
print(f"Avg Alpha Positions/day: {results_df['n_alpha'].mean():.1f}")
print(f"Avg Hedge Positions/day: {results_df['n_hedge'].mean():.1f}")

total_alpha_price = results_df['alpha_price_pnl'].sum()
total_hedge_price = results_df['hedge_price_pnl'].sum()
total_funding_paid = results_df['funding_paid'].sum()
total_funding_recv = results_df['funding_received'].sum()
total_fees = results_df['fees'].sum()
total_net = results_df['net_pnl'].sum()

print(f"\n--- RESULTS ---")
print(f"Alpha Price PnL:    ${total_alpha_price:+,.2f}")
print(f"Hedge Price PnL:    ${total_hedge_price:+,.2f}")
print(f"Funding Paid:       ${-total_funding_paid:,.2f}")
print(f"Funding Received:   ${total_funding_recv:+,.2f}")
print(f"Net Funding:        ${total_funding_recv - total_funding_paid:+,.2f}")
print(f"Fees:               ${-total_fees:,.2f}")
print(f"NET PNL:            ${total_net:+,.2f}")

hedge_ratio = total_funding_recv / total_funding_paid * 100 if total_funding_paid > 0 else 0
print(f"\nFunding Hedge Ratio: {hedge_ratio:.1f}%")
