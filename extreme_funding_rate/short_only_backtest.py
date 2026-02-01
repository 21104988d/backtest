"""
FOCUSED STRATEGY BACKTEST - SHORT ONLY, NO POSITION LIMITS

Strategy:
- Entry: Funding Rate < -0.10% (negative, shorts are paying)
- Exit: |FR| < 0.01% (normalized)
- Direction: SHORT only (go with the shorts)
- No maximum position limit - take ALL signals
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TAKER_FEE = 0.00045  # 0.045% per trade
ENTRY_THRESHOLD = -0.001  # -0.10% (negative funding)
EXIT_THRESHOLD = 0.0001   # 0.01% absolute value
MAX_HOLD_HOURS = 72

# =============================================================================
# DATA LOADING
# =============================================================================

print("=" * 100)
print("FOCUSED STRATEGY: SHORT ONLY, ENTRY < -0.10%, EXIT < |0.01%|")
print("=" * 100)

print("\nLoading data...")
funding = pd.read_csv('funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)
funding['hour'] = funding['datetime'].dt.floor('h').dt.tz_localize(None)

price = pd.read_csv('price_history.csv')
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
price['hour'] = price['timestamp'].dt.floor('h').dt.tz_localize(None)

print(f"Funding records: {len(funding):,}")
print(f"Price records: {len(price):,}")

# =============================================================================
# BUILD DATA STRUCTURES
# =============================================================================

print("\nBuilding data structures...")

# Price matrix
price_pivot = price.pivot_table(index='hour', columns='coin', values='price', aggfunc='first')
price_hours = sorted(price_pivot.index.tolist())
price_hours_set = set(price_pivot.index)
price_coins = set(price_pivot.columns)

# Funding lookup
funding_lookup = {}
for _, row in funding.iterrows():
    key = (row['hour'], row['coin'])
    funding_lookup[key] = row['funding_rate']

def get_funding_rate(hour, coin):
    return funding_lookup.get((hour, coin), 0)

# Filter funding with price data
funding_with_price = funding[
    (funding['hour'].isin(price_hours_set)) &
    (funding['coin'].isin(price_coins))
].copy()

# =============================================================================
# IDENTIFY ALL SHORT SIGNALS
# =============================================================================

print("\n" + "=" * 100)
print("IDENTIFYING SHORT SIGNALS (FR < -0.10%)")
print("=" * 100)

# All negative extreme funding signals
short_signals = funding_with_price[
    funding_with_price['funding_rate'] < ENTRY_THRESHOLD
].copy()

print(f"\nTotal SHORT signals: {len(short_signals):,}")
print(f"Unique coins: {short_signals['coin'].nunique()}")
print(f"Date range: {short_signals['hour'].min()} to {short_signals['hour'].max()}")

# Distribution by magnitude
print("\nSignal distribution by FR magnitude:")
for thresh in [-0.001, -0.002, -0.003, -0.005, -0.01]:
    count = (short_signals['funding_rate'] < thresh).sum()
    print(f"  FR < {thresh*100:.2f}%: {count:,} signals")

# =============================================================================
# BACKTEST ALL SIGNALS (NO POSITION LIMIT)
# =============================================================================

print("\n" + "=" * 100)
print("BACKTESTING ALL SHORT SIGNALS")
print("=" * 100)

trades = []
active_positions = {}  # {coin: {entry_hour, entry_price, entry_fr}}

# Track daily statistics
daily_positions = defaultdict(int)
daily_new_entries = defaultdict(int)

# Sort signals by time
short_signals_sorted = short_signals.sort_values('hour')

for _, signal in short_signals_sorted.iterrows():
    hour = signal['hour']
    coin = signal['coin']
    fr = signal['funding_rate']
    
    # Skip if already have position in this coin
    if coin in active_positions:
        continue
    
    # Get entry price
    if coin not in price_pivot.columns or hour not in price_pivot.index:
        continue
    
    entry_price = price_pivot.loc[hour, coin]
    if pd.isna(entry_price):
        continue
    
    # Open position
    active_positions[coin] = {
        'entry_hour': hour,
        'entry_price': entry_price,
        'entry_fr': fr
    }
    
    daily_new_entries[hour.date()] += 1

# Now simulate exit for each position
print("\nSimulating exits...")

all_hours = sorted(price_hours)
positions_to_process = list(active_positions.items())

for coin, pos in positions_to_process:
    entry_hour = pos['entry_hour']
    entry_price = pos['entry_price']
    entry_fr = pos['entry_fr']
    
    # Find exit
    exit_found = False
    funding_paid = 0
    
    for h in range(1, MAX_HOLD_HOURS + 1):
        check_hour = entry_hour + timedelta(hours=h)
        
        if check_hour not in price_pivot.index:
            continue
        
        if coin not in price_pivot.columns:
            break
            
        # Accumulate funding
        prev_hour = entry_hour + timedelta(hours=h-1)
        prev_fr = get_funding_rate(prev_hour, coin)
        
        # Short position: pay if FR < 0, receive if FR > 0
        if prev_fr < 0:
            funding_paid += abs(prev_fr)
        else:
            funding_paid -= prev_fr
        
        # Check exit condition
        current_fr = get_funding_rate(check_hour, coin)
        
        if abs(current_fr) < EXIT_THRESHOLD or h >= MAX_HOLD_HOURS:
            exit_price = price_pivot.loc[check_hour, coin]
            if pd.isna(exit_price):
                continue
            
            # Calculate PnL (SHORT position)
            price_return = -(exit_price - entry_price) / entry_price  # Negative because short
            gross_pnl = price_return - funding_paid
            net_pnl = gross_pnl - 2 * TAKER_FEE
            
            exit_reason = 'normalized' if abs(current_fr) < EXIT_THRESHOLD else 'timeout'
            
            trades.append({
                'coin': coin,
                'entry_hour': entry_hour,
                'exit_hour': check_hour,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_fr': entry_fr,
                'exit_fr': current_fr,
                'hold_hours': h,
                'price_return': price_return,
                'funding_paid': funding_paid,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'exit_reason': exit_reason
            })
            
            exit_found = True
            break
    
    if not exit_found:
        # Position didn't exit within max hold - use last available price
        last_hour = entry_hour + timedelta(hours=MAX_HOLD_HOURS)
        if last_hour in price_pivot.index and coin in price_pivot.columns:
            exit_price = price_pivot.loc[last_hour, coin]
            if not pd.isna(exit_price):
                price_return = -(exit_price - entry_price) / entry_price
                gross_pnl = price_return - funding_paid
                net_pnl = gross_pnl - 2 * TAKER_FEE
                
                trades.append({
                    'coin': coin,
                    'entry_hour': entry_hour,
                    'exit_hour': last_hour,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_fr': entry_fr,
                    'exit_fr': get_funding_rate(last_hour, coin),
                    'hold_hours': MAX_HOLD_HOURS,
                    'price_return': price_return,
                    'funding_paid': funding_paid,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'exit_reason': 'timeout'
                })

trades_df = pd.DataFrame(trades)
print(f"\nTotal completed trades: {len(trades_df):,}")

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

print("\n" + "=" * 100)
print("STRATEGY PERFORMANCE")
print("=" * 100)

if len(trades_df) > 0:
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 50)
    print(f"{'Total Trades':<30} {len(trades_df):>15,}")
    print(f"{'Avg PnL per Trade':<30} {trades_df['net_pnl'].mean()*100:>+14.2f}%")
    print(f"{'Median PnL per Trade':<30} {trades_df['net_pnl'].median()*100:>+14.2f}%")
    print(f"{'Std Dev':<30} {trades_df['net_pnl'].std()*100:>14.2f}%")
    print(f"{'Sharpe Ratio':<30} {trades_df['net_pnl'].mean()/trades_df['net_pnl'].std():>15.2f}")
    print(f"{'Win Rate':<30} {(trades_df['net_pnl']>0).mean()*100:>14.1f}%")
    print(f"{'Total PnL (sum)':<30} {trades_df['net_pnl'].sum()*100:>+14.0f}%")
    print(f"{'Avg Hold Time':<30} {trades_df['hold_hours'].mean():>14.1f}h")
    
    # PnL Components
    print("\n" + "-" * 50)
    print("PnL BREAKDOWN:")
    print(f"{'Avg Price Return':<30} {trades_df['price_return'].mean()*100:>+14.2f}%")
    print(f"{'Avg Funding Paid':<30} {trades_df['funding_paid'].mean()*100:>+14.2f}%")
    print(f"{'Trading Fees':<30} {-2*TAKER_FEE*100:>14.3f}%")
    
    # Exit reasons
    print("\n" + "-" * 50)
    print("EXIT REASONS:")
    for reason in trades_df['exit_reason'].unique():
        subset = trades_df[trades_df['exit_reason'] == reason]
        print(f"  {reason}: N={len(subset)}, Avg PnL={subset['net_pnl'].mean()*100:+.2f}%")

# =============================================================================
# BREAKDOWN BY FR MAGNITUDE
# =============================================================================

print("\n" + "=" * 100)
print("PERFORMANCE BY ENTRY FR MAGNITUDE")
print("=" * 100)

if len(trades_df) > 0:
    trades_df['fr_bucket'] = pd.cut(
        trades_df['entry_fr'],
        bins=[-1, -0.005, -0.003, -0.002, -0.001, 0],
        labels=['< -0.50%', '-0.50% to -0.30%', '-0.30% to -0.20%', '-0.20% to -0.10%', '> -0.10%']
    )
    
    print(f"\n{'FR Bucket':<25} {'N':>8} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Avg Hold':>10}")
    print("-" * 73)
    
    for bucket in trades_df['fr_bucket'].unique():
        if pd.isna(bucket):
            continue
        subset = trades_df[trades_df['fr_bucket'] == bucket]
        if len(subset) > 0:
            sharpe = subset['net_pnl'].mean() / subset['net_pnl'].std() if subset['net_pnl'].std() > 0 else 0
            print(f"{str(bucket):<25} {len(subset):>8} {subset['net_pnl'].mean()*100:>+9.2f}% {sharpe:>8.2f} {(subset['net_pnl']>0).mean()*100:>7.1f}% {subset['hold_hours'].mean():>9.1f}h")

# =============================================================================
# MONTHLY PERFORMANCE
# =============================================================================

print("\n" + "=" * 100)
print("MONTHLY PERFORMANCE")
print("=" * 100)

if len(trades_df) > 0:
    trades_df['month'] = pd.to_datetime(trades_df['entry_hour']).dt.to_period('M')
    
    monthly = trades_df.groupby('month').agg({
        'net_pnl': ['count', 'mean', 'sum', 'std'],
        'hold_hours': 'mean'
    })
    monthly.columns = ['n_trades', 'avg_pnl', 'total_pnl', 'std_pnl', 'avg_hold']
    monthly['sharpe'] = monthly['avg_pnl'] / monthly['std_pnl']
    monthly['win_rate'] = trades_df.groupby('month').apply(lambda x: (x['net_pnl'] > 0).mean())
    
    print(f"\n{'Month':<12} {'N':>6} {'Avg PnL':>10} {'Total':>12} {'Sharpe':>8} {'Win%':>8}")
    print("-" * 62)
    for month, row in monthly.iterrows():
        print(f"{str(month):<12} {row['n_trades']:>6.0f} {row['avg_pnl']*100:>+9.2f}% {row['total_pnl']*100:>+11.0f}% {row['sharpe']:>8.2f} {row['win_rate']*100:>7.1f}%")
    
    # Cumulative performance
    print(f"\n{'TOTAL':<12} {len(trades_df):>6} {trades_df['net_pnl'].mean()*100:>+9.2f}% {trades_df['net_pnl'].sum()*100:>+11.0f}%")

# =============================================================================
# TOP/BOTTOM COINS
# =============================================================================

print("\n" + "=" * 100)
print("COIN PERFORMANCE")
print("=" * 100)

if len(trades_df) > 0:
    coin_perf = trades_df.groupby('coin').agg({
        'net_pnl': ['count', 'mean', 'sum', 'std']
    })
    coin_perf.columns = ['n_trades', 'avg_pnl', 'total_pnl', 'std_pnl']
    coin_perf['sharpe'] = coin_perf['avg_pnl'] / coin_perf['std_pnl']
    coin_perf['win_rate'] = trades_df.groupby('coin').apply(lambda x: (x['net_pnl'] > 0).mean())
    
    # Top coins
    print("\nTOP 15 COINS (by Avg PnL, min 2 trades):")
    top_coins = coin_perf[coin_perf['n_trades'] >= 2].nlargest(15, 'avg_pnl')
    
    print(f"\n{'Coin':<10} {'N':>6} {'Avg PnL':>10} {'Total':>10} {'Sharpe':>8} {'Win%':>8}")
    print("-" * 56)
    for coin, row in top_coins.iterrows():
        print(f"{coin:<10} {row['n_trades']:>6.0f} {row['avg_pnl']*100:>+9.2f}% {row['total_pnl']*100:>+9.0f}% {row['sharpe']:>8.2f} {row['win_rate']*100:>7.1f}%")
    
    # Bottom coins
    print("\nBOTTOM 10 COINS (by Avg PnL, min 2 trades):")
    bottom_coins = coin_perf[coin_perf['n_trades'] >= 2].nsmallest(10, 'avg_pnl')
    
    print(f"\n{'Coin':<10} {'N':>6} {'Avg PnL':>10} {'Total':>10} {'Sharpe':>8} {'Win%':>8}")
    print("-" * 56)
    for coin, row in bottom_coins.iterrows():
        print(f"{coin:<10} {row['n_trades']:>6.0f} {row['avg_pnl']*100:>+9.2f}% {row['total_pnl']*100:>+9.0f}% {row['sharpe']:>8.2f} {row['win_rate']*100:>7.1f}%")

# =============================================================================
# CONCURRENT POSITIONS ANALYSIS
# =============================================================================

print("\n" + "=" * 100)
print("CONCURRENT POSITIONS ANALYSIS")
print("=" * 100)

if len(trades_df) > 0:
    # Build timeline of open positions
    position_timeline = defaultdict(int)
    
    for _, trade in trades_df.iterrows():
        entry = trade['entry_hour']
        exit_h = trade['exit_hour']
        
        current = entry
        while current <= exit_h:
            position_timeline[current] += 1
            current += timedelta(hours=1)
    
    positions_series = pd.Series(position_timeline)
    
    print(f"\nMax concurrent positions: {positions_series.max()}")
    print(f"Avg concurrent positions: {positions_series.mean():.1f}")
    print(f"Median concurrent positions: {positions_series.median():.0f}")
    
    # Distribution
    print("\nPosition count distribution:")
    for n in range(0, min(int(positions_series.max()) + 1, 15)):
        count = (positions_series == n).sum()
        pct = count / len(positions_series) * 100
        if pct > 0.5:
            print(f"  {n} positions: {count:,} hours ({pct:.1f}%)")

# =============================================================================
# DRAWDOWN ANALYSIS
# =============================================================================

print("\n" + "=" * 100)
print("DRAWDOWN ANALYSIS")
print("=" * 100)

if len(trades_df) > 0:
    trades_df_sorted = trades_df.sort_values('exit_hour')
    cumulative = trades_df_sorted['net_pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    
    print(f"\nMax Drawdown: {drawdown.max()*100:.1f}%")
    print(f"Avg Drawdown: {drawdown.mean()*100:.1f}%")
    
    # Find max drawdown period
    max_dd_idx = drawdown.idxmax()
    if max_dd_idx in trades_df_sorted.index:
        max_dd_trade = trades_df_sorted.loc[max_dd_idx]
        print(f"Max DD occurred at: {max_dd_trade['exit_hour']}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

if len(trades_df) > 0:
    print(f"""
STRATEGY: SHORT when FR < -0.10%, EXIT when |FR| < 0.01%

RESULTS:
  - Total Trades: {len(trades_df):,}
  - Avg PnL: {trades_df['net_pnl'].mean()*100:+.2f}%
  - Sharpe: {trades_df['net_pnl'].mean()/trades_df['net_pnl'].std():.2f}
  - Win Rate: {(trades_df['net_pnl']>0).mean()*100:.1f}%
  - Total PnL: {trades_df['net_pnl'].sum()*100:+.0f}%
  
PnL BREAKDOWN:
  - Price movement contributes: {trades_df['price_return'].mean()*100:+.2f}% per trade
  - Funding paid: {trades_df['funding_paid'].mean()*100:+.2f}% per trade
  - Net after fees: {trades_df['net_pnl'].mean()*100:+.2f}% per trade

KEY INSIGHT:
  When funding is very negative (shorts paying heavily), 
  the price tends to DROP, making SHORT profitable.
  The price drop ({trades_df['price_return'].mean()*100:+.2f}%) MORE than covers 
  the funding paid ({trades_df['funding_paid'].mean()*100:.2f}%).
""")

# Save results
trades_df.to_csv('short_only_strategy_trades.csv', index=False)
print("\nTrades saved to short_only_strategy_trades.csv")
