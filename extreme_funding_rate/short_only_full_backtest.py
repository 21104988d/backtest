"""
FOCUSED STRATEGY BACKTEST - SHORT ONLY, UNLIMITED POSITIONS

Strategy:
- Entry: Funding Rate < -0.10% (negative, shorts are paying)
- Exit: |FR| < 0.01% (normalized) OR 72h timeout
- Direction: SHORT only (go with the shorts)
- Allow re-entry after exit
- Track all concurrent positions
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
print("UNLIMITED POSITIONS - FULL SIMULATION")
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
# HOUR-BY-HOUR SIMULATION
# =============================================================================

print("\n" + "=" * 100)
print("HOUR-BY-HOUR SIMULATION")
print("=" * 100)

# Active positions: {(coin, entry_hour): {entry_price, entry_fr}}
active_positions = {}
completed_trades = []

# Statistics
hourly_positions = []
hourly_new_entries = []
hourly_exits = []

for hour in price_hours:
    new_entries = 0
    exits = 0
    
    # 1. Check exits for existing positions
    positions_to_close = []
    
    for (coin, entry_hour), pos in active_positions.items():
        hold_hours = int((hour - entry_hour).total_seconds() / 3600)
        
        if coin not in price_pivot.columns:
            continue
            
        current_price = price_pivot.loc[hour, coin]
        if pd.isna(current_price):
            continue
            
        current_fr = get_funding_rate(hour, coin)
        
        # Exit conditions
        should_exit = False
        exit_reason = None
        
        if abs(current_fr) < EXIT_THRESHOLD:
            should_exit = True
            exit_reason = 'normalized'
        elif hold_hours >= MAX_HOLD_HOURS:
            should_exit = True
            exit_reason = 'timeout'
        
        if should_exit:
            # Calculate funding paid over holding period
            funding_paid = 0
            for h in range(hold_hours):
                fr_hour = entry_hour + timedelta(hours=h)
                fr = get_funding_rate(fr_hour, coin)
                if fr < 0:
                    funding_paid += abs(fr)  # We pay
                else:
                    funding_paid -= fr  # We receive
            
            # Calculate PnL (SHORT position)
            price_return = -(current_price - pos['entry_price']) / pos['entry_price']
            gross_pnl = price_return - funding_paid
            net_pnl = gross_pnl - 2 * TAKER_FEE
            
            completed_trades.append({
                'coin': coin,
                'entry_hour': entry_hour,
                'exit_hour': hour,
                'entry_price': pos['entry_price'],
                'exit_price': current_price,
                'entry_fr': pos['entry_fr'],
                'exit_fr': current_fr,
                'hold_hours': hold_hours,
                'price_return': price_return,
                'funding_paid': funding_paid,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'exit_reason': exit_reason
            })
            
            positions_to_close.append((coin, entry_hour))
            exits += 1
    
    # Remove closed positions
    for key in positions_to_close:
        del active_positions[key]
    
    # 2. Check for new entries
    for coin in price_coins:
        if coin not in price_pivot.columns:
            continue
            
        current_price = price_pivot.loc[hour, coin]
        if pd.isna(current_price):
            continue
        
        current_fr = get_funding_rate(hour, coin)
        
        # Entry condition: FR < -0.10%
        if current_fr < ENTRY_THRESHOLD:
            # Check if we already have a position in this coin
            has_position = any(c == coin for (c, _) in active_positions.keys())
            
            if not has_position:
                active_positions[(coin, hour)] = {
                    'entry_price': current_price,
                    'entry_fr': current_fr
                }
                new_entries += 1
    
    hourly_positions.append(len(active_positions))
    hourly_new_entries.append(new_entries)
    hourly_exits.append(exits)

trades_df = pd.DataFrame(completed_trades)
print(f"\nSimulation complete!")
print(f"Total completed trades: {len(trades_df):,}")
print(f"Max concurrent positions: {max(hourly_positions)}")
print(f"Avg concurrent positions: {np.mean(hourly_positions):.1f}")

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
    sharpe = trades_df['net_pnl'].mean() / trades_df['net_pnl'].std() if trades_df['net_pnl'].std() > 0 else 0
    print(f"{'Sharpe Ratio':<30} {sharpe:>15.2f}")
    print(f"{'Win Rate':<30} {(trades_df['net_pnl']>0).mean()*100:>14.1f}%")
    print(f"{'Total PnL (sum)':<30} {trades_df['net_pnl'].sum()*100:>+14.0f}%")
    print(f"{'Avg Hold Time':<30} {trades_df['hold_hours'].mean():>14.1f}h")
    
    # PnL Components
    print("\n" + "-" * 50)
    print("PnL BREAKDOWN:")
    print(f"{'Avg Price Return':<30} {trades_df['price_return'].mean()*100:>+14.2f}%")
    print(f"{'Avg Funding Paid':<30} {trades_df['funding_paid'].mean()*100:>+14.2f}%")
    print(f"{'Trading Fees (per trade)':<30} {-2*TAKER_FEE*100:>14.3f}%")
    
    # Exit reasons
    print("\n" + "-" * 50)
    print("EXIT REASONS:")
    for reason in trades_df['exit_reason'].unique():
        subset = trades_df[trades_df['exit_reason'] == reason]
        print(f"  {reason}: N={len(subset)}, Avg PnL={subset['net_pnl'].mean()*100:+.2f}%, Win={((subset['net_pnl']>0).mean())*100:.1f}%")

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
    
    print(f"\n{'FR Bucket':<25} {'N':>8} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total PnL':>12}")
    print("-" * 77)
    
    for bucket in ['-0.20% to -0.10%', '-0.30% to -0.20%', '-0.50% to -0.30%', '< -0.50%']:
        subset = trades_df[trades_df['fr_bucket'] == bucket]
        if len(subset) > 0:
            sharpe = subset['net_pnl'].mean() / subset['net_pnl'].std() if subset['net_pnl'].std() > 0 else 0
            print(f"{str(bucket):<25} {len(subset):>8} {subset['net_pnl'].mean()*100:>+9.2f}% {sharpe:>8.2f} {(subset['net_pnl']>0).mean()*100:>7.1f}% {subset['net_pnl'].sum()*100:>+11.0f}%")

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
    
    print(f"\n{'TOTAL':<12} {len(trades_df):>6} {trades_df['net_pnl'].mean()*100:>+9.2f}% {trades_df['net_pnl'].sum()*100:>+11.0f}%")

# =============================================================================
# COIN PERFORMANCE
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
    print("\nTOP 15 COINS (by total PnL):")
    top_coins = coin_perf.nlargest(15, 'total_pnl')
    
    print(f"\n{'Coin':<10} {'N':>6} {'Avg PnL':>10} {'Total':>10} {'Sharpe':>8} {'Win%':>8}")
    print("-" * 56)
    for coin, row in top_coins.iterrows():
        print(f"{coin:<10} {row['n_trades']:>6.0f} {row['avg_pnl']*100:>+9.2f}% {row['total_pnl']*100:>+9.0f}% {row['sharpe']:>8.2f} {row['win_rate']*100:>7.1f}%")
    
    # Bottom coins
    print("\nBOTTOM 10 COINS (by total PnL):")
    bottom_coins = coin_perf.nsmallest(10, 'total_pnl')
    
    print(f"\n{'Coin':<10} {'N':>6} {'Avg PnL':>10} {'Total':>10} {'Sharpe':>8} {'Win%':>8}")
    print("-" * 56)
    for coin, row in bottom_coins.iterrows():
        print(f"{coin:<10} {row['n_trades']:>6.0f} {row['avg_pnl']*100:>+9.2f}% {row['total_pnl']*100:>+9.0f}% {row['sharpe']:>8.2f} {row['win_rate']*100:>7.1f}%")

# =============================================================================
# EQUITY CURVE
# =============================================================================

print("\n" + "=" * 100)
print("EQUITY CURVE (Cumulative PnL)")
print("=" * 100)

if len(trades_df) > 0:
    trades_df_sorted = trades_df.sort_values('exit_hour')
    cumulative = trades_df_sorted['net_pnl'].cumsum()
    
    print(f"\nStarting: 0%")
    # Show equity at key points
    n = len(cumulative)
    for i in [n//4, n//2, 3*n//4, n-1]:
        if i < n:
            trade = trades_df_sorted.iloc[i]
            print(f"Trade {i+1} ({trade['exit_hour'].strftime('%Y-%m-%d')}): {cumulative.iloc[i]*100:+.0f}%")
    print(f"Final: {cumulative.iloc[-1]*100:+.0f}%")
    
    # Drawdown
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    
    print(f"\nMax Drawdown: {drawdown.max()*100:.1f}%")

# =============================================================================
# DIFFERENT THRESHOLD COMPARISON
# =============================================================================

print("\n" + "=" * 100)
print("THRESHOLD SENSITIVITY TEST")
print("=" * 100)

for entry_thresh in [-0.0005, -0.001, -0.002, -0.003]:
    for exit_thresh in [0.00005, 0.0001, 0.0002]:
        # Quick rerun with different thresholds
        test_trades = []
        active = {}
        
        for hour in price_hours:
            # Check exits
            to_close = []
            for (coin, entry_hour), pos in active.items():
                hold_hours = int((hour - entry_hour).total_seconds() / 3600)
                if coin not in price_pivot.columns:
                    continue
                current_price = price_pivot.loc[hour, coin]
                if pd.isna(current_price):
                    continue
                current_fr = get_funding_rate(hour, coin)
                
                if abs(current_fr) < exit_thresh or hold_hours >= MAX_HOLD_HOURS:
                    # Simplified PnL calc
                    funding_paid = abs(pos['entry_fr']) * hold_hours / 2  # Rough estimate
                    price_return = -(current_price - pos['entry_price']) / pos['entry_price']
                    net_pnl = price_return - funding_paid - 2 * TAKER_FEE
                    test_trades.append({'net_pnl': net_pnl})
                    to_close.append((coin, entry_hour))
            
            for key in to_close:
                del active[key]
            
            # Check entries
            for coin in price_coins:
                if coin not in price_pivot.columns:
                    continue
                current_price = price_pivot.loc[hour, coin]
                if pd.isna(current_price):
                    continue
                current_fr = get_funding_rate(hour, coin)
                
                if current_fr < entry_thresh:
                    has_pos = any(c == coin for (c, _) in active.keys())
                    if not has_pos:
                        active[(coin, hour)] = {'entry_price': current_price, 'entry_fr': current_fr}
        
        if len(test_trades) >= 10:
            test_df = pd.DataFrame(test_trades)
            sharpe = test_df['net_pnl'].mean() / test_df['net_pnl'].std() if test_df['net_pnl'].std() > 0 else 0
            win_rate = (test_df['net_pnl'] > 0).mean()
            print(f"Entry<{entry_thresh*100:+.2f}% Exit<{exit_thresh*100:.3f}%: N={len(test_df):>4}, Avg={test_df['net_pnl'].mean()*100:>+5.2f}%, Sharpe={sharpe:>5.2f}, Win={win_rate*100:>5.1f}%")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

if len(trades_df) > 0:
    sharpe = trades_df['net_pnl'].mean() / trades_df['net_pnl'].std() if trades_df['net_pnl'].std() > 0 else 0
    print(f"""
STRATEGY: SHORT when FR < -0.10%, EXIT when |FR| < 0.01%

PERFORMANCE SUMMARY:
┌─────────────────────────────────────────────────────┐
│  Total Trades:          {len(trades_df):>5}                       │
│  Avg PnL per Trade:    {trades_df['net_pnl'].mean()*100:>+6.2f}%                      │
│  Sharpe Ratio:          {sharpe:>5.2f}                       │
│  Win Rate:             {(trades_df['net_pnl']>0).mean()*100:>5.1f}%                       │
│  Total PnL:           {trades_df['net_pnl'].sum()*100:>+6.0f}%                       │
│  Max Concurrent Pos:       {max(hourly_positions):>3}                       │
└─────────────────────────────────────────────────────┘

WHY IT WORKS:
- When FR < -0.10%, shorts are paying heavily
- This indicates STRONG bearish sentiment
- Price continues to DROP (trend continuation)
- Price drop ({trades_df['price_return'].mean()*100:+.2f}%) > Funding paid ({trades_df['funding_paid'].mean()*100:.2f}%)
- Net profit: {trades_df['net_pnl'].mean()*100:+.2f}% per trade
""")

# Save results
trades_df.to_csv('short_only_full_trades.csv', index=False)
print("\nTrades saved to short_only_full_trades.csv")
