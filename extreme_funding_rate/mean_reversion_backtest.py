"""
Mean Reversion Strategy on Low Funding Rate Coins (0 to 0.0015%)

Hypothesis: Low FR coins tend to mean-revert, so:
- If 0 < FR < 0.0015%: SHORT (expect price drop, receive funding)
- If -0.0015% < FR < 0: LONG (expect price rise, receive funding)

This should profit from BOTH:
1. Price appreciation (mean-reversion works)
2. Funding received (always betting against FR direction)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PARAMETERS
# =============================================================================

FR_UPPER = 0.000015    # |FR| < 0.0015% for mean-reversion
FR_LOWER = 0.0         # FR must be non-zero to have a signal
POSITION_SIZE = 100    # $100 USD per position
TAKER_FEE = 0.00045    # 0.045% per trade

print("=" * 80)
print("MEAN REVERSION STRATEGY BACKTEST")
print("=" * 80)
print(f"\nEntry Criteria: {FR_LOWER*100:.4f}% < |FR| < {FR_UPPER*100:.4f}%")
print(f"Position Size: ${POSITION_SIZE}")
print(f"Taker Fee: {TAKER_FEE*100:.3f}%")

# =============================================================================
# DATA LOADING
# =============================================================================

print("\nLoading data...")
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

# Add next hour price
merged['price_next'] = merged.groupby('coin')['price'].shift(-1)
merged['hour_next'] = merged.groupby('coin')['hour'].shift(-1)

# Filter consecutive hours
merged = merged[merged['hour_next'] == merged['hour'] + pd.Timedelta(hours=1)]
merged = merged.dropna(subset=['price_next'])

print(f"  Records: {len(merged):,}")
print(f"  Hours: {merged['hour'].nunique():,}")
print(f"  Coins: {merged['coin'].nunique():,}")
print(f"  Date range: {merged['hour'].min()} to {merged['hour'].max()}")

# =============================================================================
# COMPUTE SIGNALS AND RETURNS
# =============================================================================

merged['abs_fr'] = merged['funding_rate'].abs()
merged['price_return'] = (merged['price_next'] - merged['price']) / merged['price']

# Mean-reversion signals: |FR| in range (FR_LOWER, FR_UPPER)
# We want non-zero FR but below threshold
merged['is_signal'] = (merged['abs_fr'] > FR_LOWER) & (merged['abs_fr'] < FR_UPPER)

# Direction: SHORT if FR > 0, LONG if FR < 0 (betting against FR direction)
# Price PnL: LONG = +return, SHORT = -return
merged['direction'] = np.where(merged['funding_rate'] > 0, -1, 1)  # -1 = SHORT, 1 = LONG
merged['position_pnl'] = merged['direction'] * merged['price_return']

# Funding: We RECEIVE funding when betting against FR
# If FR > 0 and we're SHORT: we receive FR
# If FR < 0 and we're LONG: we receive |FR|
merged['funding_pnl'] = merged['abs_fr']  # Always positive (receiving)

print("\n" + "=" * 80)
print("HOURLY MEAN-REVERSION STRATEGY (Enter/Exit Every Hour)")
print("=" * 80)

# Filter to signals only
signals = merged[merged['is_signal']].copy()
print(f"\nTotal signal hours: {len(signals):,}")
print(f"Unique coins with signals: {signals['coin'].nunique():,}")

# Calculate PnL
signals['gross_price_pnl'] = signals['position_pnl'] * POSITION_SIZE
signals['gross_funding_pnl'] = signals['funding_pnl'] * POSITION_SIZE
signals['fees'] = POSITION_SIZE * TAKER_FEE * 2  # Round-trip

total_price_pnl = signals['gross_price_pnl'].sum()
total_funding_pnl = signals['gross_funding_pnl'].sum()
total_fees = signals['fees'].sum()
net_pnl = total_price_pnl + total_funding_pnl - total_fees

print(f"\n--- RESULTS (Hourly Turnover) ---")
print(f"Total Trades: {len(signals):,}")
print(f"Price PnL: ${total_price_pnl:,.2f}")
print(f"Funding Received: ${total_funding_pnl:,.2f}")
print(f"Fees: ${total_fees:,.2f}")
print(f"NET PNL: ${net_pnl:,.2f}")

print(f"\n--- Per Trade ---")
print(f"Avg Price Return: {signals['position_pnl'].mean()*100:.4f}%")
print(f"Avg Funding: {signals['funding_pnl'].mean()*100:.4f}%")
print(f"Fees: {TAKER_FEE*2*100:.3f}%")
print(f"Avg Net: {(signals['position_pnl'].mean() + signals['funding_pnl'].mean() - TAKER_FEE*2)*100:.4f}%")

# =============================================================================
# TEST WITH LONGER HOLD PERIODS
# =============================================================================

def test_hold_period(data, hold_hours):
    """Test mean-reversion with fixed hold period."""
    
    # For each signal, calculate return over hold period
    data = data.copy()
    
    # Get price at entry and exit
    # We need to look ahead hold_hours
    results = []
    
    hours = sorted(data['hour'].unique())
    hour_to_idx = {h: i for i, h in enumerate(hours)}
    
    # Group by hour
    hour_groups = {h: g for h, g in data.groupby('hour')}
    
    for hour in hours:
        if hour not in hour_groups:
            continue
        
        exit_hour = hour + pd.Timedelta(hours=hold_hours)
        if exit_hour not in hour_groups:
            continue
        
        entry_df = hour_groups[hour]
        exit_df = hour_groups[exit_hour]
        
        # Get coins present in both
        entry_coins = set(entry_df['coin'])
        exit_coins = set(exit_df['coin'])
        common_coins = entry_coins & exit_coins
        
        entry_prices = dict(zip(entry_df['coin'], entry_df['price']))
        exit_prices = dict(zip(exit_df['coin'], exit_df['price']))
        entry_fr = dict(zip(entry_df['coin'], entry_df['funding_rate']))
        entry_abs_fr = dict(zip(entry_df['coin'], entry_df['abs_fr']))
        
        for coin in common_coins:
            abs_fr = entry_abs_fr[coin]
            if abs_fr <= FR_LOWER or abs_fr >= FR_UPPER:
                continue
            
            fr = entry_fr[coin]
            direction = -1 if fr > 0 else 1  # SHORT if FR > 0
            
            entry_price = entry_prices[coin]
            exit_price = exit_prices[coin]
            price_return = (exit_price - entry_price) / entry_price
            
            # Accumulate funding over hold period (simplified: assume constant FR)
            # In reality, FR changes hourly, but this is an approximation
            funding_received = abs_fr * hold_hours
            
            results.append({
                'hour': hour,
                'coin': coin,
                'direction': direction,
                'price_return': direction * price_return,
                'funding_received': funding_received,
                'hold_hours': hold_hours
            })
    
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    total_trades = len(df)
    total_price_pnl = df['price_return'].sum() * POSITION_SIZE
    total_funding = df['funding_received'].sum() * POSITION_SIZE
    total_fees = total_trades * POSITION_SIZE * TAKER_FEE * 2
    net_pnl = total_price_pnl + total_funding - total_fees
    
    return {
        'hold_hours': hold_hours,
        'trades': total_trades,
        'price_pnl': total_price_pnl,
        'funding': total_funding,
        'fees': total_fees,
        'net_pnl': net_pnl,
        'avg_price_return': df['price_return'].mean() * 100,
        'avg_net_return': (df['price_return'].mean() + df['funding_received'].mean() - TAKER_FEE*2) * 100
    }

print("\n" + "=" * 80)
print("TESTING DIFFERENT HOLD PERIODS")
print("=" * 80)

hold_periods = [1, 4, 8, 12, 24, 48, 72]

print(f"\n{'Hold':>6} | {'Trades':>8} | {'Price PnL':>12} | {'Funding':>12} | {'Fees':>10} | {'Net PnL':>12} | {'Avg Net%':>10}")
print("-" * 90)

for hold in hold_periods:
    result = test_hold_period(merged, hold)
    if result:
        print(f"{result['hold_hours']:>4}h | {result['trades']:>8,} | ${result['price_pnl']:>10,.0f} | ${result['funding']:>10,.0f} | ${result['fees']:>8,.0f} | ${result['net_pnl']:>10,.0f} | {result['avg_net_return']:>9.3f}%")

# =============================================================================
# TEST DIFFERENT FR RANGES
# =============================================================================

print("\n" + "=" * 80)
print("TESTING DIFFERENT FR RANGES (72h hold)")
print("=" * 80)

fr_ranges = [
    (0, 0.000010),    # 0 to 0.001%
    (0, 0.000015),    # 0 to 0.0015%
    (0, 0.000020),    # 0 to 0.002%
    (0, 0.000030),    # 0 to 0.003%
    (0, 0.000050),    # 0 to 0.005%
    (0.000010, 0.000015),  # 0.001% to 0.0015%
    (0.000005, 0.000015),  # 0.0005% to 0.0015%
]

def test_fr_range(data, fr_lower, fr_upper, hold_hours=72):
    """Test mean-reversion with specific FR range."""
    
    results = []
    hours = sorted(data['hour'].unique())
    hour_groups = {h: g for h, g in data.groupby('hour')}
    
    for hour in hours:
        if hour not in hour_groups:
            continue
        
        exit_hour = hour + pd.Timedelta(hours=hold_hours)
        if exit_hour not in hour_groups:
            continue
        
        entry_df = hour_groups[hour]
        exit_df = hour_groups[exit_hour]
        
        entry_coins = set(entry_df['coin'])
        exit_coins = set(exit_df['coin'])
        common_coins = entry_coins & exit_coins
        
        entry_prices = dict(zip(entry_df['coin'], entry_df['price']))
        exit_prices = dict(zip(exit_df['coin'], exit_df['price']))
        entry_fr = dict(zip(entry_df['coin'], entry_df['funding_rate']))
        entry_abs_fr = dict(zip(entry_df['coin'], entry_df['abs_fr']))
        
        for coin in common_coins:
            abs_fr = entry_abs_fr[coin]
            if abs_fr <= fr_lower or abs_fr >= fr_upper:
                continue
            
            fr = entry_fr[coin]
            direction = -1 if fr > 0 else 1
            
            entry_price = entry_prices[coin]
            exit_price = exit_prices[coin]
            price_return = (exit_price - entry_price) / entry_price
            
            funding_received = abs_fr * hold_hours
            
            results.append({
                'price_return': direction * price_return,
                'funding_received': funding_received,
            })
    
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    total_trades = len(df)
    total_price_pnl = df['price_return'].sum() * POSITION_SIZE
    total_funding = df['funding_received'].sum() * POSITION_SIZE
    total_fees = total_trades * POSITION_SIZE * TAKER_FEE * 2
    net_pnl = total_price_pnl + total_funding - total_fees
    
    return {
        'trades': total_trades,
        'price_pnl': total_price_pnl,
        'funding': total_funding,
        'fees': total_fees,
        'net_pnl': net_pnl,
        'avg_price_return': df['price_return'].mean() * 100,
        'avg_funding': df['funding_received'].mean() * 100,
        'avg_net_return': (df['price_return'].mean() + df['funding_received'].mean() - TAKER_FEE*2) * 100
    }

print(f"\n{'FR Range':>20} | {'Trades':>8} | {'Price PnL':>12} | {'Funding':>10} | {'Net PnL':>12} | {'Avg Price%':>10} | {'Avg Net%':>10}")
print("-" * 105)

for fr_lower, fr_upper in fr_ranges:
    result = test_fr_range(merged, fr_lower, fr_upper, 72)
    if result:
        range_str = f"{fr_lower*100:.4f}-{fr_upper*100:.4f}%"
        print(f"{range_str:>20} | {result['trades']:>8,} | ${result['price_pnl']:>10,.0f} | ${result['funding']:>8,.0f} | ${result['net_pnl']:>10,.0f} | {result['avg_price_return']:>9.4f}% | {result['avg_net_return']:>9.4f}%")

# =============================================================================
# COMPARE: Mean-Reversion vs Trend-Following
# =============================================================================

print("\n" + "=" * 80)
print("COMPARISON: Mean-Reversion vs Trend-Following (72h hold)")
print("=" * 80)

def test_trend_following(data, fr_threshold, hold_hours=72):
    """Test trend-following on high FR coins."""
    
    results = []
    hours = sorted(data['hour'].unique())
    hour_groups = {h: g for h, g in data.groupby('hour')}
    
    for hour in hours:
        if hour not in hour_groups:
            continue
        
        exit_hour = hour + pd.Timedelta(hours=hold_hours)
        if exit_hour not in hour_groups:
            continue
        
        entry_df = hour_groups[hour]
        exit_df = hour_groups[exit_hour]
        
        common_coins = set(entry_df['coin']) & set(exit_df['coin'])
        
        entry_prices = dict(zip(entry_df['coin'], entry_df['price']))
        exit_prices = dict(zip(exit_df['coin'], exit_df['price']))
        entry_fr = dict(zip(entry_df['coin'], entry_df['funding_rate']))
        entry_abs_fr = dict(zip(entry_df['coin'], entry_df['abs_fr']))
        
        for coin in common_coins:
            abs_fr = entry_abs_fr[coin]
            if abs_fr < fr_threshold:
                continue
            
            fr = entry_fr[coin]
            # Trend-following: LONG if FR > 0, SHORT if FR < 0
            direction = 1 if fr > 0 else -1
            
            entry_price = entry_prices[coin]
            exit_price = exit_prices[coin]
            price_return = (exit_price - entry_price) / entry_price
            
            # Trend-following PAYS funding
            funding_paid = abs_fr * hold_hours
            
            results.append({
                'price_return': direction * price_return,
                'funding_paid': funding_paid,
            })
    
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    total_trades = len(df)
    total_price_pnl = df['price_return'].sum() * POSITION_SIZE
    total_funding = -df['funding_paid'].sum() * POSITION_SIZE  # Negative because paying
    total_fees = total_trades * POSITION_SIZE * TAKER_FEE * 2
    net_pnl = total_price_pnl + total_funding - total_fees
    
    return {
        'trades': total_trades,
        'price_pnl': total_price_pnl,
        'funding': total_funding,
        'fees': total_fees,
        'net_pnl': net_pnl,
        'avg_price_return': df['price_return'].mean() * 100,
        'avg_net_return': (df['price_return'].mean() - df['funding_paid'].mean() - TAKER_FEE*2) * 100
    }

print("\nMean-Reversion (0 < |FR| < 0.0015%):")
mr_result = test_fr_range(merged, 0, 0.000015, 72)
if mr_result:
    print(f"  Trades: {mr_result['trades']:,}")
    print(f"  Price PnL: ${mr_result['price_pnl']:,.2f}")
    print(f"  Funding RECEIVED: ${mr_result['funding']:,.2f}")
    print(f"  Fees: ${mr_result['fees']:,.2f}")
    print(f"  NET PNL: ${mr_result['net_pnl']:,.2f}")
    print(f"  Avg Price Return: {mr_result['avg_price_return']:.4f}%")
    print(f"  Avg Net Return: {mr_result['avg_net_return']:.4f}%")

print("\nTrend-Following (|FR| >= 0.0015%):")
tf_result = test_trend_following(merged, 0.000015, 72)
if tf_result:
    print(f"  Trades: {tf_result['trades']:,}")
    print(f"  Price PnL: ${tf_result['price_pnl']:,.2f}")
    print(f"  Funding PAID: ${-tf_result['funding']:,.2f}")
    print(f"  Fees: ${tf_result['fees']:,.2f}")
    print(f"  NET PNL: ${tf_result['net_pnl']:,.2f}")
    print(f"  Avg Price Return: {tf_result['avg_price_return']:.4f}%")
    print(f"  Avg Net Return: {tf_result['avg_net_return']:.4f}%")

print("\n" + "=" * 80)
print("COMBINED STRATEGY POTENTIAL")
print("=" * 80)

if mr_result and tf_result:
    combined_net = mr_result['net_pnl'] + tf_result['net_pnl']
    combined_trades = mr_result['trades'] + tf_result['trades']
    
    print(f"\nIf we run BOTH strategies simultaneously:")
    print(f"  Mean-Reversion Net: ${mr_result['net_pnl']:,.2f}")
    print(f"  Trend-Following Net: ${tf_result['net_pnl']:,.2f}")
    print(f"  COMBINED NET: ${combined_net:,.2f}")
    print(f"  Total Trades: {combined_trades:,}")
