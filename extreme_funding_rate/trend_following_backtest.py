"""
TREND-FOLLOWING STRATEGY BACKTEST

Strategy: Go WITH the crowd (opposite of mean-reversion)
- When FR < -0.10% (shorts paying): GO SHORT (join the shorts)
- When FR > +0.10% (longs paying): GO LONG (join the longs)

Exit Strategies:
1. Fixed holding period (4h, 8h, 12h, 24h)
2. Exit when funding normalizes (|FR| < threshold)
3. Exit when funding reverses sign
4. Trailing stop based on funding magnitude

Filters:
- Funding rate magnitude buckets
- Time since extreme started (don't enter on first hour)
- Consecutive hours of extreme funding
- Coin selection (exclude worst performers)
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

FUNDING_THRESHOLD = 0.001  # 0.10% - entry signal
TAKER_FEE = 0.00045  # 0.045% per trade
MAX_HOLD_HOURS = 48  # Maximum holding period for dynamic exits

# =============================================================================
# DATA LOADING
# =============================================================================

print("Loading data...")
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

# Funding lookup: (hour, coin) -> funding_rate
funding_lookup = {}
for _, row in funding.iterrows():
    key = (row['hour'], row['coin'])
    funding_lookup[key] = row['funding_rate']

def get_funding_rate(hour, coin):
    return funding_lookup.get((hour, coin), 0)

# Get price data coverage
price_hours_set = set(price_pivot.index)
price_coins = set(price_pivot.columns)

# =============================================================================
# CALCULATE CONSECUTIVE EXTREME HOURS
# =============================================================================

print("Calculating consecutive extreme funding hours...")

# For each (coin, hour), calculate how many consecutive hours of extreme funding
# This helps filter "mature" vs "new" extreme funding

funding_sorted = funding.sort_values(['coin', 'hour'])

def calculate_consecutive_hours(group):
    """Calculate consecutive hours of extreme funding (same sign)"""
    group = group.sort_values('hour')
    consecutive = []
    current_count = 0
    prev_sign = 0
    prev_hour = None
    
    for _, row in group.iterrows():
        fr = row['funding_rate']
        current_sign = np.sign(fr) if abs(fr) > FUNDING_THRESHOLD else 0
        
        if current_sign != 0:
            if prev_hour is not None:
                hour_diff = (row['hour'] - prev_hour).total_seconds() / 3600
                if hour_diff == 1 and current_sign == prev_sign:
                    current_count += 1
                else:
                    current_count = 1
            else:
                current_count = 1
            prev_sign = current_sign
        else:
            current_count = 0
            prev_sign = 0
        
        prev_hour = row['hour']
        consecutive.append(current_count)
    
    group['consecutive_extreme_hours'] = consecutive
    return group

print("  Processing coins...")
funding_with_consecutive = funding_sorted.groupby('coin', group_keys=False).apply(calculate_consecutive_hours)

# Build consecutive hours lookup
consecutive_lookup = {}
for _, row in funding_with_consecutive.iterrows():
    key = (row['hour'], row['coin'])
    consecutive_lookup[key] = row['consecutive_extreme_hours']

def get_consecutive_hours(hour, coin):
    return consecutive_lookup.get((hour, coin), 0)

# =============================================================================
# IDENTIFY ENTRY SIGNALS
# =============================================================================

print("\nIdentifying entry signals...")

# Filter funding to events with price data
extreme_funding = funding[
    (funding['funding_rate'].abs() > FUNDING_THRESHOLD) &
    (funding['hour'].isin(price_hours_set)) &
    (funding['coin'].isin(price_coins))
].copy()

# TREND-FOLLOWING: Direction is SAME as funding sign
# FR < 0 (shorts pay) -> GO SHORT
# FR > 0 (longs pay) -> GO LONG
extreme_funding['direction'] = np.where(extreme_funding['funding_rate'] < 0, 'short', 'long')

# Add consecutive hours
extreme_funding['consecutive_hours'] = extreme_funding.apply(
    lambda row: get_consecutive_hours(row['hour'], row['coin']), axis=1
)

print(f"Total entry signals: {len(extreme_funding):,}")
print(f"  Short signals (FR < -{FUNDING_THRESHOLD*100:.2f}%): {(extreme_funding['direction'] == 'short').sum():,}")
print(f"  Long signals (FR > +{FUNDING_THRESHOLD*100:.2f}%): {(extreme_funding['direction'] == 'long').sum():,}")

# =============================================================================
# BACKTEST FUNCTIONS
# =============================================================================

def backtest_fixed_hold(coin, entry_hour, direction, entry_fr, holding_hours):
    """Strategy 1: Fixed holding period"""
    result = {
        'coin': coin,
        'entry_hour': entry_hour,
        'direction': direction,
        'entry_fr': entry_fr,
        'exit_type': f'fixed_{holding_hours}h',
        'holding_hours': holding_hours,
        'entry_price': np.nan,
        'exit_price': np.nan,
        'exit_hour': None,
        'price_return': np.nan,
        'funding_paid': 0,  # We PAY funding in trend-following
        'trading_fee': 2 * TAKER_FEE,
        'gross_pnl': np.nan,
        'net_pnl': np.nan,
        'valid': False
    }
    
    if coin not in price_pivot.columns:
        return result
    if entry_hour not in price_pivot.index:
        return result
    
    entry_price = price_pivot.loc[entry_hour, coin]
    if pd.isna(entry_price):
        return result
    
    result['entry_price'] = entry_price
    
    exit_hour = entry_hour + timedelta(hours=holding_hours)
    if exit_hour not in price_pivot.index:
        return result
    
    exit_price = price_pivot.loc[exit_hour, coin]
    if pd.isna(exit_price):
        return result
    
    result['exit_price'] = exit_price
    result['exit_hour'] = exit_hour
    
    # Price return (trend-following direction)
    price_return = (exit_price - entry_price) / entry_price
    if direction == 'short':
        price_return = -price_return
    result['price_return'] = price_return
    
    # Funding PAID (we're going with the crowd, so we pay)
    funding_paid = 0
    for h in range(holding_hours):
        funding_hour = entry_hour + timedelta(hours=h)
        fr = get_funding_rate(funding_hour, coin)
        
        # Trend-following: we PAY the funding
        # If short and FR < 0, we PAY |FR|
        # If long and FR > 0, we PAY FR
        if direction == 'short' and fr < 0:
            funding_paid += abs(fr)
        elif direction == 'long' and fr > 0:
            funding_paid += fr
        elif direction == 'short' and fr > 0:
            funding_paid -= fr  # We receive
        elif direction == 'long' and fr < 0:
            funding_paid -= abs(fr)  # We receive
    
    result['funding_paid'] = funding_paid
    result['gross_pnl'] = result['price_return'] - result['funding_paid']
    result['net_pnl'] = result['gross_pnl'] - result['trading_fee']
    result['valid'] = True
    
    return result


def backtest_until_normalized(coin, entry_hour, direction, entry_fr, exit_threshold=0.0005):
    """Strategy 2: Exit when funding normalizes (|FR| < exit_threshold)"""
    result = {
        'coin': coin,
        'entry_hour': entry_hour,
        'direction': direction,
        'entry_fr': entry_fr,
        'exit_type': f'normalized_{exit_threshold*100:.2f}%',
        'holding_hours': 0,
        'entry_price': np.nan,
        'exit_price': np.nan,
        'exit_hour': None,
        'exit_fr': np.nan,
        'price_return': np.nan,
        'funding_paid': 0,
        'trading_fee': 2 * TAKER_FEE,
        'gross_pnl': np.nan,
        'net_pnl': np.nan,
        'valid': False
    }
    
    if coin not in price_pivot.columns:
        return result
    if entry_hour not in price_pivot.index:
        return result
    
    entry_price = price_pivot.loc[entry_hour, coin]
    if pd.isna(entry_price):
        return result
    
    result['entry_price'] = entry_price
    
    # Find exit hour when funding normalizes
    funding_paid = 0
    for h in range(1, MAX_HOLD_HOURS + 1):
        check_hour = entry_hour + timedelta(hours=h)
        
        if check_hour not in price_pivot.index:
            break
        
        fr = get_funding_rate(check_hour, coin)
        
        # Accumulate funding paid
        prev_hour = entry_hour + timedelta(hours=h-1)
        prev_fr = get_funding_rate(prev_hour, coin)
        if direction == 'short' and prev_fr < 0:
            funding_paid += abs(prev_fr)
        elif direction == 'long' and prev_fr > 0:
            funding_paid += prev_fr
        elif direction == 'short' and prev_fr > 0:
            funding_paid -= prev_fr
        elif direction == 'long' and prev_fr < 0:
            funding_paid -= abs(prev_fr)
        
        # Check exit condition
        if abs(fr) < exit_threshold:
            exit_price = price_pivot.loc[check_hour, coin]
            if pd.isna(exit_price):
                continue
            
            result['exit_price'] = exit_price
            result['exit_hour'] = check_hour
            result['exit_fr'] = fr
            result['holding_hours'] = h
            
            price_return = (exit_price - entry_price) / entry_price
            if direction == 'short':
                price_return = -price_return
            
            result['price_return'] = price_return
            result['funding_paid'] = funding_paid
            result['gross_pnl'] = result['price_return'] - result['funding_paid']
            result['net_pnl'] = result['gross_pnl'] - result['trading_fee']
            result['valid'] = True
            return result
    
    return result


def backtest_until_reversed(coin, entry_hour, direction, entry_fr):
    """Strategy 3: Exit when funding reverses sign"""
    result = {
        'coin': coin,
        'entry_hour': entry_hour,
        'direction': direction,
        'entry_fr': entry_fr,
        'exit_type': 'reversed',
        'holding_hours': 0,
        'entry_price': np.nan,
        'exit_price': np.nan,
        'exit_hour': None,
        'exit_fr': np.nan,
        'price_return': np.nan,
        'funding_paid': 0,
        'trading_fee': 2 * TAKER_FEE,
        'gross_pnl': np.nan,
        'net_pnl': np.nan,
        'valid': False
    }
    
    if coin not in price_pivot.columns:
        return result
    if entry_hour not in price_pivot.index:
        return result
    
    entry_price = price_pivot.loc[entry_hour, coin]
    if pd.isna(entry_price):
        return result
    
    result['entry_price'] = entry_price
    entry_sign = np.sign(entry_fr)
    
    funding_paid = 0
    for h in range(1, MAX_HOLD_HOURS + 1):
        check_hour = entry_hour + timedelta(hours=h)
        
        if check_hour not in price_pivot.index:
            break
        
        fr = get_funding_rate(check_hour, coin)
        
        # Accumulate funding
        prev_hour = entry_hour + timedelta(hours=h-1)
        prev_fr = get_funding_rate(prev_hour, coin)
        if direction == 'short' and prev_fr < 0:
            funding_paid += abs(prev_fr)
        elif direction == 'long' and prev_fr > 0:
            funding_paid += prev_fr
        elif direction == 'short' and prev_fr > 0:
            funding_paid -= prev_fr
        elif direction == 'long' and prev_fr < 0:
            funding_paid -= abs(prev_fr)
        
        # Check if sign reversed
        if np.sign(fr) != entry_sign and np.sign(fr) != 0:
            exit_price = price_pivot.loc[check_hour, coin]
            if pd.isna(exit_price):
                continue
            
            result['exit_price'] = exit_price
            result['exit_hour'] = check_hour
            result['exit_fr'] = fr
            result['holding_hours'] = h
            
            price_return = (exit_price - entry_price) / entry_price
            if direction == 'short':
                price_return = -price_return
            
            result['price_return'] = price_return
            result['funding_paid'] = funding_paid
            result['gross_pnl'] = result['price_return'] - result['funding_paid']
            result['net_pnl'] = result['gross_pnl'] - result['trading_fee']
            result['valid'] = True
            return result
    
    return result


def backtest_until_fr_drops(coin, entry_hour, direction, entry_fr, drop_pct=0.5):
    """Strategy 4: Exit when funding drops by X% from entry"""
    result = {
        'coin': coin,
        'entry_hour': entry_hour,
        'direction': direction,
        'entry_fr': entry_fr,
        'exit_type': f'fr_drop_{int(drop_pct*100)}%',
        'holding_hours': 0,
        'entry_price': np.nan,
        'exit_price': np.nan,
        'exit_hour': None,
        'exit_fr': np.nan,
        'price_return': np.nan,
        'funding_paid': 0,
        'trading_fee': 2 * TAKER_FEE,
        'gross_pnl': np.nan,
        'net_pnl': np.nan,
        'valid': False
    }
    
    if coin not in price_pivot.columns:
        return result
    if entry_hour not in price_pivot.index:
        return result
    
    entry_price = price_pivot.loc[entry_hour, coin]
    if pd.isna(entry_price):
        return result
    
    result['entry_price'] = entry_price
    entry_fr_abs = abs(entry_fr)
    target_fr = entry_fr_abs * (1 - drop_pct)
    
    funding_paid = 0
    for h in range(1, MAX_HOLD_HOURS + 1):
        check_hour = entry_hour + timedelta(hours=h)
        
        if check_hour not in price_pivot.index:
            break
        
        fr = get_funding_rate(check_hour, coin)
        
        # Accumulate funding
        prev_hour = entry_hour + timedelta(hours=h-1)
        prev_fr = get_funding_rate(prev_hour, coin)
        if direction == 'short' and prev_fr < 0:
            funding_paid += abs(prev_fr)
        elif direction == 'long' and prev_fr > 0:
            funding_paid += prev_fr
        elif direction == 'short' and prev_fr > 0:
            funding_paid -= prev_fr
        elif direction == 'long' and prev_fr < 0:
            funding_paid -= abs(prev_fr)
        
        # Check if FR dropped enough
        if abs(fr) < target_fr:
            exit_price = price_pivot.loc[check_hour, coin]
            if pd.isna(exit_price):
                continue
            
            result['exit_price'] = exit_price
            result['exit_hour'] = check_hour
            result['exit_fr'] = fr
            result['holding_hours'] = h
            
            price_return = (exit_price - entry_price) / entry_price
            if direction == 'short':
                price_return = -price_return
            
            result['price_return'] = price_return
            result['funding_paid'] = funding_paid
            result['gross_pnl'] = result['price_return'] - result['funding_paid']
            result['net_pnl'] = result['gross_pnl'] - result['trading_fee']
            result['valid'] = True
            return result
    
    return result

# =============================================================================
# RUN BACKTESTS
# =============================================================================

print("\n" + "="*100)
print("RUNNING BACKTESTS")
print("="*100)

# Store all results
all_results = {}

# Strategy 1: Fixed holding periods
for hold_hours in [4, 8, 12, 24]:
    print(f"\nProcessing fixed {hold_hours}h hold...")
    trades = []
    for i, (_, row) in enumerate(extreme_funding.iterrows()):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(extreme_funding)}...")
        result = backtest_fixed_hold(
            row['coin'], row['hour'], row['direction'], 
            row['funding_rate'], hold_hours
        )
        result['consecutive_hours'] = row['consecutive_hours']
        result['fr_magnitude'] = abs(row['funding_rate'])
        trades.append(result)
    
    df = pd.DataFrame(trades)
    all_results[f'fixed_{hold_hours}h'] = df[df['valid'] == True]

# Strategy 2: Until normalized
for exit_thresh in [0.0005, 0.0003, 0.0001]:  # 0.05%, 0.03%, 0.01%
    print(f"\nProcessing until normalized (|FR| < {exit_thresh*100:.2f}%)...")
    trades = []
    for i, (_, row) in enumerate(extreme_funding.iterrows()):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(extreme_funding)}...")
        result = backtest_until_normalized(
            row['coin'], row['hour'], row['direction'],
            row['funding_rate'], exit_thresh
        )
        result['consecutive_hours'] = row['consecutive_hours']
        result['fr_magnitude'] = abs(row['funding_rate'])
        trades.append(result)
    
    df = pd.DataFrame(trades)
    all_results[f'normalized_{exit_thresh*100:.2f}%'] = df[df['valid'] == True]

# Strategy 3: Until reversed
print("\nProcessing until reversed...")
trades = []
for i, (_, row) in enumerate(extreme_funding.iterrows()):
    if i % 500 == 0 and i > 0:
        print(f"  {i}/{len(extreme_funding)}...")
    result = backtest_until_reversed(
        row['coin'], row['hour'], row['direction'],
        row['funding_rate']
    )
    result['consecutive_hours'] = row['consecutive_hours']
    result['fr_magnitude'] = abs(row['funding_rate'])
    trades.append(result)

df = pd.DataFrame(trades)
all_results['reversed'] = df[df['valid'] == True]

# Strategy 4: FR drops by X%
for drop_pct in [0.3, 0.5, 0.7]:
    print(f"\nProcessing FR drop {int(drop_pct*100)}%...")
    trades = []
    for i, (_, row) in enumerate(extreme_funding.iterrows()):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(extreme_funding)}...")
        result = backtest_until_fr_drops(
            row['coin'], row['hour'], row['direction'],
            row['funding_rate'], drop_pct
        )
        result['consecutive_hours'] = row['consecutive_hours']
        result['fr_magnitude'] = abs(row['funding_rate'])
        trades.append(result)
    
    df = pd.DataFrame(trades)
    all_results[f'fr_drop_{int(drop_pct*100)}%'] = df[df['valid'] == True]

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

print("\n" + "="*100)
print("STRATEGY COMPARISON (All Trades)")
print("="*100)

print(f"\n{'Strategy':<25} {'N':<8} {'Avg Hold':<10} {'Avg Net PnL':<14} {'Sharpe':<10} {'Win Rate':<10} {'Total PnL':<12}")
print("-"*100)

for name, df in sorted(all_results.items()):
    if len(df) > 0:
        avg_hold = df['holding_hours'].mean()
        avg_pnl = df['net_pnl'].mean()
        std_pnl = df['net_pnl'].std()
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = (df['net_pnl'] > 0).mean()
        total = df['net_pnl'].sum()
        print(f"{name:<25} {len(df):<8} {avg_hold:>8.1f}h  {avg_pnl*100:>11.4f}%  {sharpe:>8.4f}  {win_rate*100:>8.1f}%  {total*100:>10.2f}%")

# =============================================================================
# FILTER: CONSECUTIVE HOURS >= 2
# =============================================================================

print("\n" + "="*100)
print("FILTERED: Consecutive Extreme Hours >= 2 (Not First Hour)")
print("="*100)

print(f"\n{'Strategy':<25} {'N':<8} {'Avg Hold':<10} {'Avg Net PnL':<14} {'Sharpe':<10} {'Win Rate':<10} {'Total PnL':<12}")
print("-"*100)

for name, df in sorted(all_results.items()):
    filtered = df[df['consecutive_hours'] >= 2]
    if len(filtered) > 0:
        avg_hold = filtered['holding_hours'].mean()
        avg_pnl = filtered['net_pnl'].mean()
        std_pnl = filtered['net_pnl'].std()
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = (filtered['net_pnl'] > 0).mean()
        total = filtered['net_pnl'].sum()
        print(f"{name:<25} {len(filtered):<8} {avg_hold:>8.1f}h  {avg_pnl*100:>11.4f}%  {sharpe:>8.4f}  {win_rate*100:>8.1f}%  {total*100:>10.2f}%")

# =============================================================================
# FILTER: CONSECUTIVE HOURS >= 4 (Mature Trend)
# =============================================================================

print("\n" + "="*100)
print("FILTERED: Consecutive Extreme Hours >= 4 (Mature Trend)")
print("="*100)

print(f"\n{'Strategy':<25} {'N':<8} {'Avg Hold':<10} {'Avg Net PnL':<14} {'Sharpe':<10} {'Win Rate':<10} {'Total PnL':<12}")
print("-"*100)

for name, df in sorted(all_results.items()):
    filtered = df[df['consecutive_hours'] >= 4]
    if len(filtered) > 0:
        avg_hold = filtered['holding_hours'].mean()
        avg_pnl = filtered['net_pnl'].mean()
        std_pnl = filtered['net_pnl'].std()
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = (filtered['net_pnl'] > 0).mean()
        total = filtered['net_pnl'].sum()
        print(f"{name:<25} {len(filtered):<8} {avg_hold:>8.1f}h  {avg_pnl*100:>11.4f}%  {sharpe:>8.4f}  {win_rate*100:>8.1f}%  {total*100:>10.2f}%")

# =============================================================================
# FILTER: LOW FR MAGNITUDE (0.10-0.20%)
# =============================================================================

print("\n" + "="*100)
print("FILTERED: Low FR Magnitude (0.10-0.20%)")
print("="*100)

print(f"\n{'Strategy':<25} {'N':<8} {'Avg Hold':<10} {'Avg Net PnL':<14} {'Sharpe':<10} {'Win Rate':<10} {'Total PnL':<12}")
print("-"*100)

for name, df in sorted(all_results.items()):
    filtered = df[df['fr_magnitude'].between(0.001, 0.002)]
    if len(filtered) > 0:
        avg_hold = filtered['holding_hours'].mean()
        avg_pnl = filtered['net_pnl'].mean()
        std_pnl = filtered['net_pnl'].std()
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = (filtered['net_pnl'] > 0).mean()
        total = filtered['net_pnl'].sum()
        print(f"{name:<25} {len(filtered):<8} {avg_hold:>8.1f}h  {avg_pnl*100:>11.4f}%  {sharpe:>8.4f}  {win_rate*100:>8.1f}%  {total*100:>10.2f}%")

# =============================================================================
# FILTER: HIGH FR MAGNITUDE (> 0.30%)
# =============================================================================

print("\n" + "="*100)
print("FILTERED: High FR Magnitude (> 0.30%)")
print("="*100)

print(f"\n{'Strategy':<25} {'N':<8} {'Avg Hold':<10} {'Avg Net PnL':<14} {'Sharpe':<10} {'Win Rate':<10} {'Total PnL':<12}")
print("-"*100)

for name, df in sorted(all_results.items()):
    filtered = df[df['fr_magnitude'] > 0.003]
    if len(filtered) > 0:
        avg_hold = filtered['holding_hours'].mean()
        avg_pnl = filtered['net_pnl'].mean()
        std_pnl = filtered['net_pnl'].std()
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = (filtered['net_pnl'] > 0).mean()
        total = filtered['net_pnl'].sum()
        print(f"{name:<25} {len(filtered):<8} {avg_hold:>8.1f}h  {avg_pnl*100:>11.4f}%  {sharpe:>8.4f}  {win_rate*100:>8.1f}%  {total*100:>10.2f}%")

# =============================================================================
# COMBINED FILTER: Consec >= 2 AND Low FR
# =============================================================================

print("\n" + "="*100)
print("COMBINED FILTER: Consecutive >= 2 AND Low FR (0.10-0.20%)")
print("="*100)

print(f"\n{'Strategy':<25} {'N':<8} {'Avg Hold':<10} {'Avg Net PnL':<14} {'Sharpe':<10} {'Win Rate':<10} {'Total PnL':<12}")
print("-"*100)

for name, df in sorted(all_results.items()):
    filtered = df[(df['consecutive_hours'] >= 2) & (df['fr_magnitude'].between(0.001, 0.002))]
    if len(filtered) > 0:
        avg_hold = filtered['holding_hours'].mean()
        avg_pnl = filtered['net_pnl'].mean()
        std_pnl = filtered['net_pnl'].std()
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = (filtered['net_pnl'] > 0).mean()
        total = filtered['net_pnl'].sum()
        print(f"{name:<25} {len(filtered):<8} {avg_hold:>8.1f}h  {avg_pnl*100:>11.4f}%  {sharpe:>8.4f}  {win_rate*100:>8.1f}%  {total*100:>10.2f}%")

# =============================================================================
# LONG vs SHORT BREAKDOWN (Best Strategy)
# =============================================================================

print("\n" + "="*100)
print("LONG vs SHORT BREAKDOWN (Fixed 8h Strategy)")
print("="*100)

best_df = all_results['fixed_8h']

for direction in ['long', 'short']:
    dir_df = best_df[best_df['direction'] == direction]
    if len(dir_df) > 0:
        print(f"\n### {direction.upper()} Trades ###")
        print(f"N Trades: {len(dir_df)}")
        print(f"Avg Net PnL: {dir_df['net_pnl'].mean()*100:.4f}%")
        print(f"Sharpe: {dir_df['net_pnl'].mean() / dir_df['net_pnl'].std():.4f}")
        print(f"Win Rate: {(dir_df['net_pnl'] > 0).mean()*100:.1f}%")
        print(f"Total PnL: {dir_df['net_pnl'].sum()*100:.2f}%")

# =============================================================================
# P&L BREAKDOWN
# =============================================================================

print("\n" + "="*100)
print("P&L BREAKDOWN (Fixed 8h Strategy)")
print("="*100)

df = all_results['fixed_8h']
print(f"\nAvg Price Return: {df['price_return'].mean()*100:.4f}%")
print(f"Avg Funding Paid: {df['funding_paid'].mean()*100:.4f}%")
print(f"Avg Trading Fee: {df['trading_fee'].mean()*100:.4f}%")
print(f"Avg Net PnL: {df['net_pnl'].mean()*100:.4f}%")

# =============================================================================
# TOP/BOTTOM COINS
# =============================================================================

print("\n" + "="*100)
print("COIN ANALYSIS (Fixed 8h Strategy)")
print("="*100)

coin_stats = best_df.groupby('coin').agg({
    'net_pnl': ['count', 'mean', 'std', lambda x: (x > 0).mean(), 'sum']
}).round(6)
coin_stats.columns = ['n_trades', 'avg_pnl', 'std_pnl', 'win_rate', 'total_pnl']
coin_stats = coin_stats[coin_stats['n_trades'] >= 10]
coin_stats['sharpe'] = coin_stats['avg_pnl'] / coin_stats['std_pnl']

print("\n### Top 15 Coins by Avg PnL ###")
top_coins = coin_stats.sort_values('avg_pnl', ascending=False).head(15)
print(f"{'Coin':<12} {'N':<8} {'Avg PnL':<12} {'Sharpe':<10} {'Win Rate':<10} {'Total PnL':<12}")
print("-"*70)
for coin, row in top_coins.iterrows():
    print(f"{coin:<12} {int(row['n_trades']):<8} {row['avg_pnl']*100:>9.4f}%  {row['sharpe']:>8.4f}  {row['win_rate']*100:>8.1f}%  {row['total_pnl']*100:>10.2f}%")

print("\n### Bottom 15 Coins by Avg PnL ###")
bottom_coins = coin_stats.sort_values('avg_pnl', ascending=True).head(15)
print(f"{'Coin':<12} {'N':<8} {'Avg PnL':<12} {'Sharpe':<10} {'Win Rate':<10} {'Total PnL':<12}")
print("-"*70)
for coin, row in bottom_coins.iterrows():
    print(f"{coin:<12} {int(row['n_trades']):<8} {row['avg_pnl']*100:>9.4f}%  {row['sharpe']:>8.4f}  {row['win_rate']*100:>8.1f}%  {row['total_pnl']*100:>10.2f}%")

# =============================================================================
# FILTERED BY TOP COINS
# =============================================================================

print("\n" + "="*100)
print("FILTERED: TOP COINS ONLY (Sharpe > 0)")
print("="*100)

profitable_coins = coin_stats[coin_stats['sharpe'] > 0].index.tolist()
print(f"Profitable coins: {profitable_coins}")

print(f"\n{'Strategy':<25} {'N':<8} {'Avg Hold':<10} {'Avg Net PnL':<14} {'Sharpe':<10} {'Win Rate':<10} {'Total PnL':<12}")
print("-"*100)

for name, df in sorted(all_results.items()):
    filtered = df[df['coin'].isin(profitable_coins)]
    if len(filtered) > 0:
        avg_hold = filtered['holding_hours'].mean()
        avg_pnl = filtered['net_pnl'].mean()
        std_pnl = filtered['net_pnl'].std()
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = (filtered['net_pnl'] > 0).mean()
        total = filtered['net_pnl'].sum()
        print(f"{name:<25} {len(filtered):<8} {avg_hold:>8.1f}h  {avg_pnl*100:>11.4f}%  {sharpe:>8.4f}  {win_rate*100:>8.1f}%  {total*100:>10.2f}%")

# =============================================================================
# SAVE BEST RESULTS
# =============================================================================

# Save the fixed 8h results
all_results['fixed_8h'].to_csv('trend_following_8h_results.csv', index=False)
print("\nSaved: trend_following_8h_results.csv")

# Save dynamic exit results
all_results['normalized_0.05%'].to_csv('trend_following_normalized_results.csv', index=False)
print("Saved: trend_following_normalized_results.csv")

print("\n" + "="*100)
print("BACKTEST COMPLETE")
print("="*100)
