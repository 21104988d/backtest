"""
Ultra-fast Mean Reversion Test using sampling approach
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

TAKER_FEE = 0.00045

print('Loading data...')
funding = pd.read_csv('funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)
funding['hour'] = funding['datetime'].dt.floor('h').dt.tz_localize(None)

price = pd.read_csv('price_history.csv')
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
price['hour'] = price['timestamp'].dt.floor('h').dt.tz_localize(None)

merged = pd.merge(
    funding[['hour', 'coin', 'funding_rate']],
    price[['hour', 'coin', 'price']],
    on=['hour', 'coin'],
    how='inner'
).sort_values(['coin', 'hour']).reset_index(drop=True)

print(f'Merged: {len(merged):,}')

# Pre-compute for each row: price after N hours
print('Pre-computing future prices...')

# Group by coin and compute future prices
for h in [24, 48, 72]:
    merged[f'price_{h}h'] = merged.groupby('coin')['price'].shift(-h)
    merged[f'fr_sum_{h}h'] = merged.groupby('coin')['funding_rate'].transform(
        lambda x: x.rolling(h, min_periods=1).sum().shift(-h+1)
    )

merged = merged.dropna(subset=['price_72h'])
print(f'After dropna: {len(merged):,}')

def analyze_strategy(df, fr_low, fr_high, direction, hold_hours=72):
    """
    Analyze strategy for given FR range
    direction: 'LONG' or 'SHORT'
    """
    if fr_low is not None and fr_high is not None:
        mask = (df['funding_rate'] >= fr_low) & (df['funding_rate'] < fr_high)
    elif fr_low is not None:
        mask = df['funding_rate'] < fr_low
    else:
        mask = df['funding_rate'] > fr_high
    
    subset = df[mask].copy()
    
    if len(subset) == 0:
        return None
    
    # Sample if too many signals (for speed)
    # Take one per coin per day maximum to avoid overlapping
    subset['date'] = subset['hour'].dt.date
    subset = subset.groupby(['coin', 'date']).first().reset_index()
    
    entry_price = subset['price'].values
    exit_price = subset[f'price_{hold_hours}h'].values
    fr_sum = subset[f'fr_sum_{hold_hours}h'].values
    
    if direction == 'LONG':
        price_ret = (exit_price - entry_price) / entry_price
        funding_pnl = fr_sum  # LONG: we receive positive FR
    else:
        price_ret = -(exit_price - entry_price) / entry_price
        funding_pnl = -fr_sum  # SHORT: we pay positive FR
    
    net_pnl = price_ret + funding_pnl - 2 * TAKER_FEE
    
    return {
        'n': len(net_pnl),
        'avg': np.mean(net_pnl),
        'std': np.std(net_pnl),
        'sharpe': np.mean(net_pnl) / np.std(net_pnl) if np.std(net_pnl) > 0 else 0,
        'win_rate': np.mean(net_pnl > 0),
        'total': np.sum(net_pnl),
        'pnls': net_pnl
    }

# =============================================================================
# SIGNAL FREQUENCY CHECK
# =============================================================================
print('\n' + '='*80)
print('SIGNAL FREQUENCY BY FUNDING RATE RANGE')
print('='*80)

ranges = [
    (-999, -0.001, 'FR < -0.10%'),
    (-0.001, -0.0005, '-0.10% <= FR < -0.05%'),
    (-0.0005, -0.0003, '-0.05% <= FR < -0.03%'),
    (-0.0003, 0, '-0.03% <= FR < 0%'),
    (0, 0.0003, '0% <= FR < +0.03%'),
    (0.0003, 0.0005, '+0.03% <= FR < +0.05%'),
    (0.0005, 0.001, '+0.05% <= FR < +0.10%'),
    (0.001, 999, 'FR >= +0.10%'),
]

print(f"\n{'Range':<30} {'Count':>10} {'Pct':>10}")
print('-'*55)
for low, high, label in ranges:
    if low == -999:
        count = (merged['funding_rate'] < high).sum()
    elif high == 999:
        count = (merged['funding_rate'] >= low).sum()
    else:
        count = ((merged['funding_rate'] >= low) & (merged['funding_rate'] < high)).sum()
    pct = count / len(merged) * 100
    print(f"{label:<30} {count:>10,} {pct:>9.2f}%")

# =============================================================================
# MEAN REVERSION TESTS
# =============================================================================
print('\n' + '='*80)
print('MEAN REVERSION: LONG when FR is NEGATIVE (expect bounce)')
print('Hold 72h | One signal per coin per day')
print('='*80)

mr_long_configs = [
    (-0.001, -0.0003, '-0.10% to -0.03%'),
    (-0.0005, -0.0003, '-0.05% to -0.03%'),
    (-0.001, -0.0005, '-0.10% to -0.05%'),
    (-0.0003, 0, '-0.03% to 0%'),
]

print(f"\n{'Range':<25} {'N':>7} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10}")
print('-'*75)

for low, high, label in mr_long_configs:
    result = analyze_strategy(merged, low, high, 'LONG', 72)
    if result and result['n'] >= 10:
        print(f"{label:<25} {result['n']:>7} {result['avg']*100:>+9.2f}% {result['sharpe']:>8.2f} {result['win_rate']*100:>7.1f}% {result['total']*100:>+9.0f}%")
    else:
        print(f"{label:<25} {'<10':>7}")

print('\n' + '='*80)
print('MEAN REVERSION: SHORT when FR is POSITIVE (expect drop)')
print('Hold 72h | One signal per coin per day')
print('='*80)

mr_short_configs = [
    (0.0003, 0.001, '+0.03% to +0.10%'),
    (0.0003, 0.0005, '+0.03% to +0.05%'),
    (0.0005, 0.001, '+0.05% to +0.10%'),
    (0, 0.0003, '0% to +0.03%'),
]

print(f"\n{'Range':<25} {'N':>7} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10}")
print('-'*75)

for low, high, label in mr_short_configs:
    result = analyze_strategy(merged, low, high, 'SHORT', 72)
    if result and result['n'] >= 10:
        print(f"{label:<25} {result['n']:>7} {result['avg']*100:>+9.2f}% {result['sharpe']:>8.2f} {result['win_rate']*100:>7.1f}% {result['total']*100:>+9.0f}%")
    else:
        print(f"{label:<25} {'<10':>7}")

# =============================================================================
# TREND FOLLOWING FOR COMPARISON
# =============================================================================
print('\n' + '='*80)
print('TREND FOLLOWING: For comparison (|FR| > 0.10%)')
print('Hold 72h | One signal per coin per day')
print('='*80)

# SHORT when FR < -0.10%
tf_short = analyze_strategy(merged, None, -0.001, 'SHORT', 72)
if tf_short:
    print(f"\nSHORT when FR < -0.10%:")
    print(f"  N={tf_short['n']}, Avg={tf_short['avg']*100:+.2f}%, Sharpe={tf_short['sharpe']:.2f}, Win={tf_short['win_rate']*100:.1f}%, Total={tf_short['total']*100:+.0f}%")

# LONG when FR > +0.10%
tf_long = analyze_strategy(merged, 0.001, None, 'LONG', 72)
if tf_long:
    print(f"\nLONG when FR > +0.10%:")
    print(f"  N={tf_long['n']}, Avg={tf_long['avg']*100:+.2f}%, Sharpe={tf_long['sharpe']:.2f}, Win={tf_long['win_rate']*100:.1f}%, Total={tf_long['total']*100:+.0f}%")

# =============================================================================
# COMBINED STRATEGIES
# =============================================================================
print('\n' + '='*80)
print('COMBINED: Mean Reversion + Trend Following')
print('='*80)

# Get all trades for combined strategy
all_trades = []

# Mean reversion LONG for -0.10% to -0.03%
mr_long = analyze_strategy(merged, -0.001, -0.0003, 'LONG', 72)
if mr_long:
    all_trades.extend(mr_long['pnls'].tolist())
    print(f"\nMR LONG (-0.10% to -0.03%): {mr_long['n']} trades, {mr_long['avg']*100:+.2f}% avg")

# Mean reversion SHORT for +0.03% to +0.10%
mr_short = analyze_strategy(merged, 0.0003, 0.001, 'SHORT', 72)
if mr_short:
    all_trades.extend(mr_short['pnls'].tolist())
    print(f"MR SHORT (+0.03% to +0.10%): {mr_short['n']} trades, {mr_short['avg']*100:+.2f}% avg")

# Trend following SHORT for FR < -0.10%
if tf_short:
    all_trades.extend(tf_short['pnls'].tolist())
    print(f"TF SHORT (FR < -0.10%): {tf_short['n']} trades, {tf_short['avg']*100:+.2f}% avg")

# Trend following LONG for FR > +0.10%
if tf_long:
    all_trades.extend(tf_long['pnls'].tolist())
    print(f"TF LONG (FR > +0.10%): {tf_long['n']} trades, {tf_long['avg']*100:+.2f}% avg")

if all_trades:
    print(f"\n--- COMBINED PORTFOLIO ---")
    print(f"Total Trades: {len(all_trades)}")
    print(f"Avg PnL: {np.mean(all_trades)*100:+.2f}%")
    print(f"Sharpe: {np.mean(all_trades)/np.std(all_trades):.2f}")
    print(f"Win Rate: {np.mean(np.array(all_trades) > 0)*100:.1f}%")
    print(f"Total PnL: {np.sum(all_trades)*100:+.0f}%")

# =============================================================================
# OPPOSITE TEST: What if MR is wrong direction?
# =============================================================================
print('\n' + '='*80)
print('OPPOSITE TEST: What if we flip mean reversion?')
print('Maybe trend following works even for lower FR?')
print('='*80)

# Try trend following for lower FR
tf_short_low = analyze_strategy(merged, -0.001, -0.0003, 'SHORT', 72)  # SHORT when FR negative
tf_long_low = analyze_strategy(merged, 0.0003, 0.001, 'LONG', 72)  # LONG when FR positive

print(f"\nTF SHORT (-0.10% to -0.03%): {tf_short_low['n']} trades, {tf_short_low['avg']*100:+.2f}% avg, {tf_short_low['win_rate']*100:.1f}% win")
print(f"TF LONG (+0.03% to +0.10%): {tf_long_low['n']} trades, {tf_long_low['avg']*100:+.2f}% avg, {tf_long_low['win_rate']*100:.1f}% win")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print('\n' + '='*80)
print('SUMMARY: Mean Reversion vs Trend Following')
print('='*80)
print('''
For each FR range, which direction works better?

FR Range            | Mean Reversion      | Trend Following
--------------------|---------------------|--------------------
-0.10% to -0.03%    | LONG (expect bounce)| SHORT (go with crowd)
+0.03% to +0.10%    | SHORT (expect drop) | LONG (go with crowd)
FR < -0.10%         | N/A (extreme)       | SHORT (WORKS)
FR > +0.10%         | N/A (extreme)       | LONG (WORKS)
''')
