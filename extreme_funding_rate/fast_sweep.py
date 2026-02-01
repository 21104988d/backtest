"""
Optimized parameter sweep for SHORT-only strategy with 995 days of data
Uses vectorized operations for speed
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

print(f'Funding: {len(funding):,} records')
print(f'Price: {len(price):,} records')

# Merge funding and price
print('Merging data...')
merged = pd.merge(
    funding[['hour', 'coin', 'funding_rate']],
    price[['hour', 'coin', 'price']],
    on=['hour', 'coin'],
    how='inner'
)
merged = merged.sort_values(['coin', 'hour']).reset_index(drop=True)
print(f'Merged: {len(merged):,} records')

# Calculate forward returns for different holding periods
print('Calculating forward returns...')

def calc_forward_stats(df, hours):
    """Calculate price return and funding paid for holding period"""
    df = df.copy()
    df['future_price'] = df.groupby('coin')['price'].shift(-hours)
    df['price_return'] = (df['future_price'] - df['price']) / df['price']
    
    # Calculate cumulative funding for short position
    funding_cols = []
    for h in range(hours):
        col = f'fr_{h}'
        df[col] = df.groupby('coin')['funding_rate'].shift(-h)
        funding_cols.append(col)
    
    # Sum funding (for short: negative FR = we pay, positive FR = we receive)
    df['funding_paid'] = 0
    for col in funding_cols:
        df['funding_paid'] += df[col].fillna(0).apply(lambda x: abs(x) if x < 0 else -x)
    
    # Clean up
    for col in funding_cols:
        del df[col]
    
    return df

# Calculate for different holding periods
print('\nCalculating returns for 24h, 48h, 72h holding periods...')
merged_24 = calc_forward_stats(merged, 24)
merged_48 = calc_forward_stats(merged, 48)
merged_72 = calc_forward_stats(merged, 72)

print('='*90)
print('SHORT STRATEGY PARAMETER SWEEP')
print('='*90)

results = []

for hold_df, max_hold in [(merged_24, 24), (merged_48, 48), (merged_72, 72)]:
    for entry_thresh in [-0.0005, -0.001, -0.002, -0.003, -0.005]:
        # Filter for entry condition
        entries = hold_df[hold_df['funding_rate'] < entry_thresh].copy()
        entries = entries.dropna(subset=['price_return', 'funding_paid'])
        
        if len(entries) < 20:
            continue
        
        # SHORT PnL = -price_return - funding_paid - fees
        entries['net_pnl'] = -entries['price_return'] - entries['funding_paid'] - 2 * TAKER_FEE
        
        avg_pnl = entries['net_pnl'].mean()
        std_pnl = entries['net_pnl'].std()
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = (entries['net_pnl'] > 0).mean()
        total_pnl = entries['net_pnl'].sum()
        
        results.append({
            'entry': f'{entry_thresh*100:.2f}%',
            'max_hold': max_hold,
            'n': len(entries),
            'avg': avg_pnl,
            'sharpe': sharpe,
            'win': win_rate,
            'total': total_pnl
        })

# Sort by Sharpe
results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'Entry':<10} {'Hold':<6} {'N':>8} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total PnL':>10}")
print('-'*70)
for r in results[:20]:
    print(f"{r['entry']:<10} {r['max_hold']:<6} {r['n']:>8} {r['avg']*100:>+9.2f}% {r['sharpe']:>8.2f} {r['win']*100:>7.1f}% {r['total']*100:>+9.0f}%")

print('\n' + '='*90)
print('LONG STRATEGY PARAMETER SWEEP')
print('='*90)

long_results = []

for hold_df, max_hold in [(merged_24, 24), (merged_48, 48), (merged_72, 72)]:
    for entry_thresh in [0.0005, 0.001, 0.002, 0.003, 0.005]:
        # Filter for entry condition
        entries = hold_df[hold_df['funding_rate'] > entry_thresh].copy()
        entries = entries.dropna(subset=['price_return', 'funding_paid'])
        
        if len(entries) < 20:
            continue
        
        # LONG PnL = +price_return + funding_earned - fees
        # For long: positive FR = we receive, negative FR = we pay
        entries['funding_earned'] = -entries['funding_paid']  # Flip the sign
        entries['net_pnl'] = entries['price_return'] + entries['funding_earned'] - 2 * TAKER_FEE
        
        avg_pnl = entries['net_pnl'].mean()
        std_pnl = entries['net_pnl'].std()
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = (entries['net_pnl'] > 0).mean()
        total_pnl = entries['net_pnl'].sum()
        
        long_results.append({
            'entry': f'+{entry_thresh*100:.2f}%',
            'max_hold': max_hold,
            'n': len(entries),
            'avg': avg_pnl,
            'sharpe': sharpe,
            'win': win_rate,
            'total': total_pnl
        })

long_results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'Entry':<10} {'Hold':<6} {'N':>8} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total PnL':>10}")
print('-'*70)
for r in long_results[:15]:
    print(f"{r['entry']:<10} {r['max_hold']:<6} {r['n']:>8} {r['avg']*100:>+9.2f}% {r['sharpe']:>8.2f} {r['win']*100:>7.1f}% {r['total']*100:>+9.0f}%")

print('\n' + '='*90)
print('SUMMARY')
print('='*90)

best_short = results[0] if results else None
best_long = long_results[0] if long_results else None

if best_short:
    print(f"\nBest SHORT: Entry<{best_short['entry']}, Hold={best_short['max_hold']}h")
    print(f"  N={best_short['n']}, Sharpe={best_short['sharpe']:.2f}, Win={best_short['win']*100:.1f}%, Total={best_short['total']*100:.0f}%")

if best_long:
    print(f"\nBest LONG: Entry>{best_long['entry']}, Hold={best_long['max_hold']}h")
    print(f"  N={best_long['n']}, Sharpe={best_long['sharpe']:.2f}, Win={best_long['win']*100:.1f}%, Total={best_long['total']*100:.0f}%")

# Compare with random
print('\n' + '='*90)
print('BENCHMARK: Random entry comparison')
print('='*90)

random_short_24 = merged_24.dropna(subset=['price_return', 'funding_paid']).copy()
random_short_24['net_pnl'] = -random_short_24['price_return'] - random_short_24['funding_paid'] - 2 * TAKER_FEE
print(f"Random SHORT 24h: Avg={random_short_24['net_pnl'].mean()*100:+.2f}%, Win={100*(random_short_24['net_pnl']>0).mean():.1f}%")

random_long_24 = merged_24.dropna(subset=['price_return']).copy()
random_long_24['net_pnl'] = random_long_24['price_return'] - 2 * TAKER_FEE
print(f"Random LONG 24h: Avg={random_long_24['net_pnl'].mean()*100:+.2f}%, Win={100*(random_long_24['net_pnl']>0).mean():.1f}%")
