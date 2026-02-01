"""
Simple parameter sweep - just tests fixed holding periods without early exit
"""

import pandas as pd
import numpy as np
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

print(f'Funding: {len(funding):,} | Price: {len(price):,}')

# Merge
print('Merging...')
merged = pd.merge(
    funding[['hour', 'coin', 'funding_rate']],
    price[['hour', 'coin', 'price']],
    on=['hour', 'coin'],
    how='inner'
)
merged = merged.sort_values(['coin', 'hour']).reset_index(drop=True)
print(f'Merged: {len(merged):,}')

# Calculate forward price returns
print('Calculating forward returns...')
for h in [8, 24, 48, 72]:
    merged[f'ret_{h}h'] = merged.groupby('coin')['price'].pct_change(h).shift(-h)

# Sum forward funding rates
print('Calculating forward funding...')
for h in [8, 24, 48, 72]:
    # Rolling sum of next h funding rates
    merged[f'fr_sum_{h}h'] = merged.groupby('coin')['funding_rate'].rolling(h, min_periods=1).sum().shift(-h+1).reset_index(drop=True)

print('\n' + '='*90)
print('SHORT STRATEGY: Entry when FR < threshold, hold for fixed period')
print('='*90)

results = []

for hold in [8, 24, 48, 72]:
    ret_col = f'ret_{hold}h'
    fr_col = f'fr_sum_{hold}h'
    
    for entry_thresh in [-0.0003, -0.0005, -0.001, -0.002, -0.003, -0.005]:
        entries = merged[merged['funding_rate'] < entry_thresh].copy()
        entries = entries.dropna(subset=[ret_col, fr_col])
        
        if len(entries) < 30:
            continue
        
        # SHORT PnL: -price_return - funding_paid - fees
        # For short: negative FR means we pay, positive FR means we receive
        # fr_sum is sum of future FRs. If avg FR is negative, we pay more
        entries['funding_cost'] = entries[fr_col].apply(lambda x: abs(x) if x < 0 else -x)
        entries['net_pnl'] = -entries[ret_col] - entries['funding_cost'] - 2 * TAKER_FEE
        
        avg = entries['net_pnl'].mean()
        std = entries['net_pnl'].std()
        sharpe = avg / std if std > 0 else 0
        win = (entries['net_pnl'] > 0).mean()
        total = entries['net_pnl'].sum()
        
        results.append({
            'entry': entry_thresh,
            'hold': hold,
            'n': len(entries),
            'avg': avg,
            'sharpe': sharpe,
            'win': win,
            'total': total
        })

results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'Entry':<12} {'Hold':<6} {'N':>7} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10}")
print('-'*70)
for r in results[:20]:
    print(f"{r['entry']*100:>+.2f}%      {r['hold']:>3}h   {r['n']:>7} {r['avg']*100:>+9.2f}% {r['sharpe']:>8.2f} {r['win']*100:>7.1f}% {r['total']*100:>+9.0f}%")

print('\n' + '='*90)
print('LONG STRATEGY: Entry when FR > threshold, hold for fixed period')
print('='*90)

long_results = []

for hold in [8, 24, 48, 72]:
    ret_col = f'ret_{hold}h'
    fr_col = f'fr_sum_{hold}h'
    
    for entry_thresh in [0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005]:
        entries = merged[merged['funding_rate'] > entry_thresh].copy()
        entries = entries.dropna(subset=[ret_col, fr_col])
        
        if len(entries) < 30:
            continue
        
        # LONG PnL: +price_return + funding_earned - fees
        # For long: positive FR means we receive, negative FR means we pay
        entries['funding_earned'] = entries[fr_col].apply(lambda x: x if x > 0 else -abs(x))
        entries['net_pnl'] = entries[ret_col] + entries['funding_earned'] - 2 * TAKER_FEE
        
        avg = entries['net_pnl'].mean()
        std = entries['net_pnl'].std()
        sharpe = avg / std if std > 0 else 0
        win = (entries['net_pnl'] > 0).mean()
        total = entries['net_pnl'].sum()
        
        long_results.append({
            'entry': entry_thresh,
            'hold': hold,
            'n': len(entries),
            'avg': avg,
            'sharpe': sharpe,
            'win': win,
            'total': total
        })

long_results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'Entry':<12} {'Hold':<6} {'N':>7} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10}")
print('-'*70)
for r in long_results[:20]:
    print(f"+{r['entry']*100:.2f}%      {r['hold']:>3}h   {r['n']:>7} {r['avg']*100:>+9.2f}% {r['sharpe']:>8.2f} {r['win']*100:>7.1f}% {r['total']*100:>+9.0f}%")

print('\n' + '='*90)
print('SUMMARY')
print('='*90)

if results:
    best = results[0]
    print(f"\nBest SHORT: Entry<{best['entry']*100:.2f}%, Hold={best['hold']}h")
    print(f"  -> N={best['n']}, Avg={best['avg']*100:+.2f}%, Sharpe={best['sharpe']:.2f}, Win={best['win']*100:.1f}%")

if long_results:
    best = long_results[0]
    print(f"\nBest LONG: Entry>{best['entry']*100:.2f}%, Hold={best['hold']}h")
    print(f"  -> N={best['n']}, Avg={best['avg']*100:+.2f}%, Sharpe={best['sharpe']:.2f}, Win={best['win']*100:.1f}%")

# Benchmark
print('\n' + '='*90)
print('BENCHMARK: Random entry')
print('='*90)
rand = merged.dropna(subset=['ret_24h'])
print(f"Random LONG 24h:  Avg={rand['ret_24h'].mean()*100:+.3f}%, Win={(rand['ret_24h']>0).mean()*100:.1f}%")
print(f"Random SHORT 24h: Avg={-rand['ret_24h'].mean()*100:+.3f}%, Win={(rand['ret_24h']<0).mean()*100:.1f}%")
