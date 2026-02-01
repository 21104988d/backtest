"""
Trend Following with REALISTIC Funding Rate Thresholds

Hourly FR to APY conversion:
- 0.001%/h = 8.76% APY
- 0.002%/h = 17.5% APY  
- 0.003%/h = 26.3% APY
- 0.005%/h = 43.8% APY
- 0.01%/h  = 87.6% APY
- 0.02%/h  = 175% APY (very rare)
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

merged = pd.merge(
    funding[['hour', 'coin', 'funding_rate']],
    price[['hour', 'coin', 'price']],
    on=['hour', 'coin'],
    how='inner'
).sort_values(['coin', 'hour']).reset_index(drop=True)

print(f'Merged: {len(merged):,}')

# Pre-compute future prices
print('Pre-computing...')
for h in [24, 48, 72]:
    merged[f'price_{h}h'] = merged.groupby('coin')['price'].shift(-h)
    merged[f'fr_sum_{h}h'] = merged.groupby('coin')['funding_rate'].transform(
        lambda x: x.rolling(h, min_periods=1).sum().shift(-h+1)
    )

merged = merged.dropna(subset=['price_72h'])

# Sample: one per coin per day
merged['date'] = merged['hour'].dt.date
sampled = merged.groupby(['coin', 'date']).first().reset_index()
print(f'Sampled: {len(sampled):,}')

# =============================================================================
# SIGNAL FREQUENCY BY FR THRESHOLD
# =============================================================================
print('\n' + '='*80)
print('SIGNAL FREQUENCY (Hourly FR to APY)')
print('='*80)

thresholds = [
    (0.00001, '0.001%', '8.76% APY'),
    (0.00002, '0.002%', '17.5% APY'),
    (0.00003, '0.003%', '26.3% APY'),
    (0.00005, '0.005%', '43.8% APY'),
    (0.00007, '0.007%', '61.3% APY'),
    (0.0001, '0.010%', '87.6% APY'),
    (0.00015, '0.015%', '131% APY'),
    (0.0002, '0.020%', '175% APY'),
    (0.0003, '0.030%', '263% APY'),
]

print(f"\n{'Threshold':<12} {'APY':<12} {'FR < -X':<15} {'FR > +X':<15} {'Total':<10}")
print('-'*70)

for thresh, label, apy in thresholds:
    neg = (sampled['funding_rate'] < -thresh).sum()
    pos = (sampled['funding_rate'] > thresh).sum()
    print(f"{label:<12} {apy:<12} {neg:>10,} ({neg/len(sampled)*100:.1f}%)  {pos:>10,} ({pos/len(sampled)*100:.1f}%)")

# =============================================================================
# BACKTEST FUNCTION
# =============================================================================
def fast_backtest(df, entry_thresh, hold_hours, direction):
    if direction == 'SHORT':
        signals = df[df['funding_rate'] < -entry_thresh].copy()
    else:
        signals = df[df['funding_rate'] > entry_thresh].copy()
    
    if len(signals) == 0:
        return None
    
    entry_price = signals['price'].values
    exit_price = signals[f'price_{hold_hours}h'].values
    fr_sum = signals[f'fr_sum_{hold_hours}h'].values
    
    if direction == 'SHORT':
        price_ret = -(exit_price - entry_price) / entry_price
        funding_pnl = -fr_sum
    else:
        price_ret = (exit_price - entry_price) / entry_price
        funding_pnl = fr_sum
    
    net_pnl = price_ret + funding_pnl - 2 * TAKER_FEE
    
    return {
        'n': len(net_pnl),
        'avg': np.nanmean(net_pnl),
        'std': np.nanstd(net_pnl),
        'sharpe': np.nanmean(net_pnl) / np.nanstd(net_pnl) if np.nanstd(net_pnl) > 0 else 0,
        'win': np.nanmean(net_pnl > 0),
        'total': np.nansum(net_pnl),
        'pnls': net_pnl
    }

# =============================================================================
# SWEEP ENTRY THRESHOLDS (72h hold)
# =============================================================================
print('\n' + '='*80)
print('ENTRY THRESHOLD SWEEP (Hold=72h)')
print('='*80)

entry_thresholds = [
    (0.00001, '0.001%', '8.76% APY'),
    (0.000015, '0.0015%', '13.1% APY'),
    (0.00002, '0.002%', '17.5% APY'),
    (0.000025, '0.0025%', '21.9% APY'),
    (0.00003, '0.003%', '26.3% APY'),
    (0.00004, '0.004%', '35.0% APY'),
    (0.00005, '0.005%', '43.8% APY'),
    (0.00007, '0.007%', '61.3% APY'),
    (0.0001, '0.010%', '87.6% APY'),
    (0.00015, '0.015%', '131% APY'),
    (0.0002, '0.020%', '175% APY'),
]

print(f"\n{'Entry':<10} {'APY':<12} {'Dir':<7} {'N':>7} {'Avg':>9} {'Sharpe':>8} {'Win%':>8} {'Total':>10}")
print('-'*80)

best_results = []

for thresh, label, apy in entry_thresholds:
    res_short = fast_backtest(sampled, thresh, 72, 'SHORT')
    res_long = fast_backtest(sampled, thresh, 72, 'LONG')
    
    if res_short and res_short['n'] >= 20:
        print(f"{label:<10} {apy:<12} SHORT  {res_short['n']:>7} {res_short['avg']*100:>+8.2f}% {res_short['sharpe']:>8.2f} {res_short['win']*100:>7.1f}% {res_short['total']*100:>+9.0f}%")
        best_results.append({'type': 'SHORT', 'thresh': thresh, 'label': label, 'apy': apy, **res_short})
    
    if res_long and res_long['n'] >= 20:
        print(f"{label:<10} {apy:<12} LONG   {res_long['n']:>7} {res_long['avg']*100:>+8.2f}% {res_long['sharpe']:>8.2f} {res_long['win']*100:>7.1f}% {res_long['total']*100:>+9.0f}%")
        best_results.append({'type': 'LONG', 'thresh': thresh, 'label': label, 'apy': apy, **res_long})
    
    # Combined
    if res_short and res_long and res_short['n'] >= 20 and res_long['n'] >= 20:
        total_n = res_short['n'] + res_long['n']
        total_pnl = res_short['total'] + res_long['total']
        avg_pnl = total_pnl / total_n
        win = (res_short['win']*res_short['n'] + res_long['win']*res_long['n']) / total_n
        print(f"{label:<10} {apy:<12} BOTH   {total_n:>7} {avg_pnl*100:>+8.2f}%          {win*100:>7.1f}% {total_pnl*100:>+9.0f}%")
    
    print()

# =============================================================================
# HOLD PERIOD SWEEP (for best entry threshold)
# =============================================================================
print('\n' + '='*80)
print('HOLD PERIOD SWEEP (Entry=0.005%, 43.8% APY)')
print('='*80)

entry = 0.00005
print(f"\n{'Hold':<8} {'Dir':<7} {'N':>7} {'Avg':>9} {'Sharpe':>8} {'Win%':>8} {'Total':>10}")
print('-'*65)

for hold in [24, 48, 72]:
    res_short = fast_backtest(sampled, entry, hold, 'SHORT')
    res_long = fast_backtest(sampled, entry, hold, 'LONG')
    
    if res_short and res_short['n'] >= 20:
        print(f"{hold}h      SHORT  {res_short['n']:>7} {res_short['avg']*100:>+8.2f}% {res_short['sharpe']:>8.2f} {res_short['win']*100:>7.1f}% {res_short['total']*100:>+9.0f}%")
    if res_long and res_long['n'] >= 20:
        print(f"{hold}h      LONG   {res_long['n']:>7} {res_long['avg']*100:>+8.2f}% {res_long['sharpe']:>8.2f} {res_long['win']*100:>7.1f}% {res_long['total']*100:>+9.0f}%")

# =============================================================================
# GRID: Entry x Hold
# =============================================================================
print('\n' + '='*80)
print('GRID: Entry x Hold (Combined SHORT+LONG)')
print('='*80)

entry_grid = [0.00002, 0.00003, 0.00005, 0.00007, 0.0001]
hold_grid = [24, 48, 72]

print(f"\n{'Entry':<10} {'Hold':<8} {'N':>7} {'Avg':>9} {'Sharpe':>8} {'Win%':>8} {'Total':>10}")
print('-'*65)

grid_results = []
for entry in entry_grid:
    for hold in hold_grid:
        res_short = fast_backtest(sampled, entry, hold, 'SHORT')
        res_long = fast_backtest(sampled, entry, hold, 'LONG')
        
        if res_short and res_long:
            total_n = res_short['n'] + res_long['n']
            total_pnl = res_short['total'] + res_long['total']
            avg_pnl = total_pnl / total_n
            combined_std = (res_short['std']*res_short['n'] + res_long['std']*res_long['n']) / total_n
            sharpe = avg_pnl / combined_std if combined_std > 0 else 0
            win = (res_short['win']*res_short['n'] + res_long['win']*res_long['n']) / total_n
            
            grid_results.append({
                'entry': entry, 'hold': hold, 'n': total_n,
                'avg': avg_pnl, 'sharpe': sharpe, 'win': win, 'total': total_pnl
            })
            
            print(f"{entry*100:.3f}%    {hold}h      {total_n:>7} {avg_pnl*100:>+8.2f}% {sharpe:>8.2f} {win*100:>7.1f}% {total_pnl*100:>+9.0f}%")

# =============================================================================
# TOP CONFIGURATIONS
# =============================================================================
print('\n' + '='*80)
print('TOP CONFIGURATIONS')
print('='*80)

if grid_results:
    print('\nBy Sharpe Ratio:')
    for r in sorted(grid_results, key=lambda x: x['sharpe'], reverse=True)[:5]:
        print(f"  Entry={r['entry']*100:.3f}% ({r['entry']*24*365*100:.0f}% APY), Hold={r['hold']}h: Sharpe={r['sharpe']:.2f}, N={r['n']}, Avg={r['avg']*100:+.2f}%")
    
    print('\nBy Total PnL:')
    for r in sorted(grid_results, key=lambda x: x['total'], reverse=True)[:5]:
        print(f"  Entry={r['entry']*100:.3f}% ({r['entry']*24*365*100:.0f}% APY), Hold={r['hold']}h: Total={r['total']*100:+.0f}%, N={r['n']}, Avg={r['avg']*100:+.2f}%")

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '='*80)
print('SUMMARY: Realistic Funding Rate Thresholds')
print('='*80)
print('''
Hourly FR to Annualized Rate:
  FR/hour × 24 × 365 = APY
  
Common thresholds:
  0.001%/h = 8.76% APY   (common, low signal quality)
  0.003%/h = 26.3% APY   (moderate)
  0.005%/h = 43.8% APY   (good balance)
  0.010%/h = 87.6% APY   (high, but fewer signals)
  0.020%/h = 175% APY    (very high, rare)
  
Strategy: TREND FOLLOWING
  - SHORT when FR < -threshold (bearish sentiment)
  - LONG when FR > +threshold (bullish sentiment)
  - Exit after fixed hold period (24-72h)
''')
