"""
Trend Following Threshold Optimization
Find optimal entry and exit thresholds for:
- SHORT when FR < -entry_threshold
- LONG when FR > +entry_threshold
- Exit when |FR| < exit_threshold or timeout
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

# Build lookups
price_lookup = merged.set_index(['hour', 'coin'])['price'].to_dict()
fr_lookup = merged.set_index(['hour', 'coin'])['funding_rate'].to_dict()
all_hours = sorted(merged['hour'].unique())

def get_price(hour, coin):
    return price_lookup.get((hour, coin), np.nan)

def get_fr(hour, coin):
    return fr_lookup.get((hour, coin), 0)

def run_portfolio_backtest(entry_thresh, exit_thresh, max_hold):
    """
    Run trend following backtest with proper portfolio simulation:
    - SHORT when FR < -entry_thresh
    - LONG when FR > +entry_thresh
    - Exit when |FR| < exit_thresh or max_hold hours
    """
    
    # Get all signals
    short_signals = merged[merged['funding_rate'] < -entry_thresh][['hour', 'coin', 'funding_rate', 'price']].copy()
    short_signals['direction'] = 'SHORT'
    
    long_signals = merged[merged['funding_rate'] > entry_thresh][['hour', 'coin', 'funding_rate', 'price']].copy()
    long_signals['direction'] = 'LONG'
    
    signals = pd.concat([short_signals, long_signals]).sort_values('hour').reset_index(drop=True)
    
    active = {}  # (coin, entry_hour) -> position info
    trades = []
    concurrent_history = []
    signal_idx = 0
    
    for hour in all_hours:
        concurrent_history.append(len(active))
        
        # Check exits
        to_close = []
        for (coin, entry_hour), pos in active.items():
            hold_hours = int((hour - entry_hour).total_seconds() / 3600)
            if hold_hours < 1:
                continue
            
            current_fr = get_fr(hour, coin)
            current_price = get_price(hour, coin)
            if pd.isna(current_price):
                continue
            
            # Exit when FR normalizes or timeout
            should_exit = abs(current_fr) < exit_thresh or hold_hours >= max_hold
            
            if should_exit:
                # Calculate funding flow
                funding_flow = 0
                for h in range(hold_hours):
                    fr_h = entry_hour + timedelta(hours=h)
                    fr = get_fr(fr_h, coin)
                    if pos['direction'] == 'SHORT':
                        funding_flow += (-abs(fr) if fr < 0 else fr)
                    else:
                        funding_flow += (fr if fr > 0 else -abs(fr))
                
                # Price return
                if pos['direction'] == 'SHORT':
                    price_ret = -(current_price - pos['entry_price']) / pos['entry_price']
                else:
                    price_ret = (current_price - pos['entry_price']) / pos['entry_price']
                
                net_pnl = price_ret + funding_flow - 2 * TAKER_FEE
                trades.append({
                    'net_pnl': net_pnl,
                    'direction': pos['direction'],
                    'hold_hours': hold_hours,
                    'entry_fr': pos['entry_fr']
                })
                to_close.append((coin, entry_hour))
        
        for key in to_close:
            del active[key]
        
        # Process new signals
        while signal_idx < len(signals) and signals.iloc[signal_idx]['hour'] == hour:
            row = signals.iloc[signal_idx]
            coin = row['coin']
            # One position per coin
            coins_in_pos = {c for (c, _) in active.keys()}
            if coin not in coins_in_pos:
                active[(coin, hour)] = {
                    'entry_price': row['price'],
                    'entry_fr': row['funding_rate'],
                    'direction': row['direction']
                }
            signal_idx += 1
    
    return trades, concurrent_history

# =============================================================================
# SWEEP: Entry Thresholds
# =============================================================================
print('\n' + '='*90)
print('SWEEP 1: Entry Threshold (Fixed Exit=0.02%, MaxHold=72h)')
print('='*90)

entry_thresholds = [0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005]
exit_fixed = 0.0002  # 0.02%
max_hold = 72

print(f"\n{'Entry':<10} {'N':>7} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>12} {'MaxConc':>10}")
print('-'*75)

entry_results = []
for entry in entry_thresholds:
    trades, conc = run_portfolio_backtest(entry, exit_fixed, max_hold)
    if len(trades) < 10:
        print(f"{entry*100:.3f}%     {'<10':>7}")
        continue
    pnls = [t['net_pnl'] for t in trades]
    avg = np.mean(pnls)
    std = np.std(pnls)
    sharpe = avg / std if std > 0 else 0
    win = sum(1 for p in pnls if p > 0) / len(pnls)
    total = sum(pnls)
    max_conc = max(conc) if conc else 0
    
    entry_results.append({
        'entry': entry, 'n': len(trades), 'avg': avg, 'sharpe': sharpe, 
        'win': win, 'total': total, 'max_conc': max_conc
    })
    print(f"{entry*100:.3f}%    {len(trades):>7} {avg*100:>+9.2f}% {sharpe:>8.2f} {win*100:>7.1f}% {total*100:>+11.0f}% {max_conc:>10}")

# =============================================================================
# SWEEP: Exit Thresholds (with best entry from above)
# =============================================================================
print('\n' + '='*90)
print('SWEEP 2: Exit Threshold (Fixed Entry=0.03%, MaxHold=72h)')
print('='*90)

entry_fixed = 0.0003  # 0.03%
exit_thresholds = [0.00005, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.0005, 0.001]

print(f"\n{'Exit':<10} {'N':>7} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>12} {'AvgHold':>10}")
print('-'*75)

exit_results = []
for exit_t in exit_thresholds:
    trades, conc = run_portfolio_backtest(entry_fixed, exit_t, max_hold)
    if len(trades) < 10:
        print(f"{exit_t*100:.4f}%    {'<10':>7}")
        continue
    pnls = [t['net_pnl'] for t in trades]
    avg = np.mean(pnls)
    std = np.std(pnls)
    sharpe = avg / std if std > 0 else 0
    win = sum(1 for p in pnls if p > 0) / len(pnls)
    total = sum(pnls)
    avg_hold = np.mean([t['hold_hours'] for t in trades])
    
    exit_results.append({
        'exit': exit_t, 'n': len(trades), 'avg': avg, 'sharpe': sharpe,
        'win': win, 'total': total, 'avg_hold': avg_hold
    })
    print(f"{exit_t*100:.4f}%   {len(trades):>7} {avg*100:>+9.2f}% {sharpe:>8.2f} {win*100:>7.1f}% {total*100:>+11.0f}% {avg_hold:>9.1f}h")

# =============================================================================
# SWEEP: Max Hold Period
# =============================================================================
print('\n' + '='*90)
print('SWEEP 3: Max Hold Period (Fixed Entry=0.03%, Exit=0.02%)')
print('='*90)

hold_periods = [12, 24, 36, 48, 72, 96, 120, 168]

print(f"\n{'MaxHold':<10} {'N':>7} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>12} {'AvgHold':>10}")
print('-'*75)

for hold in hold_periods:
    trades, conc = run_portfolio_backtest(entry_fixed, exit_fixed, hold)
    if len(trades) < 10:
        print(f"{hold}h        {'<10':>7}")
        continue
    pnls = [t['net_pnl'] for t in trades]
    avg = np.mean(pnls)
    std = np.std(pnls)
    sharpe = avg / std if std > 0 else 0
    win = sum(1 for p in pnls if p > 0) / len(pnls)
    total = sum(pnls)
    avg_hold = np.mean([t['hold_hours'] for t in trades])
    
    print(f"{hold}h        {len(trades):>7} {avg*100:>+9.2f}% {sharpe:>8.2f} {win*100:>7.1f}% {total*100:>+11.0f}% {avg_hold:>9.1f}h")

# =============================================================================
# GRID SEARCH: Entry x Exit combinations
# =============================================================================
print('\n' + '='*90)
print('GRID SEARCH: Entry x Exit Threshold Combinations (MaxHold=72h)')
print('='*90)

entry_grid = [0.0001, 0.0002, 0.0003, 0.0005, 0.001]
exit_grid = [0.0001, 0.00015, 0.0002, 0.0003]

print(f"\n{'Entry':<8} {'Exit':<8} {'N':>6} {'Avg PnL':>9} {'Sharpe':>7} {'Win%':>7} {'Total':>10}")
print('-'*65)

grid_results = []
for entry in entry_grid:
    for exit_t in exit_grid:
        trades, conc = run_portfolio_backtest(entry, exit_t, 72)
        if len(trades) < 10:
            continue
        pnls = [t['net_pnl'] for t in trades]
        avg = np.mean(pnls)
        std = np.std(pnls)
        sharpe = avg / std if std > 0 else 0
        win = sum(1 for p in pnls if p > 0) / len(pnls)
        total = sum(pnls)
        
        grid_results.append({
            'entry': entry, 'exit': exit_t, 'n': len(trades), 
            'avg': avg, 'sharpe': sharpe, 'win': win, 'total': total
        })
        print(f"{entry*100:.2f}%   {exit_t*100:.3f}%  {len(trades):>6} {avg*100:>+8.2f}% {sharpe:>7.2f} {win*100:>6.1f}% {total*100:>+9.0f}%")

# =============================================================================
# FIND BEST CONFIGURATIONS
# =============================================================================
print('\n' + '='*90)
print('TOP 5 CONFIGURATIONS')
print('='*90)

if grid_results:
    # By Sharpe
    print('\nBest by Sharpe Ratio:')
    by_sharpe = sorted(grid_results, key=lambda x: x['sharpe'], reverse=True)[:5]
    for r in by_sharpe:
        print(f"  Entry={r['entry']*100:.2f}%, Exit={r['exit']*100:.3f}%: Sharpe={r['sharpe']:.2f}, N={r['n']}, Avg={r['avg']*100:+.2f}%, Total={r['total']*100:+.0f}%")
    
    # By Total PnL
    print('\nBest by Total PnL:')
    by_total = sorted(grid_results, key=lambda x: x['total'], reverse=True)[:5]
    for r in by_total:
        print(f"  Entry={r['entry']*100:.2f}%, Exit={r['exit']*100:.3f}%: Total={r['total']*100:+.0f}%, N={r['n']}, Avg={r['avg']*100:+.2f}%, Sharpe={r['sharpe']:.2f}")
    
    # By Win Rate
    print('\nBest by Win Rate:')
    by_win = sorted(grid_results, key=lambda x: x['win'], reverse=True)[:5]
    for r in by_win:
        print(f"  Entry={r['entry']*100:.2f}%, Exit={r['exit']*100:.3f}%: Win={r['win']*100:.1f}%, N={r['n']}, Avg={r['avg']*100:+.2f}%, Sharpe={r['sharpe']:.2f}")

# =============================================================================
# SEPARATE SHORT vs LONG ANALYSIS
# =============================================================================
print('\n' + '='*90)
print('SHORT vs LONG BREAKDOWN (Entry=0.03%, Exit=0.02%)')
print('='*90)

trades, _ = run_portfolio_backtest(0.0003, 0.0002, 72)
short_trades = [t for t in trades if t['direction'] == 'SHORT']
long_trades = [t for t in trades if t['direction'] == 'LONG']

print(f"\nSHORT (FR < -0.03%):")
if short_trades:
    short_pnls = [t['net_pnl'] for t in short_trades]
    print(f"  N={len(short_trades)}, Avg={np.mean(short_pnls)*100:+.2f}%, Win={(sum(1 for p in short_pnls if p>0)/len(short_pnls))*100:.1f}%, Total={sum(short_pnls)*100:+.0f}%")

print(f"\nLONG (FR > +0.03%):")
if long_trades:
    long_pnls = [t['net_pnl'] for t in long_trades]
    print(f"  N={len(long_trades)}, Avg={np.mean(long_pnls)*100:+.2f}%, Win={(sum(1 for p in long_pnls if p>0)/len(long_pnls))*100:.1f}%, Total={sum(long_pnls)*100:+.0f}%")

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '='*90)
print('SUMMARY')
print('='*90)
print('''
TREND FOLLOWING STRATEGY:
  - SHORT when funding_rate < -entry_threshold
  - LONG when funding_rate > +entry_threshold
  - Exit when |funding_rate| < exit_threshold OR max_hold timeout

KEY PARAMETERS TO OPTIMIZE:
  1. Entry Threshold: How extreme should FR be to enter?
  2. Exit Threshold: How normalized should FR be to exit?
  3. Max Hold: Maximum hours to hold before forced exit
''')
