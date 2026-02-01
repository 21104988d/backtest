"""
Extended parameter sweep for SHORT-only strategy with 995 days of data
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

price_pivot = price.pivot_table(index='hour', columns='coin', values='price', aggfunc='first')
price_hours = sorted(price_pivot.index.tolist())
price_hours_set = set(price_pivot.index)
price_coins = set(price_pivot.columns)

funding_lookup = {}
for _, row in funding.iterrows():
    key = (row['hour'], row['coin'])
    funding_lookup[key] = row['funding_rate']

def get_funding_rate(hour, coin):
    return funding_lookup.get((hour, coin), 0)

print('Testing different entry/exit combinations...')
print('='*90)

results = []

# Test different parameter combinations
for entry_thresh in [-0.0005, -0.001, -0.002, -0.003, -0.005]:
    for exit_thresh in [0.0005, 0.001, 0.002, 0.003, 0.005]:
        for max_hold in [24, 48, 72]:
            active = {}
            trades = []
            
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
                    
                    should_exit = abs(current_fr) < exit_thresh or hold_hours >= max_hold
                    
                    if should_exit:
                        funding_paid = 0
                        for h in range(hold_hours):
                            fr_hour = entry_hour + timedelta(hours=h)
                            fr = get_funding_rate(fr_hour, coin)
                            if fr < 0:
                                funding_paid += abs(fr)
                            else:
                                funding_paid -= fr
                        
                        price_return = -(current_price - pos['entry_price']) / pos['entry_price']
                        net_pnl = price_return - funding_paid - 2 * TAKER_FEE
                        trades.append(net_pnl)
                        to_close.append((coin, entry_hour))
                
                for key in to_close:
                    del active[key]
                
                # Check entries
                for coin in price_coins:
                    if (coin, hour) in active:
                        continue
                    fr = get_funding_rate(hour, coin)
                    if fr < entry_thresh:
                        if coin not in price_pivot.columns:
                            continue
                        entry_price = price_pivot.loc[hour, coin]
                        if pd.isna(entry_price):
                            continue
                        active[(coin, hour)] = {'entry_price': entry_price, 'entry_fr': fr}
            
            if len(trades) > 20:
                avg_pnl = np.mean(trades)
                std_pnl = np.std(trades)
                sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
                win_rate = sum(1 for t in trades if t > 0) / len(trades)
                total_pnl = sum(trades)
                
                results.append({
                    'entry': f'{entry_thresh*100:.2f}%',
                    'exit': f'{exit_thresh*100:.2f}%',
                    'max_hold': max_hold,
                    'n': len(trades),
                    'avg': avg_pnl,
                    'sharpe': sharpe,
                    'win': win_rate,
                    'total': total_pnl
                })

# Sort by Sharpe
results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'Entry':<10} {'Exit':<10} {'Hold':<6} {'N':>6} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total PnL':>10}")
print('-'*80)
for r in results[:25]:
    print(f"{r['entry']:<10} {r['exit']:<10} {r['max_hold']:<6} {r['n']:>6} {r['avg']*100:>+9.2f}% {r['sharpe']:>8.2f} {r['win']*100:>7.1f}% {r['total']*100:>+9.0f}%")

print('\n' + '='*80)
print('ALSO TESTING LONG STRATEGY (when FR > threshold)')
print('='*80)

# Test LONG strategy
long_results = []
for entry_thresh in [0.0005, 0.001, 0.002, 0.003, 0.005]:
    for exit_thresh in [0.0005, 0.001, 0.002, 0.003]:
        for max_hold in [24, 48, 72]:
            active = {}
            trades = []
            
            for hour in price_hours:
                to_close = []
                for (coin, entry_hour), pos in active.items():
                    hold_hours = int((hour - entry_hour).total_seconds() / 3600)
                    if coin not in price_pivot.columns:
                        continue
                    current_price = price_pivot.loc[hour, coin]
                    if pd.isna(current_price):
                        continue
                    current_fr = get_funding_rate(hour, coin)
                    
                    should_exit = abs(current_fr) < exit_thresh or hold_hours >= max_hold
                    
                    if should_exit:
                        funding_earned = 0
                        for h in range(hold_hours):
                            fr_hour = entry_hour + timedelta(hours=h)
                            fr = get_funding_rate(fr_hour, coin)
                            if fr > 0:
                                funding_earned += fr  # We receive
                            else:
                                funding_earned -= abs(fr)  # We pay
                        
                        # LONG position
                        price_return = (current_price - pos['entry_price']) / pos['entry_price']
                        net_pnl = price_return + funding_earned - 2 * TAKER_FEE
                        trades.append(net_pnl)
                        to_close.append((coin, entry_hour))
                
                for key in to_close:
                    del active[key]
                
                for coin in price_coins:
                    if (coin, hour) in active:
                        continue
                    fr = get_funding_rate(hour, coin)
                    if fr > entry_thresh:  # LONG when FR is positive
                        if coin not in price_pivot.columns:
                            continue
                        entry_price = price_pivot.loc[hour, coin]
                        if pd.isna(entry_price):
                            continue
                        active[(coin, hour)] = {'entry_price': entry_price, 'entry_fr': fr}
            
            if len(trades) > 20:
                avg_pnl = np.mean(trades)
                std_pnl = np.std(trades)
                sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
                win_rate = sum(1 for t in trades if t > 0) / len(trades)
                total_pnl = sum(trades)
                
                long_results.append({
                    'entry': f'+{entry_thresh*100:.2f}%',
                    'exit': f'{exit_thresh*100:.2f}%',
                    'max_hold': max_hold,
                    'n': len(trades),
                    'avg': avg_pnl,
                    'sharpe': sharpe,
                    'win': win_rate,
                    'total': total_pnl
                })

long_results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'Entry':<10} {'Exit':<10} {'Hold':<6} {'N':>6} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total PnL':>10}")
print('-'*80)
for r in long_results[:15]:
    print(f"{r['entry']:<10} {r['exit']:<10} {r['max_hold']:<6} {r['n']:>6} {r['avg']*100:>+9.2f}% {r['sharpe']:>8.2f} {r['win']*100:>7.1f}% {r['total']*100:>+9.0f}%")

print('\n' + '='*80)
print('SUMMARY')
print('='*80)
best_short = results[0] if results else None
best_long = long_results[0] if long_results else None

if best_short:
    print(f"\nBest SHORT config: Entry<{best_short['entry']}, Exit<{best_short['exit']}, Hold={best_short['max_hold']}h")
    print(f"  -> N={best_short['n']}, Sharpe={best_short['sharpe']:.2f}, Win={best_short['win']*100:.1f}%, Total={best_short['total']*100:.0f}%")

if best_long:
    print(f"\nBest LONG config: Entry>{best_long['entry']}, Exit<{best_long['exit']}, Hold={best_long['max_hold']}h")
    print(f"  -> N={best_long['n']}, Sharpe={best_long['sharpe']:.2f}, Win={best_long['win']*100:.1f}%, Total={best_long['total']*100:.0f}%")
