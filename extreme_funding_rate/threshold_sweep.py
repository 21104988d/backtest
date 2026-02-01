"""
Test various entry thresholds for SHORT and LONG strategies
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

price_lookup = merged.set_index(['hour', 'coin'])['price'].to_dict()
fr_lookup = merged.set_index(['hour', 'coin'])['funding_rate'].to_dict()
all_hours = sorted(merged['hour'].unique())

def get_price(hour, coin):
    return price_lookup.get((hour, coin), np.nan)

def get_fr(hour, coin):
    return fr_lookup.get((hour, coin), 0)

def run_backtest(entry_thresh, exit_thresh, max_hold, direction='SHORT'):
    if direction == 'SHORT':
        signals = merged[merged['funding_rate'] < entry_thresh][['hour', 'coin', 'funding_rate', 'price']].copy()
    else:
        signals = merged[merged['funding_rate'] > entry_thresh][['hour', 'coin', 'funding_rate', 'price']].copy()
    
    signals = signals.sort_values('hour').reset_index(drop=True)
    active = {}
    trades = []
    concurrent = []
    signal_idx = 0
    
    for hour in all_hours:
        concurrent.append(len(active))
        
        to_close = []
        for (coin, entry_hour), pos in active.items():
            hold_hours = int((hour - entry_hour).total_seconds() / 3600)
            if hold_hours < 1:
                continue
            
            current_fr = get_fr(hour, coin)
            current_price = get_price(hour, coin)
            if pd.isna(current_price):
                continue
            
            should_exit = abs(current_fr) < exit_thresh or hold_hours >= max_hold
            
            if should_exit:
                funding_flow = 0
                for h in range(hold_hours):
                    fr_h = entry_hour + timedelta(hours=h)
                    fr = get_fr(fr_h, coin)
                    if direction == 'SHORT':
                        funding_flow += (-abs(fr) if fr < 0 else fr)
                    else:
                        funding_flow += (fr if fr > 0 else -abs(fr))
                
                if direction == 'SHORT':
                    price_ret = -(current_price - pos['entry_price']) / pos['entry_price']
                else:
                    price_ret = (current_price - pos['entry_price']) / pos['entry_price']
                
                net_pnl = price_ret + funding_flow - 2 * TAKER_FEE
                trades.append(net_pnl)
                to_close.append((coin, entry_hour))
        
        for key in to_close:
            del active[key]
        
        while signal_idx < len(signals) and signals.iloc[signal_idx]['hour'] == hour:
            row = signals.iloc[signal_idx]
            coin = row['coin']
            coins_in_pos = {c for (c, _) in active.keys()}
            if coin not in coins_in_pos:
                active[(coin, hour)] = {'entry_price': row['price'], 'entry_fr': row['funding_rate']}
            signal_idx += 1
    
    return trades, concurrent

# Signal frequency check first
print('\n' + '='*90)
print('SIGNAL FREQUENCY')
print('='*90)

print('\nNegative FR (SHORT signals):')
for thresh in [-0.0003, -0.0005, -0.001, -0.0015, -0.002, -0.003, -0.005]:
    n = (merged['funding_rate'] < thresh).sum()
    print(f'  FR < {thresh*100:.2f}%: {n:>7,} signals ({n/len(merged)*100:.3f}%)')

print('\nPositive FR (LONG signals):')
for thresh in [0.0003, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005]:
    n = (merged['funding_rate'] > thresh).sum()
    print(f'  FR > +{thresh*100:.2f}%: {n:>7,} signals ({n/len(merged)*100:.3f}%)')

# SHORT strategy tests
print('\n' + '='*90)
print('SHORT STRATEGY - VARIOUS ENTRY THRESHOLDS (Exit |FR| < 0.02%, Max 72h)')
print('='*90)

short_configs = [
    (-0.0003, 0.0002, 72),  # -0.03%
    (-0.0005, 0.0002, 72),  # -0.05%
    (-0.001, 0.0002, 72),   # -0.10%
    (-0.0015, 0.0002, 72),  # -0.15%
    (-0.002, 0.0002, 72),   # -0.20%
    (-0.003, 0.0002, 72),   # -0.30%
    (-0.005, 0.0002, 72),   # -0.50%
]

print(f"\n{'Entry':<10} {'Exit':<10} {'N':>6} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10} {'MaxConc':>8}")
print('-'*80)

for entry, exit_t, hold in short_configs:
    trades, conc = run_backtest(entry, exit_t, hold, 'SHORT')
    if len(trades) < 5:
        print(f"{entry*100:>+.2f}%     {exit_t*100:.2f}%        <5 trades")
        continue
    avg = np.mean(trades)
    std = np.std(trades)
    sharpe = avg/std if std > 0 else 0
    win = sum(1 for p in trades if p > 0) / len(trades)
    total = sum(trades)
    max_c = max(conc) if conc else 0
    print(f"{entry*100:>+.2f}%     {exit_t*100:.2f}%     {len(trades):>6} {avg*100:>+9.2f}% {sharpe:>8.2f} {win*100:>7.1f}% {total*100:>+9.0f}% {max_c:>8}")

# LONG strategy tests
print('\n' + '='*90)
print('LONG STRATEGY - VARIOUS ENTRY THRESHOLDS (Exit |FR| < 0.02%, Max 72h)')
print('='*90)

long_configs = [
    (0.0003, 0.0002, 72),   # +0.03%
    (0.0005, 0.0002, 72),   # +0.05%
    (0.001, 0.0002, 72),    # +0.10%
    (0.0015, 0.0002, 72),   # +0.15%
    (0.002, 0.0002, 72),    # +0.20%
    (0.003, 0.0002, 72),    # +0.30%
    (0.005, 0.0002, 72),    # +0.50%
]

print(f"\n{'Entry':<10} {'Exit':<10} {'N':>6} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10} {'MaxConc':>8}")
print('-'*80)

for entry, exit_t, hold in long_configs:
    trades, conc = run_backtest(entry, exit_t, hold, 'LONG')
    if len(trades) < 5:
        print(f"+{entry*100:.2f}%     {exit_t*100:.2f}%        <5 trades")
        continue
    avg = np.mean(trades)
    std = np.std(trades)
    sharpe = avg/std if std > 0 else 0
    win = sum(1 for p in trades if p > 0) / len(trades)
    total = sum(trades)
    max_c = max(conc) if conc else 0
    print(f"+{entry*100:.2f}%     {exit_t*100:.2f}%     {len(trades):>6} {avg*100:>+9.2f}% {sharpe:>8.2f} {win*100:>7.1f}% {total*100:>+9.0f}% {max_c:>8}")

# Summary
print('\n' + '='*90)
print('SUMMARY')
print('='*90)
print('''
KEY INSIGHTS:
- Lower thresholds = more trades but possibly lower edge
- Higher thresholds = fewer trades but stronger signal
- Trade-off between frequency and quality
''')
