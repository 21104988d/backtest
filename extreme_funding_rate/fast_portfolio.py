"""
FAST Portfolio backtest - only iterate through extreme funding signals
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

print(f'Funding: {len(funding):,} | Price: {len(price):,}')

# Merge for fast lookup
print('Merging data...')
merged = pd.merge(
    funding[['hour', 'coin', 'funding_rate']],
    price[['hour', 'coin', 'price']],
    on=['hour', 'coin'],
    how='inner'
).sort_values(['coin', 'hour']).reset_index(drop=True)

print(f'Merged: {len(merged):,}')

# Create lookups
price_lookup = merged.set_index(['hour', 'coin'])['price'].to_dict()
fr_lookup = merged.set_index(['hour', 'coin'])['funding_rate'].to_dict()

def get_price(hour, coin):
    return price_lookup.get((hour, coin), np.nan)

def get_fr(hour, coin):
    return fr_lookup.get((hour, coin), 0)

# Get all hours sorted
all_hours = sorted(merged['hour'].unique())

# =============================================================================
# PORTFOLIO SIMULATION
# =============================================================================

def run_backtest(entry_thresh, exit_thresh, max_hold, direction='SHORT'):
    """Fast portfolio simulation"""
    
    # Find entry signals
    if direction == 'SHORT':
        signals = merged[merged['funding_rate'] < entry_thresh][['hour', 'coin', 'funding_rate', 'price']].copy()
    else:
        signals = merged[merged['funding_rate'] > entry_thresh][['hour', 'coin', 'funding_rate', 'price']].copy()
    
    signals = signals.sort_values('hour').reset_index(drop=True)
    
    active = {}  # {(coin, entry_hour): entry_data}
    trades = []
    concurrent = []
    
    # Track which (coin, entry_hour) we've already processed
    processed_signals = set()
    
    signal_idx = 0
    
    for hour in all_hours:
        # Record concurrent positions at start of hour
        concurrent.append(len(active))
        
        # 1. Check exits
        to_close = []
        for (coin, entry_hour), pos in active.items():
            hold_hours = int((hour - entry_hour).total_seconds() / 3600)
            if hold_hours < 1:
                continue
            
            current_fr = get_fr(hour, coin)
            current_price = get_price(hour, coin)
            
            if pd.isna(current_price):
                continue
            
            # Exit conditions
            should_exit = False
            exit_reason = None
            
            if abs(current_fr) < exit_thresh:
                should_exit = True
                exit_reason = 'normalized'
            elif hold_hours >= max_hold:
                should_exit = True
                exit_reason = 'timeout'
            
            if should_exit:
                # Calculate funding
                funding_flow = 0
                for h in range(hold_hours):
                    fr_h = entry_hour + timedelta(hours=h)
                    fr = get_fr(fr_h, coin)
                    if direction == 'SHORT':
                        funding_flow += (-abs(fr) if fr < 0 else fr)
                    else:
                        funding_flow += (fr if fr > 0 else -abs(fr))
                
                # Price return
                if direction == 'SHORT':
                    price_ret = -(current_price - pos['entry_price']) / pos['entry_price']
                else:
                    price_ret = (current_price - pos['entry_price']) / pos['entry_price']
                
                net_pnl = price_ret + funding_flow - 2 * TAKER_FEE
                
                trades.append({
                    'coin': coin,
                    'entry_hour': entry_hour,
                    'exit_hour': hour,
                    'hold_hours': hold_hours,
                    'entry_fr': pos['entry_fr'],
                    'exit_fr': current_fr,
                    'price_return': price_ret,
                    'funding_flow': funding_flow,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason
                })
                to_close.append((coin, entry_hour))
        
        for key in to_close:
            del active[key]
        
        # 2. Check new entries from signals
        while signal_idx < len(signals) and signals.iloc[signal_idx]['hour'] == hour:
            row = signals.iloc[signal_idx]
            coin = row['coin']
            
            # Check if we already have position in this coin
            coins_in_pos = {c for (c, _) in active.keys()}
            
            if coin not in coins_in_pos:
                active[(coin, hour)] = {
                    'entry_price': row['price'],
                    'entry_fr': row['funding_rate']
                }
            
            signal_idx += 1
    
    return trades, concurrent

# =============================================================================
# RUN TESTS
# =============================================================================

print('\n' + '='*90)
print('SHORT STRATEGY: Entry FR < threshold, Exit |FR| < exit_threshold')
print('='*90)

configs = [
    (-0.005, 0.0001, 72),  # -0.50%, exit 0.01%
    (-0.005, 0.0002, 72),  # -0.50%, exit 0.02%
    (-0.005, 0.0005, 72),  # -0.50%, exit 0.05%
    (-0.005, 0.001, 72),   # -0.50%, exit 0.10%
    (-0.003, 0.0002, 72),  # -0.30%, exit 0.02%
    (-0.002, 0.0002, 72),  # -0.20%, exit 0.02%
    (-0.001, 0.0001, 72),  # -0.10%, exit 0.01%
    (-0.001, 0.0002, 72),  # -0.10%, exit 0.02%
]

print(f"\n{'Entry':<10} {'Exit':<10} {'N':>6} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10} {'MaxConc':>8}")
print('-'*80)

for entry, exit_t, hold in configs:
    trades, conc = run_backtest(entry, exit_t, hold, 'SHORT')
    
    if len(trades) < 5:
        continue
    
    pnls = [t['net_pnl'] for t in trades]
    avg = np.mean(pnls)
    std = np.std(pnls)
    sharpe = avg/std if std > 0 else 0
    win = sum(1 for p in pnls if p > 0) / len(pnls)
    total = sum(pnls)
    max_c = max(conc) if conc else 0
    
    print(f"{entry*100:>+.2f}%     {exit_t*100:.2f}%     {len(trades):>6} {avg*100:>+9.2f}% {sharpe:>8.2f} {win*100:>7.1f}% {total*100:>+9.0f}% {max_c:>8}")

print('\n' + '='*90)
print('LONG STRATEGY: Entry FR > threshold, Exit |FR| < exit_threshold')
print('='*90)

long_configs = [
    (0.001, 0.0001, 72),  # +0.10%, exit 0.01%
    (0.001, 0.0002, 72),  # +0.10%, exit 0.02%
    (0.002, 0.0002, 72),  # +0.20%, exit 0.02%
    (0.003, 0.0002, 72),  # +0.30%, exit 0.02%
    (0.005, 0.0002, 72),  # +0.50%, exit 0.02%
]

print(f"\n{'Entry':<10} {'Exit':<10} {'N':>6} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10} {'MaxConc':>8}")
print('-'*80)

for entry, exit_t, hold in long_configs:
    trades, conc = run_backtest(entry, exit_t, hold, 'LONG')
    
    if len(trades) < 5:
        continue
    
    pnls = [t['net_pnl'] for t in trades]
    avg = np.mean(pnls)
    std = np.std(pnls)
    sharpe = avg/std if std > 0 else 0
    win = sum(1 for p in pnls if p > 0) / len(pnls)
    total = sum(pnls)
    max_c = max(conc) if conc else 0
    
    print(f"+{entry*100:.2f}%     {exit_t*100:.2f}%     {len(trades):>6} {avg*100:>+9.2f}% {sharpe:>8.2f} {win*100:>7.1f}% {total*100:>+9.0f}% {max_c:>8}")

# =============================================================================
# DETAILED ANALYSIS
# =============================================================================

print('\n' + '='*90)
print('DETAILED: SHORT FR < -0.50%, Exit |FR| < 0.02%')
print('='*90)

trades, conc = run_backtest(-0.005, 0.0002, 72, 'SHORT')

if trades:
    df = pd.DataFrame(trades)
    
    print(f"\nTotal trades: {len(df)}")
    print(f"Avg PnL: {df['net_pnl'].mean()*100:+.2f}%")
    print(f"Sharpe: {df['net_pnl'].mean()/df['net_pnl'].std():.2f}")
    print(f"Win rate: {(df['net_pnl']>0).mean()*100:.1f}%")
    print(f"Max concurrent: {max(conc)}")
    print(f"Avg hold time: {df['hold_hours'].mean():.1f}h")
    
    print("\nExit reasons:")
    for reason, grp in df.groupby('exit_reason'):
        print(f"  {reason}: N={len(grp)}, Avg={grp['net_pnl'].mean()*100:+.2f}%, AvgHold={grp['hold_hours'].mean():.0f}h")
    
    # Show concurrent position distribution
    print("\nConcurrent positions:")
    conc_s = pd.Series(conc)
    for n in range(max(conc)+1):
        pct = (conc_s == n).mean() * 100
        if pct > 0.5:
            print(f"  {n} positions: {pct:.1f}% of time")
    
    # Show overlapping trades example
    print("\nExample of concurrent positions:")
    df_sorted = df.sort_values('entry_hour')
    
    # Find hours with multiple entries
    entry_counts = df_sorted.groupby('entry_hour').size()
    multi_entry = entry_counts[entry_counts > 1].head(5)
    
    for hour, count in multi_entry.items():
        print(f"\n  {hour} - {count} positions opened:")
        for _, row in df_sorted[df_sorted['entry_hour'] == hour].iterrows():
            print(f"    {row['coin']:<8} FR={row['entry_fr']*100:>+.2f}% -> PnL={row['net_pnl']*100:>+6.2f}%")

print('\n' + '='*90)
print('ANSWER TO YOUR QUESTIONS')
print('='*90)
print('''
1. EXIT THRESHOLD:
   - Previous simple_sweep used FIXED HOLD PERIOD (no exit threshold)
   - This backtest uses EXIT when |FR| < threshold (normalized)
   - E.g., Exit when |FR| < 0.02% means close when funding normalizes
   - Also has MAX HOLD = 72h timeout

2. MULTIPLE ASSETS:
   - YES, we track concurrent positions
   - When coin A and B both have FR < -0.50% at same hour, we open BOTH
   - No position limit (unlimited concurrent)
   - One position per coin at a time (no doubling down)
   - See "Concurrent positions" section for distribution
''')
