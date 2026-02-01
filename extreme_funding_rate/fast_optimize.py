"""
FAST Trend Following Threshold Optimization
Using vectorized operations and sampling
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

# Pre-compute future prices for various hold periods
print('Pre-computing future prices...')
for h in [24, 48, 72]:
    merged[f'price_{h}h'] = merged.groupby('coin')['price'].shift(-h)
    merged[f'fr_sum_{h}h'] = merged.groupby('coin')['funding_rate'].transform(
        lambda x: x.rolling(h, min_periods=1).sum().shift(-h+1)
    )

# For dynamic exit, compute when FR crosses threshold
def compute_exit_hour(df, exit_thresh, max_hold):
    """Find first hour where |FR| < exit_thresh, within max_hold"""
    result = []
    for coin in df['coin'].unique():
        coin_df = df[df['coin'] == coin].copy()
        coin_df = coin_df.sort_values('hour').reset_index(drop=True)
        
        # For each row, find when |FR| first goes below exit_thresh
        n = len(coin_df)
        exit_idx = np.full(n, max_hold)  # Default to max_hold
        
        for i in range(n):
            for j in range(1, min(max_hold + 1, n - i)):
                if abs(coin_df.iloc[i + j]['funding_rate']) < exit_thresh:
                    exit_idx[i] = j
                    break
        
        coin_df['exit_hours'] = exit_idx
        result.append(coin_df)
    
    return pd.concat(result)

# Sample: one signal per coin per day to avoid overlapping
print('Sampling signals...')
merged['date'] = merged['hour'].dt.date
sampled = merged.groupby(['coin', 'date']).first().reset_index()
sampled = sampled.dropna(subset=['price_72h'])
print(f'Sampled signals: {len(sampled):,}')

def fast_backtest(df, entry_thresh, hold_hours, direction):
    """
    Fast vectorized backtest
    direction: 'SHORT' or 'LONG'
    """
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
        funding_pnl = -fr_sum  # SHORT: we pay negative FR, receive positive FR
    else:
        price_ret = (exit_price - entry_price) / entry_price
        funding_pnl = fr_sum  # LONG: we receive positive FR
    
    net_pnl = price_ret + funding_pnl - 2 * TAKER_FEE
    
    return {
        'n': len(net_pnl),
        'avg': np.nanmean(net_pnl),
        'std': np.nanstd(net_pnl),
        'sharpe': np.nanmean(net_pnl) / np.nanstd(net_pnl) if np.nanstd(net_pnl) > 0 else 0,
        'win': np.nanmean(net_pnl > 0),
        'total': np.nansum(net_pnl)
    }

# =============================================================================
# SWEEP: Entry Thresholds (Fixed 72h hold)
# =============================================================================
print('\n' + '='*90)
print('SWEEP 1: Entry Threshold (Fixed Hold=72h)')
print('='*90)

entry_thresholds = [0.00005, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.0004, 0.0005, 
                   0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005]

print(f"\n{'Entry':<12} {'Dir':<7} {'N':>7} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>12}")
print('-'*75)

for entry in entry_thresholds:
    # SHORT
    res_short = fast_backtest(sampled, entry, 72, 'SHORT')
    if res_short and res_short['n'] >= 10:
        print(f"{entry*100:.3f}%      SHORT  {res_short['n']:>7} {res_short['avg']*100:>+9.2f}% {res_short['sharpe']:>8.2f} {res_short['win']*100:>7.1f}% {res_short['total']*100:>+11.0f}%")
    
    # LONG  
    res_long = fast_backtest(sampled, entry, 72, 'LONG')
    if res_long and res_long['n'] >= 10:
        print(f"{entry*100:.3f}%      LONG   {res_long['n']:>7} {res_long['avg']*100:>+9.2f}% {res_long['sharpe']:>8.2f} {res_long['win']*100:>7.1f}% {res_long['total']*100:>+11.0f}%")
    
    # Combined
    if res_short and res_long and res_short['n'] >= 10 and res_long['n'] >= 10:
        total_n = res_short['n'] + res_long['n']
        total_pnl = res_short['total'] + res_long['total']
        avg_pnl = total_pnl / total_n
        print(f"{entry*100:.3f}%      BOTH   {total_n:>7} {avg_pnl*100:>+9.2f}%          {(res_short['win']*res_short['n']+res_long['win']*res_long['n'])/total_n*100:>7.1f}% {total_pnl*100:>+11.0f}%")
    print()

# =============================================================================
# SWEEP: Hold Period
# =============================================================================
print('\n' + '='*90)
print('SWEEP 2: Hold Period (Fixed Entry=0.03%)')
print('='*90)

entry_fixed = 0.0003
hold_periods = [24, 48, 72]

print(f"\n{'Hold':<10} {'Dir':<7} {'N':>7} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>12}")
print('-'*75)

for hold in hold_periods:
    res_short = fast_backtest(sampled, entry_fixed, hold, 'SHORT')
    res_long = fast_backtest(sampled, entry_fixed, hold, 'LONG')
    
    if res_short and res_short['n'] >= 10:
        print(f"{hold}h        SHORT  {res_short['n']:>7} {res_short['avg']*100:>+9.2f}% {res_short['sharpe']:>8.2f} {res_short['win']*100:>7.1f}% {res_short['total']*100:>+11.0f}%")
    if res_long and res_long['n'] >= 10:
        print(f"{hold}h        LONG   {res_long['n']:>7} {res_long['avg']*100:>+9.2f}% {res_long['sharpe']:>8.2f} {res_long['win']*100:>7.1f}% {res_long['total']*100:>+11.0f}%")
    print()

# =============================================================================
# COMPREHENSIVE GRID: Entry x Hold
# =============================================================================
print('\n' + '='*90)
print('GRID: Entry x Hold (Combined SHORT+LONG)')
print('='*90)

entry_grid = [0.0001, 0.0002, 0.0003, 0.0005, 0.001]
hold_grid = [24, 48, 72]

results = []
print(f"\n{'Entry':<10} {'Hold':<8} {'N':>7} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>12}")
print('-'*70)

for entry in entry_grid:
    for hold in hold_grid:
        res_short = fast_backtest(sampled, entry, hold, 'SHORT')
        res_long = fast_backtest(sampled, entry, hold, 'LONG')
        
        if res_short and res_long:
            total_n = res_short['n'] + res_long['n']
            total_pnl = res_short['total'] + res_long['total']
            avg_pnl = total_pnl / total_n
            
            # Combined sharpe (simplified)
            combined_std = (res_short['std'] * res_short['n'] + res_long['std'] * res_long['n']) / total_n
            sharpe = avg_pnl / combined_std if combined_std > 0 else 0
            
            win_rate = (res_short['win'] * res_short['n'] + res_long['win'] * res_long['n']) / total_n
            
            results.append({
                'entry': entry, 'hold': hold, 'n': total_n, 
                'avg': avg_pnl, 'sharpe': sharpe, 'win': win_rate, 'total': total_pnl
            })
            
            print(f"{entry*100:.2f}%     {hold}h      {total_n:>7} {avg_pnl*100:>+9.2f}% {sharpe:>8.2f} {win_rate*100:>7.1f}% {total_pnl*100:>+11.0f}%")

# =============================================================================
# BEST CONFIGURATIONS
# =============================================================================
print('\n' + '='*90)
print('TOP CONFIGURATIONS')
print('='*90)

if results:
    print('\nBy Sharpe Ratio:')
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True)[:5]:
        print(f"  Entry={r['entry']*100:.2f}%, Hold={r['hold']}h: Sharpe={r['sharpe']:.2f}, N={r['n']}, Avg={r['avg']*100:+.2f}%, Total={r['total']*100:+.0f}%")
    
    print('\nBy Total PnL:')
    for r in sorted(results, key=lambda x: x['total'], reverse=True)[:5]:
        print(f"  Entry={r['entry']*100:.2f}%, Hold={r['hold']}h: Total={r['total']*100:+.0f}%, N={r['n']}, Avg={r['avg']*100:+.2f}%, Sharpe={r['sharpe']:.2f}")

# =============================================================================
# DYNAMIC EXIT: Test exit thresholds by simulation
# =============================================================================
print('\n' + '='*90)
print('EXIT THRESHOLD ANALYSIS')
print('Simulating dynamic exit when |FR| < exit_threshold')
print('='*90)

# For this we need to compute exit points more carefully
# Let's use the full merged data but with proper coin-level iteration

def simulate_with_exit(entry_thresh, exit_thresh, max_hold, direction):
    """Simulate with dynamic exit threshold"""
    if direction == 'SHORT':
        signals = merged[merged['funding_rate'] < -entry_thresh].copy()
    else:
        signals = merged[merged['funding_rate'] > entry_thresh].copy()
    
    # Sample: one per coin per day
    signals['date'] = signals['hour'].dt.date
    signals = signals.groupby(['coin', 'date']).first().reset_index()
    
    pnls = []
    hold_hours_list = []
    
    for _, row in signals.iterrows():
        coin = row['coin']
        entry_hour = row['hour']
        entry_price = row['price']
        entry_fr = row['funding_rate']
        
        # Find future data
        future = merged[(merged['coin'] == coin) & 
                       (merged['hour'] > entry_hour) &
                       (merged['hour'] <= entry_hour + pd.Timedelta(hours=max_hold))]
        
        if len(future) == 0:
            continue
        
        # Find exit point
        exit_row = None
        for _, frow in future.iterrows():
            if abs(frow['funding_rate']) < exit_thresh:
                exit_row = frow
                break
        
        if exit_row is None:
            exit_row = future.iloc[-1]  # Max hold reached
        
        exit_price = exit_row['price']
        hold_h = int((exit_row['hour'] - entry_hour).total_seconds() / 3600)
        
        # Calculate PnL
        if direction == 'SHORT':
            price_ret = -(exit_price - entry_price) / entry_price
        else:
            price_ret = (exit_price - entry_price) / entry_price
        
        # Funding (simplified: just use entry FR * hold hours as approximation)
        funding_pnl = entry_fr * hold_h * (-1 if direction == 'SHORT' else 1)
        
        net_pnl = price_ret + funding_pnl - 2 * TAKER_FEE
        pnls.append(net_pnl)
        hold_hours_list.append(hold_h)
    
    if len(pnls) < 10:
        return None
    
    return {
        'n': len(pnls),
        'avg': np.mean(pnls),
        'sharpe': np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0,
        'win': np.mean(np.array(pnls) > 0),
        'total': np.sum(pnls),
        'avg_hold': np.mean(hold_hours_list)
    }

print('\nTesting exit thresholds (Entry=0.03%, MaxHold=72h):')
exit_thresholds = [0.00005, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003]

print(f"\n{'Exit':<12} {'N':>7} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10} {'AvgHold':>10}")
print('-'*70)

for exit_t in exit_thresholds:
    res_short = simulate_with_exit(0.0003, exit_t, 72, 'SHORT')
    res_long = simulate_with_exit(0.0003, exit_t, 72, 'LONG')
    
    if res_short and res_long:
        total_n = res_short['n'] + res_long['n']
        total_pnl = res_short['total'] + res_long['total']
        avg_pnl = total_pnl / total_n
        avg_hold = (res_short['avg_hold'] * res_short['n'] + res_long['avg_hold'] * res_long['n']) / total_n
        win = (res_short['win'] * res_short['n'] + res_long['win'] * res_long['n']) / total_n
        
        print(f"{exit_t*100:.3f}%      {total_n:>7} {avg_pnl*100:>+9.2f}%          {win*100:>7.1f}% {total_pnl*100:>+9.0f}% {avg_hold:>9.1f}h")

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '='*90)
print('OPTIMIZATION SUMMARY')
print('='*90)
print('''
TREND FOLLOWING STRATEGY:
  - SHORT when FR < -entry_threshold (go with bearish sentiment)
  - LONG when FR > +entry_threshold (go with bullish sentiment)
  
PARAMETERS:
  1. Entry Threshold: When |FR| exceeds this, enter position
  2. Exit Threshold: When |FR| drops below this, exit position
  3. Max Hold: Maximum hours to hold (safety timeout)

FINDINGS:
  - Lower entry thresholds = more trades but lower quality
  - Higher entry thresholds = fewer trades but higher quality
  - Optimal balance depends on risk preference
''')
