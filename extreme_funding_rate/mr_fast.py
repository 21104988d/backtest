"""
Optimized Mean Reversion Test for |FR| < 0.10%
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

# Pre-compute future prices and funding for each row
print('Pre-computing future data...')

# Create coin-grouped data for faster lookups
merged['hour_idx'] = merged.groupby('coin').cumcount()
merged['next_hour'] = merged.groupby('coin')['hour'].shift(-1)

# Pre-compute 72h forward windows
def compute_trades_vectorized(df, entry_low, entry_high, exit_thresh, max_hold, direction):
    """
    Vectorized backtest - much faster
    """
    # Select signals based on funding rate range
    if direction == 'LONG_MR':  # Mean reversion: LONG when FR negative
        mask = (df['funding_rate'] >= entry_low) & (df['funding_rate'] < entry_high)
    elif direction == 'SHORT_MR':  # Mean reversion: SHORT when FR positive
        mask = (df['funding_rate'] > entry_low) & (df['funding_rate'] <= entry_high)
    elif direction == 'SHORT_TF':  # Trend following: SHORT when FR very negative
        mask = df['funding_rate'] < entry_low
    elif direction == 'LONG_TF':  # Trend following: LONG when FR very positive
        mask = df['funding_rate'] > entry_high
    
    signals = df[mask].copy()
    print(f'  Signals: {len(signals):,}')
    
    if len(signals) == 0:
        return []
    
    # For each signal, find exit point
    trades = []
    processed_coins = {}  # Track last exit time per coin
    
    for idx, row in signals.iterrows():
        coin = row['coin']
        entry_hour = row['hour']
        entry_price = row['price']
        entry_fr = row['funding_rate']
        
        # Skip if we still have a position in this coin
        if coin in processed_coins and entry_hour < processed_coins[coin]:
            continue
        
        # Find future data for this coin
        coin_data = df[(df['coin'] == coin) & (df['hour'] > entry_hour) & 
                       (df['hour'] <= entry_hour + timedelta(hours=max_hold))]
        
        if len(coin_data) == 0:
            continue
        
        # Find exit point
        exit_idx = None
        for i, (_, future_row) in enumerate(coin_data.iterrows()):
            if abs(future_row['funding_rate']) < exit_thresh:
                exit_idx = i
                break
        
        if exit_idx is None:
            exit_idx = len(coin_data) - 1
        
        exit_row = coin_data.iloc[exit_idx]
        exit_hour = exit_row['hour']
        exit_price = exit_row['price']
        
        hold_hours = int((exit_hour - entry_hour).total_seconds() / 3600)
        if hold_hours < 1:
            hold_hours = 1
        
        # Calculate funding flow (simplified - assume avg funding rate)
        funding_between = df[(df['coin'] == coin) & 
                            (df['hour'] >= entry_hour) & 
                            (df['hour'] < exit_hour)]
        
        if 'LONG' in direction:
            # LONG: receive positive FR, pay negative FR
            funding_flow = funding_between['funding_rate'].sum()
        else:
            # SHORT: receive negative FR, pay positive FR  
            funding_flow = -funding_between['funding_rate'].sum()
        
        # Price return
        if 'LONG' in direction:
            price_ret = (exit_price - entry_price) / entry_price
        else:
            price_ret = -(exit_price - entry_price) / entry_price
        
        net_pnl = price_ret + funding_flow - 2 * TAKER_FEE
        trades.append(net_pnl)
        
        # Mark this coin as having position until exit
        processed_coins[coin] = exit_hour
    
    return trades

# =============================================================================
# TEST MEAN REVERSION
# =============================================================================
print('\n' + '='*90)
print('MEAN REVERSION: LONG when FR is NEGATIVE (expect bounce)')
print('='*90)

configs_long = [
    # (entry_low, entry_high, exit_thresh, max_hold, label)
    (-0.001, -0.0003, 0.0002, 72, '-0.10% to -0.03%'),  # Exclude very negative
    (-0.0005, -0.0003, 0.0002, 72, '-0.05% to -0.03%'),
    (-0.001, -0.0005, 0.0002, 72, '-0.10% to -0.05%'),
]

print(f"\n{'Range':<25} {'N':>6} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10}")
print('-'*75)

for entry_low, entry_high, exit_t, hold, label in configs_long:
    trades = compute_trades_vectorized(merged, entry_low, entry_high, exit_t, hold, 'LONG_MR')
    if len(trades) < 10:
        print(f"{label:<25} {'<10':>6}")
        continue
    avg = np.mean(trades)
    std = np.std(trades)
    sharpe = avg/std if std > 0 else 0
    win = sum(1 for p in trades if p > 0) / len(trades)
    total = sum(trades)
    print(f"{label:<25} {len(trades):>6} {avg*100:>+9.2f}% {sharpe:>8.2f} {win*100:>7.1f}% {total*100:>+9.0f}%")

print('\n' + '='*90)
print('MEAN REVERSION: SHORT when FR is POSITIVE (expect drop)')
print('='*90)

configs_short = [
    (0.0003, 0.001, 0.0002, 72, '+0.03% to +0.10%'),  # Exclude very positive
    (0.0003, 0.0005, 0.0002, 72, '+0.03% to +0.05%'),
    (0.0005, 0.001, 0.0002, 72, '+0.05% to +0.10%'),
]

print(f"\n{'Range':<25} {'N':>6} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10}")
print('-'*75)

for entry_low, entry_high, exit_t, hold, label in configs_short:
    trades = compute_trades_vectorized(merged, entry_low, entry_high, exit_t, hold, 'SHORT_MR')
    if len(trades) < 10:
        print(f"{label:<25} {'<10':>6}")
        continue
    avg = np.mean(trades)
    std = np.std(trades)
    sharpe = avg/std if std > 0 else 0
    win = sum(1 for p in trades if p > 0) / len(trades)
    total = sum(trades)
    print(f"{label:<25} {len(trades):>6} {avg*100:>+9.2f}% {sharpe:>8.2f} {win*100:>7.1f}% {total*100:>+9.0f}%")

# =============================================================================
# COMPARISON: Trend Following for extreme FR
# =============================================================================
print('\n' + '='*90)
print('TREND FOLLOWING: For comparison (|FR| > 0.10%)')
print('='*90)

print('\nTrend Follow SHORT (FR < -0.10%):')
tf_short = compute_trades_vectorized(merged, -0.001, 0, 0.0002, 72, 'SHORT_TF')
if len(tf_short) >= 10:
    print(f"  N={len(tf_short)}, Avg={np.mean(tf_short)*100:+.2f}%, Win={(sum(1 for p in tf_short if p>0)/len(tf_short))*100:.1f}%, Total={sum(tf_short)*100:+.0f}%")

print('\nTrend Follow LONG (FR > +0.10%):')
tf_long = compute_trades_vectorized(merged, 0, 0.001, 0.0002, 72, 'LONG_TF')
if len(tf_long) >= 10:
    print(f"  N={len(tf_long)}, Avg={np.mean(tf_long)*100:+.2f}%, Win={(sum(1 for p in tf_long if p>0)/len(tf_long))*100:.1f}%, Total={sum(tf_long)*100:+.0f}%")

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '='*90)
print('SUMMARY: Comparison of Strategies')
print('='*90)
print('''
MEAN REVERSION (for |FR| < 0.10%):
  - LONG when FR is slightly negative (-0.03% to -0.10%)
  - SHORT when FR is slightly positive (+0.03% to +0.10%)
  
TREND FOLLOWING (for |FR| > 0.10%):
  - SHORT when FR < -0.10%
  - LONG when FR > +0.10%
  
Key question: Can we use BOTH strategies together?
''')
