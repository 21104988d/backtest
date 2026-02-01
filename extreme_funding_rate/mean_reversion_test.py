"""
Test MEAN REVERSION for lower funding rates (|FR| < 0.10%)

Mean Reversion Logic:
- Slightly negative FR (-0.03% to -0.10%): GO LONG (expect price to bounce)
- Slightly positive FR (+0.03% to +0.10%): GO SHORT (expect price to drop)

This is OPPOSITE of trend following.
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

def run_backtest(entry_low, entry_high, exit_thresh, max_hold, direction):
    """
    Run backtest for funding rate BETWEEN entry_low and entry_high
    
    For mean reversion:
    - LONG when FR between entry_low and entry_high (negative range)
    - SHORT when FR between entry_low and entry_high (positive range)
    """
    
    if direction == 'LONG':
        # LONG when FR is negative (mean reversion: expect bounce)
        signals = merged[(merged['funding_rate'] >= entry_low) & 
                        (merged['funding_rate'] < entry_high)][['hour', 'coin', 'funding_rate', 'price']].copy()
    else:
        # SHORT when FR is positive (mean reversion: expect drop)
        signals = merged[(merged['funding_rate'] > entry_low) & 
                        (merged['funding_rate'] <= entry_high)][['hour', 'coin', 'funding_rate', 'price']].copy()
    
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
            
            # Exit when FR normalizes (close to 0) or timeout
            should_exit = abs(current_fr) < exit_thresh or hold_hours >= max_hold
            
            if should_exit:
                # Calculate funding flow
                funding_flow = 0
                for h in range(hold_hours):
                    fr_h = entry_hour + timedelta(hours=h)
                    fr = get_fr(fr_h, coin)
                    if direction == 'LONG':
                        # LONG: positive FR = we receive, negative FR = we pay
                        funding_flow += (fr if fr > 0 else -abs(fr))
                    else:
                        # SHORT: negative FR = we pay, positive FR = we receive
                        funding_flow += (-abs(fr) if fr < 0 else fr)
                
                if direction == 'LONG':
                    price_ret = (current_price - pos['entry_price']) / pos['entry_price']
                else:
                    price_ret = -(current_price - pos['entry_price']) / pos['entry_price']
                
                net_pnl = price_ret + funding_flow - 2 * TAKER_FEE
                trades.append({
                    'net_pnl': net_pnl,
                    'hold_hours': hold_hours,
                    'entry_fr': pos['entry_fr'],
                    'exit_fr': current_fr
                })
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

# =============================================================================
# MEAN REVERSION: LONG when FR is negative (between -0.10% and -0.03%)
# =============================================================================
print('\n' + '='*90)
print('MEAN REVERSION: LONG when FR is NEGATIVE (expect bounce)')
print('Exit when |FR| < threshold or 72h timeout')
print('='*90)

long_configs = [
    # (entry_low, entry_high, exit_thresh, max_hold)
    (-0.001, 0, 0.0001, 72),      # -0.10% to 0%, exit 0.01%
    (-0.001, 0, 0.0002, 72),      # -0.10% to 0%, exit 0.02%
    (-0.001, -0.0003, 0.0002, 72), # -0.10% to -0.03%, exit 0.02%
    (-0.0005, 0, 0.0002, 72),     # -0.05% to 0%, exit 0.02%
    (-0.0005, -0.0003, 0.0002, 72), # -0.05% to -0.03%, exit 0.02%
    (-0.001, -0.0005, 0.0002, 72), # -0.10% to -0.05%, exit 0.02%
]

print(f"\n{'Entry Range':<20} {'Exit':<8} {'N':>6} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10}")
print('-'*80)

for entry_low, entry_high, exit_t, hold in long_configs:
    trades, conc = run_backtest(entry_low, entry_high, exit_t, hold, 'LONG')
    if len(trades) < 10:
        print(f"{entry_low*100:.2f}% to {entry_high*100:.2f}%    {exit_t*100:.2f}%       <10 trades")
        continue
    pnls = [t['net_pnl'] for t in trades]
    avg = np.mean(pnls)
    std = np.std(pnls)
    sharpe = avg/std if std > 0 else 0
    win = sum(1 for p in pnls if p > 0) / len(pnls)
    total = sum(pnls)
    print(f"{entry_low*100:.2f}% to {entry_high*100:.2f}%    {exit_t*100:.2f}%   {len(trades):>6} {avg*100:>+9.2f}% {sharpe:>8.2f} {win*100:>7.1f}% {total*100:>+9.0f}%")

# =============================================================================
# MEAN REVERSION: SHORT when FR is positive (between +0.03% and +0.10%)
# =============================================================================
print('\n' + '='*90)
print('MEAN REVERSION: SHORT when FR is POSITIVE (expect drop)')
print('Exit when |FR| < threshold or 72h timeout')
print('='*90)

short_configs = [
    # (entry_low, entry_high, exit_thresh, max_hold)
    (0, 0.001, 0.0001, 72),       # 0% to +0.10%, exit 0.01%
    (0, 0.001, 0.0002, 72),       # 0% to +0.10%, exit 0.02%
    (0.0003, 0.001, 0.0002, 72),  # +0.03% to +0.10%, exit 0.02%
    (0, 0.0005, 0.0002, 72),      # 0% to +0.05%, exit 0.02%
    (0.0003, 0.0005, 0.0002, 72), # +0.03% to +0.05%, exit 0.02%
    (0.0005, 0.001, 0.0002, 72),  # +0.05% to +0.10%, exit 0.02%
]

print(f"\n{'Entry Range':<20} {'Exit':<8} {'N':>6} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10}")
print('-'*80)

for entry_low, entry_high, exit_t, hold in short_configs:
    trades, conc = run_backtest(entry_low, entry_high, exit_t, hold, 'SHORT')
    if len(trades) < 10:
        print(f"+{entry_low*100:.2f}% to +{entry_high*100:.2f}%   {exit_t*100:.2f}%       <10 trades")
        continue
    pnls = [t['net_pnl'] for t in trades]
    avg = np.mean(pnls)
    std = np.std(pnls)
    sharpe = avg/std if std > 0 else 0
    win = sum(1 for p in pnls if p > 0) / len(pnls)
    total = sum(pnls)
    print(f"+{entry_low*100:.2f}% to +{entry_high*100:.2f}%   {exit_t*100:.2f}%   {len(trades):>6} {avg*100:>+9.2f}% {sharpe:>8.2f} {win*100:>7.1f}% {total*100:>+9.0f}%")

# =============================================================================
# COMPARISON: Trend Following vs Mean Reversion
# =============================================================================
print('\n' + '='*90)
print('COMPARISON: TREND FOLLOWING vs MEAN REVERSION')
print('='*90)

print('\n--- TREND FOLLOWING (go WITH the crowd) ---')
print('SHORT when FR < -0.10% | LONG when FR > +0.10%')

# Trend following SHORT
tf_short, _ = run_backtest(-999, -0.001, 0.0002, 72, 'SHORT')  # Hack: use -999 as no lower bound
# Actually need to rewrite for trend following...

print('\n--- Recalculating Trend Following ---')

def run_trend_following(entry_thresh, exit_thresh, max_hold, direction):
    """Trend following: go WITH extreme funding"""
    if direction == 'SHORT':
        signals = merged[merged['funding_rate'] < entry_thresh][['hour', 'coin', 'funding_rate', 'price']].copy()
    else:
        signals = merged[merged['funding_rate'] > entry_thresh][['hour', 'coin', 'funding_rate', 'price']].copy()
    
    signals = signals.sort_values('hour').reset_index(drop=True)
    active = {}
    trades = []
    signal_idx = 0
    
    for hour in all_hours:
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
    
    return trades

# Trend Following
tf_short = run_trend_following(-0.001, 0.0002, 72, 'SHORT')
tf_long = run_trend_following(0.001, 0.0002, 72, 'LONG')

print(f"\nTREND FOLLOWING SHORT (FR < -0.10%):")
print(f"  N={len(tf_short)}, Avg={np.mean(tf_short)*100:+.2f}%, Win={(sum(1 for p in tf_short if p>0)/len(tf_short))*100:.1f}%, Total={sum(tf_short)*100:+.0f}%")

print(f"\nTREND FOLLOWING LONG (FR > +0.10%):")
print(f"  N={len(tf_long)}, Avg={np.mean(tf_long)*100:+.2f}%, Win={(sum(1 for p in tf_long if p>0)/len(tf_long))*100:.1f}%, Total={sum(tf_long)*100:+.0f}%")

# Mean Reversion for comparison
mr_long, _ = run_backtest(-0.001, -0.0003, 0.0002, 72, 'LONG')
mr_short, _ = run_backtest(0.0003, 0.001, 0.0002, 72, 'SHORT')

print(f"\nMEAN REVERSION LONG (FR between -0.10% and -0.03%):")
mr_long_pnls = [t['net_pnl'] for t in mr_long]
print(f"  N={len(mr_long)}, Avg={np.mean(mr_long_pnls)*100:+.2f}%, Win={(sum(1 for p in mr_long_pnls if p>0)/len(mr_long_pnls))*100:.1f}%, Total={sum(mr_long_pnls)*100:+.0f}%")

print(f"\nMEAN REVERSION SHORT (FR between +0.03% and +0.10%):")
mr_short_pnls = [t['net_pnl'] for t in mr_short]
print(f"  N={len(mr_short)}, Avg={np.mean(mr_short_pnls)*100:+.2f}%, Win={(sum(1 for p in mr_short_pnls if p>0)/len(mr_short_pnls))*100:.1f}%, Total={sum(mr_short_pnls)*100:+.0f}%")

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '='*90)
print('SUMMARY')
print('='*90)
print('''
STRATEGY COMPARISON:

1. TREND FOLLOWING (|FR| > 0.10%):
   - SHORT when FR < -0.10% (go with bearish sentiment)
   - LONG when FR > +0.10% (go with bullish sentiment)

2. MEAN REVERSION (|FR| < 0.10%):
   - LONG when FR is slightly negative (expect bounce)
   - SHORT when FR is slightly positive (expect drop)

The question: Does mean reversion work for lower funding rates?
''')
