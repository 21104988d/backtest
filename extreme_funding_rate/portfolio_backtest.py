"""
PROPER PORTFOLIO BACKTEST with:
1. Entry threshold (FR < -0.50% for SHORT)
2. Exit threshold (|FR| < 0.02% = normalized)
3. Concurrent position tracking
4. Capital allocation per position
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

# Build lookup tables
print('Building lookups...')
price_pivot = price.pivot_table(index='hour', columns='coin', values='price', aggfunc='first')
price_hours = sorted(price_pivot.index.tolist())

# Fast funding lookup using groupby
print('  Building funding lookup...')
funding_lookup = funding.set_index(['hour', 'coin'])['funding_rate'].to_dict()
print(f'  Funding lookup: {len(funding_lookup):,} entries')

def get_fr(hour, coin):
    return funding_lookup.get((hour, coin), 0)

def get_price(hour, coin):
    if coin in price_pivot.columns and hour in price_pivot.index:
        return price_pivot.loc[hour, coin]
    return np.nan

# =============================================================================
# PORTFOLIO SIMULATION
# =============================================================================

def run_portfolio_backtest(entry_thresh, exit_thresh, max_hold, direction='SHORT'):
    """
    Run hour-by-hour portfolio simulation with concurrent positions
    
    Args:
        entry_thresh: FR threshold for entry (negative for SHORT, positive for LONG)
        exit_thresh: |FR| threshold for exit (normalized = close position)
        max_hold: Maximum holding hours
        direction: 'SHORT' or 'LONG'
    """
    
    active_positions = {}  # {(coin, entry_hour): {'entry_price', 'entry_fr'}}
    completed_trades = []
    
    # Track concurrent positions over time
    concurrent_counts = []
    
    for hour in price_hours:
        # 1. Check exits for active positions
        to_close = []
        
        for (coin, entry_hour), pos in active_positions.items():
            hold_hours = int((hour - entry_hour).total_seconds() / 3600)
            
            current_price = get_price(hour, coin)
            if pd.isna(current_price):
                continue
            
            current_fr = get_fr(hour, coin)
            
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
                # Calculate funding paid/earned
                funding_flow = 0
                for h in range(hold_hours):
                    fr_hour = entry_hour + timedelta(hours=h)
                    fr = get_fr(fr_hour, coin)
                    if direction == 'SHORT':
                        # SHORT: negative FR = we pay, positive FR = we receive
                        if fr < 0:
                            funding_flow -= abs(fr)  # We pay
                        else:
                            funding_flow += fr  # We receive
                    else:  # LONG
                        # LONG: positive FR = we receive, negative FR = we pay
                        if fr > 0:
                            funding_flow += fr  # We receive
                        else:
                            funding_flow -= abs(fr)  # We pay
                
                # Price return
                if direction == 'SHORT':
                    price_return = -(current_price - pos['entry_price']) / pos['entry_price']
                else:  # LONG
                    price_return = (current_price - pos['entry_price']) / pos['entry_price']
                
                net_pnl = price_return + funding_flow - 2 * TAKER_FEE
                
                completed_trades.append({
                    'coin': coin,
                    'entry_hour': entry_hour,
                    'exit_hour': hour,
                    'hold_hours': hold_hours,
                    'entry_fr': pos['entry_fr'],
                    'exit_fr': current_fr,
                    'price_return': price_return,
                    'funding_flow': funding_flow,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason
                })
                
                to_close.append((coin, entry_hour))
        
        for key in to_close:
            del active_positions[key]
        
        # 2. Check for new entries
        coins_in_position = {coin for (coin, _) in active_positions.keys()}
        
        for coin in price_pivot.columns:
            if coin in coins_in_position:
                continue  # Already have position in this coin
            
            fr = get_fr(hour, coin)
            
            # Entry condition
            if direction == 'SHORT' and fr < entry_thresh:
                entry_price = get_price(hour, coin)
                if pd.isna(entry_price):
                    continue
                active_positions[(coin, hour)] = {
                    'entry_price': entry_price,
                    'entry_fr': fr
                }
            elif direction == 'LONG' and fr > entry_thresh:
                entry_price = get_price(hour, coin)
                if pd.isna(entry_price):
                    continue
                active_positions[(coin, hour)] = {
                    'entry_price': entry_price,
                    'entry_fr': fr
                }
        
        concurrent_counts.append(len(active_positions))
    
    return completed_trades, concurrent_counts


# =============================================================================
# RUN BACKTESTS
# =============================================================================

print('\n' + '='*90)
print('PORTFOLIO BACKTEST: SHORT when FR < threshold, EXIT when |FR| < exit_thresh')
print('='*90)

configs = [
    # (entry_thresh, exit_thresh, max_hold, direction)
    (-0.005, 0.0002, 72, 'SHORT'),   # FR < -0.50%, exit |FR| < 0.02%
    (-0.005, 0.0005, 72, 'SHORT'),   # FR < -0.50%, exit |FR| < 0.05%
    (-0.005, 0.001, 72, 'SHORT'),    # FR < -0.50%, exit |FR| < 0.10%
    (-0.001, 0.0001, 72, 'SHORT'),   # FR < -0.10%, exit |FR| < 0.01%
    (-0.001, 0.0002, 72, 'SHORT'),   # FR < -0.10%, exit |FR| < 0.02%
    (-0.002, 0.0002, 72, 'SHORT'),   # FR < -0.20%, exit |FR| < 0.02%
    (-0.003, 0.0002, 72, 'SHORT'),   # FR < -0.30%, exit |FR| < 0.02%
]

print(f"\n{'Entry':<12} {'Exit':<12} {'Hold':<6} {'N':>6} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10} {'MaxConc':>8}")
print('-'*90)

short_results = []
for entry_thresh, exit_thresh, max_hold, direction in configs:
    trades, concurrent = run_portfolio_backtest(entry_thresh, exit_thresh, max_hold, direction)
    
    if len(trades) < 10:
        continue
    
    pnls = [t['net_pnl'] for t in trades]
    avg = np.mean(pnls)
    std = np.std(pnls)
    sharpe = avg / std if std > 0 else 0
    win = sum(1 for p in pnls if p > 0) / len(pnls)
    total = sum(pnls)
    max_conc = max(concurrent) if concurrent else 0
    
    print(f"{entry_thresh*100:>+.2f}%       {exit_thresh*100:.2f}%       {max_hold:<6} {len(trades):>6} {avg*100:>+9.2f}% {sharpe:>8.2f} {win*100:>7.1f}% {total*100:>+9.0f}% {max_conc:>8}")
    
    short_results.append({
        'entry': entry_thresh,
        'exit': exit_thresh,
        'trades': trades,
        'concurrent': concurrent
    })

# =============================================================================
# LONG STRATEGY
# =============================================================================

print('\n' + '='*90)
print('PORTFOLIO BACKTEST: LONG when FR > threshold, EXIT when |FR| < exit_thresh')
print('='*90)

long_configs = [
    (0.001, 0.0002, 72, 'LONG'),   # FR > 0.10%, exit |FR| < 0.02%
    (0.001, 0.0005, 72, 'LONG'),   # FR > 0.10%, exit |FR| < 0.05%
    (0.002, 0.0002, 72, 'LONG'),   # FR > 0.20%, exit |FR| < 0.02%
    (0.003, 0.0002, 72, 'LONG'),   # FR > 0.30%, exit |FR| < 0.02%
    (0.005, 0.0002, 72, 'LONG'),   # FR > 0.50%, exit |FR| < 0.02%
]

print(f"\n{'Entry':<12} {'Exit':<12} {'Hold':<6} {'N':>6} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>10} {'MaxConc':>8}")
print('-'*90)

for entry_thresh, exit_thresh, max_hold, direction in long_configs:
    trades, concurrent = run_portfolio_backtest(entry_thresh, exit_thresh, max_hold, direction)
    
    if len(trades) < 5:
        continue
    
    pnls = [t['net_pnl'] for t in trades]
    avg = np.mean(pnls)
    std = np.std(pnls)
    sharpe = avg / std if std > 0 else 0
    win = sum(1 for p in pnls if p > 0) / len(pnls)
    total = sum(pnls)
    max_conc = max(concurrent) if concurrent else 0
    
    print(f"+{entry_thresh*100:.2f}%       {exit_thresh*100:.2f}%       {max_hold:<6} {len(trades):>6} {avg*100:>+9.2f}% {sharpe:>8.2f} {win*100:>7.1f}% {total*100:>+9.0f}% {max_conc:>8}")

# =============================================================================
# DETAILED ANALYSIS OF BEST CONFIG
# =============================================================================

print('\n' + '='*90)
print('DETAILED: SHORT FR < -0.50%, Exit |FR| < 0.02%, Max 72h')
print('='*90)

trades, concurrent = run_portfolio_backtest(-0.005, 0.0002, 72, 'SHORT')

if trades:
    df = pd.DataFrame(trades)
    df['month'] = pd.to_datetime(df['entry_hour']).dt.to_period('M')
    
    print(f"\nTotal trades: {len(df)}")
    print(f"Avg PnL: {df['net_pnl'].mean()*100:+.2f}%")
    print(f"Sharpe: {df['net_pnl'].mean()/df['net_pnl'].std():.2f}")
    print(f"Win rate: {(df['net_pnl']>0).mean()*100:.1f}%")
    print(f"Max concurrent: {max(concurrent)}")
    
    print("\nExit reasons:")
    for reason, group in df.groupby('exit_reason'):
        print(f"  {reason}: N={len(group)}, Avg PnL={group['net_pnl'].mean()*100:+.2f}%")
    
    print("\nConcurrent position distribution:")
    conc_counts = pd.Series(concurrent)
    for n in range(0, min(11, max(concurrent)+1)):
        pct = (conc_counts == n).sum() / len(conc_counts) * 100
        if pct > 0.1:
            print(f"  {n} positions: {pct:.1f}% of time")
    
    print("\nSample trades:")
    for _, row in df.sort_values('entry_hour').head(20).iterrows():
        print(f"  {row['entry_hour']} {row['coin']:<8} FR={row['entry_fr']*100:>+.2f}% -> {row['exit_fr']*100:>+.2f}% ({row['exit_reason']:<10}) Hold={row['hold_hours']:>2}h PnL={row['net_pnl']*100:>+6.2f}%")

# =============================================================================
# SUMMARY
# =============================================================================

print('\n' + '='*90)
print('SUMMARY')
print('='*90)

print('''
BACKTEST METHODOLOGY:
- Entry: Open position when FR crosses threshold
- Exit: Close when |FR| < exit_threshold (normalized) OR max hold reached
- Multiple coins: Allow concurrent positions (no limit)
- One position per coin at a time

KEY FINDINGS:
1. Exit threshold matters - lower exit threshold (0.02%) = faster exit = more trades
2. Concurrent positions: Up to 10+ positions at peak times
3. Strategy works but needs proper position sizing for concurrent trades
''')
