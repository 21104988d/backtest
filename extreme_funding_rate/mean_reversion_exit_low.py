"""
Mean Reversion - Two-Sided Exit Optimization

Entry: 0.0010% < |FR| < 0.0015%
Exit High: 0.0050% (take profit when FR spikes)
Exit Low: TEST different values

Also analyze concurrent positions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

POSITION_SIZE = 100
TAKER_FEE = 0.00045

# Fixed parameters
ENTRY_LOW = 0.000010   # 0.0010%
ENTRY_HIGH = 0.000015  # 0.0015%
EXIT_HIGH = 0.000050   # 0.0050%

print("=" * 80)
print("MEAN REVERSION - EXIT LOW OPTIMIZATION")
print("=" * 80)
print(f"\nFixed Parameters:")
print(f"  Entry: {ENTRY_LOW*100:.4f}% < |FR| < {ENTRY_HIGH*100:.4f}%")
print(f"  Exit High: {EXIT_HIGH*100:.4f}%")

# Load data
print("\nLoading data...")
base_path = Path(__file__).parent

funding = pd.read_csv(base_path / 'funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)
funding['hour'] = funding['datetime'].dt.floor('h').dt.tz_localize(None)

price = pd.read_csv(base_path / 'price_history.csv')
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
price['hour'] = price['timestamp'].dt.floor('h').dt.tz_localize(None)

merged = pd.merge(
    funding[['hour', 'coin', 'funding_rate']],
    price[['hour', 'coin', 'price']],
    on=['hour', 'coin'],
    how='inner'
).sort_values(['hour', 'coin']).reset_index(drop=True)

merged['abs_fr'] = merged['funding_rate'].abs()

print(f"  Records: {len(merged):,}")
print(f"  Hours: {merged['hour'].nunique():,}")

hours = sorted(merged['hour'].unique())
hour_groups = {h: g for h, g in merged.groupby('hour')}


def test_strategy(exit_low, return_details=False):
    """
    Test two-sided exit strategy.
    """
    
    positions = {}
    trades = []
    hourly_stats = []
    
    for hour in hours:
        if hour not in hour_groups:
            continue
        
        df = hour_groups[hour]
        coin_data = dict(zip(
            df['coin'],
            [{'fr': fr, 'abs_fr': afr, 'price': p} 
             for fr, afr, p in zip(df['funding_rate'], df['abs_fr'], df['price'])]
        ))
        
        available_coins = set(coin_data.keys())
        
        # Check exits
        for coin in list(positions.keys()):
            if coin not in available_coins:
                continue
            
            data = coin_data[coin]
            pos = positions[coin]
            
            hold_hours = (hour - pos['entry_hour']).total_seconds() / 3600
            
            exit_up = data['abs_fr'] >= EXIT_HIGH
            exit_down = data['abs_fr'] <= exit_low
            
            if exit_up or exit_down:
                exit_price = data['price']
                entry_price = pos['entry_price']
                direction = pos['direction']
                
                price_return = (exit_price - entry_price) / entry_price
                position_pnl = direction * price_return * POSITION_SIZE
                funding_pnl = pos['funding_accumulated']
                fees = POSITION_SIZE * TAKER_FEE * 2
                
                exit_reason = 'up' if exit_up else 'down'
                
                trades.append({
                    'coin': coin,
                    'entry_hour': pos['entry_hour'],
                    'exit_hour': hour,
                    'hold_hours': hold_hours,
                    'direction': 'SHORT' if direction == -1 else 'LONG',
                    'price_pnl': position_pnl,
                    'funding_pnl': funding_pnl,
                    'fees': fees,
                    'exit_reason': exit_reason,
                    'net_pnl': position_pnl + funding_pnl - fees
                })
                
                del positions[coin]
        
        # Update funding
        for coin in positions:
            if coin in coin_data:
                positions[coin]['funding_accumulated'] += coin_data[coin]['abs_fr'] * POSITION_SIZE
        
        # Check entries
        for coin in available_coins:
            if coin in positions:
                continue
            
            data = coin_data[coin]
            
            if data['abs_fr'] > ENTRY_LOW and data['abs_fr'] < ENTRY_HIGH:
                direction = -1 if data['fr'] > 0 else 1
                
                positions[coin] = {
                    'direction': direction,
                    'entry_hour': hour,
                    'entry_price': data['price'],
                    'funding_accumulated': 0.0
                }
        
        # Track hourly stats
        n_long = sum(1 for p in positions.values() if p['direction'] == 1)
        n_short = sum(1 for p in positions.values() if p['direction'] == -1)
        
        hourly_stats.append({
            'hour': hour,
            'n_positions': len(positions),
            'n_long': n_long,
            'n_short': n_short
        })
    
    if not trades:
        return None
    
    trades_df = pd.DataFrame(trades)
    hourly_df = pd.DataFrame(hourly_stats)
    
    total_net = trades_df['net_pnl'].sum()
    wins = (trades_df['net_pnl'] > 0).sum()
    exit_counts = trades_df['exit_reason'].value_counts().to_dict()
    
    # PnL by exit reason
    pnl_by_reason = trades_df.groupby('exit_reason')['net_pnl'].agg(['sum', 'mean', 'count'])
    
    result = {
        'trades': len(trades_df),
        'net_pnl': total_net,
        'price_pnl': trades_df['price_pnl'].sum(),
        'funding_pnl': trades_df['funding_pnl'].sum(),
        'fees': trades_df['fees'].sum(),
        'avg_hold': trades_df['hold_hours'].mean(),
        'win_rate': wins / len(trades_df) * 100,
        'avg_net_pct': total_net / len(trades_df) / POSITION_SIZE * 100,
        'open_positions': len(positions),
        'exit_up': exit_counts.get('up', 0),
        'exit_down': exit_counts.get('down', 0),
        'avg_positions': hourly_df['n_positions'].mean(),
        'max_positions': hourly_df['n_positions'].max(),
        'p50_positions': hourly_df['n_positions'].quantile(0.50),
        'p75_positions': hourly_df['n_positions'].quantile(0.75),
        'p95_positions': hourly_df['n_positions'].quantile(0.95),
        'p99_positions': hourly_df['n_positions'].quantile(0.99),
        'pnl_by_reason': pnl_by_reason
    }
    
    if return_details:
        result['trades_df'] = trades_df
        result['hourly_df'] = hourly_df
    
    return result


# =============================================================================
# TEST DIFFERENT EXIT_LOW THRESHOLDS
# =============================================================================

print("\n" + "=" * 80)
print("TEST: Different Exit Low Thresholds")
print("=" * 80)

exit_lows = [0.000000, 0.000002, 0.000004, 0.000005, 0.000006, 0.000007, 0.000008, 0.000009, 0.000010]

print(f"\n{'Exit Low':>10} | {'Trades':>7} | {'Hold':>7} | {'Win%':>6} | {'Net PnL':>10} | {'Avg%':>8} | {'Exit Up':>8} | {'Exit Down':>9} | {'Avg Pos':>8} | {'Max Pos':>8}")
print("-" * 120)

for exit_low in exit_lows:
    result = test_strategy(exit_low)
    if result:
        exit_str = f"{exit_low*100:.4f}%"
        print(f"{exit_str:>10} | {result['trades']:>7,} | {result['avg_hold']:>5.1f}h | {result['win_rate']:>5.1f}% | ${result['net_pnl']:>8,.0f} | {result['avg_net_pct']:>7.3f}% | {result['exit_up']:>8,} | {result['exit_down']:>9,} | {result['avg_positions']:>7.1f} | {result['max_positions']:>8}")

# =============================================================================
# DETAILED ANALYSIS FOR BEST EXIT_LOW
# =============================================================================

print("\n" + "=" * 80)
print("DETAILED ANALYSIS: Exit Low = 0.0005%")
print("=" * 80)

result = test_strategy(0.000005, return_details=True)

if result:
    print(f"\n--- TRADE STATISTICS ---")
    print(f"Total Trades: {result['trades']:,}")
    print(f"Exit Up (Take Profit): {result['exit_up']:,}")
    print(f"Exit Down (Cut Loss): {result['exit_down']:,}")
    print(f"Still Open: {result['open_positions']}")
    
    print(f"\n--- HOLDING TIME ---")
    print(f"Avg Hold: {result['avg_hold']:.1f} hours")
    
    print(f"\n--- PNL BREAKDOWN ---")
    print(f"Price PnL:        ${result['price_pnl']:>12,.2f}")
    print(f"Funding Received: ${result['funding_pnl']:>12,.2f}")
    print(f"Fees:             ${-result['fees']:>12,.2f}")
    print(f"{'─' * 30}")
    print(f"NET PNL:          ${result['net_pnl']:>12,.2f}")
    
    print(f"\n--- PNL BY EXIT REASON ---")
    print(result['pnl_by_reason'].to_string())
    
    print(f"\n--- CONCURRENT POSITIONS ---")
    print(f"Average: {result['avg_positions']:.1f}")
    print(f"Max: {result['max_positions']}")
    print(f"P50 (Median): {result['p50_positions']:.0f}")
    print(f"P75: {result['p75_positions']:.0f}")
    print(f"P95: {result['p95_positions']:.0f}")
    print(f"P99: {result['p99_positions']:.0f}")
    
    # Win rate by exit reason
    trades_df = result['trades_df']
    print(f"\n--- WIN RATE BY EXIT REASON ---")
    for reason in ['up', 'down']:
        subset = trades_df[trades_df['exit_reason'] == reason]
        if len(subset) > 0:
            wins = (subset['net_pnl'] > 0).sum()
            avg_pnl = subset['net_pnl'].mean()
            avg_hold = subset['hold_hours'].mean()
            print(f"  Exit {reason.upper()}: {len(subset):,} trades, Win Rate: {wins/len(subset)*100:.1f}%, Avg PnL: ${avg_pnl:.2f}, Avg Hold: {avg_hold:.1f}h")

# =============================================================================
# COMPARE: Exit Low = 0, 0.0005%, 0.0008%, 0.0010%
# =============================================================================

print("\n" + "=" * 80)
print("COMPARISON: Key Exit Low Values")
print("=" * 80)

key_exit_lows = [0.000000, 0.000005, 0.000008, 0.000010]

for exit_low in key_exit_lows:
    result = test_strategy(exit_low, return_details=True)
    if result:
        print(f"\n{'='*60}")
        print(f"Exit Low: {exit_low*100:.4f}%")
        print(f"{'='*60}")
        print(f"Trades: {result['trades']:,} (Up: {result['exit_up']:,}, Down: {result['exit_down']:,})")
        print(f"Avg Hold: {result['avg_hold']:.1f}h")
        print(f"Net PnL: ${result['net_pnl']:,.2f} ({result['avg_net_pct']:.3f}% per trade)")
        print(f"Win Rate: {result['win_rate']:.1f}%")
        print(f"Concurrent Positions: Avg={result['avg_positions']:.1f}, Max={result['max_positions']}, P95={result['p95_positions']:.0f}")
        
        trades_df = result['trades_df']
        print(f"\nBreakdown by Exit Reason:")
        for reason in ['up', 'down']:
            subset = trades_df[trades_df['exit_reason'] == reason]
            if len(subset) > 0:
                wins = (subset['net_pnl'] > 0).sum()
                total_pnl = subset['net_pnl'].sum()
                avg_pnl = subset['net_pnl'].mean()
                print(f"  {reason.upper():>5}: {len(subset):>6,} trades | Total: ${total_pnl:>10,.0f} | Avg: ${avg_pnl:>6.2f} | Win: {wins/len(subset)*100:>5.1f}%")

# =============================================================================
# YEARLY BREAKDOWN FOR BEST CONFIG
# =============================================================================

print("\n" + "=" * 80)
print("YEARLY BREAKDOWN: Exit Low = 0.0005%")
print("=" * 80)

result = test_strategy(0.000005, return_details=True)
if result:
    trades_df = result['trades_df']
    trades_df['year'] = trades_df['exit_hour'].dt.year
    
    yearly = trades_df.groupby('year').agg({
        'net_pnl': ['sum', 'count', 'mean'],
        'hold_hours': 'mean'
    }).round(2)
    yearly.columns = ['Net PnL', 'Trades', 'Avg PnL', 'Avg Hold']
    print(yearly.to_string())
    
    # Position distribution over time
    hourly_df = result['hourly_df']
    hourly_df['year'] = hourly_df['hour'].dt.year
    
    print("\n--- AVG CONCURRENT POSITIONS BY YEAR ---")
    yearly_pos = hourly_df.groupby('year')['n_positions'].agg(['mean', 'max', 'std']).round(1)
    yearly_pos.columns = ['Avg Pos', 'Max Pos', 'Std']
    print(yearly_pos.to_string())
