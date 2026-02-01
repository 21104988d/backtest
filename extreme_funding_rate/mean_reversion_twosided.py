"""
Mean Reversion - Two-Sided Exit Test

Entry: entry_low < |FR| < entry_high
Exit: |FR| >= exit_high OR |FR| <= exit_low (two-sided)

This ensures we exit when FR either:
1. Spikes up (mean reversion worked, take profit)
2. Drops too low (signal weakened, cut loss)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

POSITION_SIZE = 100
TAKER_FEE = 0.00045

print("=" * 80)
print("MEAN REVERSION - TWO-SIDED EXIT TEST")
print("=" * 80)

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


def test_two_sided_exit(entry_low, entry_high, exit_high, exit_low=None, max_hold=None):
    """
    Entry: entry_low < |FR| < entry_high
    Exit: |FR| >= exit_high OR |FR| <= exit_low (if specified) OR hold > max_hold
    """
    
    positions = {}
    trades = []
    
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
            
            # Exit conditions
            exit_up = data['abs_fr'] >= exit_high  # FR spiked up (take profit)
            exit_down = exit_low is not None and data['abs_fr'] <= exit_low  # FR dropped (cut loss)
            exit_time = max_hold is not None and hold_hours >= max_hold
            
            if exit_up or exit_down or exit_time:
                exit_price = data['price']
                entry_price = pos['entry_price']
                direction = pos['direction']
                
                price_return = (exit_price - entry_price) / entry_price
                position_pnl = direction * price_return * POSITION_SIZE
                funding_pnl = pos['funding_accumulated']
                fees = POSITION_SIZE * TAKER_FEE * 2
                
                exit_reason = 'up' if exit_up else ('down' if exit_down else 'time')
                
                trades.append({
                    'price_pnl': position_pnl,
                    'funding_pnl': funding_pnl,
                    'fees': fees,
                    'hold_hours': hold_hours,
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
            
            if data['abs_fr'] > entry_low and data['abs_fr'] < entry_high:
                direction = -1 if data['fr'] > 0 else 1
                
                positions[coin] = {
                    'direction': direction,
                    'entry_hour': hour,
                    'entry_price': data['price'],
                    'funding_accumulated': 0.0
                }
    
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    
    total_net = df['net_pnl'].sum()
    wins = (df['net_pnl'] > 0).sum()
    
    # Exit reason breakdown
    exit_counts = df['exit_reason'].value_counts().to_dict()
    
    return {
        'trades': len(df),
        'net_pnl': total_net,
        'avg_hold': df['hold_hours'].mean(),
        'win_rate': wins / len(df) * 100,
        'avg_net_pct': total_net / len(df) / POSITION_SIZE * 100,
        'open_positions': len(positions),
        'exit_up': exit_counts.get('up', 0),
        'exit_down': exit_counts.get('down', 0),
        'exit_time': exit_counts.get('time', 0),
    }


# =============================================================================
# TEST 1: Compare one-sided vs two-sided exit
# =============================================================================

print("\n" + "=" * 80)
print("COMPARISON: One-Sided vs Two-Sided Exit")
print("=" * 80)
print("\nEntry: 0.0010% < |FR| < 0.0015%")

print("\n--- ONE-SIDED EXIT (only exit when FR spikes up) ---")
print(f"{'Exit High':>12} | {'Trades':>7} | {'Avg Hold':>9} | {'Win%':>6} | {'Net PnL':>11} | {'Avg%':>8} | {'Open':>6}")
print("-" * 75)

for exit_high in [0.000020, 0.000025, 0.000030, 0.000040, 0.000050]:
    result = test_two_sided_exit(0.000010, 0.000015, exit_high, exit_low=None)
    if result:
        print(f"{exit_high*100:.4f}% | {result['trades']:>7,} | {result['avg_hold']:>7.1f}h | {result['win_rate']:>5.1f}% | ${result['net_pnl']:>9,.0f} | {result['avg_net_pct']:>7.3f}% | {result['open_positions']:>6}")

print("\n--- TWO-SIDED EXIT (exit when FR < 0.0010% OR FR >= exit_high) ---")
print(f"{'Exit High':>12} | {'Trades':>7} | {'Avg Hold':>9} | {'Win%':>6} | {'Net PnL':>11} | {'Avg%':>8} | {'Exit Up':>8} | {'Exit Down':>10}")
print("-" * 105)

for exit_high in [0.000015, 0.000020, 0.000025, 0.000030, 0.000040, 0.000050]:
    result = test_two_sided_exit(0.000010, 0.000015, exit_high, exit_low=0.000010)
    if result:
        print(f"{exit_high*100:.4f}% | {result['trades']:>7,} | {result['avg_hold']:>7.1f}h | {result['win_rate']:>5.1f}% | ${result['net_pnl']:>9,.0f} | {result['avg_net_pct']:>7.3f}% | {result['exit_up']:>8,} | {result['exit_down']:>10,}")

# =============================================================================
# TEST 2: Different exit_low thresholds
# =============================================================================

print("\n" + "=" * 80)
print("TEST: Different Exit Low Thresholds (with Exit High = 0.0030%)")
print("=" * 80)
print("\nEntry: 0.0010% < |FR| < 0.0015%, Exit High: 0.0030%")

print(f"\n{'Exit Low':>12} | {'Trades':>7} | {'Avg Hold':>9} | {'Win%':>6} | {'Net PnL':>11} | {'Avg%':>8} | {'Exit Up':>8} | {'Exit Down':>10}")
print("-" * 105)

for exit_low in [None, 0.000005, 0.000008, 0.000010, 0.000012]:
    result = test_two_sided_exit(0.000010, 0.000015, 0.000030, exit_low=exit_low)
    if result:
        exit_low_str = "None" if exit_low is None else f"{exit_low*100:.4f}%"
        print(f"{exit_low_str:>12} | {result['trades']:>7,} | {result['avg_hold']:>7.1f}h | {result['win_rate']:>5.1f}% | ${result['net_pnl']:>9,.0f} | {result['avg_net_pct']:>7.3f}% | {result['exit_up']:>8,} | {result['exit_down']:>10,}")

# =============================================================================
# TEST 3: Grid search with two-sided exit
# =============================================================================

print("\n" + "=" * 80)
print("GRID SEARCH: Two-Sided Exit (Exit Low = Entry Low)")
print("=" * 80)

entry_ranges = [
    (0.000005, 0.000015),
    (0.000008, 0.000015),
    (0.000010, 0.000015),
    (0.000010, 0.000020),
    (0.000012, 0.000015),
]

exit_highs = [0.000015, 0.000020, 0.000025, 0.000030, 0.000040]

print(f"\n{'Entry Range':>22} | {'Exit High':>10} | {'Trades':>7} | {'Avg Hold':>9} | {'Win%':>6} | {'Net PnL':>11} | {'Avg%':>8}")
print("-" * 100)

for entry_low, entry_high in entry_ranges:
    for exit_high in exit_highs:
        if exit_high < entry_high:
            continue
        
        # Exit low = entry low (exit if FR drops out of entry range)
        result = test_two_sided_exit(entry_low, entry_high, exit_high, exit_low=entry_low)
        if result:
            entry_str = f"{entry_low*100:.4f}-{entry_high*100:.4f}%"
            exit_str = f"{exit_high*100:.4f}%"
            print(f"{entry_str:>22} | {exit_str:>10} | {result['trades']:>7,} | {result['avg_hold']:>7.1f}h | {result['win_rate']:>5.1f}% | ${result['net_pnl']:>9,.0f} | {result['avg_net_pct']:>7.3f}%")

# =============================================================================
# TEST 4: Best config with max hold
# =============================================================================

print("\n" + "=" * 80)
print("BEST CONFIG + MAX HOLD")
print("=" * 80)
print("\nEntry: 0.0010% < |FR| < 0.0015%")
print("Exit Low: 0.0010% (same as entry low)")

print(f"\n{'Exit High':>10} | {'Max Hold':>9} | {'Trades':>7} | {'Avg Hold':>9} | {'Win%':>6} | {'Net PnL':>11} | {'Avg%':>8}")
print("-" * 85)

for exit_high in [0.000020, 0.000025, 0.000030]:
    for max_hold in [None, 72, 48, 24]:
        result = test_two_sided_exit(0.000010, 0.000015, exit_high, exit_low=0.000010, max_hold=max_hold)
        if result:
            exit_str = f"{exit_high*100:.4f}%"
            max_str = "None" if max_hold is None else f"{max_hold}h"
            print(f"{exit_str:>10} | {max_str:>9} | {result['trades']:>7,} | {result['avg_hold']:>7.1f}h | {result['win_rate']:>5.1f}% | ${result['net_pnl']:>9,.0f} | {result['avg_net_pct']:>7.3f}%")
