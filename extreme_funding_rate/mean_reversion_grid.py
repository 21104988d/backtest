"""
Mean Reversion Strategy - Grid Search Entry/Exit Thresholds

Test different combinations:
- Entry: FR_LOW < |FR| < FR_HIGH
- Exit: |FR| >= EXIT_THRESHOLD

Find optimal entry range and exit threshold for best performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

POSITION_SIZE = 100
TAKER_FEE = 0.00045

print("=" * 80)
print("MEAN REVERSION - GRID SEARCH ENTRY/EXIT THRESHOLDS")
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
print(f"  Date range: {merged['hour'].min()} to {merged['hour'].max()}")

# Pre-compute hour groups
hours = sorted(merged['hour'].unique())
hour_groups = {h: g for h, g in merged.groupby('hour')}


def test_strategy(entry_low, entry_high, exit_thresh, max_hold=None):
    """
    Test mean-reversion strategy with specific thresholds.
    
    Entry: entry_low < |FR| < entry_high
    Exit: |FR| >= exit_thresh OR hold > max_hold (if specified)
    """
    
    positions = {}
    trades = []
    hourly_positions = []
    
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
            exit_fr = data['abs_fr'] >= exit_thresh
            exit_time = max_hold is not None and hold_hours >= max_hold
            
            if exit_fr or exit_time:
                exit_price = data['price']
                entry_price = pos['entry_price']
                direction = pos['direction']
                
                price_return = (exit_price - entry_price) / entry_price
                position_pnl = direction * price_return * POSITION_SIZE
                funding_pnl = pos['funding_accumulated']
                fees = POSITION_SIZE * TAKER_FEE * 2
                
                trades.append({
                    'price_pnl': position_pnl,
                    'funding_pnl': funding_pnl,
                    'fees': fees,
                    'hold_hours': hold_hours,
                    'exit_reason': 'fr' if exit_fr else 'time',
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
            
            # Entry: entry_low < |FR| < entry_high
            if data['abs_fr'] > entry_low and data['abs_fr'] < entry_high:
                direction = -1 if data['fr'] > 0 else 1
                
                positions[coin] = {
                    'direction': direction,
                    'entry_hour': hour,
                    'entry_price': data['price'],
                    'funding_accumulated': 0.0
                }
        
        hourly_positions.append(len(positions))
    
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    
    total_price = df['price_pnl'].sum()
    total_funding = df['funding_pnl'].sum()
    total_fees = df['fees'].sum()
    total_net = total_price + total_funding - total_fees
    
    wins = (df['net_pnl'] > 0).sum()
    win_rate = wins / len(df) * 100
    
    return {
        'trades': len(df),
        'price_pnl': total_price,
        'funding': total_funding,
        'fees': total_fees,
        'net_pnl': total_net,
        'avg_hold': df['hold_hours'].mean(),
        'win_rate': win_rate,
        'avg_net_pct': total_net / len(df) / POSITION_SIZE * 100,
        'avg_positions': np.mean(hourly_positions),
        'max_positions': max(hourly_positions),
        'open_positions': len(positions)
    }


# =============================================================================
# GRID SEARCH
# =============================================================================

print("\n" + "=" * 80)
print("GRID SEARCH: Entry Range & Exit Threshold (No Max Hold)")
print("=" * 80)

# Entry ranges to test
entry_ranges = [
    (0.000000, 0.000015),  # 0 to 0.0015%
    (0.000005, 0.000015),  # 0.0005% to 0.0015%
    (0.000010, 0.000015),  # 0.0010% to 0.0015%
    (0.000010, 0.000020),  # 0.0010% to 0.0020%
    (0.000012, 0.000015),  # 0.0012% to 0.0015%
    (0.000000, 0.000010),  # 0 to 0.0010%
    (0.000005, 0.000010),  # 0.0005% to 0.0010%
]

# Exit thresholds to test
exit_thresholds = [0.000015, 0.000020, 0.000025, 0.000030]

print(f"\n{'Entry Range':>22} | {'Exit':>8} | {'Trades':>7} | {'Avg Hold':>9} | {'Win%':>6} | {'Price PnL':>11} | {'Funding':>9} | {'Net PnL':>11} | {'Avg%':>8} | {'Avg Pos':>8}")
print("-" * 135)

results = []

for entry_low, entry_high in entry_ranges:
    for exit_t in exit_thresholds:
        # Only test where exit >= entry_high (otherwise no trades)
        if exit_t < entry_high:
            continue
            
        result = test_strategy(entry_low, entry_high, exit_t)
        if result:
            entry_str = f"{entry_low*100:.4f}-{entry_high*100:.4f}%"
            exit_str = f"{exit_t*100:.4f}%"
            
            results.append({
                'entry_low': entry_low,
                'entry_high': entry_high,
                'exit': exit_t,
                **result
            })
            
            print(f"{entry_str:>22} | {exit_str:>8} | {result['trades']:>7,} | {result['avg_hold']:>7.1f}h | {result['win_rate']:>5.1f}% | ${result['price_pnl']:>9,.0f} | ${result['funding']:>7,.0f} | ${result['net_pnl']:>9,.0f} | {result['avg_net_pct']:>7.3f}% | {result['avg_positions']:>7.1f}")

# =============================================================================
# WITH MAX HOLD TIME
# =============================================================================

print("\n" + "=" * 80)
print("GRID SEARCH: With Max Hold = 72h")
print("=" * 80)

print(f"\n{'Entry Range':>22} | {'Exit':>8} | {'Trades':>7} | {'Avg Hold':>9} | {'Win%':>6} | {'Price PnL':>11} | {'Funding':>9} | {'Net PnL':>11} | {'Avg%':>8} | {'Avg Pos':>8}")
print("-" * 135)

for entry_low, entry_high in entry_ranges:
    for exit_t in exit_thresholds:
        if exit_t < entry_high:
            continue
            
        result = test_strategy(entry_low, entry_high, exit_t, max_hold=72)
        if result:
            entry_str = f"{entry_low*100:.4f}-{entry_high*100:.4f}%"
            exit_str = f"{exit_t*100:.4f}%"
            
            print(f"{entry_str:>22} | {exit_str:>8} | {result['trades']:>7,} | {result['avg_hold']:>7.1f}h | {result['win_rate']:>5.1f}% | ${result['price_pnl']:>9,.0f} | ${result['funding']:>7,.0f} | ${result['net_pnl']:>9,.0f} | {result['avg_net_pct']:>7.3f}% | {result['avg_positions']:>7.1f}")

# =============================================================================
# BEST COMBINATIONS
# =============================================================================

print("\n" + "=" * 80)
print("DETAILED TEST: Best Entry Range (0.0010% - 0.0015%)")
print("=" * 80)

# Test 0.0010% - 0.0015% entry with various exit thresholds and max holds
entry_low = 0.000010
entry_high = 0.000015

max_holds = [24, 48, 72, 96, 120, 168, None]
exit_thresholds = [0.000015, 0.000020, 0.000025, 0.000030, 0.000040, 0.000050]

print(f"\nEntry: {entry_low*100:.4f}% < |FR| < {entry_high*100:.4f}%")
print(f"\n{'Exit':>10} | {'MaxHold':>8} | {'Trades':>7} | {'Avg Hold':>9} | {'Win%':>6} | {'Net PnL':>11} | {'Avg%':>8} | {'Avg Pos':>8} | {'Open':>6}")
print("-" * 105)

for exit_t in exit_thresholds:
    for max_h in max_holds:
        result = test_strategy(entry_low, entry_high, exit_t, max_hold=max_h)
        if result:
            exit_str = f"{exit_t*100:.4f}%"
            max_str = f"{max_h}h" if max_h else "None"
            
            print(f"{exit_str:>10} | {max_str:>8} | {result['trades']:>7,} | {result['avg_hold']:>7.1f}h | {result['win_rate']:>5.1f}% | ${result['net_pnl']:>9,.0f} | {result['avg_net_pct']:>7.3f}% | {result['avg_positions']:>7.1f} | {result['open_positions']:>6}")

# =============================================================================
# TIGHTER ENTRY RANGE
# =============================================================================

print("\n" + "=" * 80)
print("TIGHTER ENTRY RANGES")
print("=" * 80)

tight_ranges = [
    (0.000011, 0.000015),
    (0.000012, 0.000015),
    (0.000013, 0.000015),
    (0.000014, 0.000015),
    (0.000010, 0.000014),
    (0.000010, 0.000013),
    (0.000010, 0.000012),
]

print(f"\n{'Entry Range':>22} | {'Exit':>10} | {'Trades':>7} | {'Avg Hold':>9} | {'Win%':>6} | {'Net PnL':>11} | {'Avg%':>8}")
print("-" * 95)

for entry_low, entry_high in tight_ranges:
    for exit_t in [0.000015, 0.000020, 0.000025]:
        result = test_strategy(entry_low, entry_high, exit_t, max_hold=72)
        if result and result['trades'] > 100:
            entry_str = f"{entry_low*100:.4f}-{entry_high*100:.4f}%"
            exit_str = f"{exit_t*100:.4f}%"
            
            print(f"{entry_str:>22} | {exit_str:>10} | {result['trades']:>7,} | {result['avg_hold']:>7.1f}h | {result['win_rate']:>5.1f}% | ${result['net_pnl']:>9,.0f} | {result['avg_net_pct']:>7.3f}%")
