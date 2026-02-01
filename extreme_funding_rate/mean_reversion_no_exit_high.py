"""
Mean Reversion - Exit Only When FR Drops (No Exit High)

Entry: entry_low < |FR| < entry_high (within 0.0010% - 0.0015%)
Exit: |FR| < exit_low (FR drops, close position)

No exit high - keep position even if FR spikes.
Test tighter entry ranges to reduce concurrent positions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

POSITION_SIZE = 100
TAKER_FEE = 0.00045

print("=" * 80)
print("MEAN REVERSION - EXIT ONLY WHEN FR DROPS")
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


def test_strategy(entry_low, entry_high, exit_low, return_details=False):
    """
    Entry: entry_low < |FR| < entry_high
    Exit: |FR| <= exit_low (FR drops)
    
    No exit high - keep position even if FR spikes above entry_high.
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
        
        # Check exits - ONLY exit when FR drops
        for coin in list(positions.keys()):
            if coin not in available_coins:
                continue
            
            data = coin_data[coin]
            pos = positions[coin]
            
            hold_hours = (hour - pos['entry_hour']).total_seconds() / 3600
            
            # Exit ONLY when FR drops below threshold
            if data['abs_fr'] <= exit_low:
                exit_price = data['price']
                entry_price = pos['entry_price']
                direction = pos['direction']
                
                price_return = (exit_price - entry_price) / entry_price
                position_pnl = direction * price_return * POSITION_SIZE
                funding_pnl = pos['funding_accumulated']
                fees = POSITION_SIZE * TAKER_FEE * 2
                
                trades.append({
                    'coin': coin,
                    'entry_hour': pos['entry_hour'],
                    'exit_hour': hour,
                    'hold_hours': hold_hours,
                    'direction': 'SHORT' if direction == -1 else 'LONG',
                    'price_pnl': position_pnl,
                    'funding_pnl': funding_pnl,
                    'fees': fees,
                    'net_pnl': position_pnl + funding_pnl - fees
                })
                
                del positions[coin]
        
        # Update funding for all open positions
        for coin in positions:
            if coin in coin_data:
                positions[coin]['funding_accumulated'] += coin_data[coin]['abs_fr'] * POSITION_SIZE
        
        # Check entries - only enter within entry range
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
        'avg_positions': hourly_df['n_positions'].mean(),
        'max_positions': hourly_df['n_positions'].max(),
        'p50_positions': hourly_df['n_positions'].quantile(0.50),
        'p75_positions': hourly_df['n_positions'].quantile(0.75),
        'p95_positions': hourly_df['n_positions'].quantile(0.95),
        'p99_positions': hourly_df['n_positions'].quantile(0.99),
    }
    
    if return_details:
        result['trades_df'] = trades_df
        result['hourly_df'] = hourly_df
    
    return result


# =============================================================================
# TEST 1: Full range 0.0010% - 0.0015% with different exit_low
# =============================================================================

print("\n" + "=" * 80)
print("TEST 1: Entry 0.0010% - 0.0015%, Different Exit Low")
print("=" * 80)

exit_lows = [0.000002, 0.000003, 0.000004, 0.000005, 0.000006, 0.000007, 0.000008]

print(f"\n{'Exit Low':>10} | {'Trades':>7} | {'Hold':>7} | {'Win%':>6} | {'Net PnL':>10} | {'Avg%':>8} | {'Avg Pos':>8} | {'Max Pos':>8} | {'P95 Pos':>8}")
print("-" * 105)

for exit_low in exit_lows:
    result = test_strategy(0.000010, 0.000015, exit_low)
    if result:
        exit_str = f"{exit_low*100:.4f}%"
        print(f"{exit_str:>10} | {result['trades']:>7,} | {result['avg_hold']:>5.1f}h | {result['win_rate']:>5.1f}% | ${result['net_pnl']:>8,.0f} | {result['avg_net_pct']:>7.3f}% | {result['avg_positions']:>7.1f} | {result['max_positions']:>8} | {result['p95_positions']:>7.0f}")


# =============================================================================
# TEST 2: Tighter Entry Ranges (reduce concurrent positions)
# =============================================================================

print("\n" + "=" * 80)
print("TEST 2: Tighter Entry Ranges (Exit Low = 0.0005%)")
print("=" * 80)

entry_ranges = [
    (0.000010, 0.000015),  # Full range
    (0.000011, 0.000015),  # 0.0011% - 0.0015%
    (0.000012, 0.000015),  # 0.0012% - 0.0015%
    (0.000013, 0.000015),  # 0.0013% - 0.0015%
    (0.000010, 0.000014),  # 0.0010% - 0.0014%
    (0.000010, 0.000013),  # 0.0010% - 0.0013%
    (0.000011, 0.000014),  # 0.0011% - 0.0014%
    (0.000012, 0.000014),  # 0.0012% - 0.0014%
]

print(f"\n{'Entry Range':>22} | {'Trades':>7} | {'Hold':>7} | {'Win%':>6} | {'Net PnL':>10} | {'Avg%':>8} | {'Avg Pos':>8} | {'Max Pos':>8} | {'P95 Pos':>8}")
print("-" * 120)

for entry_low, entry_high in entry_ranges:
    result = test_strategy(entry_low, entry_high, 0.000005)
    if result:
        entry_str = f"{entry_low*100:.4f}-{entry_high*100:.4f}%"
        print(f"{entry_str:>22} | {result['trades']:>7,} | {result['avg_hold']:>5.1f}h | {result['win_rate']:>5.1f}% | ${result['net_pnl']:>8,.0f} | {result['avg_net_pct']:>7.3f}% | {result['avg_positions']:>7.1f} | {result['max_positions']:>8} | {result['p95_positions']:>7.0f}")


# =============================================================================
# TEST 3: Grid Search - Entry Range + Exit Low
# =============================================================================

print("\n" + "=" * 80)
print("TEST 3: Grid Search - Best Entry Range + Exit Low Combinations")
print("=" * 80)

# Tighter entry ranges
entry_ranges = [
    (0.000011, 0.000015),
    (0.000012, 0.000015),
    (0.000011, 0.000014),
    (0.000012, 0.000014),
    (0.000012, 0.000013),
]

exit_lows = [0.000003, 0.000004, 0.000005, 0.000006]

print(f"\n{'Entry Range':>22} | {'Exit Low':>10} | {'Trades':>7} | {'Hold':>7} | {'Win%':>6} | {'Net PnL':>10} | {'Avg%':>8} | {'Avg Pos':>8} | {'Max Pos':>8}")
print("-" * 125)

best_result = None
best_config = None

for entry_low, entry_high in entry_ranges:
    for exit_low in exit_lows:
        result = test_strategy(entry_low, entry_high, exit_low)
        if result:
            entry_str = f"{entry_low*100:.4f}-{entry_high*100:.4f}%"
            exit_str = f"{exit_low*100:.4f}%"
            print(f"{entry_str:>22} | {exit_str:>10} | {result['trades']:>7,} | {result['avg_hold']:>5.1f}h | {result['win_rate']:>5.1f}% | ${result['net_pnl']:>8,.0f} | {result['avg_net_pct']:>7.3f}% | {result['avg_positions']:>7.1f} | {result['max_positions']:>8}")
            
            # Track best by net PnL with reasonable position count
            if result['avg_positions'] <= 50 and (best_result is None or result['net_pnl'] > best_result['net_pnl']):
                best_result = result
                best_config = (entry_low, entry_high, exit_low)


# =============================================================================
# TEST 4: Very Tight Ranges (Target < 30 avg positions)
# =============================================================================

print("\n" + "=" * 80)
print("TEST 4: Very Tight Ranges (Target < 30 Avg Positions)")
print("=" * 80)

# Very narrow ranges
tight_ranges = [
    (0.000013, 0.000015),  # 0.0013% - 0.0015%
    (0.000014, 0.000015),  # 0.0014% - 0.0015%
    (0.000013, 0.000014),  # 0.0013% - 0.0014%
    (0.000012, 0.0000135), # 0.0012% - 0.00135%
    (0.0000125, 0.000015), # 0.00125% - 0.0015%
    (0.0000130, 0.0000145),# 0.0013% - 0.00145%
]

print(f"\n{'Entry Range':>25} | {'Exit Low':>10} | {'Trades':>7} | {'Hold':>7} | {'Win%':>6} | {'Net PnL':>10} | {'Avg%':>8} | {'Avg Pos':>8} | {'Max Pos':>8}")
print("-" * 130)

for entry_low, entry_high in tight_ranges:
    for exit_low in [0.000003, 0.000004, 0.000005]:
        result = test_strategy(entry_low, entry_high, exit_low)
        if result and result['trades'] > 500:  # At least 500 trades
            entry_str = f"{entry_low*100:.5f}-{entry_high*100:.5f}%"
            exit_str = f"{exit_low*100:.4f}%"
            print(f"{entry_str:>25} | {exit_str:>10} | {result['trades']:>7,} | {result['avg_hold']:>5.1f}h | {result['win_rate']:>5.1f}% | ${result['net_pnl']:>8,.0f} | {result['avg_net_pct']:>7.3f}% | {result['avg_positions']:>7.1f} | {result['max_positions']:>8}")


# =============================================================================
# DETAILED ANALYSIS: Best Config
# =============================================================================

if best_config:
    print("\n" + "=" * 80)
    print(f"DETAILED ANALYSIS: Best Config with Avg Pos <= 50")
    print("=" * 80)
    
    entry_low, entry_high, exit_low = best_config
    result = test_strategy(entry_low, entry_high, exit_low, return_details=True)
    
    print(f"\nConfig: Entry {entry_low*100:.4f}%-{entry_high*100:.4f}%, Exit Low {exit_low*100:.4f}%")
    print(f"\n--- TRADE STATISTICS ---")
    print(f"Total Trades: {result['trades']:,}")
    print(f"Still Open: {result['open_positions']}")
    print(f"Win Rate: {result['win_rate']:.1f}%")
    
    print(f"\n--- HOLDING TIME ---")
    print(f"Avg Hold: {result['avg_hold']:.1f} hours")
    
    print(f"\n--- PNL BREAKDOWN ---")
    print(f"Price PnL:        ${result['price_pnl']:>12,.2f}")
    print(f"Funding Received: ${result['funding_pnl']:>12,.2f}")
    print(f"Fees:             ${-result['fees']:>12,.2f}")
    print(f"{'─' * 30}")
    print(f"NET PNL:          ${result['net_pnl']:>12,.2f}")
    
    print(f"\n--- CONCURRENT POSITIONS ---")
    print(f"Average: {result['avg_positions']:.1f}")
    print(f"Max: {result['max_positions']}")
    print(f"P50: {result['p50_positions']:.0f}")
    print(f"P75: {result['p75_positions']:.0f}")
    print(f"P95: {result['p95_positions']:.0f}")
    print(f"P99: {result['p99_positions']:.0f}")
    
    # Yearly breakdown
    trades_df = result['trades_df']
    trades_df['year'] = trades_df['exit_hour'].dt.year
    
    print(f"\n--- YEARLY BREAKDOWN ---")
    yearly = trades_df.groupby('year').agg({
        'net_pnl': ['sum', 'count', 'mean'],
        'hold_hours': 'mean'
    }).round(2)
    yearly.columns = ['Net PnL', 'Trades', 'Avg PnL', 'Avg Hold']
    print(yearly.to_string())


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY: Position Count vs Profitability Trade-off")
print("=" * 80)

summary_configs = [
    (0.000010, 0.000015, 0.000005, "Full Range"),
    (0.000011, 0.000015, 0.000005, "Tighter Low"),
    (0.000012, 0.000015, 0.000005, "Tighter Low 2"),
    (0.000012, 0.000014, 0.000005, "Narrow Band"),
    (0.000013, 0.000015, 0.000005, "High Only"),
]

print(f"\n{'Config':>20} | {'Trades':>7} | {'Net PnL':>10} | {'Avg Pos':>8} | {'Max Pos':>8} | {'Avg%':>8}")
print("-" * 85)

for entry_low, entry_high, exit_low, name in summary_configs:
    result = test_strategy(entry_low, entry_high, exit_low)
    if result:
        print(f"{name:>20} | {result['trades']:>7,} | ${result['net_pnl']:>8,.0f} | {result['avg_positions']:>7.1f} | {result['max_positions']:>8} | {result['avg_net_pct']:>7.3f}%")
