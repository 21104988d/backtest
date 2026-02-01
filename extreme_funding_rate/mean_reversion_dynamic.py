"""
Mean Reversion Strategy - Dynamic Exit

Entry: 0 < |FR| < 0.0015%
Exit: |FR| >= 0.0015% (FR has reverted to extreme)

Direction:
- If FR > 0: SHORT (expect price drop, receive funding)
- If FR < 0: LONG (expect price rise, receive funding)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PARAMETERS
# =============================================================================

ENTRY_THRESHOLD = 0.000015    # Enter when |FR| < 0.0015%
EXIT_THRESHOLD = 0.000015     # Exit when |FR| >= 0.0015%
POSITION_SIZE = 100           # $100 USD per position
TAKER_FEE = 0.00045           # 0.045% per trade

print("=" * 80)
print("MEAN REVERSION STRATEGY - DYNAMIC EXIT")
print("=" * 80)
print(f"\nEntry: |FR| < {ENTRY_THRESHOLD*100:.4f}%")
print(f"Exit: |FR| >= {EXIT_THRESHOLD*100:.4f}%")
print(f"Position Size: ${POSITION_SIZE}")
print(f"Taker Fee: {TAKER_FEE*100:.3f}%")

# =============================================================================
# DATA LOADING
# =============================================================================

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
print(f"  Coins: {merged['coin'].nunique():,}")
print(f"  Date range: {merged['hour'].min()} to {merged['hour'].max()}")

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

print("\nRunning backtest...")

hours = sorted(merged['hour'].unique())
hour_groups = {h: g for h, g in merged.groupby('hour')}

# Position tracking: {coin: {'direction': 1/-1, 'entry_hour': hour, 'entry_price': price}}
positions = {}

# Results
trades = []
hourly_stats = []

for h_idx, hour in enumerate(hours):
    if hour not in hour_groups:
        continue
    
    df = hour_groups[hour]
    coin_data = dict(zip(
        df['coin'],
        [{'fr': fr, 'abs_fr': afr, 'price': p} 
         for fr, afr, p in zip(df['funding_rate'], df['abs_fr'], df['price'])]
    ))
    
    available_coins = set(coin_data.keys())
    
    # Track hourly funding for open positions
    hourly_funding_received = 0.0
    
    # ---------------------------------------------------------------------
    # STEP 1: Check exits - close positions where |FR| >= EXIT_THRESHOLD
    # ---------------------------------------------------------------------
    for coin in list(positions.keys()):
        if coin not in available_coins:
            continue
        
        data = coin_data[coin]
        pos = positions[coin]
        
        # Exit condition: |FR| >= threshold (FR has reverted to extreme)
        if data['abs_fr'] >= EXIT_THRESHOLD:
            exit_price = data['price']
            entry_price = pos['entry_price']
            direction = pos['direction']
            
            # Calculate PnL
            price_return = (exit_price - entry_price) / entry_price
            position_pnl = direction * price_return * POSITION_SIZE
            
            # Funding accumulated during hold
            funding_pnl = pos['funding_accumulated']
            
            # Fees (round-trip)
            fees = POSITION_SIZE * TAKER_FEE * 2
            
            # Hold time
            hold_hours = (hour - pos['entry_hour']).total_seconds() / 3600
            
            trades.append({
                'coin': coin,
                'entry_hour': pos['entry_hour'],
                'exit_hour': hour,
                'hold_hours': hold_hours,
                'direction': 'SHORT' if direction == -1 else 'LONG',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_fr': pos['entry_fr'],
                'exit_fr': data['fr'],
                'price_pnl': position_pnl,
                'funding_pnl': funding_pnl,
                'fees': fees,
                'net_pnl': position_pnl + funding_pnl - fees
            })
            
            del positions[coin]
    
    # ---------------------------------------------------------------------
    # STEP 2: Update funding for remaining positions
    # ---------------------------------------------------------------------
    for coin in positions:
        if coin in coin_data:
            # We receive funding (betting against FR direction)
            positions[coin]['funding_accumulated'] += coin_data[coin]['abs_fr'] * POSITION_SIZE
            hourly_funding_received += coin_data[coin]['abs_fr'] * POSITION_SIZE
    
    # ---------------------------------------------------------------------
    # STEP 3: Check entries - open positions where |FR| < ENTRY_THRESHOLD
    # ---------------------------------------------------------------------
    for coin in available_coins:
        if coin in positions:
            continue
        
        data = coin_data[coin]
        
        # Entry condition: 0 < |FR| < threshold
        if data['abs_fr'] > 0 and data['abs_fr'] < ENTRY_THRESHOLD:
            # Direction: SHORT if FR > 0, LONG if FR < 0 (betting against FR)
            direction = -1 if data['fr'] > 0 else 1
            
            positions[coin] = {
                'direction': direction,
                'entry_hour': hour,
                'entry_price': data['price'],
                'entry_fr': data['fr'],
                'funding_accumulated': 0.0
            }
    
    # Track hourly stats
    hourly_stats.append({
        'hour': hour,
        'n_positions': len(positions),
        'funding_received': hourly_funding_received
    })
    
    if (h_idx + 1) % 5000 == 0:
        print(f"  Processed {h_idx + 1:,} / {len(hours):,} hours...")

print(f"  Completed: {len(hours):,} hours")

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("MEAN REVERSION BACKTEST RESULTS")
print("=" * 80)

if not trades:
    print("\nNo trades executed!")
else:
    trades_df = pd.DataFrame(trades)
    hourly_df = pd.DataFrame(hourly_stats)
    
    print(f"\n--- TRADE STATISTICS ---")
    print(f"Total Trades: {len(trades_df):,}")
    print(f"Unique Coins Traded: {trades_df['coin'].nunique()}")
    
    print(f"\n--- HOLDING TIME ---")
    print(f"Avg Hold: {trades_df['hold_hours'].mean():.1f} hours")
    print(f"Min Hold: {trades_df['hold_hours'].min():.1f} hours")
    print(f"Max Hold: {trades_df['hold_hours'].max():.1f} hours")
    print(f"Median Hold: {trades_df['hold_hours'].median():.1f} hours")
    
    print(f"\n--- DIRECTION BREAKDOWN ---")
    direction_counts = trades_df['direction'].value_counts()
    for d, count in direction_counts.items():
        pct = count / len(trades_df) * 100
        avg_pnl = trades_df[trades_df['direction'] == d]['net_pnl'].mean()
        print(f"  {d}: {count:,} trades ({pct:.1f}%), Avg Net PnL: ${avg_pnl:.2f}")
    
    print(f"\n--- POSITION STATISTICS ---")
    print(f"Avg Concurrent Positions: {hourly_df['n_positions'].mean():.1f}")
    print(f"Max Concurrent Positions: {hourly_df['n_positions'].max()}")
    print(f"P50 Positions: {hourly_df['n_positions'].quantile(0.50):.0f}")
    print(f"P75 Positions: {hourly_df['n_positions'].quantile(0.75):.0f}")
    print(f"P95 Positions: {hourly_df['n_positions'].quantile(0.95):.0f}")
    print(f"P99 Positions: {hourly_df['n_positions'].quantile(0.99):.0f}")
    
    print(f"\n--- PNL BREAKDOWN ---")
    total_price_pnl = trades_df['price_pnl'].sum()
    total_funding = trades_df['funding_pnl'].sum()
    total_fees = trades_df['fees'].sum()
    total_net = trades_df['net_pnl'].sum()
    
    print(f"Price PnL:        ${total_price_pnl:>12,.2f}")
    print(f"Funding Received: ${total_funding:>12,.2f}")
    print(f"Fees:             ${-total_fees:>12,.2f}")
    print(f"{'─' * 30}")
    print(f"NET PNL:          ${total_net:>12,.2f}")
    
    print(f"\n--- PER TRADE AVERAGES ---")
    print(f"Avg Price Return: {trades_df['price_pnl'].mean() / POSITION_SIZE * 100:.4f}%")
    print(f"Avg Funding: {trades_df['funding_pnl'].mean() / POSITION_SIZE * 100:.4f}%")
    print(f"Avg Fees: {TAKER_FEE * 2 * 100:.3f}%")
    print(f"Avg Net PnL: ${trades_df['net_pnl'].mean():.2f} ({trades_df['net_pnl'].mean() / POSITION_SIZE * 100:.4f}%)")
    
    # Win rate
    wins = (trades_df['net_pnl'] > 0).sum()
    print(f"\n--- WIN RATE ---")
    print(f"Winning Trades: {wins:,} ({wins/len(trades_df)*100:.1f}%)")
    print(f"Losing Trades: {len(trades_df) - wins:,} ({(len(trades_df) - wins)/len(trades_df)*100:.1f}%)")
    
    # Yearly breakdown
    trades_df['year'] = trades_df['exit_hour'].dt.year
    print(f"\n--- YEARLY BREAKDOWN ---")
    yearly = trades_df.groupby('year').agg({
        'net_pnl': ['sum', 'count', 'mean'],
        'hold_hours': 'mean'
    }).round(2)
    yearly.columns = ['Net PnL', 'Trades', 'Avg PnL', 'Avg Hold']
    print(yearly.to_string())
    
    # Still open positions
    print(f"\n--- OPEN POSITIONS (Not Exited) ---")
    print(f"Positions still open: {len(positions)}")
    if positions:
        open_funding = sum(p['funding_accumulated'] for p in positions.values())
        print(f"Accumulated funding in open positions: ${open_funding:.2f}")

# =============================================================================
# TEST DIFFERENT EXIT THRESHOLDS
# =============================================================================

print("\n" + "=" * 80)
print("TESTING DIFFERENT EXIT THRESHOLDS")
print("=" * 80)

def test_exit_threshold(data, entry_thresh, exit_thresh):
    """Test with specific entry/exit thresholds."""
    
    hours = sorted(data['hour'].unique())
    hour_groups = {h: g for h, g in data.groupby('hour')}
    
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
            
            data_coin = coin_data[coin]
            pos = positions[coin]
            
            if data_coin['abs_fr'] >= exit_thresh:
                exit_price = data_coin['price']
                entry_price = pos['entry_price']
                direction = pos['direction']
                
                price_return = (exit_price - entry_price) / entry_price
                position_pnl = direction * price_return * POSITION_SIZE
                funding_pnl = pos['funding_accumulated']
                fees = POSITION_SIZE * TAKER_FEE * 2
                hold_hours = (hour - pos['entry_hour']).total_seconds() / 3600
                
                trades.append({
                    'price_pnl': position_pnl,
                    'funding_pnl': funding_pnl,
                    'fees': fees,
                    'hold_hours': hold_hours
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
            
            data_coin = coin_data[coin]
            
            if data_coin['abs_fr'] > 0 and data_coin['abs_fr'] < entry_thresh:
                direction = -1 if data_coin['fr'] > 0 else 1
                
                positions[coin] = {
                    'direction': direction,
                    'entry_hour': hour,
                    'entry_price': data_coin['price'],
                    'entry_fr': data_coin['fr'],
                    'funding_accumulated': 0.0
                }
    
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    
    return {
        'trades': len(df),
        'price_pnl': df['price_pnl'].sum(),
        'funding': df['funding_pnl'].sum(),
        'fees': df['fees'].sum(),
        'net_pnl': df['price_pnl'].sum() + df['funding_pnl'].sum() - df['fees'].sum(),
        'avg_hold': df['hold_hours'].mean(),
        'avg_net_pct': (df['price_pnl'].sum() + df['funding_pnl'].sum() - df['fees'].sum()) / len(df) / POSITION_SIZE * 100
    }

# Test different exit thresholds
exit_thresholds = [0.000010, 0.000015, 0.000020, 0.000025, 0.000030, 0.000050]

print(f"\nEntry: |FR| < 0.0015%")
print(f"\n{'Exit Thresh':>12} | {'Trades':>8} | {'Avg Hold':>10} | {'Price PnL':>12} | {'Funding':>10} | {'Net PnL':>12} | {'Avg Net%':>10}")
print("-" * 95)

for exit_t in exit_thresholds:
    result = test_exit_threshold(merged, 0.000015, exit_t)
    if result:
        print(f"{exit_t*100:.4f}% | {result['trades']:>8,} | {result['avg_hold']:>8.1f}h | ${result['price_pnl']:>10,.0f} | ${result['funding']:>8,.0f} | ${result['net_pnl']:>10,.0f} | {result['avg_net_pct']:>9.4f}%")
