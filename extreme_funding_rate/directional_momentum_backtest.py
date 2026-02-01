"""
DIRECTIONAL MOMENTUM STRATEGY BACKTEST

Strategy:
- Long when funding rate < -0.10% (shorts paying longs, potential short squeeze)
- Short when funding rate > +0.10% (longs paying shorts, potential long liquidation)
- Hold for 4h or 8h (test both)
- Collect funding during holding period
- No hedge - pure directional bet on mean reversion

Hypothesis: Extreme funding signals crowded positioning that tends to unwind
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

FUNDING_THRESHOLD = 0.001  # 0.10% - entry signal
TAKER_FEE = 0.00045  # 0.045% per trade
HOLDING_PERIODS = [1, 2, 4, 6, 8, 12, 24]  # Hours to test

# =============================================================================
# DATA LOADING
# =============================================================================

print("Loading data...")
funding = pd.read_csv('funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)
funding['hour'] = funding['datetime'].dt.floor('h').dt.tz_localize(None)

price = pd.read_csv('price_history.csv')
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
price['hour'] = price['timestamp'].dt.floor('h').dt.tz_localize(None)

print(f"Funding records: {len(funding):,}")
print(f"Price records: {len(price):,}")

# =============================================================================
# BUILD PRICE MATRIX
# =============================================================================

print("\nBuilding price matrix...")
price_pivot = price.pivot_table(index='hour', columns='coin', values='price', aggfunc='first')
print(f"Price matrix shape: {price_pivot.shape}")

# Build funding lookup
funding_lookup = {}
for _, row in funding.iterrows():
    key = (row['hour'], row['coin'])
    funding_lookup[key] = row['funding_rate']

def get_funding_rate(hour, coin):
    return funding_lookup.get((hour, coin), 0)

# =============================================================================
# IDENTIFY EXTREME FUNDING EVENTS
# =============================================================================

print(f"\nIdentifying extreme funding events (|FR| > {FUNDING_THRESHOLD*100:.2f}%)...")

# Get price data coverage
price_hours = set(price_pivot.index)
price_coins = set(price_pivot.columns)

# Filter funding to events with price data
extreme_funding = funding[
    (funding['funding_rate'].abs() > FUNDING_THRESHOLD) &
    (funding['hour'].isin(price_hours)) &
    (funding['coin'].isin(price_coins))
].copy()

extreme_funding['direction'] = np.where(extreme_funding['funding_rate'] < 0, 'long', 'short')

print(f"Extreme funding events: {len(extreme_funding):,}")
print(f"  Long signals (FR < -{FUNDING_THRESHOLD*100:.2f}%): {(extreme_funding['direction'] == 'long').sum():,}")
print(f"  Short signals (FR > +{FUNDING_THRESHOLD*100:.2f}%): {(extreme_funding['direction'] == 'short').sum():,}")

# =============================================================================
# BACKTEST FUNCTION
# =============================================================================

def backtest_trade(coin, entry_hour, direction, holding_hours, price_matrix, funding_lookup):
    """
    Simulate a single directional trade
    
    Returns dict with:
    - entry/exit prices and times
    - price return
    - funding collected during hold
    - fees
    - net PnL
    """
    result = {
        'coin': coin,
        'entry_hour': entry_hour,
        'direction': direction,
        'holding_hours': holding_hours,
        'entry_price': np.nan,
        'exit_price': np.nan,
        'exit_hour': None,
        'price_return': np.nan,
        'funding_collected': 0,
        'entry_funding': np.nan,
        'trading_fee': 2 * TAKER_FEE,  # Entry + exit
        'gross_pnl': np.nan,
        'net_pnl': np.nan,
        'valid': False
    }
    
    # Check if coin exists
    if coin not in price_matrix.columns:
        return result
    
    # Get entry price
    if entry_hour not in price_matrix.index:
        return result
    
    entry_price = price_matrix.loc[entry_hour, coin]
    if pd.isna(entry_price):
        return result
    
    result['entry_price'] = entry_price
    result['entry_funding'] = get_funding_rate(entry_hour, coin)
    
    # Get exit price
    exit_hour = entry_hour + timedelta(hours=holding_hours)
    if exit_hour not in price_matrix.index:
        return result
    
    exit_price = price_matrix.loc[exit_hour, coin]
    if pd.isna(exit_price):
        return result
    
    result['exit_price'] = exit_price
    result['exit_hour'] = exit_hour
    
    # Calculate price return
    price_return = (exit_price - entry_price) / entry_price
    if direction == 'short':
        price_return = -price_return
    
    result['price_return'] = price_return
    
    # Calculate funding collected during hold
    # Funding is paid every hour
    funding_collected = 0
    for h in range(holding_hours):
        funding_hour = entry_hour + timedelta(hours=h)
        fr = get_funding_rate(funding_hour, coin)
        
        # If long and FR < 0, we receive |FR|
        # If short and FR > 0, we receive FR
        if direction == 'long' and fr < 0:
            funding_collected += abs(fr)
        elif direction == 'short' and fr > 0:
            funding_collected += fr
        elif direction == 'long' and fr > 0:
            funding_collected -= fr  # We pay
        elif direction == 'short' and fr < 0:
            funding_collected -= abs(fr)  # We pay
    
    result['funding_collected'] = funding_collected
    
    # Calculate PnL
    result['gross_pnl'] = result['price_return'] + result['funding_collected']
    result['net_pnl'] = result['gross_pnl'] - result['trading_fee']
    result['valid'] = True
    
    return result

# =============================================================================
# RUN BACKTEST FOR EACH HOLDING PERIOD
# =============================================================================

print("\n" + "="*100)
print("BACKTEST RESULTS BY HOLDING PERIOD")
print("="*100)

all_results = {}

for holding_hours in HOLDING_PERIODS:
    print(f"\nProcessing {holding_hours}h holding period...")
    
    trades = []
    for i, (_, row) in enumerate(extreme_funding.iterrows()):
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{len(extreme_funding)} events...")
        
        result = backtest_trade(
            coin=row['coin'],
            entry_hour=row['hour'],
            direction=row['direction'],
            holding_hours=holding_hours,
            price_matrix=price_pivot,
            funding_lookup=funding_lookup
        )
        trades.append(result)
    
    df = pd.DataFrame(trades)
    valid = df[df['valid'] == True]
    all_results[holding_hours] = valid
    
    print(f"  Valid trades: {len(valid)} / {len(df)}")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "="*100)
print("PERFORMANCE SUMMARY (GROSS PnL - before fees)")
print("="*100)

print(f"\n{'Hold':<8} {'N Trades':<12} {'Avg Gross':<14} {'Std Dev':<12} {'Sharpe':<10} {'Win Rate':<10} {'Avg FR Coll':<12}")
print("-"*90)

for hours in HOLDING_PERIODS:
    valid = all_results[hours]
    if len(valid) > 0:
        avg = valid['gross_pnl'].mean()
        std = valid['gross_pnl'].std()
        sharpe = avg / std if std > 0 else 0
        win_rate = (valid['gross_pnl'] > 0).mean()
        avg_fr = valid['funding_collected'].mean()
        print(f"{hours}h{'':<6} {len(valid):<12} {avg*100:>11.4f}%  {std*100:>10.2f}%  {sharpe:>8.4f}  {win_rate*100:>8.1f}%  {avg_fr*100:>10.4f}%")

print("\n" + "="*100)
print("PERFORMANCE SUMMARY (NET PnL - after fees)")
print("="*100)

print(f"\n{'Hold':<8} {'N Trades':<12} {'Avg Net':<14} {'Std Dev':<12} {'Sharpe':<10} {'Win Rate':<10} {'Total PnL':<12}")
print("-"*90)

for hours in HOLDING_PERIODS:
    valid = all_results[hours]
    if len(valid) > 0:
        avg = valid['net_pnl'].mean()
        std = valid['net_pnl'].std()
        sharpe = avg / std if std > 0 else 0
        win_rate = (valid['net_pnl'] > 0).mean()
        total = valid['net_pnl'].sum()
        print(f"{hours}h{'':<6} {len(valid):<12} {avg*100:>11.4f}%  {std*100:>10.2f}%  {sharpe:>8.4f}  {win_rate*100:>8.1f}%  {total*100:>10.2f}%")

# =============================================================================
# P&L BREAKDOWN
# =============================================================================

print("\n" + "="*100)
print("P&L BREAKDOWN BY HOLDING PERIOD")
print("="*100)

print(f"\n{'Hold':<8} {'Price PnL':<14} {'Funding Coll':<14} {'Fees':<12} {'Net PnL':<12}")
print("-"*65)

for hours in HOLDING_PERIODS:
    valid = all_results[hours]
    if len(valid) > 0:
        price_pnl = valid['price_return'].mean()
        funding = valid['funding_collected'].mean()
        fees = valid['trading_fee'].mean()
        net = valid['net_pnl'].mean()
        print(f"{hours}h{'':<6} {price_pnl*100:>11.4f}%  {funding*100:>12.4f}%  {fees*100:>10.4f}%  {net*100:>10.4f}%")

# =============================================================================
# LONG VS SHORT ANALYSIS
# =============================================================================

print("\n" + "="*100)
print("LONG vs SHORT BREAKDOWN (4h and 8h holds)")
print("="*100)

for hours in [4, 8]:
    valid = all_results[hours]
    if len(valid) > 0:
        print(f"\n### {hours}h Holding Period ###")
        
        longs = valid[valid['direction'] == 'long']
        shorts = valid[valid['direction'] == 'short']
        
        print(f"\n{'Direction':<10} {'N':<10} {'Avg Net PnL':<14} {'Sharpe':<10} {'Win Rate':<10}")
        print("-"*60)
        
        if len(longs) > 0:
            avg = longs['net_pnl'].mean()
            std = longs['net_pnl'].std()
            sharpe = avg / std if std > 0 else 0
            win = (longs['net_pnl'] > 0).mean()
            print(f"{'LONG':<10} {len(longs):<10} {avg*100:>11.4f}%  {sharpe:>8.4f}  {win*100:>8.1f}%")
        
        if len(shorts) > 0:
            avg = shorts['net_pnl'].mean()
            std = shorts['net_pnl'].std()
            sharpe = avg / std if std > 0 else 0
            win = (shorts['net_pnl'] > 0).mean()
            print(f"{'SHORT':<10} {len(shorts):<10} {avg*100:>11.4f}%  {sharpe:>8.4f}  {win*100:>8.1f}%")

# =============================================================================
# FUNDING RATE MAGNITUDE ANALYSIS
# =============================================================================

print("\n" + "="*100)
print("PERFORMANCE BY FUNDING RATE MAGNITUDE (8h hold)")
print("="*100)

valid_8h = all_results[8].copy()
valid_8h['fr_magnitude'] = valid_8h['entry_funding'].abs()
valid_8h['fr_bucket'] = pd.cut(
    valid_8h['fr_magnitude'], 
    bins=[0.001, 0.002, 0.003, 0.005, 0.01, 1],
    labels=['0.10-0.20%', '0.20-0.30%', '0.30-0.50%', '0.50-1.0%', '>1.0%']
)

print(f"\n{'FR Bucket':<15} {'N':<10} {'Avg Net PnL':<14} {'Sharpe':<10} {'Win Rate':<10}")
print("-"*65)

for bucket in ['0.10-0.20%', '0.20-0.30%', '0.30-0.50%', '0.50-1.0%', '>1.0%']:
    bucket_data = valid_8h[valid_8h['fr_bucket'] == bucket]
    if len(bucket_data) > 10:
        avg = bucket_data['net_pnl'].mean()
        std = bucket_data['net_pnl'].std()
        sharpe = avg / std if std > 0 else 0
        win = (bucket_data['net_pnl'] > 0).mean()
        print(f"{bucket:<15} {len(bucket_data):<10} {avg*100:>11.4f}%  {sharpe:>8.4f}  {win*100:>8.1f}%")

# =============================================================================
# TOP PERFORMING COINS
# =============================================================================

print("\n" + "="*100)
print("TOP PERFORMING COINS (8h hold, min 10 trades)")
print("="*100)

coin_stats = valid_8h.groupby('coin').agg({
    'net_pnl': ['count', 'mean', 'std', lambda x: (x > 0).mean()]
}).round(6)
coin_stats.columns = ['n_trades', 'avg_pnl', 'std_pnl', 'win_rate']
coin_stats = coin_stats[coin_stats['n_trades'] >= 10]
coin_stats['sharpe'] = coin_stats['avg_pnl'] / coin_stats['std_pnl']
coin_stats = coin_stats.sort_values('avg_pnl', ascending=False)

print("\n### Top 15 by Average PnL ###")
print(f"{'Coin':<12} {'N Trades':<10} {'Avg Net PnL':<14} {'Sharpe':<10} {'Win Rate':<10}")
print("-"*60)
for coin, row in coin_stats.head(15).iterrows():
    print(f"{coin:<12} {int(row['n_trades']):<10} {row['avg_pnl']*100:>11.4f}%  {row['sharpe']:>8.4f}  {row['win_rate']*100:>8.1f}%")

print("\n### Bottom 15 by Average PnL ###")
print(f"{'Coin':<12} {'N Trades':<10} {'Avg Net PnL':<14} {'Sharpe':<10} {'Win Rate':<10}")
print("-"*60)
for coin, row in coin_stats.tail(15).iterrows():
    print(f"{coin:<12} {int(row['n_trades']):<10} {row['avg_pnl']*100:>11.4f}%  {row['sharpe']:>8.4f}  {row['win_rate']*100:>8.1f}%")

# =============================================================================
# TIME-BASED ANALYSIS
# =============================================================================

print("\n" + "="*100)
print("PERFORMANCE BY ENTRY HOUR (8h hold)")
print("="*100)

valid_8h['entry_hour_of_day'] = valid_8h['entry_hour'].dt.hour

hourly_perf = valid_8h.groupby('entry_hour_of_day').agg({
    'net_pnl': ['count', 'mean', lambda x: (x > 0).mean()]
}).round(4)
hourly_perf.columns = ['n_trades', 'avg_pnl', 'win_rate']

print(f"\n{'Hour':<8} {'N Trades':<12} {'Avg Net PnL':<14} {'Win Rate':<10}")
print("-"*50)
for hour, row in hourly_perf.iterrows():
    print(f"{hour:02d}:00{'':<3} {int(row['n_trades']):<12} {row['avg_pnl']*100:>11.4f}%  {row['win_rate']*100:>8.1f}%")

# =============================================================================
# CUMULATIVE PnL CHART DATA
# =============================================================================

print("\n" + "="*100)
print("CUMULATIVE P&L OVER TIME (8h hold)")
print("="*100)

valid_8h_sorted = valid_8h.sort_values('entry_hour')
valid_8h_sorted['cumulative_pnl'] = valid_8h_sorted['net_pnl'].cumsum()

# Show monthly stats
valid_8h_sorted['month'] = valid_8h_sorted['entry_hour'].dt.to_period('M')
monthly_stats = valid_8h_sorted.groupby('month').agg({
    'net_pnl': ['count', 'sum', 'mean', lambda x: (x > 0).mean()]
}).round(4)
monthly_stats.columns = ['n_trades', 'total_pnl', 'avg_pnl', 'win_rate']

print(f"\n{'Month':<12} {'N Trades':<12} {'Total PnL':<14} {'Avg PnL':<12} {'Win Rate':<10}")
print("-"*65)
for month, row in monthly_stats.iterrows():
    print(f"{month}{'':<4} {int(row['n_trades']):<12} {row['total_pnl']*100:>11.2f}%  {row['avg_pnl']*100:>10.4f}%  {row['win_rate']*100:>8.1f}%")

print(f"\n### Final Cumulative PnL (8h hold): {valid_8h_sorted['cumulative_pnl'].iloc[-1]*100:.2f}% ###")

# =============================================================================
# SAVE DETAILED RESULTS
# =============================================================================

# Save 8h results for further analysis
valid_8h.to_csv('directional_momentum_8h_results.csv', index=False)
print("\nSaved: directional_momentum_8h_results.csv")

# Save 4h results
all_results[4].to_csv('directional_momentum_4h_results.csv', index=False)
print("Saved: directional_momentum_4h_results.csv")

print("\n" + "="*100)
print("BACKTEST COMPLETE")
print("="*100)

# =============================================================================
# SUMMARY
# =============================================================================

print("""
### KEY FINDINGS ###

The directional momentum strategy:
- Long when funding < -0.10% (shorts paying longs)
- Short when funding > +0.10% (longs paying shorts)

Hypothesis: Extreme funding signals crowded positioning that tends to unwind.
When shorts are heavily paying, a short squeeze may occur (go long).
When longs are heavily paying, a long liquidation cascade may occur (go short).

Compare the Sharpe ratios and win rates across holding periods to find the optimal hold time.
""")
