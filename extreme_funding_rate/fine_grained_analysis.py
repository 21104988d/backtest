"""
Fine-grained threshold sweep + Monthly performance breakdown
To find the sweet spot and check for crypto cycle dependency
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
print(f'Date range: {merged["hour"].min()} to {merged["hour"].max()}')

# Pre-compute future prices
print('Pre-computing...')
for h in [72]:
    merged[f'price_{h}h'] = merged.groupby('coin')['price'].shift(-h)
    merged[f'fr_sum_{h}h'] = merged.groupby('coin')['funding_rate'].transform(
        lambda x: x.rolling(h, min_periods=1).sum().shift(-h+1)
    )

merged = merged.dropna(subset=['price_72h'])

# Sample: one per coin per day
merged['date'] = merged['hour'].dt.date
merged['year'] = merged['hour'].dt.year
merged['month'] = merged['hour'].dt.to_period('M')
sampled = merged.groupby(['coin', 'date']).first().reset_index()
print(f'Sampled: {len(sampled):,}')

# =============================================================================
# BACKTEST FUNCTION WITH TRADE DETAILS
# =============================================================================
def backtest_with_details(df, entry_thresh, hold_hours=72):
    """Returns both summary stats and individual trade details"""
    
    # SHORT signals
    short_signals = df[df['funding_rate'] < -entry_thresh].copy()
    short_signals['direction'] = 'SHORT'
    
    # LONG signals
    long_signals = df[df['funding_rate'] > entry_thresh].copy()
    long_signals['direction'] = 'LONG'
    
    all_signals = pd.concat([short_signals, long_signals])
    
    if len(all_signals) == 0:
        return None, None
    
    entry_price = all_signals['price'].values
    exit_price = all_signals[f'price_{hold_hours}h'].values
    fr_sum = all_signals[f'fr_sum_{hold_hours}h'].values
    directions = all_signals['direction'].values
    
    price_ret = np.where(
        directions == 'SHORT',
        -(exit_price - entry_price) / entry_price,
        (exit_price - entry_price) / entry_price
    )
    
    funding_pnl = np.where(
        directions == 'SHORT',
        -fr_sum,
        fr_sum
    )
    
    net_pnl = price_ret + funding_pnl - 2 * TAKER_FEE
    
    # Create trades dataframe
    trades = all_signals[['hour', 'coin', 'funding_rate', 'direction', 'month', 'year']].copy()
    trades['net_pnl'] = net_pnl
    trades['price_pnl'] = price_ret
    trades['funding_pnl'] = funding_pnl
    
    # Summary stats
    valid_pnl = net_pnl[~np.isnan(net_pnl)]
    summary = {
        'n': len(valid_pnl),
        'avg': np.mean(valid_pnl),
        'std': np.std(valid_pnl),
        'sharpe': np.mean(valid_pnl) / np.std(valid_pnl) if np.std(valid_pnl) > 0 else 0,
        'win': np.mean(valid_pnl > 0),
        'total': np.sum(valid_pnl)
    }
    
    return summary, trades

# =============================================================================
# FINE-GRAINED THRESHOLD SWEEP
# =============================================================================
print('\n' + '='*90)
print('FINE-GRAINED ENTRY THRESHOLD SWEEP (Hold=72h, Combined SHORT+LONG)')
print('='*90)

# Test from 0.0005% to 0.015% in small increments
thresholds = [
    0.000005,  # 0.0005% = 4.4% APY
    0.00001,   # 0.001% = 8.8% APY
    0.000015,  # 0.0015% = 13.1% APY
    0.00002,   # 0.002% = 17.5% APY
    0.000025,  # 0.0025% = 21.9% APY
    0.00003,   # 0.003% = 26.3% APY
    0.000035,  # 0.0035% = 30.7% APY
    0.00004,   # 0.004% = 35.0% APY
    0.000045,  # 0.0045% = 39.4% APY
    0.00005,   # 0.005% = 43.8% APY
    0.000055,  # 0.0055% = 48.2% APY
    0.00006,   # 0.006% = 52.6% APY
    0.000065,  # 0.0065% = 56.9% APY
    0.00007,   # 0.007% = 61.3% APY
    0.000075,  # 0.0075% = 65.7% APY
    0.00008,   # 0.008% = 70.1% APY
    0.000085,  # 0.0085% = 74.5% APY
    0.00009,   # 0.009% = 78.8% APY
    0.000095,  # 0.0095% = 83.2% APY
    0.0001,    # 0.01% = 87.6% APY
    0.00012,   # 0.012% = 105% APY
    0.00015,   # 0.015% = 131% APY
]

print(f"\n{'Entry':<10} {'APY':<10} {'N':>8} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Total':>12} {'$/Trade':>10}")
print('-'*90)

results = []
for thresh in thresholds:
    summary, trades = backtest_with_details(sampled, thresh, 72)
    if summary and summary['n'] >= 50:
        apy = thresh * 24 * 365 * 100
        results.append({
            'thresh': thresh,
            'apy': apy,
            **summary,
            'trades': trades
        })
        # Assuming $1000 per trade
        dollar_per_trade = summary['avg'] * 1000
        print(f"{thresh*100:.4f}%   {apy:>6.1f}%   {summary['n']:>8} {summary['avg']*100:>+9.2f}% {summary['sharpe']:>8.3f} {summary['win']*100:>7.1f}% {summary['total']*100:>+11.0f}% ${dollar_per_trade:>+8.1f}")

# =============================================================================
# FIND OPTIMAL THRESHOLD
# =============================================================================
print('\n' + '='*90)
print('OPTIMAL THRESHOLD ANALYSIS')
print('='*90)

if results:
    # Best by Sharpe
    best_sharpe = max(results, key=lambda x: x['sharpe'])
    print(f"\nBest by Sharpe: {best_sharpe['thresh']*100:.4f}% ({best_sharpe['apy']:.1f}% APY)")
    print(f"  N={best_sharpe['n']}, Sharpe={best_sharpe['sharpe']:.3f}, Avg={best_sharpe['avg']*100:+.2f}%, Win={best_sharpe['win']*100:.1f}%")
    
    # Best by Total PnL
    best_total = max(results, key=lambda x: x['total'])
    print(f"\nBest by Total PnL: {best_total['thresh']*100:.4f}% ({best_total['apy']:.1f}% APY)")
    print(f"  N={best_total['n']}, Total={best_total['total']*100:+.0f}%, Avg={best_total['avg']*100:+.2f}%")
    
    # Best by trade count (most liquid)
    best_n = max(results, key=lambda x: x['n'])
    print(f"\nMost trades: {best_n['thresh']*100:.4f}% ({best_n['apy']:.1f}% APY)")
    print(f"  N={best_n['n']}, Avg={best_n['avg']*100:+.2f}%, Sharpe={best_n['sharpe']:.3f}")
    
    # Sweet spot: good Sharpe with decent volume
    # Filter for Sharpe > 0.05 and N > 2000
    good_configs = [r for r in results if r['sharpe'] > 0.05 and r['n'] > 2000]
    if good_configs:
        sweet_spot = max(good_configs, key=lambda x: x['total'])
        print(f"\nSweet Spot (Sharpe>0.05, N>2000): {sweet_spot['thresh']*100:.4f}% ({sweet_spot['apy']:.1f}% APY)")
        print(f"  N={sweet_spot['n']}, Sharpe={sweet_spot['sharpe']:.3f}, Avg={sweet_spot['avg']*100:+.2f}%, Total={sweet_spot['total']*100:+.0f}%")

# =============================================================================
# MONTHLY PERFORMANCE BREAKDOWN
# =============================================================================
print('\n' + '='*90)
print('MONTHLY PERFORMANCE (Entry=0.005%, 43.8% APY, Hold=72h)')
print('='*90)

# Use 0.005% threshold for detailed analysis
summary, trades = backtest_with_details(sampled, 0.00005, 72)

if trades is not None:
    trades = trades.dropna(subset=['net_pnl'])
    
    # Group by month
    monthly = trades.groupby('month').agg({
        'net_pnl': ['count', 'mean', 'sum', lambda x: (x > 0).mean()],
        'price_pnl': 'mean',
        'funding_pnl': 'mean'
    }).round(4)
    monthly.columns = ['N', 'Avg_PnL', 'Total_PnL', 'Win%', 'Price_PnL', 'Funding_PnL']
    
    print(f"\n{'Month':<10} {'N':>6} {'Avg PnL':>10} {'Total':>10} {'Win%':>8} {'Price':>10} {'Funding':>10}")
    print('-'*75)
    
    for month, row in monthly.iterrows():
        print(f"{str(month):<10} {int(row['N']):>6} {row['Avg_PnL']*100:>+9.2f}% {row['Total_PnL']*100:>+9.0f}% {row['Win%']*100:>7.1f}% {row['Price_PnL']*100:>+9.2f}% {row['Funding_PnL']*100:>+9.2f}%")

# =============================================================================
# YEARLY PERFORMANCE
# =============================================================================
print('\n' + '='*90)
print('YEARLY PERFORMANCE (Entry=0.005%, 43.8% APY, Hold=72h)')
print('='*90)

if trades is not None:
    yearly = trades.groupby('year').agg({
        'net_pnl': ['count', 'mean', 'sum', lambda x: (x > 0).mean()],
        'price_pnl': 'mean',
        'funding_pnl': 'mean'
    }).round(4)
    yearly.columns = ['N', 'Avg_PnL', 'Total_PnL', 'Win%', 'Price_PnL', 'Funding_PnL']
    
    print(f"\n{'Year':<8} {'N':>6} {'Avg PnL':>10} {'Total':>10} {'Win%':>8} {'Price':>10} {'Funding':>10}")
    print('-'*70)
    
    for year, row in yearly.iterrows():
        print(f"{year:<8} {int(row['N']):>6} {row['Avg_PnL']*100:>+9.2f}% {row['Total_PnL']*100:>+9.0f}% {row['Win%']*100:>7.1f}% {row['Price_PnL']*100:>+9.2f}% {row['Funding_PnL']*100:>+9.2f}%")

# =============================================================================
# QUARTERLY PERFORMANCE (to see cycle effects)
# =============================================================================
print('\n' + '='*90)
print('QUARTERLY PERFORMANCE (Entry=0.005%)')
print('='*90)

if trades is not None:
    trades['quarter'] = trades['hour'].dt.to_period('Q')
    quarterly = trades.groupby('quarter').agg({
        'net_pnl': ['count', 'mean', 'sum', lambda x: (x > 0).mean()]
    }).round(4)
    quarterly.columns = ['N', 'Avg_PnL', 'Total_PnL', 'Win%']
    
    print(f"\n{'Quarter':<10} {'N':>6} {'Avg PnL':>10} {'Total':>10} {'Win%':>8}")
    print('-'*50)
    
    for q, row in quarterly.iterrows():
        print(f"{str(q):<10} {int(row['N']):>6} {row['Avg_PnL']*100:>+9.2f}% {row['Total_PnL']*100:>+9.0f}% {row['Win%']*100:>7.1f}%")

# =============================================================================
# BULL vs BEAR MARKET ANALYSIS
# =============================================================================
print('\n' + '='*90)
print('MARKET CONDITION ANALYSIS')
print('Using BTC price trend as proxy for market condition')
print('='*90)

# Check if BTC data exists to determine market conditions
btc_data = sampled[sampled['coin'] == 'BTC'][['hour', 'price']].copy()
if len(btc_data) > 0:
    btc_data = btc_data.sort_values('hour')
    btc_data['btc_ma_30d'] = btc_data['price'].rolling(30*24, min_periods=1).mean()
    btc_data['market'] = np.where(btc_data['price'] > btc_data['btc_ma_30d'], 'BULL', 'BEAR')
    
    # Merge market condition to trades
    trades_with_market = trades.merge(
        btc_data[['hour', 'market']],
        on='hour',
        how='left'
    )
    
    market_perf = trades_with_market.groupby('market').agg({
        'net_pnl': ['count', 'mean', 'sum', lambda x: (x > 0).mean()]
    }).round(4)
    market_perf.columns = ['N', 'Avg_PnL', 'Total_PnL', 'Win%']
    
    print(f"\n{'Market':<8} {'N':>6} {'Avg PnL':>10} {'Total':>10} {'Win%':>8}")
    print('-'*45)
    
    for market, row in market_perf.iterrows():
        if pd.notna(market):
            print(f"{market:<8} {int(row['N']):>6} {row['Avg_PnL']*100:>+9.2f}% {row['Total_PnL']*100:>+9.0f}% {row['Win%']*100:>7.1f}%")

# =============================================================================
# SHORT vs LONG BY YEAR
# =============================================================================
print('\n' + '='*90)
print('SHORT vs LONG BY YEAR (Entry=0.005%)')
print('='*90)

if trades is not None:
    direction_yearly = trades.groupby(['year', 'direction']).agg({
        'net_pnl': ['count', 'mean', 'sum']
    }).round(4)
    direction_yearly.columns = ['N', 'Avg_PnL', 'Total_PnL']
    
    print(f"\n{'Year':<8} {'Dir':<8} {'N':>6} {'Avg PnL':>10} {'Total':>10}")
    print('-'*50)
    
    for (year, direction), row in direction_yearly.iterrows():
        print(f"{year:<8} {direction:<8} {int(row['N']):>6} {row['Avg_PnL']*100:>+9.2f}% {row['Total_PnL']*100:>+9.0f}%")

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '='*90)
print('SUMMARY')
print('='*90)
print('''
KEY FINDINGS:
1. Threshold Sweet Spot: Look for balance between Sharpe and trade volume
2. Monthly breakdown shows if strategy is consistent or cycle-dependent
3. Price PnL vs Funding PnL shows which component drives returns

CRYPTO CYCLE CONTEXT:
- 2023: Recovery from 2022 bear market
- 2024: Bitcoin halving year, typically bullish
- 2025: Usually post-halving bull run
- 2026: Often cooling off period

If strategy works across all periods, it's robust to cycles.
''')
