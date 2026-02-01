"""
Fine-grained threshold test with detailed PnL breakdown
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

# Pre-compute
for h in [72]:
    merged[f'price_{h}h'] = merged.groupby('coin')['price'].shift(-h)
    merged[f'fr_sum_{h}h'] = merged.groupby('coin')['funding_rate'].transform(
        lambda x: x.rolling(h, min_periods=1).sum().shift(-h+1)
    )

merged = merged.dropna(subset=['price_72h'])
merged['date'] = merged['hour'].dt.date
sampled = merged.groupby(['coin', 'date']).first().reset_index()
print(f'Sampled: {len(sampled):,}')

def backtest_detailed(df, entry_thresh, hold_hours=72):
    # SHORT signals
    short_signals = df[df['funding_rate'] < -entry_thresh].copy()
    short_signals['direction'] = 'SHORT'
    
    # LONG signals
    long_signals = df[df['funding_rate'] > entry_thresh].copy()
    long_signals['direction'] = 'LONG'
    
    all_signals = pd.concat([short_signals, long_signals])
    
    if len(all_signals) == 0:
        return None
    
    entry_price = all_signals['price'].values
    exit_price = all_signals[f'price_{hold_hours}h'].values
    fr_sum = all_signals[f'fr_sum_{hold_hours}h'].values
    directions = all_signals['direction'].values
    
    # Price return (trend following)
    price_ret = np.where(
        directions == 'SHORT',
        -(exit_price - entry_price) / entry_price,
        (exit_price - entry_price) / entry_price
    )
    
    # Funding PnL
    funding_pnl = np.where(
        directions == 'SHORT',
        -fr_sum,  # SHORT receives when FR negative
        fr_sum    # LONG pays when FR positive
    )
    
    fees = 2 * TAKER_FEE
    net_pnl = price_ret + funding_pnl - fees
    
    valid = ~np.isnan(net_pnl)
    
    return {
        'n': valid.sum(),
        'price_pnl': np.nanmean(price_ret),
        'funding_pnl': np.nanmean(funding_pnl),
        'fees': fees,
        'net_pnl': np.nanmean(net_pnl),
        'win': np.nanmean(net_pnl > 0),
        'total': np.nansum(net_pnl),
        'sharpe': np.nanmean(net_pnl) / np.nanstd(net_pnl) if np.nanstd(net_pnl) > 0 else 0
    }

# =============================================================================
# FINE-GRAINED SWEEP
# =============================================================================
print('\n' + '='*130)
print('DETAILED PNL BREAKDOWN: Entry Threshold Sweep (Hold=72h)')
print('Net PnL = Price PnL + Funding PnL - Fees')
print('='*130)

thresholds = [
    0.000005,  # 0.0005%
    0.000006,  # 0.0006%
    0.000007,  # 0.0007%
    0.000008,  # 0.0008%
    0.000009,  # 0.0009%
    0.00001,   # 0.001%
    0.000011,  # 0.0011%
    0.000012,  # 0.0012%
    0.000013,  # 0.0013%
    0.000014,  # 0.0014%
    0.000015,  # 0.0015%
    0.000016,  # 0.0016%
    0.000018,  # 0.0018%
    0.00002,   # 0.002%
    0.000025,  # 0.0025%
    0.00003,   # 0.003%
    0.00004,   # 0.004%
    0.00005,   # 0.005%
]

print(f"\n{'Entry':<10} {'APY':<8} {'N':>8} | {'Price PnL':>11} {'Fund PnL':>11} {'Fees':>8} | {'Net PnL':>11} {'Sharpe':>8} {'Win%':>7} {'Total':>12}")
print('-'*130)

for thresh in thresholds:
    result = backtest_detailed(sampled, thresh, 72)
    if result and result['n'] >= 50:
        apy = thresh * 24 * 365 * 100
        print(f"{thresh*100:.4f}%   {apy:>5.1f}%   {result['n']:>8} | {result['price_pnl']*100:>+10.2f}% {result['funding_pnl']*100:>+10.2f}% {result['fees']*100:>7.2f}% | {result['net_pnl']*100:>+10.2f}% {result['sharpe']:>8.3f} {result['win']*100:>6.1f}% {result['total']*100:>+11.0f}%")

# =============================================================================
# KEY ANALYSIS
# =============================================================================
print('\n' + '='*130)
print('KEY ANALYSIS: Why threshold matters')
print('='*130)

print('''
EXPLANATION:
- Price PnL: Return from trend following (betting price moves WITH funding direction)
- Funding PnL: Net funding payments over 72h holding period
  * SHORT when FR<0: We RECEIVE funding (positive)
  * LONG when FR>0: We PAY funding (negative)
- Fees: 0.09% round trip (entry + exit)

KEY INSIGHT:
At very low thresholds (< 0.001%), Price PnL is NEGATIVE because:
- Weak funding signals don't predict price direction well
- Strategy loses on price movement, barely compensated by funding

At higher thresholds (> 0.0015%), Price PnL turns POSITIVE because:
- Stronger funding signals ARE predictive of price direction
- Strategy profits from BOTH price movement AND funding
''')

# =============================================================================
# SHORT vs LONG breakdown
# =============================================================================
print('\n' + '='*130)
print('SHORT vs LONG BREAKDOWN at key thresholds')
print('='*130)

def backtest_by_direction(df, entry_thresh, direction, hold_hours=72):
    if direction == 'SHORT':
        signals = df[df['funding_rate'] < -entry_thresh].copy()
    else:
        signals = df[df['funding_rate'] > entry_thresh].copy()
    
    if len(signals) == 0:
        return None
    
    entry_price = signals['price'].values
    exit_price = signals[f'price_{hold_hours}h'].values
    fr_sum = signals[f'fr_sum_{hold_hours}h'].values
    
    if direction == 'SHORT':
        price_ret = -(exit_price - entry_price) / entry_price
        funding_pnl = -fr_sum
    else:
        price_ret = (exit_price - entry_price) / entry_price
        funding_pnl = fr_sum
    
    net_pnl = price_ret + funding_pnl - 2 * TAKER_FEE
    valid = ~np.isnan(net_pnl)
    
    return {
        'n': valid.sum(),
        'price_pnl': np.nanmean(price_ret),
        'funding_pnl': np.nanmean(funding_pnl),
        'net_pnl': np.nanmean(net_pnl),
        'win': np.nanmean(net_pnl > 0)
    }

print(f"\n{'Thresh':<10} {'Dir':<6} {'N':>7} | {'Price PnL':>11} {'Fund PnL':>11} | {'Net PnL':>11} {'Win%':>7}")
print('-'*80)

for thresh in [0.000008, 0.00001, 0.000012, 0.000015, 0.00002, 0.00003]:
    short = backtest_by_direction(sampled, thresh, 'SHORT', 72)
    long = backtest_by_direction(sampled, thresh, 'LONG', 72)
    
    if short and short['n'] >= 20:
        print(f"{thresh*100:.4f}%   SHORT  {short['n']:>7} | {short['price_pnl']*100:>+10.2f}% {short['funding_pnl']*100:>+10.2f}% | {short['net_pnl']*100:>+10.2f}% {short['win']*100:>6.1f}%")
    if long and long['n'] >= 20:
        print(f"{thresh*100:.4f}%   LONG   {long['n']:>7} | {long['price_pnl']*100:>+10.2f}% {long['funding_pnl']*100:>+10.2f}% | {long['net_pnl']*100:>+10.2f}% {long['win']*100:>6.1f}%")
    print()
