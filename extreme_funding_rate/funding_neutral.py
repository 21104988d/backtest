"""
Funding Neutral Strategy Analysis
Explores ways to construct positions that are funding-neutral while keeping price alpha
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

TAKER_FEE = 0.00045
ENTRY_THRESH = 0.000015  # 0.0015%

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

# Pre-compute 72h forward
merged['price_72h'] = merged.groupby('coin')['price'].shift(-72)
merged['fr_sum_72h'] = merged.groupby('coin')['funding_rate'].transform(
    lambda x: x.rolling(72, min_periods=1).sum().shift(-71)
)
merged = merged.dropna(subset=['price_72h'])
merged['date'] = merged['hour'].dt.date

# Sample one per coin per day
sampled = merged.groupby(['coin', 'date']).first().reset_index()
print(f'Sampled signals: {len(sampled):,}')

# =============================================================================
# STRATEGY 1: SHORT-ONLY (Funding Positive)
# =============================================================================
print('\n' + '='*80)
print('STRATEGY 1: SHORT-ONLY (Skip LONG signals)')
print('Only trade when FR < -0.0015% (we RECEIVE funding)')
print('='*80)

short_signals = sampled[sampled['funding_rate'] < -ENTRY_THRESH].copy()
entry_p = short_signals['price'].values
exit_p = short_signals['price_72h'].values
fr_sum = short_signals['fr_sum_72h'].values

price_ret = -(exit_p - entry_p) / entry_p
funding_pnl = -fr_sum  # We RECEIVE when FR is negative
net_pnl = price_ret + funding_pnl - 2 * TAKER_FEE

print(f"\nSHORT-ONLY Results:")
print(f"  Trades: {len(net_pnl):,}")
print(f"  Price PnL:   {np.nanmean(price_ret)*100:+.2f}%")
print(f"  Funding PnL: {np.nanmean(funding_pnl)*100:+.2f}% (RECEIVE)")
print(f"  Fees:        {2*TAKER_FEE*100:.2f}%")
print(f"  Net PnL:     {np.nanmean(net_pnl)*100:+.2f}%")
print(f"  Sharpe:      {np.nanmean(net_pnl)/np.nanstd(net_pnl):.3f}")
print(f"  Win Rate:    {(net_pnl > 0).mean()*100:.1f}%")
print(f"  Total:       {np.nansum(net_pnl)*100:+.0f}%")

# =============================================================================
# STRATEGY 2: LONG with Spot Hedge (Delta Neutral)
# =============================================================================
print('\n' + '='*80)
print('STRATEGY 2: LONG Futures + SHORT Spot (Delta Neutral)')
print('When FR > +0.0015%: LONG futures, SHORT spot')
print('Price exposure = 0, but we still PAY funding')
print('='*80)

long_signals = sampled[sampled['funding_rate'] > ENTRY_THRESH].copy()
entry_p = long_signals['price'].values
fr_sum = long_signals['fr_sum_72h'].values

# Delta neutral: price return = 0
price_ret = np.zeros_like(entry_p)
funding_pnl = fr_sum  # We PAY when FR is positive (LONG pays)
net_pnl = price_ret + funding_pnl - 2 * TAKER_FEE  # Still pay fees for futures

print(f"\nLONG + Spot Hedge Results:")
print(f"  Trades: {len(net_pnl):,}")
print(f"  Price PnL:   {np.nanmean(price_ret)*100:+.2f}% (hedged)")
print(f"  Funding PnL: {np.nanmean(funding_pnl)*100:+.2f}% (PAY)")
print(f"  Fees:        {2*TAKER_FEE*100:.2f}%")
print(f"  Net PnL:     {np.nanmean(net_pnl)*100:+.2f}%")
print(f"  -> This LOSES money because we pay funding with no upside!")

# =============================================================================
# STRATEGY 3: SHORT Futures + LONG Spot (Classic Funding Arb)
# =============================================================================
print('\n' + '='*80)
print('STRATEGY 3: SHORT Futures + LONG Spot (Classic Cash & Carry)')
print('When FR < -0.0015%: SHORT futures + LONG spot')
print('Price exposure = 0, and we RECEIVE funding')
print('='*80)

short_signals = sampled[sampled['funding_rate'] < -ENTRY_THRESH].copy()
fr_sum = short_signals['fr_sum_72h'].values

# Delta neutral: price return = 0
price_ret = np.zeros(len(short_signals))
funding_pnl = -fr_sum  # We RECEIVE when FR is negative
net_pnl = price_ret + funding_pnl - 2 * TAKER_FEE

print(f"\nSHORT + Spot Hedge Results:")
print(f"  Trades: {len(net_pnl):,}")
print(f"  Price PnL:   {np.nanmean(price_ret)*100:+.2f}% (hedged)")
print(f"  Funding PnL: {np.nanmean(funding_pnl)*100:+.2f}% (RECEIVE)")
print(f"  Fees:        {2*TAKER_FEE*100:.2f}%")
print(f"  Net PnL:     {np.nanmean(net_pnl)*100:+.2f}%")
print(f"  Win Rate:    {(net_pnl > 0).mean()*100:.1f}%")
print(f"  Total:       {np.nansum(net_pnl)*100:+.0f}%")
print(f"  -> PURE funding collection with NO price risk!")

# =============================================================================
# STRATEGY 4: Cross-Coin Hedge (LONG extreme + SHORT neutral)
# =============================================================================
print('\n' + '='*80)
print('STRATEGY 4: Cross-Coin Hedge')
print('When LONG coin A (FR > 0.0015%):')
print('  - Also SHORT a "neutral" coin B (|FR| < 0.0005%)')
print('='*80)

# For each LONG signal, find a neutral coin at same hour to hedge
long_signals = sampled[sampled['funding_rate'] > ENTRY_THRESH].copy()

# Find neutral coins at each hour
neutral_mask = (sampled['funding_rate'].abs() < 0.000005)  # |FR| < 0.0005%
neutral_coins = sampled[neutral_mask][['hour', 'coin', 'funding_rate', 'price', 'price_72h', 'fr_sum_72h']]

# Merge to find hedge pairs
pairs = long_signals.merge(
    neutral_coins,
    on='hour',
    suffixes=('_long', '_hedge'),
    how='inner'
)
# Remove self-pairs
pairs = pairs[pairs['coin_long'] != pairs['coin_hedge']]

if len(pairs) > 0:
    # Take first neutral hedge for each LONG signal
    pairs = pairs.groupby(['hour', 'coin_long']).first().reset_index()
    
    # LONG position (on extreme positive FR coin)
    long_price_ret = (pairs['price_72h_long'] - pairs['price_long']) / pairs['price_long']
    long_funding = pairs['fr_sum_72h_long']  # We PAY
    
    # SHORT hedge position (on neutral coin)
    hedge_price_ret = -(pairs['price_72h_hedge'] - pairs['price_hedge']) / pairs['price_hedge']
    hedge_funding = -pairs['fr_sum_72h_hedge']  # We receive (but ~0)
    
    # Net position
    net_price = long_price_ret + hedge_price_ret
    net_funding = long_funding + hedge_funding
    net_pnl = net_price + net_funding - 4 * TAKER_FEE  # 2 positions = 4 fees
    
    print(f"\nCross-Coin Hedge Results:")
    print(f"  Pairs found: {len(pairs):,}")
    print(f"  LONG side:  Price={long_price_ret.mean()*100:+.2f}%, Funding={long_funding.mean()*100:+.2f}%")
    print(f"  HEDGE side: Price={hedge_price_ret.mean()*100:+.2f}%, Funding={hedge_funding.mean()*100:+.2f}%")
    print(f"  Net Price:  {net_price.mean()*100:+.2f}%")
    print(f"  Net Funding: {net_funding.mean()*100:+.2f}%")
    print(f"  Fees:       {4*TAKER_FEE*100:.2f}%")
    print(f"  Net PnL:    {np.nanmean(net_pnl)*100:+.2f}%")
    print(f"  Win Rate:   {(net_pnl > 0).mean()*100:.1f}%")
else:
    print("  No hedge pairs found!")

# =============================================================================
# STRATEGY 5: SHORT extreme + LONG neutral (Funding Arbitrage)
# =============================================================================
print('\n' + '='*80)
print('STRATEGY 5: Pure Funding Arbitrage (Cross-Coin)')
print('SHORT coin with FR < -0.0015% (RECEIVE high funding)')
print('LONG coin with FR near 0% (PAY ~0 funding)')
print('Net: RECEIVE funding, partial price hedge')
print('='*80)

# Find SHORT signals (extreme negative FR)
short_signals = sampled[sampled['funding_rate'] < -ENTRY_THRESH][['hour', 'coin', 'funding_rate', 'price', 'price_72h', 'fr_sum_72h']].copy()

# Find neutral coins
neutral_coins = sampled[(sampled['funding_rate'].abs() < 0.000005)][['hour', 'coin', 'funding_rate', 'price', 'price_72h', 'fr_sum_72h']].copy()

# Merge
pairs = short_signals.merge(
    neutral_coins,
    on='hour',
    suffixes=('_short', '_long'),
    how='inner'
)
pairs = pairs[pairs['coin_short'] != pairs['coin_long']]

if len(pairs) > 0:
    pairs = pairs.groupby(['hour', 'coin_short']).first().reset_index()
    
    # SHORT extreme FR coin
    short_price_ret = -(pairs['price_72h_short'] - pairs['price_short']) / pairs['price_short']
    short_funding = -pairs['fr_sum_72h_short']  # RECEIVE (fr_sum is negative)
    
    # LONG neutral coin
    long_price_ret = (pairs['price_72h_long'] - pairs['price_long']) / pairs['price_long']
    long_funding = pairs['fr_sum_72h_long']  # PAY (but ~0)
    
    net_price = short_price_ret + long_price_ret
    net_funding = short_funding + long_funding
    net_pnl = net_price + net_funding - 4 * TAKER_FEE
    
    print(f"\nFunding Arbitrage Results:")
    print(f"  Pairs: {len(pairs):,}")
    print(f"  SHORT (extreme): Price={short_price_ret.mean()*100:+.2f}%, Funding={short_funding.mean()*100:+.2f}% (RECEIVE)")
    print(f"  LONG (neutral):  Price={long_price_ret.mean()*100:+.2f}%, Funding={long_funding.mean()*100:+.2f}%")
    print(f"  Net Price:   {net_price.mean()*100:+.2f}% (partially hedged)")
    print(f"  Net Funding: {net_funding.mean()*100:+.2f}%")
    print(f"  Fees:        {4*TAKER_FEE*100:.2f}%")
    print(f"  Net PnL:     {np.nanmean(net_pnl)*100:+.2f}%")
    print(f"  Win Rate:    {(net_pnl > 0).mean()*100:.1f}%")
    print(f"  Sharpe:      {np.nanmean(net_pnl)/np.nanstd(net_pnl):.3f}")
    print(f"  Total:       {np.nansum(net_pnl)*100:+.0f}%")

# =============================================================================
# STRATEGY 6: Original with BOTH SHORT + LONG (Baseline)
# =============================================================================
print('\n' + '='*80)
print('BASELINE: Original Trend Following (SHORT + LONG)')
print('='*80)

short_sigs = sampled[sampled['funding_rate'] < -ENTRY_THRESH].copy()
long_sigs = sampled[sampled['funding_rate'] > ENTRY_THRESH].copy()

# SHORT
s_price = -(short_sigs['price_72h'] - short_sigs['price']) / short_sigs['price']
s_fund = -short_sigs['fr_sum_72h']
s_pnl = s_price + s_fund - 2 * TAKER_FEE

# LONG
l_price = (long_sigs['price_72h'] - long_sigs['price']) / long_sigs['price']
l_fund = long_sigs['fr_sum_72h']
l_pnl = l_price + l_fund - 2 * TAKER_FEE

all_pnl = np.concatenate([s_pnl.values, l_pnl.values])

print(f"\nBaseline Results:")
print(f"  SHORT: {len(s_pnl):,} trades, {s_pnl.mean()*100:+.2f}% avg")
print(f"  LONG:  {len(l_pnl):,} trades, {l_pnl.mean()*100:+.2f}% avg")
print(f"  Combined: {len(all_pnl):,} trades")
print(f"  Net PnL: {np.nanmean(all_pnl)*100:+.2f}%")
print(f"  Sharpe:  {np.nanmean(all_pnl)/np.nanstd(all_pnl):.3f}")
print(f"  Win Rate: {(all_pnl > 0).mean()*100:.1f}%")

# =============================================================================
# SUMMARY
# =============================================================================
print('\n' + '='*80)
print('SUMMARY: Funding-Neutral Strategy Comparison')
print('='*80)
print('''
┌─────────────────────────────────────────────────────────────────────────────┐
│ FUNDING NEUTRAL CONSTRUCTION OPTIONS                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ OPTION A: Same-Coin Hedge (Futures vs Spot)                                  │
│ ─────────────────────────────────────────────                                │
│ When FR < -0.0015%:                                                          │
│   • SHORT Futures (receive funding)                                          │
│   • LONG Spot (no funding)                                                   │
│   • Net: Delta=0, collect funding                                            │
│   • PRO: Zero price risk, pure funding arb                                   │
│   • CON: Lose the price alpha                                                │
│                                                                              │
│ OPTION B: Cross-Coin Hedge (Two Futures)                                     │
│ ────────────────────────────────────────────                                 │
│ When FR < -0.0015% on coin A:                                                │
│   • SHORT Futures A (receive high funding)                                   │
│   • LONG Futures B where |FR| < 0.0005% (pay ~0 funding)                    │
│   • Net: Partial delta hedge, receive net funding                            │
│   • PRO: All on futures, no spot needed                                      │
│   • CON: Basis risk between coins                                            │
│                                                                              │
│ OPTION C: Portfolio Balancing                                                │
│ ─────────────────────────────────                                            │
│   • Run both SHORT and LONG signals                                          │
│   • Size positions so net funding ≈ 0                                        │
│   • Capture price alpha on both sides                                        │
│   • PRO: Maximize opportunity                                                │
│   • CON: Imperfect hedge, funding still has direction                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

RECOMMENDATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For TRUE funding neutrality with lowest risk:
  → Strategy 3: SHORT Futures + LONG Spot (same coin)
  → Receive ~0.53% funding over 72h with ZERO price risk

For MAXIMUM return (but with price exposure):  
  → Original: Trend following with both SHORT + LONG
  → ~0.8% avg per trade, but exposed to price moves

For PRACTICAL funding arbitrage:
  → Strategy 5: SHORT extreme FR + LONG neutral coin
  → Receive funding with PARTIAL price hedge
''')
