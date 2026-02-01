"""
Deep dive analysis: Why is hedged funding capture not profitable?

Hypothesis:
1. Price moves against funding direction (mean reversion) - already captured in unhedged
2. Hedge instruments (BTC/ETH) don't move with altcoins as expected
3. Timing issue - price moves before funding settlement
4. Need to look at different holding periods
"""

import pandas as pd
import numpy as np
from datetime import timedelta

# Load data
funding = pd.read_csv('funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)
funding['hour'] = funding['datetime'].dt.floor('h').dt.tz_localize(None)

price = pd.read_csv('price_history.csv')
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
price['hour'] = price['timestamp'].dt.floor('h').dt.tz_localize(None)

events = pd.read_csv('extreme_funding_1h_events.csv')
events['hour'] = pd.to_datetime(events['hour'])

# Build price lookup
price_pivot = price.pivot_table(index='hour', columns='coin', values='price', aggfunc='first')

THRESHOLD = 0.001

print("=" * 80)
print("DEEP DIVE: WHY HEDGED FUNDING CAPTURE IS NOT PROFITABLE")
print("=" * 80)

# =============================================================================
# 1. ANALYSIS: When does funding get paid?
# =============================================================================
print("\n### 1. FUNDING TIMING ANALYSIS ###")
print("Hyperliquid pays funding every hour at the hour mark.")
print("Strategy: Enter BEFORE funding, exit AFTER receiving funding.")

# =============================================================================
# 2. PRICE MOVEMENT AROUND FUNDING
# =============================================================================
print("\n### 2. PRICE MOVEMENT AROUND EXTREME FUNDING ###")

neg = events[(events['type'] == 'negative') & (events['funding_rate'] < -THRESHOLD)]
pos = events[(events['type'] == 'positive') & (events['funding_rate'] > THRESHOLD)]

print(f"\nNegative extreme funding (Long position to receive funding):")
print(f"  Count: {len(neg)}")
print(f"  Avg funding rate: {neg['funding_rate'].mean()*100:.4f}%")
print(f"  Avg 1h return: {neg['return_1h_pct'].mean():.4f}%")
print(f"  If you LONG: funding +{abs(neg['funding_rate'].mean())*100:.4f}%, price {neg['return_1h_pct'].mean():.4f}%")
print(f"  Net (unhedged): {(abs(neg['funding_rate'].mean())*100 + neg['return_1h_pct'].mean()):.4f}%")

print(f"\nPositive extreme funding (Short position to receive funding):")
print(f"  Count: {len(pos)}")
print(f"  Avg funding rate: {pos['funding_rate'].mean()*100:.4f}%")
print(f"  Avg 1h return: {pos['return_1h_pct'].mean():.4f}%")
print(f"  If you SHORT: funding +{pos['funding_rate'].mean()*100:.4f}%, price {-pos['return_1h_pct'].mean():.4f}%")
print(f"  Net (unhedged): {(pos['funding_rate'].mean()*100 - pos['return_1h_pct'].mean()):.4f}%")

# =============================================================================
# 3. THE PROBLEM: Price moves against you!
# =============================================================================
print("\n### 3. THE CORE ISSUE ###")
print("""
The problem is clear:
- When funding is NEGATIVE (crowded shorts), price tends to DROP further
- When funding is POSITIVE (crowded longs), price tends to... also DROP

This suggests:
1. Extreme funding predicts CONTINUATION, not mean reversion in 1h
2. The funding you earn is less than the adverse price move
""")

# =============================================================================
# 4. TEST DIFFERENT HOLDING PERIODS
# =============================================================================
print("\n### 4. OPTIMAL HOLDING PERIOD ANALYSIS ###")
print("Testing if longer holding periods show mean reversion...")

def get_return_at_offset(coin, start_hour, offset_hours, price_pivot):
    """Get return from start_hour to start_hour + offset"""
    end_hour = start_hour + timedelta(hours=offset_hours)
    if start_hour not in price_pivot.index or end_hour not in price_pivot.index:
        return np.nan
    if coin not in price_pivot.columns:
        return np.nan
    
    start_price = price_pivot.loc[start_hour, coin]
    end_price = price_pivot.loc[end_hour, coin]
    
    if pd.isna(start_price) or pd.isna(end_price):
        return np.nan
    
    return (end_price / start_price - 1) * 100

print(f"\n{'Offset':<10} {'Neg FR Ret':<14} {'Pos FR Ret':<14} {'Neg+FR Net':<14} {'Pos+FR Net':<14}")
print("-" * 70)

for offset in [1, 2, 4, 8, 12, 24, 48, 72]:
    neg_rets = []
    pos_rets = []
    
    for _, row in neg.iterrows():
        ret = get_return_at_offset(row['coin'], row['hour'], offset, price_pivot)
        if not pd.isna(ret):
            neg_rets.append(ret)
    
    for _, row in pos.iterrows():
        ret = get_return_at_offset(row['coin'], row['hour'], offset, price_pivot)
        if not pd.isna(ret):
            pos_rets.append(ret)
    
    if neg_rets and pos_rets:
        avg_neg = np.mean(neg_rets)
        avg_pos = np.mean(pos_rets)
        
        # Net for negative (long position)
        net_neg = abs(neg['funding_rate'].mean()) * 100 + avg_neg
        # Net for positive (short position)  
        net_pos = pos['funding_rate'].mean() * 100 - avg_pos
        
        print(f"{offset}h        {avg_neg:>12.4f}%  {avg_pos:>12.4f}%  {net_neg:>12.4f}%  {net_pos:>12.4f}%")

# =============================================================================
# 5. ALTERNATIVE STRATEGY: Only trade when FR is VERY extreme
# =============================================================================
print("\n### 5. HIGHER THRESHOLD ANALYSIS ###")
print("Perhaps only VERY extreme funding rates show profit...")

for t in [0.002, 0.003, 0.005, 0.01]:
    neg_t = events[(events['type'] == 'negative') & (events['funding_rate'] < -t)]
    pos_t = events[(events['type'] == 'positive') & (events['funding_rate'] > t)]
    
    if len(neg_t) > 5:
        net_neg = abs(neg_t['funding_rate'].mean())*100 + neg_t['return_1h_pct'].mean()
        print(f"\n|FR| > {t*100:.2f}%:")
        print(f"  Negative: n={len(neg_t)}, FR={neg_t['funding_rate'].mean()*100:.4f}%, Ret={neg_t['return_1h_pct'].mean():.4f}%, Net={net_neg:.4f}%")
    
    if len(pos_t) > 5:
        net_pos = pos_t['funding_rate'].mean()*100 - pos_t['return_1h_pct'].mean()
        print(f"  Positive: n={len(pos_t)}, FR={pos_t['funding_rate'].mean()*100:.4f}%, Ret={pos_t['return_1h_pct'].mean():.4f}%, Net={net_pos:.4f}%")

# =============================================================================
# 6. CHECK BTC/ETH CORRELATION DURING EXTREME FUNDING
# =============================================================================
print("\n### 6. HEDGE EFFECTIVENESS DURING EXTREME FUNDING ###")

returns = price_pivot.pct_change()

def get_correlation_during_events(events_df, returns_df):
    """Calculate correlation between altcoin and BTC/ETH during extreme funding events"""
    correlations_btc = []
    correlations_eth = []
    
    for _, row in events_df.iterrows():
        coin = row['coin']
        hour = row['hour']
        
        if coin in ['BTC', 'ETH']:
            continue
        
        if hour not in returns_df.index:
            continue
        
        next_hour = hour + timedelta(hours=1)
        if next_hour not in returns_df.index:
            continue
        
        coin_ret = returns_df.loc[next_hour, coin] if coin in returns_df.columns else np.nan
        btc_ret = returns_df.loc[next_hour, 'BTC'] if 'BTC' in returns_df.columns else np.nan
        eth_ret = returns_df.loc[next_hour, 'ETH'] if 'ETH' in returns_df.columns else np.nan
        
        if not pd.isna(coin_ret) and not pd.isna(btc_ret):
            correlations_btc.append((coin_ret, btc_ret))
        if not pd.isna(coin_ret) and not pd.isna(eth_ret):
            correlations_eth.append((coin_ret, eth_ret))
    
    if correlations_btc:
        arr = np.array(correlations_btc)
        corr_btc = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
    else:
        corr_btc = np.nan
    
    if correlations_eth:
        arr = np.array(correlations_eth)
        corr_eth = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
    else:
        corr_eth = np.nan
    
    return corr_btc, corr_eth, len(correlations_btc)

neg_extreme = events[(events['type'] == 'negative') & (events['funding_rate'] < -THRESHOLD)]
pos_extreme = events[(events['type'] == 'positive') & (events['funding_rate'] > THRESHOLD)]

corr_btc_neg, corr_eth_neg, n_neg = get_correlation_during_events(neg_extreme, returns)
corr_btc_pos, corr_eth_pos, n_pos = get_correlation_during_events(pos_extreme, returns)

print(f"\nCorrelation with BTC during negative extreme funding: {corr_btc_neg:.4f} (n={n_neg})")
print(f"Correlation with ETH during negative extreme funding: {corr_eth_neg:.4f}")
print(f"Correlation with BTC during positive extreme funding: {corr_btc_pos:.4f} (n={n_pos})")
print(f"Correlation with ETH during positive extreme funding: {corr_eth_pos:.4f}")

print("""
Low correlation explains why hedging doesn't help much:
- During extreme funding moments, altcoins move idiosyncratically
- BTC/ETH hedge doesn't capture altcoin-specific moves
""")

# =============================================================================
# 7. CONCLUSION
# =============================================================================
print("\n" + "=" * 80)
print("CONCLUSION & ALTERNATIVE STRATEGIES")
print("=" * 80)
print("""
Key Findings:
1. Extreme funding does NOT lead to immediate mean reversion (1h)
2. Price tends to continue moving against the funding position
3. BTC/ETH hedges don't effectively neutralize altcoin-specific risk
4. The funding earned is smaller than the adverse price movement

Alternative Strategies to Consider:
1. DELAYED ENTRY: Wait for price to stabilize after funding payment
2. SAME-COIN HEDGE: Use spot vs perp on same coin (perfect hedge, but need spot access)
3. FUNDING RATE MOMENTUM: Trade the same direction as extreme funding (not against it)
4. LONGER HOLDING: Hold through multiple funding periods to accumulate more funding
5. PORTFOLIO APPROACH: Diversify across many extreme funding coins to reduce idiosyncratic risk
""")
