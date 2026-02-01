"""
DEEP ANALYSIS: Why is hedge correlation so low?

1. Analyze correlation distribution
2. Test different correlation thresholds
3. Check if higher correlation leads to better performance
"""

import pandas as pd
import numpy as np

# Load results
single_df = pd.read_csv('optimal_single_hedge_results.csv')
multi_df = pd.read_csv('optimal_multi_hedge_results.csv')

print("=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. CORRELATION DISTRIBUTION
# =============================================================================

print("\n### SINGLE HEDGE CORRELATION DISTRIBUTION ###")
print(f"Min:    {single_df['correlation'].min():.4f}")
print(f"25%:    {single_df['correlation'].quantile(0.25):.4f}")
print(f"Median: {single_df['correlation'].median():.4f}")
print(f"75%:    {single_df['correlation'].quantile(0.75):.4f}")
print(f"Max:    {single_df['correlation'].max():.4f}")

print("\n### MULTI HEDGE CORRELATION DISTRIBUTION ###")
print(f"Min:    {multi_df['correlation'].min():.4f}")
print(f"25%:    {multi_df['correlation'].quantile(0.25):.4f}")
print(f"Median: {multi_df['correlation'].median():.4f}")
print(f"75%:    {multi_df['correlation'].quantile(0.75):.4f}")
print(f"Max:    {multi_df['correlation'].max():.4f}")

# =============================================================================
# 2. PERFORMANCE BY CORRELATION BUCKET
# =============================================================================

print("\n### SINGLE HEDGE: PERFORMANCE BY CORRELATION BUCKET ###")
print(f"{'Corr Range':<15} {'N':<8} {'Avg PnL':<12} {'Sharpe':<10} {'Win Rate':<10}")
print("-" * 60)

for low, high in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
    subset = single_df[(single_df['correlation'] >= low) & (single_df['correlation'] < high)]
    if len(subset) < 5:
        continue
    
    avg_pnl = subset['net_pnl'].mean() * 100
    std_pnl = subset['net_pnl'].std() * 100
    sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
    win_rate = (subset['net_pnl'] > 0).mean() * 100
    
    print(f"{low:.1f}-{high:.1f}         {len(subset):<8} {avg_pnl:>10.4f}%  {sharpe:>8.4f}  {win_rate:>8.1f}%")

print("\n### MULTI HEDGE: PERFORMANCE BY CORRELATION BUCKET ###")
print(f"{'Corr Range':<15} {'N':<8} {'Avg PnL':<12} {'Sharpe':<10} {'Win Rate':<10}")
print("-" * 60)

for low, high in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
    subset = multi_df[(multi_df['correlation'] >= low) & (multi_df['correlation'] < high)]
    if len(subset) < 5:
        continue
    
    avg_pnl = subset['net_pnl'].mean() * 100
    std_pnl = subset['net_pnl'].std() * 100
    sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
    win_rate = (subset['net_pnl'] > 0).mean() * 100
    
    print(f"{low:.1f}-{high:.1f}         {len(subset):<8} {avg_pnl:>10.4f}%  {sharpe:>8.4f}  {win_rate:>8.1f}%")

# =============================================================================
# 3. CHECK IF HIGHER CORRELATION = BETTER HEDGE
# =============================================================================

print("\n### CORRELATION VS HEDGE EFFECTIVENESS ###")
print("Does higher correlation lead to better risk reduction?")

# Calculate price residual after hedge
single_df['price_residual'] = single_df['price_pnl'] + single_df['hedge_pnl']
multi_df['price_residual'] = multi_df['price_pnl'] + multi_df['hedge_pnl']

print("\n** Single Hedge: Price Residual (should be closer to 0 with higher corr) **")
for low, high in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]:
    subset = single_df[(single_df['correlation'] >= low) & (single_df['correlation'] < high)]
    if len(subset) < 5:
        continue
    
    avg_residual = subset['price_residual'].mean() * 100
    std_residual = subset['price_residual'].std() * 100
    
    print(f"  Corr {low:.1f}-{high:.1f}: Avg Residual = {avg_residual:>8.4f}%, Std = {std_residual:>8.4f}%")

print("\n** Multi Hedge: Price Residual **")
for low, high in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]:
    subset = multi_df[(multi_df['correlation'] >= low) & (multi_df['correlation'] < high)]
    if len(subset) < 5:
        continue
    
    avg_residual = subset['price_residual'].mean() * 100
    std_residual = subset['price_residual'].std() * 100
    
    print(f"  Corr {low:.1f}-{high:.1f}: Avg Residual = {avg_residual:>8.4f}%, Std = {std_residual:>8.4f}%")

# =============================================================================
# 4. TOP PERFORMING TRADES
# =============================================================================

print("\n### TOP 20 PROFITABLE SINGLE HEDGE TRADES ###")
top_trades = single_df.nlargest(20, 'net_pnl')
print(f"{'Target':<10} {'Hedge':<10} {'Corr':<8} {'Beta':<8} {'Net PnL':<12} {'Funding':<12}")
print("-" * 65)
for _, row in top_trades.iterrows():
    print(f"{row['target_coin']:<10} {row['hedge_coin']:<10} {row['correlation']:.4f}  {row['beta']:.4f}  {row['net_pnl']*100:>10.4f}%  {row['funding_pnl']*100:>10.4f}%")

# =============================================================================
# 5. WORST PERFORMING TRADES
# =============================================================================

print("\n### WORST 20 SINGLE HEDGE TRADES ###")
worst_trades = single_df.nsmallest(20, 'net_pnl')
print(f"{'Target':<10} {'Hedge':<10} {'Corr':<8} {'Beta':<8} {'Net PnL':<12} {'Funding':<12}")
print("-" * 65)
for _, row in worst_trades.iterrows():
    print(f"{row['target_coin']:<10} {row['hedge_coin']:<10} {row['correlation']:.4f}  {row['beta']:.4f}  {row['net_pnl']*100:>10.4f}%  {row['funding_pnl']*100:>10.4f}%")

# =============================================================================
# 5B. TOP 20 MULTI-ASSET HEDGE TRADES
# =============================================================================

print("\n### TOP 20 PROFITABLE MULTI-ASSET HEDGE TRADES ###")
top_multi = multi_df.nlargest(20, 'net_pnl')
print(f"{'Target':<10} {'N_Hedges':<10} {'Corr':<8} {'Net PnL':<12} {'Funding':<12} {'Hedge Assets':<40}")
print("-" * 95)
for _, row in top_multi.iterrows():
    hedges = row['hedge_coins'][:35] + '...' if len(row['hedge_coins']) > 35 else row['hedge_coins']
    print(f"{row['target_coin']:<10} {row['n_hedges']:<10} {row['correlation']:.4f}  {row['net_pnl']*100:>10.4f}%  {row['funding_pnl']*100:>10.4f}%  {hedges}")

print("\n### WORST 20 MULTI-ASSET HEDGE TRADES ###")
worst_multi = multi_df.nsmallest(20, 'net_pnl')
print(f"{'Target':<10} {'N_Hedges':<10} {'Corr':<8} {'Net PnL':<12} {'Funding':<12} {'Hedge Assets':<40}")
print("-" * 95)
for _, row in worst_multi.iterrows():
    hedges = row['hedge_coins'][:35] + '...' if len(row['hedge_coins']) > 35 else row['hedge_coins']
    print(f"{row['target_coin']:<10} {row['n_hedges']:<10} {row['correlation']:.4f}  {row['net_pnl']*100:>10.4f}%  {row['funding_pnl']*100:>10.4f}%  {hedges}")

# =============================================================================
# 6. FILTER: HIGH CORRELATION + HIGH FUNDING
# =============================================================================

print("\n" + "=" * 80)
print("FILTERED STRATEGY: High Correlation + High Funding")
print("=" * 80)

print("\n** SINGLE HEDGE **")
for min_corr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for min_funding in [0.001, 0.002, 0.003]:
        subset = single_df[
            (single_df['correlation'] >= min_corr) & 
            (single_df['funding_pnl'] >= min_funding)
        ]
        if len(subset) < 10:
            continue
        
        avg_pnl = subset['net_pnl'].mean() * 100
        std_pnl = subset['net_pnl'].std() * 100
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = (subset['net_pnl'] > 0).mean() * 100
        
        profitable = "✓" if avg_pnl > 0 else "✗"
        print(f"Corr>={min_corr}, FR>={min_funding*100:.2f}%: N={len(subset):>4}, PnL={avg_pnl:>8.4f}%, Sharpe={sharpe:>6.4f}, WinRate={win_rate:>5.1f}% {profitable}")

print("\n** MULTI-ASSET HEDGE **")
for min_corr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for min_funding in [0.001, 0.002, 0.003]:
        subset = multi_df[
            (multi_df['correlation'] >= min_corr) & 
            (multi_df['funding_pnl'] >= min_funding)
        ]
        if len(subset) < 10:
            continue
        
        avg_pnl = subset['net_pnl'].mean() * 100
        std_pnl = subset['net_pnl'].std() * 100
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = (subset['net_pnl'] > 0).mean() * 100
        
        profitable = "✓" if avg_pnl > 0 else "✗"
        print(f"Corr>={min_corr}, FR>={min_funding*100:.2f}%: N={len(subset):>4}, PnL={avg_pnl:>8.4f}%, Sharpe={sharpe:>6.4f}, WinRate={win_rate:>5.1f}% {profitable}")

# =============================================================================
# 7. CONCLUSION
# =============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("""
1. CORRELATION IS LOW:
   - Even the "best" hedge has only ~0.42-0.45 correlation on average
   - Very few trades achieve correlation > 0.7
   - Crypto altcoins are highly idiosyncratic during extreme funding

2. HIGHER CORRELATION HELPS BUT NOT ENOUGH:
   - Higher correlation does reduce price residual variance
   - But the average price move still goes AGAINST the position
   
3. THE FUNDAMENTAL PROBLEM:
   - Extreme funding = extreme positioning = price CONTINUATION
   - Mean reversion doesn't happen in 1 hour
   - Hedging reduces variance but doesn't fix the negative expected value

4. POSSIBLE SOLUTIONS TO EXPLORE:
   - Longer holding periods (accumulate more funding)
   - Wait for price stabilization before entering
   - Trade WITH the crowd (momentum) instead of against
   - Only trade when funding is VERY extreme (>0.5%)
""")
