"""
DEEPER ANALYSIS OF DIRECTIONAL MOMENTUM STRATEGY

The basic strategy didn't work - let's understand why and try filters
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("Loading results...")
results_8h = pd.read_csv('directional_momentum_8h_results.csv')
results_8h['entry_hour'] = pd.to_datetime(results_8h['entry_hour'])

print(f"Total trades: {len(results_8h)}")

# =============================================================================
# ISSUE 1: IMBALANCED LONG/SHORT SIGNALS
# =============================================================================

print("\n" + "="*80)
print("ISSUE 1: SIGNAL IMBALANCE")
print("="*80)

print(f"\nLong signals: {(results_8h['direction'] == 'long').sum()}")
print(f"Short signals: {(results_8h['direction'] == 'short').sum()}")
print("\n** Most signals are LONG (extreme negative funding)**")
print("** This means we're betting on coins that are heavily shorted **")
print("** These might be fundamentally weak coins that continue down **")

# =============================================================================
# ISSUE 2: PRICE CONTINUES IN SAME DIRECTION
# =============================================================================

print("\n" + "="*80)
print("ISSUE 2: PRICE TREND ANALYSIS")
print("="*80)

# The hypothesis was wrong - extreme funding might signal the START of a trend
# not the END

longs = results_8h[results_8h['direction'] == 'long']
shorts = results_8h[results_8h['direction'] == 'short']

print("\n### LONG trades (when FR < -0.10%, expecting short squeeze) ###")
print(f"Avg Price Return: {longs['price_return'].mean()*100:.4f}%")
print(f"Avg Funding Collected: {longs['funding_collected'].mean()*100:.4f}%")
print(f"Avg Net PnL: {longs['net_pnl'].mean()*100:.4f}%")
print(f"Win Rate: {(longs['net_pnl'] > 0).mean()*100:.1f}%")

print("\n### SHORT trades (when FR > +0.10%, expecting long liquidation) ###")
print(f"Avg Price Return: {shorts['price_return'].mean()*100:.4f}%")
print(f"Avg Funding Collected: {shorts['funding_collected'].mean()*100:.4f}%")
print(f"Avg Net PnL: {shorts['net_pnl'].mean()*100:.4f}%")
print(f"Win Rate: {(shorts['net_pnl'] > 0).mean()*100:.1f}%")

# =============================================================================
# STRATEGY REVERSAL: GO WITH THE CROWD?
# =============================================================================

print("\n" + "="*80)
print("ALTERNATIVE: TREND-FOLLOWING (Go WITH the crowd)")
print("="*80)

print("\nInstead of betting AGAINST extreme funding, what if we go WITH it?")
print("- When FR < -0.10% (shorts are paying): GO SHORT (join the shorts)")
print("- When FR > +0.10% (longs are paying): GO LONG (join the longs)")

# Reverse the strategy
results_8h['reversed_price_return'] = -results_8h['price_return']

# Funding changes too - we now pay instead of receive
results_8h['reversed_funding'] = -results_8h['funding_collected']
results_8h['reversed_gross'] = results_8h['reversed_price_return'] + results_8h['reversed_funding']
results_8h['reversed_net'] = results_8h['reversed_gross'] - results_8h['trading_fee']

print("\n### REVERSED Strategy Performance ###")
print(f"Avg Price Return: {results_8h['reversed_price_return'].mean()*100:.4f}%")
print(f"Avg Funding (we PAY): {results_8h['reversed_funding'].mean()*100:.4f}%")
print(f"Avg Gross PnL: {results_8h['reversed_gross'].mean()*100:.4f}%")
print(f"Avg Net PnL: {results_8h['reversed_net'].mean()*100:.4f}%")
print(f"Sharpe: {results_8h['reversed_net'].mean() / results_8h['reversed_net'].std():.4f}")
print(f"Win Rate: {(results_8h['reversed_net'] > 0).mean()*100:.1f}%")

# =============================================================================
# COIN-SPECIFIC FILTERING
# =============================================================================

print("\n" + "="*80)
print("FILTERED STRATEGY: ONLY PROFITABLE COINS")
print("="*80)

# From the previous analysis, some coins were profitable
profitable_coins = ['COMP', 'AXS', 'TURBO', 'SUPER', 'OM', 'ACE', 'MERL', '0G']

filtered = results_8h[results_8h['coin'].isin(profitable_coins)]
print(f"\nFiltered to {len(profitable_coins)} historically profitable coins")
print(f"N trades: {len(filtered)}")

if len(filtered) > 0:
    print(f"Avg Net PnL: {filtered['net_pnl'].mean()*100:.4f}%")
    print(f"Sharpe: {filtered['net_pnl'].mean() / filtered['net_pnl'].std():.4f}")
    print(f"Win Rate: {(filtered['net_pnl'] > 0).mean()*100:.1f}%")
    print(f"Total PnL: {filtered['net_pnl'].sum()*100:.2f}%")

# =============================================================================
# HIGHER THRESHOLD FILTER
# =============================================================================

print("\n" + "="*80)
print("FILTERED STRATEGY: ONLY LOW FUNDING RATE (0.10-0.20%)")
print("="*80)

# The 0.10-0.20% bucket had the best performance
low_fr = results_8h[results_8h['entry_funding'].abs().between(0.001, 0.002)]
print(f"\nFiltered to FR between 0.10-0.20%")
print(f"N trades: {len(low_fr)}")

if len(low_fr) > 0:
    print(f"Avg Net PnL: {low_fr['net_pnl'].mean()*100:.4f}%")
    print(f"Sharpe: {low_fr['net_pnl'].mean() / low_fr['net_pnl'].std():.4f}")
    print(f"Win Rate: {(low_fr['net_pnl'] > 0).mean()*100:.1f}%")
    print(f"Total PnL: {low_fr['net_pnl'].sum()*100:.2f}%")

# =============================================================================
# COMBINED FILTERS
# =============================================================================

print("\n" + "="*80)
print("COMBINED FILTERS: Profitable Coins + Low FR")
print("="*80)

combined = results_8h[
    (results_8h['coin'].isin(profitable_coins)) & 
    (results_8h['entry_funding'].abs().between(0.001, 0.002))
]
print(f"\nFiltered to profitable coins AND FR 0.10-0.20%")
print(f"N trades: {len(combined)}")

if len(combined) > 0:
    print(f"Avg Net PnL: {combined['net_pnl'].mean()*100:.4f}%")
    print(f"Sharpe: {combined['net_pnl'].mean() / combined['net_pnl'].std():.4f}")
    print(f"Win Rate: {(combined['net_pnl'] > 0).mean()*100:.1f}%")
    print(f"Total PnL: {combined['net_pnl'].sum()*100:.2f}%")

# =============================================================================
# TIME-BASED FILTER
# =============================================================================

print("\n" + "="*80)
print("TIME-BASED FILTER: Best Hours Only")
print("="*80)

# Hours 01, 09, 13, 16, 17, 18, 20 had positive avg returns
best_hours = [1, 9, 13, 16, 17, 18, 20]
results_8h['entry_hour_of_day'] = results_8h['entry_hour'].dt.hour
time_filtered = results_8h[results_8h['entry_hour_of_day'].isin(best_hours)]

print(f"\nFiltered to best hours: {best_hours}")
print(f"N trades: {len(time_filtered)}")

if len(time_filtered) > 0:
    print(f"Avg Net PnL: {time_filtered['net_pnl'].mean()*100:.4f}%")
    print(f"Sharpe: {time_filtered['net_pnl'].mean() / time_filtered['net_pnl'].std():.4f}")
    print(f"Win Rate: {(time_filtered['net_pnl'] > 0).mean()*100:.1f}%")
    print(f"Total PnL: {time_filtered['net_pnl'].sum()*100:.2f}%")

# =============================================================================
# MONTH FILTER (Recent performance)
# =============================================================================

print("\n" + "="*80)
print("RECENT MONTHS: Oct 2025 - Jan 2026")
print("="*80)

# Oct 2025, Jan 2026 were profitable
recent = results_8h[results_8h['entry_hour'] >= '2025-10-01']
print(f"\nFiltered to Oct 2025 onwards")
print(f"N trades: {len(recent)}")

if len(recent) > 0:
    print(f"Avg Net PnL: {recent['net_pnl'].mean()*100:.4f}%")
    print(f"Sharpe: {recent['net_pnl'].mean() / recent['net_pnl'].std():.4f}")
    print(f"Win Rate: {(recent['net_pnl'] > 0).mean()*100:.1f}%")
    print(f"Total PnL: {recent['net_pnl'].sum()*100:.2f}%")

# =============================================================================
# WHAT ABOUT SHORTER HOLD?
# =============================================================================

print("\n" + "="*80)
print("TESTING 4H HOLD WITH FILTERS")
print("="*80)

results_4h = pd.read_csv('directional_momentum_4h_results.csv')
results_4h['entry_hour'] = pd.to_datetime(results_4h['entry_hour'])
results_4h['entry_hour_of_day'] = results_4h['entry_hour'].dt.hour

# Apply profitable coins filter
filtered_4h = results_4h[results_4h['coin'].isin(profitable_coins)]
print(f"\n4H Hold - Profitable Coins Only")
print(f"N trades: {len(filtered_4h)}")

if len(filtered_4h) > 0:
    print(f"Avg Net PnL: {filtered_4h['net_pnl'].mean()*100:.4f}%")
    print(f"Sharpe: {filtered_4h['net_pnl'].mean() / filtered_4h['net_pnl'].std():.4f}")
    print(f"Win Rate: {(filtered_4h['net_pnl'] > 0).mean()*100:.1f}%")
    print(f"Total PnL: {filtered_4h['net_pnl'].sum()*100:.2f}%")

# =============================================================================
# CONCLUSION
# =============================================================================

print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)

print("""
### Why The Strategy Doesn't Work ###

1. SIGNAL IMBALANCE: 97% of signals are LONG (negative funding)
   - Extreme negative funding often means the coin is fundamentally weak
   - Shorts are paying because they're RIGHT about the direction
   - Betting against them = betting on bad coins

2. TREND CONTINUATION: Extreme funding often signals START of trend, not END
   - Price continues in the same direction
   - Mean reversion doesn't happen fast enough

3. FUNDING IS TOO SMALL: Even collecting funding for 8h (~1.3%),
   price moves (~-1.9%) dominate the outcome

### Possible Improvements ###

1. COIN SELECTION: Only trade coins that historically mean-revert
   - AXS, TURBO, SUPER, COMP showed positive results
   
2. LOWER THRESHOLD: 0.10-0.20% range worked better than higher FR

3. ADDITIONAL FILTERS NEEDED:
   - Volume filters
   - Volatility filters  
   - Open interest change
   - Time in extreme state (don't enter immediately)

4. CONSIDER THE OPPOSITE: Go WITH the crowd (trend-following)
   - The reversed strategy showed positive price returns
   - But you PAY funding instead of receiving
""")
