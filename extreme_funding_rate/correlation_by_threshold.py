import pandas as pd
import numpy as np

events = pd.read_csv('/Users/leeisaackaiyui/Desktop/backtest/extreme_funding_rate/extreme_funding_1h_events.csv')

neg = events[events['type'] == 'negative'].copy()
pos = events[events['type'] == 'positive'].copy()

print("=" * 70)
print("CORRELATION ANALYSIS AT DIFFERENT FUNDING RATE THRESHOLDS")
print("=" * 70)

print("\n### NEGATIVE FUNDING (Long position receives funding) ###")
print(f"{'Threshold':<12} {'N':<8} {'FR Mean':<12} {'Ret Mean':<12} {'Correlation':<12} {'Win Rate':<10}")
print("-" * 70)

for t in [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
    subset = neg[neg['funding_rate'] < -t]
    if len(subset) < 10:
        continue
    corr = subset['funding_rate'].corr(subset['return_1h_pct'])
    win_rate = (subset['return_1h_pct'] > 0).mean() * 100
    print(f"|FR|>{t*100:.2f}%   {len(subset):<8} {subset['funding_rate'].mean()*100:>10.4f}%  {subset['return_1h_pct'].mean():>10.4f}%  {corr:>10.4f}    {win_rate:>8.1f}%")

print("\n### POSITIVE FUNDING (Short position receives funding) ###")
print(f"{'Threshold':<12} {'N':<8} {'FR Mean':<12} {'Ret Mean':<12} {'Correlation':<12} {'Win Rate':<10}")
print("-" * 70)

for t in [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
    subset = pos[pos['funding_rate'] > t]
    if len(subset) < 10:
        continue
    corr = subset['funding_rate'].corr(subset['return_1h_pct'])
    # For short, negative return is profit
    win_rate = (subset['return_1h_pct'] < 0).mean() * 100
    print(f"FR>{t*100:.2f}%     {len(subset):<8} {subset['funding_rate'].mean()*100:>10.4f}%  {subset['return_1h_pct'].mean():>10.4f}%  {corr:>10.4f}    {win_rate:>8.1f}%")

print("\n" + "=" * 70)
print("HEDGED STRATEGY PROFITABILITY BY THRESHOLD")
print("=" * 70)
print(f"{'Threshold':<12} {'Trades':<8} {'Avg FR':<12} {'Net Profit':<14} {'After 0.08% Cost':<16} {'After 0.12% Cost':<16}")
print("-" * 90)

for t in [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
    nf = neg[neg['funding_rate'] < -t]
    pf = pos[pos['funding_rate'] > t]
    n = len(nf) + len(pf)
    if n < 10:
        continue
    
    # Funding earned (hedged = no price exposure)
    funding_earned = abs(nf['funding_rate']).sum() + pf['funding_rate'].sum()
    avg_funding = funding_earned / n * 100
    
    # After costs
    cost_low = 0.0008  # 0.08% round trip
    cost_high = 0.0012  # 0.12% round trip
    
    net_low = (funding_earned - n * cost_low) * 100
    net_high = (funding_earned - n * cost_high) * 100
    
    print(f"|FR|>{t*100:.2f}%   {n:<8} {avg_funding:>10.4f}%  {funding_earned*100:>12.2f}%  {net_low:>14.2f}%    {net_high:>14.2f}%")

print("\n" + "=" * 70)
print("DETAILED BREAKDOWN: WHY |FR| > 0.10% IS RECOMMENDED")
print("=" * 70)

t = 0.001  # 0.10%
nf = neg[neg['funding_rate'] < -t]
pf = pos[pos['funding_rate'] > t]

print(f"\nAt |FR| > 0.10% threshold:")
print(f"  - Negative funding trades: {len(nf)}")
print(f"  - Positive funding trades: {len(pf)}")
print(f"  - Total trades: {len(nf) + len(pf)}")

n = len(nf) + len(pf)
funding_earned = abs(nf['funding_rate']).sum() + pf['funding_rate'].sum()
avg_funding = funding_earned / n

print(f"\n  Avg funding per trade: {avg_funding*100:.4f}%")
print(f"  Total funding earned:  {funding_earned*100:.2f}%")

print(f"\n  Cost scenarios (round-trip for perp+spot):")
for cost_name, cost in [("Low (0.05%)", 0.0005), ("Medium (0.08%)", 0.0008), ("High (0.12%)", 0.0012), ("Very High (0.15%)", 0.0015)]:
    net = funding_earned - n * cost
    net_per_trade = avg_funding - cost
    print(f"    {cost_name}: Net profit = {net*100:.2f}% ({net_per_trade*100:.4f}%/trade)")

print(f"\n  Break-even cost: {avg_funding*100:.4f}% per trade")

# Time distribution
print("\n" + "=" * 70)
print("OPPORTUNITY FREQUENCY")
print("=" * 70)

events['hour'] = pd.to_datetime(events['hour'])
date_range = (events['hour'].max() - events['hour'].min()).days

for t in [0.001, 0.005, 0.01]:
    nf = neg[neg['funding_rate'] < -t]
    pf = pos[pos['funding_rate'] > t]
    n = len(nf) + len(pf)
    per_day = n / date_range if date_range > 0 else 0
    print(f"|FR|>{t*100:.2f}%: {n} trades over {date_range} days = {per_day:.2f} trades/day")
