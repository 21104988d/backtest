"""
Analyze data coverage and plan Binance price fetching
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("ANALYZING DATA COVERAGE")
print("=" * 80)

# Load funding history
funding = pd.read_csv('funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)

# Load current price history
price = pd.read_csv('price_history.csv')
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)

print(f"\nFunding data: {len(funding):,} records, {funding['coin'].nunique()} coins")
print(f"Price data: {len(price):,} records, {price['coin'].nunique()} coins")

# Get coverage per coin for funding
funding_coverage = funding.groupby('coin').agg({
    'datetime': ['min', 'max', 'count']
}).reset_index()
funding_coverage.columns = ['coin', 'funding_start', 'funding_end', 'funding_count']

# Get coverage per coin for price
price_coverage = price.groupby('coin').agg({
    'timestamp': ['min', 'max', 'count']
}).reset_index()
price_coverage.columns = ['coin', 'price_start', 'price_end', 'price_count']

# Merge
coverage = funding_coverage.merge(price_coverage, on='coin', how='left')

# Calculate gaps
coverage['has_price'] = coverage['price_start'].notna()
coverage['gap_days'] = (coverage['price_start'] - coverage['funding_start']).dt.days
coverage['gap_days'] = coverage['gap_days'].fillna(
    (pd.Timestamp.now(tz='UTC') - coverage['funding_start']).dt.days
)

print("\n" + "=" * 80)
print("COVERAGE SUMMARY")
print("=" * 80)

print(f"\nCoins with funding data: {len(coverage)}")
print(f"Coins with price data: {coverage['has_price'].sum()}")
print(f"Coins without price data: {(~coverage['has_price']).sum()}")

# Show coins with biggest gaps (have both funding and price, but gap exists)
print("\n" + "-" * 80)
print("COINS WITH PRICE DATA - GAP ANALYSIS")
print("-" * 80)

coins_with_price = coverage[coverage['has_price']].copy()
coins_with_price = coins_with_price.sort_values('gap_days', ascending=False)

print(f"\n{'Coin':<10} {'Funding Start':<22} {'Price Start':<22} {'Gap (days)':>12}")
print("-" * 70)
for _, row in coins_with_price.head(30).iterrows():
    print(f"{row['coin']:<10} {str(row['funding_start'])[:19]:<22} {str(row['price_start'])[:19]:<22} {row['gap_days']:>12.0f}")

# Summary statistics
print(f"\nTotal gap to fill: {coins_with_price['gap_days'].sum():,.0f} coin-days")
print(f"Average gap per coin: {coins_with_price['gap_days'].mean():.0f} days")
print(f"Max gap: {coins_with_price['gap_days'].max():.0f} days")

# Coins without any price data
print("\n" + "-" * 80)
print("COINS WITHOUT PRICE DATA (need full fetch)")
print("-" * 80)

coins_no_price = coverage[~coverage['has_price']].copy()
print(f"\n{len(coins_no_price)} coins have funding but no price data:")
for _, row in coins_no_price.iterrows():
    days = (pd.Timestamp.now(tz='UTC') - row['funding_start']).days
    print(f"  {row['coin']}: funding from {str(row['funding_start'])[:10]} ({days} days)")

# Estimate data volume
print("\n" + "=" * 80)
print("DATA VOLUME ESTIMATE")
print("=" * 80)

total_gap_days = coins_with_price['gap_days'].sum()
coins_no_price_days = coins_no_price.apply(
    lambda r: (pd.Timestamp.now(tz='UTC') - r['funding_start']).days, axis=1
).sum() if len(coins_no_price) > 0 else 0

total_days_needed = total_gap_days + coins_no_price_days
total_hours_needed = total_days_needed * 24

print(f"\nTotal data needed:")
print(f"  - Gap filling for existing coins: {total_gap_days:,.0f} coin-days")
print(f"  - Full fetch for missing coins: {coins_no_price_days:,.0f} coin-days")
print(f"  - Total: {total_days_needed:,.0f} coin-days = {total_hours_needed:,.0f} hourly candles")

# Common coins that likely exist on Binance
major_coins = ['BTC', 'ETH', 'SOL', 'AVAX', 'DOGE', 'XRP', 'ADA', 'DOT', 'LINK', 
               'MATIC', 'UNI', 'AAVE', 'SNX', 'CRV', 'COMP', 'MKR', 'SUSHI',
               'YFI', 'LTC', 'BCH', 'ETC', 'XLM', 'ATOM', 'FIL', 'NEAR',
               'APT', 'ARB', 'OP', 'SUI', 'SEI', 'TIA', 'INJ', 'JUP']

binance_likely = coverage[coverage['coin'].isin(major_coins)]
print(f"\nMajor coins (likely on Binance): {len(binance_likely)}")
print(f"  Total gap for major coins: {binance_likely['gap_days'].sum():,.0f} days")

# Show top coins by funding activity
print("\n" + "=" * 80)
print("TOP COINS BY FUNDING ACTIVITY (priority for backfill)")
print("=" * 80)

# Count extreme funding events per coin
extreme_funding = funding[funding['funding_rate'].abs() > 0.001].copy()  # > 0.10%
extreme_counts = extreme_funding.groupby('coin').size().reset_index(name='extreme_events')

coverage_with_extreme = coverage.merge(extreme_counts, on='coin', how='left')
coverage_with_extreme['extreme_events'] = coverage_with_extreme['extreme_events'].fillna(0)

top_by_activity = coverage_with_extreme.nlargest(30, 'extreme_events')

print(f"\n{'Coin':<10} {'Extreme Events':>15} {'Gap Days':>12} {'Has Price':>12}")
print("-" * 55)
for _, row in top_by_activity.iterrows():
    has_price = 'Yes' if row['has_price'] else 'No'
    print(f"{row['coin']:<10} {row['extreme_events']:>15.0f} {row['gap_days']:>12.0f} {has_price:>12}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("""
YES, fetching price data from Binance is recommended!

Benefits:
1. Extend backtest from 8 months to 2+ years
2. More robust strategy validation
3. See performance across different market conditions (bull/bear)

Approach:
1. Priority 1: Major coins with high extreme funding events
2. Priority 2: Coins already listed on both exchanges
3. Skip: Very new coins or Hyperliquid-only coins

Considerations:
- Binance prices may differ slightly from Hyperliquid
- Use USDT perpetual futures prices (most liquid)
- Some coins may not exist on Binance (newer/smaller ones)
""")

# Save coverage analysis
coverage_with_extreme.to_csv('data_coverage_analysis.csv', index=False)
print("\nCoverage analysis saved to data_coverage_analysis.csv")
