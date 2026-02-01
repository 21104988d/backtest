"""
ALTERNATIVE FUNDING STRATEGIES ANALYSIS

Exploring different approaches beyond single-asset hedging
"""
import pandas as pd
import numpy as np

print("Loading data...")
funding = pd.read_csv('funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)
funding['hour'] = funding['datetime'].dt.floor('h').dt.tz_localize(None)

price = pd.read_csv('price_history.csv')
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
price['hour'] = price['timestamp'].dt.floor('h').dt.tz_localize(None)

THRESHOLD = 0.001  # 0.10%
TAKER_FEE = 0.00045  # 0.045%

print("\n" + "="*80)
print("STRATEGY 1: MULTI-PERIOD FUNDING ACCUMULATION")
print("="*80)
print("\nIdea: Hold position through multiple funding periods instead of 1 hour")
print("This reduces fee impact per funding earned")

# Simulate holding for different periods
print("\n### Fee Impact Analysis ###")
print(f"Round-trip fee (2 legs): {4*TAKER_FEE*100:.3f}%")
print(f"Avg funding per hour (extreme events): ~0.20%")
print("\nBreak-even analysis (fees vs funding accumulated):")
print(f"{'Hours Held':<15} {'Funding Earned':<20} {'Fees':<15} {'Net':<15}")
print("-"*70)

for hours in [1, 2, 4, 8, 12, 24]:
    funding_earned = 0.002 * hours  # ~0.20% per hour
    fees = 4 * TAKER_FEE  # Entry + exit, 2 legs
    net = funding_earned - fees
    print(f"{hours:<15} {funding_earned*100:.3f}%{'':<14} {fees*100:.3f}%{'':<9} {net*100:.3f}%")

print("\n" + "="*80)
print("STRATEGY 2: FUNDING PERSISTENCE ANALYSIS")
print("="*80)
print("\nQuestion: How long does extreme funding typically persist?")
print("If it persists, multi-period accumulation makes sense.")

# Analyze funding persistence
extreme_events = funding[funding['funding_rate'].abs() > THRESHOLD].copy()
extreme_events = extreme_events.sort_values(['coin', 'hour'])

# For each extreme event, count consecutive hours of extreme funding
def count_consecutive_extreme(group):
    group = group.sort_values('hour')
    results = []
    
    for i, (_, row) in enumerate(group.iterrows()):
        # Count consecutive hours with same sign extreme funding
        count = 1
        sign = np.sign(row['funding_rate'])
        
        # Look forward
        for j in range(i+1, len(group)):
            next_row = group.iloc[j]
            hour_diff = (next_row['hour'] - row['hour']).total_seconds() / 3600
            
            if hour_diff == count and np.sign(next_row['funding_rate']) == sign and abs(next_row['funding_rate']) > THRESHOLD:
                count += 1
            else:
                break
        
        results.append({
            'coin': row['coin'],
            'hour': row['hour'],
            'funding_rate': row['funding_rate'],
            'consecutive_hours': count
        })
    
    return pd.DataFrame(results)

print("\nAnalyzing funding persistence (this may take a moment)...")

# Sample to speed up
sample_coins = extreme_events['coin'].value_counts().head(20).index.tolist()
sample_events = extreme_events[extreme_events['coin'].isin(sample_coins)]

persistence_results = []
for coin in sample_coins:
    coin_data = sample_events[sample_events['coin'] == coin]
    result = count_consecutive_extreme(coin_data)
    persistence_results.append(result)

persistence_df = pd.concat(persistence_results, ignore_index=True)

print("\n### Consecutive Extreme Funding Hours Distribution ###")
print(persistence_df['consecutive_hours'].describe())

print("\n### Persistence by bucket ###")
for hours in [1, 2, 3, 4, 5, '6+']:
    if hours == '6+':
        count = (persistence_df['consecutive_hours'] >= 6).sum()
    else:
        count = (persistence_df['consecutive_hours'] == hours).sum()
    pct = count / len(persistence_df) * 100
    print(f"{hours} hours: {count} ({pct:.1f}%)")

print("\n" + "="*80)
print("STRATEGY 3: FUNDING + MOMENTUM (Directional)")  
print("="*80)
print("\nIdea: Extreme funding signals crowded positioning")
print("Very negative funding → heavy shorts → potential short squeeze")
print("Very positive funding → heavy longs → potential long liquidation")

# Merge funding with price returns
price_returns = price.groupby(['coin', 'hour'])['price'].first().reset_index()
price_returns = price_returns.sort_values(['coin', 'hour'])
price_returns['return_1h'] = price_returns.groupby('coin')['price'].pct_change()
price_returns['return_4h'] = price_returns.groupby('coin')['price'].pct_change(4)
price_returns['return_8h'] = price_returns.groupby('coin')['price'].pct_change(8)

# Merge
analysis = funding.merge(price_returns[['coin', 'hour', 'return_1h', 'return_4h', 'return_8h']], 
                         on=['coin', 'hour'], how='inner')

# Filter extreme funding
extreme = analysis[analysis['funding_rate'].abs() > THRESHOLD].copy()
extreme['direction'] = np.where(extreme['funding_rate'] < 0, 'long', 'short')
extreme['position_return_1h'] = np.where(extreme['direction'] == 'long', 
                                          extreme['return_1h'], 
                                          -extreme['return_1h'])
extreme['position_return_4h'] = np.where(extreme['direction'] == 'long', 
                                          extreme['return_4h'], 
                                          -extreme['return_4h'])
extreme['position_return_8h'] = np.where(extreme['direction'] == 'long', 
                                          extreme['return_8h'], 
                                          -extreme['return_8h'])

print("\n### Directional Strategy Performance (NO HEDGE) ###")
print("Position: Long when funding < -0.10%, Short when funding > +0.10%")
print("\n(This includes price movement, NOT just funding)")

for period, col in [('1h', 'position_return_1h'), ('4h', 'position_return_4h'), ('8h', 'position_return_8h')]:
    valid = extreme[col].dropna()
    if len(valid) > 0:
        avg = valid.mean()
        std = valid.std()
        sharpe = avg / std if std > 0 else 0
        win_rate = (valid > 0).mean()
        print(f"\n{period} Holding Period:")
        print(f"  Avg Return: {avg*100:.4f}%")
        print(f"  Std Dev: {std*100:.2f}%")
        print(f"  Sharpe: {sharpe:.4f}")
        print(f"  Win Rate: {win_rate*100:.1f}%")
        print(f"  N trades: {len(valid)}")

print("\n" + "="*80)
print("STRATEGY 4: FUNDING RATE BUCKETS (More Extreme = Better?)")
print("="*80)
print("\nAnalyze if MORE extreme funding gives better returns")

# Create funding rate buckets
extreme['fr_bucket'] = pd.cut(extreme['funding_rate'].abs(), 
                               bins=[0.001, 0.002, 0.003, 0.005, 0.01, 1],
                               labels=['0.10-0.20%', '0.20-0.30%', '0.30-0.50%', '0.50-1.0%', '>1.0%'])

print("\n### Performance by Funding Rate Magnitude ###")
print(f"{'Bucket':<15} {'N':<10} {'Avg 1h Ret':<15} {'Win Rate':<12} {'Avg FR':<12}")
print("-"*70)

for bucket in ['0.10-0.20%', '0.20-0.30%', '0.30-0.50%', '0.50-1.0%', '>1.0%']:
    bucket_data = extreme[extreme['fr_bucket'] == bucket]
    if len(bucket_data) > 10:
        valid = bucket_data['position_return_1h'].dropna()
        avg_ret = valid.mean() * 100
        win_rate = (valid > 0).mean() * 100
        avg_fr = bucket_data['funding_rate'].abs().mean() * 100
        print(f"{bucket:<15} {len(valid):<10} {avg_ret:>12.4f}%  {win_rate:>10.1f}%  {avg_fr:>10.4f}%")

print("\n" + "="*80)
print("STRATEGY 5: TIME-OF-DAY ANALYSIS")
print("="*80)
print("\nQuestion: Is there a better time to trade funding?")

extreme['hour_of_day'] = extreme['hour'].dt.hour

print("\n### Extreme Funding by Hour of Day ###")
hourly_stats = extreme.groupby('hour_of_day').agg({
    'funding_rate': ['count', lambda x: x.abs().mean()],
    'position_return_1h': ['mean', lambda x: (x > 0).mean()]
}).round(4)
hourly_stats.columns = ['count', 'avg_abs_fr', 'avg_ret', 'win_rate']
print(hourly_stats.to_string())

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("""
Based on this analysis, here are the most promising directions:

1. CROSS-EXCHANGE ARBITRAGE (Not analyzed here - needs multi-exchange data)
   - Same coin, different exchanges = perfect hedge
   - Collect funding differential
   - This is the "cleanest" funding arbitrage

2. MULTI-PERIOD ACCUMULATION
   - Hold positions 4-8+ hours instead of 1 hour
   - Only works if funding persists (check persistence data above)
   - Reduces fee impact significantly

3. DIRECTIONAL MOMENTUM (Higher risk, higher reward)
   - Use extreme funding as a signal, not something to hedge
   - Very negative funding → potential short squeeze
   - Combine with other momentum indicators

4. HIGHER THRESHOLD FILTER
   - Only trade when funding is MORE extreme (>0.30% or >0.50%)
   - Fewer trades but potentially better risk/reward

5. SELECTIVE TIMING
   - Certain hours may be more profitable
   - Combine with volume/volatility filters
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
