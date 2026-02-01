import pandas as pd
from collections import defaultdict
from datetime import timedelta

# Load data
funding = pd.read_csv('/Users/leeisaackaiyui/Desktop/backtest/extreme_funding_rate/funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)

price = pd.read_csv('/Users/leeisaackaiyui/Desktop/backtest/extreme_funding_rate/price_history.csv')
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)

print('=' * 60)
print('DATA DATE RANGES')
print('=' * 60)

print(f'\nFunding data:')
print(f'  Start: {funding["datetime"].min()}')
print(f'  End:   {funding["datetime"].max()}')
funding_days = (funding["datetime"].max() - funding["datetime"].min()).days
print(f'  Duration: {funding_days} days')

print(f'\nPrice data:')
print(f'  Start: {price["timestamp"].min()}')
print(f'  End:   {price["timestamp"].max()}')
price_days = (price["timestamp"].max() - price["timestamp"].min()).days
print(f'  Duration: {price_days} days')

# Overlap period
overlap_start = max(funding['datetime'].min(), price['timestamp'].min())
overlap_end = min(funding['datetime'].max(), price['timestamp'].max())
backtest_days = (overlap_end - overlap_start).days

print(f'\n*** BACKTEST PERIOD (where we have BOTH funding + price): ***')
print(f'  Start: {overlap_start}')
print(f'  End:   {overlap_end}')
print(f'  Duration: {backtest_days} days = {backtest_days/30:.1f} months = {backtest_days/365:.2f} years')

# Load trades
trades = pd.read_csv('/Users/leeisaackaiyui/Desktop/backtest/extreme_funding_rate/short_only_full_trades.csv')
trades['entry_hour'] = pd.to_datetime(trades['entry_hour'])
trades['exit_hour'] = pd.to_datetime(trades['exit_hour'])

print(f'\n' + '=' * 60)
print('WHAT IS MAX CONCURRENT POSITIONS?')
print('=' * 60)

# Calculate concurrent positions over time
position_count = defaultdict(int)
for _, trade in trades.iterrows():
    current = trade['entry_hour']
    while current <= trade['exit_hour']:
        position_count[current] += 1
        current += timedelta(hours=1)

max_time = max(position_count.items(), key=lambda x: x[1])
print(f'\nMax concurrent positions = {max_time[1]}')
print(f'This occurred at: {max_time[0]}')

# Show the positions at that time
at_max = trades[(trades['entry_hour'] <= max_time[0]) & (trades['exit_hour'] >= max_time[0])]
print(f'\nMeaning: At {max_time[0]}, we had {len(at_max)} SHORT positions open simultaneously:')
for _, t in at_max.iterrows():
    print(f'  - {t["coin"]}: entered {t["entry_hour"].strftime("%Y-%m-%d %H:%M")}, FR={t["entry_fr"]*100:.2f}%')

print(f'\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'''
BACKTEST PERIOD: ~{backtest_days} days = ~{backtest_days/30:.0f} months = ~{backtest_days/365:.1f} years

MAX CONCURRENT POSITIONS = 10 means:
  At the PEAK moment during the backtest, we had 10 different 
  coins with SHORT positions open at the same time.
  
  This is useful for:
  - Capital allocation (need to split capital across positions)
  - Risk management (more positions = more diversified)
  - Margin requirements
''')
