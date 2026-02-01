import pandas as pd

# Check the date range of events
events = pd.read_csv('/Users/leeisaackaiyui/Desktop/backtest/extreme_funding_rate/extreme_funding_1h_events.csv')
events['hour'] = pd.to_datetime(events['hour'])

# Check price data range
price = pd.read_csv('/Users/leeisaackaiyui/Desktop/backtest/extreme_funding_rate/price_history.csv')
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True).dt.tz_localize(None)

print('=== EVENTS DATE RANGE ===')
print(f'Start: {events["hour"].min()}')
print(f'End: {events["hour"].max()}')
print(f'Total events: {len(events)}')

print('\n=== PRICE DATA DATE RANGE ===')
print(f'Start: {price["timestamp"].min()}')
print(f'End: {price["timestamp"].max()}')

# Filter extreme events
THRESHOLD = 0.001
extreme = events[
    ((events['type'] == 'negative') & (events['funding_rate'] < -THRESHOLD)) |
    ((events['type'] == 'positive') & (events['funding_rate'] > THRESHOLD))
]
print(f'\nExtreme events (|FR|>0.10%): {len(extreme)}')
print(f'Extreme events date range: {extreme["hour"].min()} to {extreme["hour"].max()}')

# Check how many events are within price data range
price_start = price['timestamp'].min()
price_end = price['timestamp'].max()

within_range = extreme[(extreme['hour'] >= price_start) & (extreme['hour'] <= price_end)]
print(f'\nEvents within price data range: {len(within_range)}')

# Lookback period check (168 hours = 7 days)
lookback = 168
early_cutoff = price_start + pd.Timedelta(hours=lookback)
print(f'\n=== LOOKBACK PERIOD ===')
print(f'Lookback period: {lookback} hours (7 days)')
print(f'Price data starts: {price_start}')
print(f'First valid hour for lookback: {early_cutoff}')

after_lookback = extreme[extreme['hour'] >= early_cutoff]
print(f'Events after lookback cutoff: {len(after_lookback)}')

# Why are some invalid?
print('\n=== REASONS FOR INVALID TRADES ===')
print('1. Event hour is BEFORE price data starts')
print('2. Event hour is in first 168 hours (need lookback for correlation)')
print('3. Missing price data for the specific coin at that hour')

# Count by reason
before_price = len(extreme[extreme['hour'] < price_start])
in_lookback = len(extreme[(extreme['hour'] >= price_start) & (extreme['hour'] < early_cutoff)])
print(f'\nEvents before price data: {before_price}')
print(f'Events in lookback period (first 7 days): {in_lookback}')
print(f'Events with sufficient history data: {len(extreme) - before_price - in_lookback}')
