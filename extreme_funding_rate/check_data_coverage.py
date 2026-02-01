"""
Check why we only have 713 extreme events when we have more historical data
"""
import pandas as pd

# Check funding history
print("=== FUNDING HISTORY ===")
funding = pd.read_csv('funding_history.csv')
print(f"Total rows: {len(funding):,}")
print(f"Unique coins: {funding['coin'].nunique()}")
print(f"Date range: {funding['datetime'].min()} to {funding['datetime'].max()}")

# Check price history
print("\n=== PRICE HISTORY ===")
price = pd.read_csv('price_history.csv')
print(f"Total rows: {len(price):,}")
print(f"Unique coins: {price['coin'].nunique()}")
print(f"Date range: {price['timestamp'].min()} to {price['timestamp'].max()}")

# Check extreme events file
print("\n=== EXTREME EVENTS FILE ===")
events = pd.read_csv('extreme_funding_1h_events.csv')
print(f"Total rows: {len(events):,}")
print(f"Date range: {events['hour'].min()} to {events['hour'].max()}")

# Check threshold - the extreme_funding_1h_analysis.py uses different thresholds
print("\n=== EXTREME FUNDING COUNTS ===")
threshold = 0.001  # 0.10%
extreme_in_file = events[(events['funding_rate'].abs() > threshold)]
print(f"Events in file with |FR| > {threshold*100:.2f}%: {len(extreme_in_file)}")

# Check from raw funding data
funding['fr_abs'] = funding['funding_rate'].abs()
extreme_from_raw = funding[funding['fr_abs'] > threshold]
print(f"From RAW funding data, events with |FR| > {threshold*100:.2f}%: {len(extreme_from_raw)}")

# The issue is the extreme_funding_1h_events.csv was created with different logic
# Let's check what the extreme_funding_1h_analysis.py does
print("\n=== CHECKING EXTREME EVENTS FILE GENERATION ===")
print("The extreme_funding_1h_events.csv file contains:")
print(events.head(10).to_string())
print(f"\nUnique types: {events['type'].unique()}")
print(f"Type counts:\n{events['type'].value_counts()}")

# The file has 'negative' and 'positive' types
# negative = funding_rate < -threshold (we long to receive)
# positive = funding_rate > +threshold (we short to receive)
print("\n=== FILTERING FOR EXTREME EVENTS ===")
negative = events[(events['type'] == 'negative') & (events['funding_rate'] < -threshold)]
positive = events[(events['type'] == 'positive') & (events['funding_rate'] > threshold)]
print(f"Negative extreme (FR < -{threshold*100:.2f}%): {len(negative)}")
print(f"Positive extreme (FR > +{threshold*100:.2f}%): {len(positive)}")
print(f"Total: {len(negative) + len(positive)}")

# This matches the 713 - so the issue is in how extreme_funding_1h_events.csv was generated
# Let's regenerate it from raw funding data to get ALL extreme events
print("\n" + "="*60)
print("REGENERATING EXTREME EVENTS FROM RAW FUNDING DATA")
print("="*60)

# Parse datetime
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)
funding['hour'] = funding['datetime'].dt.floor('h').dt.tz_localize(None)

# Find extreme funding events
extreme_negative = funding[funding['funding_rate'] < -threshold].copy()
extreme_negative['type'] = 'negative'
extreme_negative['position'] = 'long'

extreme_positive = funding[funding['funding_rate'] > threshold].copy()
extreme_positive['type'] = 'positive'  
extreme_positive['position'] = 'short'

all_extreme = pd.concat([extreme_negative, extreme_positive])
print(f"\nTotal extreme funding events from raw data: {len(all_extreme)}")
print(f"  Negative (FR < -{threshold*100:.2f}%): {len(extreme_negative)}")
print(f"  Positive (FR > +{threshold*100:.2f}%): {len(extreme_positive)}")
print(f"  Unique coins: {all_extreme['coin'].nunique()}")
print(f"  Date range: {all_extreme['hour'].min()} to {all_extreme['hour'].max()}")

# Check overlap with price data
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
price['hour'] = price['timestamp'].dt.floor('h').dt.tz_localize(None)
price_hours = set(price['hour'].unique())
price_coins = set(price['coin'].unique())

# Filter extreme events to those with price data
events_with_price = all_extreme[
    (all_extreme['hour'].isin(price_hours)) & 
    (all_extreme['coin'].isin(price_coins))
]
print(f"\nExtreme events with matching price data: {len(events_with_price)}")
