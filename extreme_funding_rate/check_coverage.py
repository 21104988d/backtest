import pandas as pd

# Load updated price data
price = pd.read_csv('price_history.csv')
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)

print('=' * 60)
print('UPDATED PRICE DATA COVERAGE')
print('=' * 60)

print(f'Total records: {len(price):,}')
print(f'Unique coins: {price["coin"].nunique()}')
print(f'Date range: {price["timestamp"].min()} to {price["timestamp"].max()}')
print(f'Days covered: {(price["timestamp"].max() - price["timestamp"].min()).days}')

# Coverage by coin
coverage = price.groupby('coin').agg({
    'timestamp': ['min', 'max', 'count']
}).reset_index()
coverage.columns = ['coin', 'start', 'end', 'records']
coverage['days'] = (coverage['end'] - coverage['start']).dt.days

print(f'\nAverage records per coin: {coverage["records"].mean():,.0f}')
print(f'Average days per coin: {coverage["days"].mean():.0f}')

# Show major coins
print('\nMajor coins coverage:')
for coin in ['BTC', 'ETH', 'SOL', 'ARB', 'OP', 'DOGE', 'LINK', 'WIF']:
    if coin in coverage['coin'].values:
        row = coverage[coverage['coin'] == coin].iloc[0]
        print(f'  {coin:<8} {str(row["start"])[:10]} to {str(row["end"])[:10]} ({row["days"]} days, {row["records"]:,} records)')

# Compare with funding data
print('\n' + '=' * 60)
print('COMPARISON WITH FUNDING DATA')
print('=' * 60)

funding = pd.read_csv('funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)

print(f'\nFunding records: {len(funding):,}')
print(f'Funding coins: {funding["coin"].nunique()}')
print(f'Funding range: {funding["datetime"].min()} to {funding["datetime"].max()}')

# Overlap
common_coins = set(price['coin'].unique()) & set(funding['coin'].unique())
print(f'\nCoins with both price AND funding: {len(common_coins)}')

# Calculate backtest-able range
print('\n' + '=' * 60)
print('BACKTEST-ABLE DATA')
print('=' * 60)

# Normalize funding timestamps to hour (remove milliseconds)
funding['hour'] = funding['datetime'].dt.floor('h')
price['hour'] = price['timestamp'].dt.floor('h')

merged = pd.merge(
    price[['hour', 'coin', 'price']],
    funding[['hour', 'coin', 'funding_rate']],
    on=['hour', 'coin'],
    how='inner'
)

print(f'\nMatched records (price + funding): {len(merged):,}')
print(f'Coins in matched data: {merged["coin"].nunique()}')
print(f'Date range: {merged["hour"].min()} to {merged["hour"].max()}')
print(f'Days covered: {(merged["hour"].max() - merged["hour"].min()).days}')
