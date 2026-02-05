import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
N = 3  # Number of top/bottom assets to track

# Read the price history CSV
df = pd.read_csv('price_history.csv')

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract date (without time)
df['date'] = df['timestamp'].dt.date

# Sort by coin and timestamp
df = df.sort_values(['coin', 'timestamp'])

# Get first and last price of each day for each coin to calculate daily returns
daily_prices = df.groupby(['date', 'coin'])['price'].agg(['first', 'last']).reset_index()
daily_prices['daily_return'] = (daily_prices['last'] / daily_prices['first'] - 1) * 100

# Drop any NaN or infinite returns
daily_prices = daily_prices.replace([np.inf, -np.inf], np.nan).dropna(subset=['daily_return'])

# For each day, get top N highest and bottom N lowest returning assets
results = []

for date in sorted(daily_prices['date'].unique()):
    day_data = daily_prices[daily_prices['date'] == date].copy()
    
    # Get top N and bottom N
    top_n = day_data.nlargest(N, 'daily_return')[['coin', 'daily_return']].values.tolist()
    bottom_n = day_data.nsmallest(N, 'daily_return')[['coin', 'daily_return']].values.tolist()
    
    row = {'date': date}
    
    # Add top N
    for i, (coin, ret) in enumerate(top_n, 1):
        row[f'top_{i}_coin'] = coin
        row[f'top_{i}_return'] = ret
    
    # Add bottom N
    for i, (coin, ret) in enumerate(bottom_n, 1):
        row[f'bottom_{i}_coin'] = coin
        row[f'bottom_{i}_return'] = ret
    
    results.append(row)

# Create DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv('daily_top_bottom_returns.csv', index=False)

print(f"Daily Top {N} and Bottom {N} Returning Assets")
print("=" * 100)
print(f"\nTotal days analyzed: {len(results_df)}")
print(f"Date range: {results_df['date'].min()} to {results_df['date'].max()}")
print(f"Total unique coins: {daily_prices['coin'].nunique()}")

print("\n" + "=" * 100)
print(f"\nAll Days - Top {N} Gainers and Bottom {N} Losers:")
print("-" * 100)

for _, row in results_df.iterrows():
    print(f"\n{row['date']}:")
    print(f"  TOP {N} GAINERS:")
    for i in range(1, N + 1):
        coin = row.get(f'top_{i}_coin', 'N/A')
        ret = row.get(f'top_{i}_return', 0)
        coin_str = str(coin) if pd.notna(coin) else 'N/A'
        ret_val = float(ret) if pd.notna(ret) else 0.0
        print(f"    {i}. {coin_str:12s} {ret_val:+8.2f}%")
    print(f"  BOTTOM {N} LOSERS:")
    for i in range(1, N + 1):
        coin = row.get(f'bottom_{i}_coin', 'N/A')
        ret = row.get(f'bottom_{i}_return', 0)
        coin_str = str(coin) if pd.notna(coin) else 'N/A'
        ret_val = float(ret) if pd.notna(ret) else 0.0
        print(f"    {i}. {coin_str:12s} {ret_val:+8.2f}%")

print("\n" + "=" * 100)
print(f"\nResults saved to: daily_top_bottom_returns.csv")
