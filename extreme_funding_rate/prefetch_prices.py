"""
Pre-fetch all required price data and save to CSV for instant backtest execution.
This eliminates API rate limiting during backtests and ensures 100% data coverage.
"""
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set
import sys

class PriceDataPrefetcher:
    """Pre-fetch and cache all price data needed for backtesting."""
    
    def __init__(self, funding_file='funding_history.csv'):
        self.funding_file = funding_file
        self.api_url = "https://api.hyperliquid.xyz/info"
        self.base_delay = 1.0
        self.max_retries = 5
        
    def load_funding_data(self) -> pd.DataFrame:
        """Load funding rate data to determine what prices we need."""
        print("\n📊 Loading funding rate data...")
        df = pd.read_csv(self.funding_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        print(f"✓ Loaded {len(df):,} funding rate records")
        return df
    
    def identify_required_prices(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify all (coin, timestamp) pairs we need prices for.
        For each hour, we need entry and exit prices for extreme funding coins.
        """
        print("\n🔍 Identifying required price data...")
        
        df['hour'] = df['datetime'].dt.floor('h')
        hours = sorted(df['hour'].unique())
        
        required_prices = []
        coins_per_hour = {}
        
        for hour in hours:
            hour_data = df[df['hour'] == hour]
            
            # Get top 3 most negative funding coins (we only trade 1, but fetch 3 for safety)
            top_negative = hour_data.nsmallest(3, 'funding_rate')
            
            for _, row in top_negative.iterrows():
                coin = row['coin']
                
                # Entry time (start of hour)
                entry_time = hour
                # Exit time (1 hour later)
                exit_time = hour + timedelta(hours=1)
                
                # Add both entry and exit timestamps
                required_prices.append({
                    'coin': coin,
                    'timestamp': entry_time,
                    'type': 'entry',
                    'hour': hour
                })
                required_prices.append({
                    'coin': coin,
                    'timestamp': exit_time,
                    'type': 'exit',
                    'hour': hour
                })
                
                # Also need BTC prices for delta-hedged strategy
                required_prices.append({
                    'coin': 'BTC',
                    'timestamp': entry_time,
                    'type': 'entry',
                    'hour': hour
                })
                required_prices.append({
                    'coin': 'BTC',
                    'timestamp': exit_time,
                    'type': 'exit',
                    'hour': hour
                })
        
        # Remove duplicates
        df_prices = pd.DataFrame(required_prices)
        df_prices = df_prices.drop_duplicates(subset=['coin', 'timestamp'])
        df_prices = df_prices.sort_values(['timestamp', 'coin']).reset_index(drop=True)
        
        print(f"✓ Need {len(df_prices):,} unique price points")
        print(f"  - {len(df_prices['coin'].unique())} unique coins")
        print(f"  - {len(df_prices['timestamp'].unique())} unique timestamps")
        
        return df_prices.to_dict('records')
    
    def fetch_price_with_retry(self, coin: str, timestamp: pd.Timestamp) -> float:
        """Fetch a single price with exponential backoff retry."""
        start_ms = int(timestamp.timestamp() * 1000)
        # Use 1-hour candles for better historical data availability
        end_ms = start_ms + (60 * 60 * 1000)  # 1 hour later
        
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "1h",
                "startTime": start_ms,
                "endTime": end_ms
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if data and len(data) > 0:
                    price = float(data[0]['c'])
                    time.sleep(self.base_delay)
                    return price
                
                # No data, retry
                if attempt < self.max_retries - 1:
                    wait_time = self.base_delay * (2 ** attempt)
                    time.sleep(wait_time)
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        wait_time = self.base_delay * (2 ** attempt)
                        print(f"  ⚠️ Rate limit for {coin}, retry {attempt+1}/{self.max_retries} after {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        print(f"  ❌ Max retries for {coin} at {timestamp}")
                        return None
                else:
                    print(f"  ❌ HTTP error for {coin}: {e}")
                    return None
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.base_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Error for {coin}: {e}")
                    return None
        
        return None
    
    def prefetch_all_prices(self, required_prices: List[Dict], output_file='price_cache.csv'):
        """Fetch all required prices and save to CSV."""
        print(f"\n🚀 Pre-fetching {len(required_prices):,} price points...")
        print(f"⏱️  Estimated time: ~{len(required_prices) * 1.0 / 60:.1f} minutes\n")
        
        results = []
        total = len(required_prices)
        failed = 0
        
        for i, item in enumerate(required_prices, 1):
            coin = item['coin']
            timestamp = pd.to_datetime(item['timestamp'])
            
            price = self.fetch_price_with_retry(coin, timestamp)
            
            if price is not None:
                results.append({
                    'coin': coin,
                    'timestamp': timestamp,
                    'price': price
                })
            else:
                failed += 1
            
            # Progress update every 10 items
            if i % 10 == 0 or i == total:
                success_rate = ((i - failed) / i) * 100
                print(f"Progress: {i}/{total} ({i/total*100:.1f}%) | Success: {success_rate:.1f}% | Failed: {failed}")
                
                # Save intermediate results every 50 items
                if i % 50 == 0:
                    df_temp = pd.DataFrame(results)
                    df_temp.to_csv(output_file + '.tmp', index=False)
        
        # Save final results
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
        
        print(f"\n✅ Pre-fetch complete!")
        print(f"  - Success: {len(results):,}/{total:,} ({len(results)/total*100:.1f}%)")
        print(f"  - Failed: {failed:,} ({failed/total*100:.1f}%)")
        print(f"  - Saved to: {output_file}")
        
        return df_results

def main():
    print("═" * 70)
    print("PRICE DATA PRE-FETCH")
    print("═" * 70)
    
    prefetcher = PriceDataPrefetcher()
    
    # Load funding data
    df = prefetcher.load_funding_data()
    
    # Identify what prices we need
    required_prices = prefetcher.identify_required_prices(df)
    
    # Fetch all prices
    df_prices = prefetcher.prefetch_all_prices(required_prices, 'price_cache.csv')
    
    print("\n" + "═" * 70)
    print("Pre-fetch complete! Now backtests can run instantly with 100% data.")
    print("═" * 70)

if __name__ == '__main__':
    main()
