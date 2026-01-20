import requests

def get_top_losers(limit=20):
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/ticker/24hr"
    
    try:
        response = requests.get(base_url + endpoint)
        data = response.json()
        
        # Check if list (API returns list of objects)
        if not isinstance(data, list):
            print("Unexpected response format from Binance API.")
            return

        # Filtering:
        # 1. We generally only care about USDT pairs for general market sentiment.
        # 2. We want to sort by priceChangePercent.
        
        # Filter for pairs ending in USDT
        usdt_pairs = [item for item in data if item['symbol'].endswith('USDT')]
        
        # Convert numerical columns for sorting/display
        for item in usdt_pairs:
            item['priceChangePercent'] = float(item['priceChangePercent'])
            item['lastPrice'] = float(item['lastPrice'])
            item['quoteVolume'] = float(item['quoteVolume']) # Volume in USDT
            
        # Sort by priceChangePercent ascending (Top Losers)
        sorted_pairs = sorted(usdt_pairs, key=lambda x: x['priceChangePercent'])
        
        print(f"\n=== Binance Top {limit} Losers / 跌幅榜 (24h Change) ===")
        print(f"{'Symbol':<15} {'Price':<15} {'24h Change %':<15} {'Volume (USDT)':<20}")
        print("-" * 75)
        
        for row in sorted_pairs[:limit]:
            symbol = row['symbol']
            price = row['lastPrice']
            change = row['priceChangePercent']
            volume = row['quoteVolume']
            
            # Format volume
            if volume > 1_000_000:
                vol_str = f"{volume/1_000_000:.2f}M"
            elif volume > 1_000:
                vol_str = f"{volume/1_000:.2f}K"
            else:
                vol_str = f"{volume:.2f}"
            
            print(f"{symbol:<15} {price:<15.4f} {change:<15.2f}% {vol_str:<20}")

    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    get_top_losers()
