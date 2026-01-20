import requests
import pandas as pd
from datetime import datetime

def fetch_hyperliquid_funding():
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}
    payload = {"type": "metaAndAssetCtxs"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # data[0] is 'meta' (universe details)
        # data[1] is 'assetCtxs' (current market state including funding)
        meta = data[0]
        asset_ctxs = data[1]
        
        universe = meta['universe']
        
        results = []
        
        for i, asset_info in enumerate(universe):
            coin_name = asset_info['name']
            ctx = asset_ctxs[i]
            
            # Extract price and funding
            # 'markPx' is the current mark price
            # 'funding' is the funding rate (hourly on Hyperliquid usually)
            mark_price = float(ctx['markPx'])
            funding_rate_raw = float(ctx['funding'])
            
            # Hyperliquid funding is essentially 1h rate.
            # To make it comparable to 8h rates (like Binance/Bybit), strictly speaking we simulate 8h.
            # But usually displaying Hourly and Annualized is best.
            funding_rate_1h = funding_rate_raw
            funding_rate_apr = funding_rate_raw * 24 * 365 * 100
            
            results.append({
                "Coin": coin_name,
                "Price": mark_price,
                "Funding Rate (1h)": funding_rate_1h,
                "Funding Rate (Apr %)": funding_rate_apr
            })
            
        df = pd.DataFrame(results)
        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    print(f"Fetching Hyperliquid funding rates at {datetime.now()}...")
    df = fetch_hyperliquid_funding()
    
    if not df.empty:
        # Sort by Funding Rate (ascending) to find most negative
        df_sorted = df.sort_values(by="Funding Rate (1h)", ascending=True)
        
        print("\n--- Top 10 Most NEGATIVE Funding Rates (Shorts pay Longs) ---")
        print(df_sorted.head(10).to_string(index=False))
        
        print("\n--- Top 10 Most POSITIVE Funding Rates (Longs pay Shorts) ---")
        print(df_sorted.tail(10).iloc[::-1].to_string(index=False))
        
        # Save to CSV
        filename = "hyperliquid_funding_snapshot.csv"
        df_sorted.to_csv(filename, index=False)
        print(f"\nFull list saved to {filename}")
    else:
        print("No data found.")
