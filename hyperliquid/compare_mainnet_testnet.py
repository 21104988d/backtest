import requests
import pandas as pd
from datetime import datetime

def fetch_funding_rates(url, network_name):
    headers = {"Content-Type": "application/json"}
    payload = {"type": "metaAndAssetCtxs"}

    try:
        print(f"Fetching {network_name} data from {url}...")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        meta = data[0]
        asset_ctxs = data[1]
        
        universe = meta['universe']
        
        results = []
        
        for i, asset_info in enumerate(universe):
            coin_name = asset_info['name']
            if i < len(asset_ctxs):
                ctx = asset_ctxs[i]
                funding = float(ctx['funding'])
                results.append({
                    "Coin": coin_name,
                    f"Funding_{network_name}": funding
                })
            
        return pd.DataFrame(results)

    except Exception as e:
        print(f"Error fetching {network_name} data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    mainnet_url = "https://api.hyperliquid.xyz/info"
    testnet_url = "https://api.hyperliquid-testnet.xyz/info"
    
    df_main = fetch_funding_rates(mainnet_url, "Mainnet")
    df_test = fetch_funding_rates(testnet_url, "Testnet")
    
    if not df_main.empty and not df_test.empty:
        # Merge on Coin
        merged = pd.merge(df_main, df_test, on="Coin", how="inner")
        
        # Calculate Difference
        merged['Diff_1h'] = merged['Funding_Mainnet'] - merged['Funding_Testnet']
        merged['Abs_Diff'] = merged['Diff_1h'].abs()
        
        # Add APR for context (approximate)
        merged['Mainnet_APR%'] = merged['Funding_Mainnet'] * 24 * 365 * 100
        merged['Testnet_APR%'] = merged['Funding_Testnet'] * 24 * 365 * 100
        
        # Sort by biggest absolute difference
        merged_sorted = merged.sort_values(by="Abs_Diff", ascending=False)
        
        print("\n--- Funding Rate Comparison (Mainnet vs Testnet) ---")
        print("Units are 1h Funding Rate. 'Diff' = Mainnet - Testnet")
        # Format columns for readability
        pd.set_option('display.float_format', '{:.8f}'.format)
        print(merged_sorted.head(20).to_string(index=False))
        
        filename = "hyperliquid_mainnet_vs_testnet.csv"
        merged_sorted.to_csv(filename, index=False)
        print(f"\nSaved comparison to {filename}")
        
    elif df_main.empty:
        print("Failed to fetch Mainnet data.")
    elif df_test.empty:
        print("Failed to fetch Testnet data.")
