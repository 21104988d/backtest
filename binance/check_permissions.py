import os
import time
import hmac
import hashlib
import requests
from dotenv import load_dotenv
from urllib.parse import urlencode

# Load environment variables
load_dotenv()

API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

if not API_KEY or not API_SECRET:
    print("Error: BINANCE_API_KEY or BINANCE_API_SECRET not found in .env")
    exit(1)

def get_signature(params, secret):
    query_string = urlencode(params)
    return hmac.new(secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

def check_spot_permissions():
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/account"
    
    timestamp = int(time.time() * 1000)
    params = {
        'timestamp': timestamp
    }
    
    signature = get_signature(params, API_SECRET)
    params['signature'] = signature
    
    headers = {
        'X-MBX-APIKEY': API_KEY
    }
    
    try:
        response = requests.get(base_url + endpoint, params=params, headers=headers)
        data = response.json()
        
        if response.status_code == 200:
            print("\n=== Spot Account Permissions ===")
            print(f"Can Trade: {data.get('canTrade')}")
            print(f"Can Withdraw: {data.get('canWithdraw')}")
            print(f"Can Deposit: {data.get('canDeposit')}")
            print(f"Account Type: {data.get('accountType')}")
            print(f"Permissions: {data.get('permissions')}")
            
            # Check balances
            balances = [b for b in data.get('balances', []) if float(b['free']) > 0 or float(b['locked']) > 0]
            if balances:
                print("\nSpot Balances (> 0):")
                for b in balances:
                    print(f"- {b['asset']}: Free={b['free']}, Locked={b['locked']}")
            else:
                print("\nNo non-zero Spot balances found.")
                
        else:
            print(f"\nError checking Spot permissions: {data.get('msg', 'Unknown error')}")
            
    except Exception as e:
        print(f"\nException checking Spot permissions: {e}")

def check_futures_permissions():
    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v2/account"
    
    timestamp = int(time.time() * 1000)
    params = {
        'timestamp': timestamp
    }
    
    signature = get_signature(params, API_SECRET)
    params['signature'] = signature
    
    headers = {
        'X-MBX-APIKEY': API_KEY
    }
    
    try:
        response = requests.get(base_url + endpoint, params=params, headers=headers)
        data = response.json()
        
        if response.status_code == 200:
            print("\n=== Futures Account Permissions ===")
            print(f"Can Trade: {data.get('canTrade')}")
            print(f"Can Withdraw: {data.get('canWithdraw')}")
            print(f"Can Deposit: {data.get('canDeposit')}")
            
            # Check balances
            assets = [a for a in data.get('assets', []) if float(a['walletBalance']) > 0]
            if assets:
                print("\nFutures Assets (> 0):")
                for a in assets:
                    print(f"- {a['asset']}: Wallet Balance={a['walletBalance']}, Margin Balance={a['marginBalance']}")
            else:
                print("\nNo non-zero Futures assets found.")
        else:
            print(f"\nError checking Futures permissions: {data.get('msg', 'Unknown error')}")
            
    except Exception as e:
        print(f"\nException checking Futures permissions: {e}")

if __name__ == "__main__":
    print("Testing Binance API Keys...")
    check_spot_permissions()
    check_futures_permissions()
