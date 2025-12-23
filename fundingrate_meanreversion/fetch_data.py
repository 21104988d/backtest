import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def fetch_funding_rates(instrument_name, start_date, end_date, save_path="funding_rates.csv"):
    # Adjust save_path to be absolute or relative to script
    if not os.path.isabs(save_path):
        save_path = os.path.join(os.path.dirname(__file__), save_path)

    if os.path.exists(save_path):
        print(f"Loading data from {save_path}...")
        df = pd.read_csv(save_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.sort_index()
        # Check if data covers the range (roughly)
        if df.index.min() <= start_date + timedelta(days=1) and df.index.max() >= end_date - timedelta(days=1):
             # Filter to requested range
            return df.loc[start_date:end_date]
        else:
            print("Existing data insufficient, fetching new data...")
    
    url = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
    
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    all_data = []
    
    current_start = start_ts
    
    print(f"Fetching data from {start_date} to {end_date}...")
    
    while current_start < end_ts:
        params = {
            "instrument_name": instrument_name,
            "start_timestamp": current_start,
            "end_timestamp": end_ts
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'result' in data and data['result']:
                batch = data['result']
                all_data.extend(batch)
                last_ts = batch[-1]['timestamp']
                if last_ts >= end_ts or last_ts <= current_start:
                    break
                current_start = last_ts + 1
            else:
                break
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(1)
            continue
            
        time.sleep(0.1) 
        
    df = pd.DataFrame(all_data)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df.sort_index()
        # Rename interest_8h to funding_rate for clarity
        df = df.rename(columns={'interest_8h': 'funding_rate'})
        df.to_csv(save_path)
    return df

def fetch_ohlc(instrument_name, start_date, end_date, resolution="60", save_path="ohlc_data.csv"):
    # Adjust save_path
    if not os.path.isabs(save_path):
        save_path = os.path.join(os.path.dirname(__file__), save_path)

    if os.path.exists(save_path):
        print(f"Loading OHLC data from {save_path}...")
        df = pd.read_csv(save_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.sort_index()
        if df.index.min() <= start_date + timedelta(days=1) and df.index.max() >= end_date - timedelta(days=1):
            return df.loc[start_date:end_date]
        else:
            print("Existing OHLC data insufficient, fetching new data...")

    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    all_ticks = []
    all_open = []
    all_high = []
    all_low = []
    all_close = []
    
    # Deribit OHLC endpoint returns all data in one go if range is small, 
    # but for long ranges we might need to chunk? 
    # Documentation says "The maximum number of data points is 10000".
    # 5 min resolution: 12 points/hour * 24 hours = 288 points/day.
    # 30 days = 8640 points. So one call might be enough for 30 days.
    # Let's try chunking by 10 days to be safe.
    
    current_start = start_ts
    chunk_size_ms = 10 * 24 * 3600 * 1000 # 10 days
    
    print(f"Fetching OHLC data from {start_date} to {end_date}...")
    
    while current_start < end_ts:
        current_end = min(current_start + chunk_size_ms, end_ts)
        
        params = {
            "instrument_name": instrument_name,
            "start_timestamp": current_start,
            "end_timestamp": current_end,
            "resolution": resolution
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'result' in data and data['result']['status'] == 'ok':
                res = data['result']
                if 'ticks' in res:
                    all_ticks.extend(res['ticks'])
                    all_open.extend(res['open'])
                    all_high.extend(res['high'])
                    all_low.extend(res['low'])
                    all_close.extend(res['close'])
            else:
                print(f"Error or no data in chunk: {data}")
                
        except Exception as e:
            print(f"Error fetching OHLC: {e}")
        
        current_start = current_end
        time.sleep(0.2)

    df = pd.DataFrame({
        'timestamp': all_ticks,
        'open': all_open,
        'high': all_high,
        'low': all_low,
        'close': all_close
    })
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df.sort_index()
        # Remove duplicates if any
        df = df[~df.index.duplicated(keep='first')]
        df.to_csv(save_path)
        
    return df

if __name__ == "__main__":
    # Fetch last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print("Fetching Funding Rates...")
    df_funding = fetch_funding_rates("BTC-PERPETUAL", start_date, end_date)
    print(f"Fetched {len(df_funding)} funding records.")
    
    print("Fetching 5-min OHLC...")
    df_ohlc = fetch_ohlc("BTC-PERPETUAL", start_date, end_date, resolution="5", save_path="ohlc_5m.csv")
    print(f"Fetched {len(df_ohlc)} OHLC records.")
