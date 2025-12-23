import requests
import time
from datetime import datetime

def inspect_records():
    url = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
    # Request just 2 hours of data
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - 3600 * 1000 * 2 
    
    params = {
        "instrument_name": "BTC-PERPETUAL",
        "start_timestamp": start_ts,
        "end_timestamp": end_ts
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'result' in data and data['result']:
        print(f"Fetched {len(data['result'])} records")
        for i in range(len(data['result'])):
            ts = data['result'][i]['timestamp']
            dt = datetime.fromtimestamp(ts/1000)
            print(f"Record {i}: {dt} (ts: {ts})")
            
        if len(data['result']) > 1:
            diff = data['result'][1]['timestamp'] - data['result'][0]['timestamp']
            print(f"Time difference: {diff} ms ({diff/1000/60} minutes)")
    else:
        print("No data:", data)

if __name__ == "__main__":
    inspect_records()