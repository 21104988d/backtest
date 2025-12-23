import requests
import time
from datetime import datetime

def inspect_ohlc():
    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - 3600 * 1000 * 2 # 2 hours ago
    
    params = {
        "instrument_name": "BTC-PERPETUAL",
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "resolution": "5" # 5 minutes
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'result' in data:
        res = data['result']
        print("Keys:", res.keys())
        print("Count:", len(res['ticks']))
        print("First tick:", res['ticks'][0], datetime.fromtimestamp(res['ticks'][0]/1000))
        print("First close:", res['close'][0])
    else:
        print("No data:", data)

if __name__ == "__main__":
    inspect_ohlc()