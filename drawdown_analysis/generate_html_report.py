import csv
import json
import math
import os
import datetime

def main():
    data_path = '../daily_returns_analysis/daily_ohlc.csv'
    
    btc_data = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['coin'] == 'BTC':
                    # Extract date, open, close, high, low
                    btc_data.append({
                        'date': row['date'],
                        'open': float(row['open']),
                        'close': float(row['close']),
                        'high': float(row['high']),
                        'low': float(row['low'])
                    })
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    print(f"Loaded {len(btc_data)} BTC records.")
    
    # Sort chronologically just in case
    btc_data.sort(key=lambda x: x['date'])
    
    # For each day, if it closes down from the previous day's close (or some past high),
    # how long does it take to get back to that pre-drop level?
    # We will compute: for ANY day 'i', if there is a day 'j' > 'i' where price recovers to price at 'i',
    # and during that period price drops, what is the max drop and how many days did it take to recover.
    # A cleaner approach for user's request:
    # "If we get a drawdown today (e.g. price drops from 100 to 90), how many days we recover back to 100"
    
    # 1. Identify all local peaks before a drop.
    # A simpler way is to act like a buyer: If I buy today and price goes down tomorrow, 
    # what is my max drawdown until I break even, and how long does it take?
    
    dates = []
    dd_close_series = []
    drawdowns_close = []
    
    recovered_drawdowns = [] # Store dict: {'depth': pct, 'duration': days}
    
    # Pre-parse dates
    for i in range(len(btc_data)):
        btc_data[i]['dt'] = datetime.datetime.strptime(btc_data[i]['date'], '%Y-%m-%d').date()

    for i in range(len(btc_data) - 1):
        price_i = btc_data[i]['close']
        price_next = btc_data[i+1]['close']
        
        # Did we experience a drop immediately?
        if price_next < price_i:
            # We entered a drawdown starting from price_i.
            # Look forward to find the first day it recovers >= price_i
            trough_price = price_i
            recovered = False
            recovery_days = 0
            
            for j in range(i+1, len(btc_data)):
                current_p = btc_data[j]['close']
                if current_p < trough_price:
                    trough_price = current_p
                    
                if current_p >= price_i:
                    recovered = True
                    recovery_days = (btc_data[j]['dt'] - btc_data[i]['dt']).days
                    break
            
            if recovered:
                depth_pct = ((trough_price - price_i) / price_i) * 100
                recovered_drawdowns.append({
                    'depth': depth_pct,
                    'duration': recovery_days
                })

    # New Requirement: For ANY negative daily return (close < open), 
    # how many days does it take for the close price to recover to the OPEN price of that negative candle?
    # e.g., if price opened at 100 and closed at 90, how many days to get back to >= 100?
    daily_neg_recovery_data = [] # Store {'ret': pct, 'days': num}
    
    for i in range(len(btc_data)):
        o_price = btc_data[i]['open']
        c_price = btc_data[i]['close']
        
        # Is it a negative daily candle?
        if c_price < o_price:
            daily_ret = ((c_price - o_price) / o_price) * 100
            recovered = False
            r_days = 0
            for j in range(i+1, len(btc_data)):
                if btc_data[j]['close'] >= o_price:
                    recovered = True
                    r_days = (btc_data[j]['dt'] - btc_data[i]['dt']).days
                    break
            
            if recovered:
                daily_neg_recovery_data.append({
                    'ret': daily_ret,
                    'days': r_days
                })
                
    # Group by negative return bins (e.g., 0 to -1%, -1 to -2%, etc.)
    neg_ret_bins = {}
    for d in daily_neg_recovery_data:
        # e.g., -1.5 -> bin -2. Bin width 1%
        b_idx = math.floor(d['ret'] / 1.0) * 1
        if b_idx not in neg_ret_bins:
            neg_ret_bins[b_idx] = []
        neg_ret_bins[b_idx].append(d['days'])
        
    neg_ret_sorted = sorted(neg_ret_bins.keys())
    neg_rec_labels = [f"[{b}%, {b+1}%)" for b in neg_ret_sorted]
    neg_rec_avg_days = [sum(neg_ret_bins[b])/len(neg_ret_bins[b]) for b in neg_ret_sorted]
    neg_rec_counts = [len(neg_ret_bins[b]) for b in neg_ret_sorted]

    # Keep a cumulative max series for the overall timeseries chart
    cum_max_close = 0
    for row in btc_data:
        c = row['close']
        if c > cum_max_close: cum_max_close = c
        dd_c = ((c - cum_max_close) / cum_max_close) * 100 if cum_max_close > 0 else 0
        drawdowns_close.append(dd_c)
        dates.append(row['date'])
        dd_close_series.append(dd_c)
        
    # Create histogram data (binning)
    # Let's create bins of 2%
    bins = {}
    for d in drawdowns_close:
        # DD is negative, e.g., -15.4%
        bin_idx = math.floor(d / 2.0) * 2
        bins[bin_idx] = bins.get(bin_idx, 0) + 1
        
    # Sort bins
    sorted_bins = sorted(bins.keys())
    bin_labels = [f"[{b}%, {b+2}%)" for b in sorted_bins]
    bin_counts = [bins[b] for b in sorted_bins]
    
    # Calculate recovery bins (by max drawdown depth)
    # Define intervals, eg. 0 to -5, -5 to -10, etc.
    recovery_bins_data = {}
    for rd in recovered_drawdowns:
        # Bin by 5% increments for recovery to ensure enough samples per bin
        # e.g., -12% -> -15% bin
        b_idx = math.floor(rd['depth'] / 5.0) * 5
        if b_idx not in recovery_bins_data:
            recovery_bins_data[b_idx] = []
        recovery_bins_data[b_idx].append(rd['duration'])
        
    rec_sorted_bins = sorted(recovery_bins_data.keys())
    rec_labels = [f"[{b}%, {b+5}%)" for b in rec_sorted_bins]
    rec_avg_durations = [sum(recovery_bins_data[b])/len(recovery_bins_data[b]) for b in rec_sorted_bins]
    rec_counts = [len(recovery_bins_data[b]) for b in rec_sorted_bins]
    
    # Calculate stats
    all_recovery_days = [rd['duration'] for rd in recovered_drawdowns]
    avg_recovery = sum(all_recovery_days) / len(all_recovery_days) if all_recovery_days else 0
    max_recovery = max(all_recovery_days) if all_recovery_days else 0
    
    # Generate HTML report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Drawdown Distribution & Recovery</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .chart-wrapper {{ margin-top: 30px; position: relative; height: 400px; width: 100%; }}
        h1 {{ text-align: center; color: #333; }}
        .stats {{ display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 6px; gap: 15px; }}
        .stat-box {{ text-align: center; flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #d32f2f; }}
        .stat-value.blue {{ color: #1976d2; }}
        .stat-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Bitcoin (BTC) Drawdown & Recovery Analysis</h1>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{min(drawdowns_close):.2f}%</div>
                <div class="stat-label">Maximum Drawdown</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{sum(drawdowns_close)/len(drawdowns_close):.2f}%</div>
                <div class="stat-label">Average Drawdown</div>
            </div>
            <div class="stat-box">
                <div class="stat-value blue">{avg_recovery:.1f} days</div>
                <div class="stat-label">Average Recovery Time</div>
            </div>
            <div class="stat-box">
                <div class="stat-value blue">{max_recovery} days</div>
                <div class="stat-label">Max Recovery Time</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color:#555;">{len(btc_data)}</div>
                <div class="stat-label">Total Days analyzed</div>
            </div>
        </div>

        <div class="chart-wrapper">
            <canvas id="distChart"></canvas>
        </div>
        
        <div class="chart-wrapper">
            <canvas id="recoveryChart"></canvas>
        </div>
        
        <div class="chart-wrapper">
            <canvas id="dailyNegRecoveryChart"></canvas>
        </div>
        
        <div class="chart-wrapper">
            <canvas id="tsChart"></canvas>
        </div>
    </div>

    <script>
        // Distribution Chart
        const ctxDist = document.getElementById('distChart').getContext('2d');
        new Chart(ctxDist, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(bin_labels)},
                datasets: [{{
                    label: 'Frequency (Days)',
                    data: {json.dumps(bin_counts)},
                    backgroundColor: 'rgba(235, 87, 87, 0.7)',
                    borderColor: 'rgba(211, 47, 47, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{ display: true, text: 'Distribution of BTC Daily Drawdowns (Close vs Peak Close)' }},
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: 'Number of Days' }} }},
                    x: {{ title: {{ display: true, text: 'Drawdown Range (%)' }} }}
                }}
            }}
        }});

        // Recovery Chart (by drawdown depth)
        const ctxRec = document.getElementById('recoveryChart').getContext('2d');
        new Chart(ctxRec, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(rec_labels)},
                datasets: [{{
                    label: 'Average Recovery Time (Days)',
                    data: {json.dumps(rec_avg_durations)},
                    backgroundColor: 'rgba(25, 118, 210, 0.7)',
                    borderColor: 'rgba(21, 101, 192, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{ display: true, text: 'Average Recovery Time by Maximum Drawdown Depth' }},
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            afterLabel: function(context) {{
                                const counts = {json.dumps(rec_counts)};
                                return 'Sample Count: ' + counts[context.dataIndex] + ' drawdowns';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: 'Days to Recover' }} }},
                    x: {{ title: {{ display: true, text: 'Max Drawdown Depth (%)' }} }}
                }}
            }}
        }});

        // Daily Negative Candle: return bin (x) vs avg recovery days (y)
        const ctxNegRec = document.getElementById('dailyNegRecoveryChart').getContext('2d');
        new Chart(ctxNegRec, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(neg_rec_labels)},
                datasets: [{{
                    label: 'Average Recovery Time (Days)',
                    data: {json.dumps(neg_rec_avg_days)},
                    backgroundColor: 'rgba(255, 152, 0, 0.7)',
                    borderColor: 'rgba(245, 124, 0, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{ display: true, text: 'Daily Negative Return vs Recovery Days (to reclaim candle open)' }},
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            afterLabel: function(context) {{
                                const counts = {json.dumps(neg_rec_counts)};
                                return 'Sample Count: ' + counts[context.dataIndex] + ' candles';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: 'Days to Recover' }} }},
                    x: {{ title: {{ display: true, text: 'Daily Negative Return (%)' }} }}
                }}
            }}
        }});

        // Time Series Chart
        const allDates = {json.dumps(dates)};
        const allDds = {json.dumps(dd_close_series)};
        const ctxTs = document.getElementById('tsChart').getContext('2d');
        new Chart(ctxTs, {{
            type: 'line',
            data: {{
                labels: allDates,
                datasets: [{{
                    label: 'Close-to-Close Drawdown (%)',
                    data: allDds,
                    borderColor: 'rgba(211, 47, 47, 0.8)',
                    backgroundColor: 'rgba(235, 87, 87, 0.2)',
                    borderWidth: 1.5,
                    fill: true,
                    pointRadius: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{ display: true, text: 'BTC Drawdown Over Time' }}
                }},
                scales: {{
                    y: {{ title: {{ display: true, text: 'Drawdown (%)' }} }},
                    x: {{ title: {{ display: true, text: 'Date' }}, ticks: {{ maxTicksLimit: 20 }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    with open('btc_drawdown_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print("Successfully generated btc_drawdown_report.html!")

if __name__ == '__main__':
    main()
