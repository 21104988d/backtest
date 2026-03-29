import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_btc_drawdown():
    # Define paths based on workspace structure
    data_path = 'daily_returns_analysis/daily_ohlc.csv'

    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file at {data_path}")
        return

    # Filter for BTC
    btc_df = df[df['coin'] == 'BTC'].copy()
    
    if len(btc_df) == 0:
        print("Warning: No BTC data found in daily_ohlc.csv.")
        return

    print(f"Found {len(btc_df)} days of BTC data.")

    # Ensure chronological order
    btc_df['date'] = pd.to_datetime(btc_df['date'])
    btc_df = btc_df.sort_values('date').reset_index(drop=True)

    # Convert prices to numeric
    btc_df['close'] = pd.to_numeric(btc_df['close'])
    btc_df['high'] = pd.to_numeric(btc_df['high'])
    btc_df['low'] = pd.to_numeric(btc_df['low'])

    # Calculate Drawdown based on Close price
    btc_df['cum_max_close'] = btc_df['close'].cummax()
    btc_df['drawdown_close'] = (btc_df['close'] - btc_df['cum_max_close']) / btc_df['cum_max_close'] * 100

    # Calculate Drawdown based on High-Low (max possible drawdown intraday)
    btc_df['cum_max_high'] = btc_df['high'].cummax()
    btc_df['drawdown_high_low'] = (btc_df['low'] - btc_df['cum_max_high']) / btc_df['cum_max_high'] * 100

    print("Generating distribution plots...")

    # Plot 1: Drawdown distribution histogram (Close-to-Close)
    plt.figure(figsize=(10, 6))
    sns.histplot(btc_df['drawdown_close'], bins=50, kde=True, color='red', alpha=0.6)
    plt.title('Distribution of BTC Daily Drawdowns (Close vs Peak Close)')
    plt.xlabel('Drawdown (%)')
    plt.ylabel('Frequency (Days)')
    plt.grid(True, alpha=0.3)
    
    output_hist = 'btc_drawdown_dist_close.png'
    plt.tight_layout()
    plt.savefig(output_hist)
    
    with open("success.txt", "w") as f:
        f.write("Distribution plot saved.")
        
    print(f"Distribution plot saved to {os.path.abspath(output_hist)}")
    plt.close()

    # Plot 2: Drawdown distribution histogram (High-to-Low)
    plt.figure(figsize=(10, 6))
    sns.histplot(btc_df['drawdown_high_low'], bins=50, kde=True, color='purple', alpha=0.6)
    plt.title('Distribution of BTC Daily Drawdowns (Low vs Peak High)')
    plt.xlabel('Drawdown (%)')
    plt.ylabel('Frequency (Days)')
    plt.grid(True, alpha=0.3)

    output_hist2 = 'btc_drawdown_dist_high_low.png'
    plt.tight_layout()
    plt.savefig(output_hist2)
    print(f"High-Low Distribution plot saved to {os.path.abspath(output_hist2)}")
    plt.close()

    # Plot 3: Drawdown over time
    plt.figure(figsize=(12, 6))
    plt.fill_between(btc_df['date'], btc_df['drawdown_close'], 0, color='red', alpha=0.3)
    plt.plot(btc_df['date'], btc_df['drawdown_close'], color='red', linewidth=1, label='Close-to-Close DD')
    plt.title('BTC Drawdown Over Time')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_ts = 'btc_drawdown_timeseries.png'
    plt.tight_layout()
    plt.savefig(output_ts)
    print(f"Time series plot saved to {os.path.abspath(output_ts)}")
    plt.close()

    # Print basic stats
    print("\n--- BTC Drawdown Stats (Close-to-Close) ---")
    print(f"Mean Drawdown: {btc_df['drawdown_close'].mean():.2f}%")
    print(f"Median Drawdown: {btc_df['drawdown_close'].median():.2f}%")
    print(f"Max Drawdown: {btc_df['drawdown_close'].min():.2f}%")
    print(f"5th Percentile: {btc_df['drawdown_close'].quantile(0.05):.2f}%")

if __name__ == "__main__":
    analyze_btc_drawdown()
