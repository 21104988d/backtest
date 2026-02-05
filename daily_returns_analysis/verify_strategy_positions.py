"""
Analyze stop loss behavior specifically for TOP/BOTTOM 3 movers
(the actual positions we trade in the strategy)
"""

import pandas as pd
import numpy as np

# Load OHLC data
ohlc_df = pd.read_csv('daily_ohlc.csv')
ohlc_df['date'] = pd.to_datetime(ohlc_df['date']).dt.date
ohlc_df = ohlc_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['daily_return'])

dates = sorted(ohlc_df['date'].unique())
N = 3  # Top/bottom N to trade

print("="*80)
print("STOP LOSS ANALYSIS FOR TOP/BOTTOM 3 MOVERS ONLY")
print("="*80)

# Collect actual positions we would trade
top_positions = []  # Top 3 gainers each day
bottom_positions = []  # Bottom 3 losers each day

for i in range(1, len(dates)):
    signal_date = dates[i-1]
    trade_date = dates[i]
    
    signal_day_data = ohlc_df[ohlc_df['date'] == signal_date].copy()
    if len(signal_day_data) < N * 2:
        continue
    
    top_coins = signal_day_data.nlargest(N, 'daily_return')['coin'].tolist()
    bottom_coins = signal_day_data.nsmallest(N, 'daily_return')['coin'].tolist()
    
    trade_day_data = ohlc_df[ohlc_df['date'] == trade_date]
    
    for coin in top_coins:
        coin_data = trade_day_data[trade_day_data['coin'] == coin]
        if len(coin_data) > 0:
            top_positions.append(coin_data.iloc[0].to_dict())
    
    for coin in bottom_coins:
        coin_data = trade_day_data[trade_day_data['coin'] == coin]
        if len(coin_data) > 0:
            bottom_positions.append(coin_data.iloc[0].to_dict())

top_df = pd.DataFrame(top_positions)
bottom_df = pd.DataFrame(bottom_positions)

# Calculate drawdowns for these specific positions
top_df['long_drawdown'] = (top_df['low'] / top_df['open'] - 1) * 100
top_df['short_drawdown'] = -(top_df['high'] / top_df['open'] - 1) * 100
bottom_df['long_drawdown'] = (bottom_df['low'] / bottom_df['open'] - 1) * 100
bottom_df['short_drawdown'] = -(bottom_df['high'] / bottom_df['open'] - 1) * 100

print(f"\nTop 3 gainers (we SHORT in Mean Reversion): {len(top_df)} positions")
print(f"Bottom 3 losers (we LONG in Mean Reversion): {len(bottom_df)} positions")

print("\n" + "="*80)
print("MEAN REVERSION STRATEGY: SHORT top gainers, LONG bottom losers")
print("="*80)

print("\n--- SHORTING TOP 3 GAINERS (next day) ---")
print(f"  Next-day return (close vs open): {top_df['daily_return'].mean():.2f}%")
print(f"  % that go DOWN (good for short): {(top_df['daily_return'] < 0).mean()*100:.1f}%")
print(f"  Intraday high vs open (adverse for short): {(top_df['high']/top_df['open']-1).mean()*100:.2f}%")

# For SHORT positions, stop loss triggers if high >= open * (1 + SL%)
print("\n  Stop loss hit rates for SHORT positions:")
for sl in [0.5, 1.0, 1.5, 2.0, 3.0]:
    hit_rate = (top_df['short_drawdown'] <= -sl).mean() * 100
    print(f"    {sl}% SL: {hit_rate:.1f}% of positions stopped")

# Positions stopped at 0.5% but NOT at 1% for shorts
short_diff = top_df[(top_df['short_drawdown'] <= -0.5) & (top_df['short_drawdown'] > -1.0)]
print(f"\n  Positions stopped at 0.5% but NOT at 1% ({len(short_diff)} positions):")
# For shorts, profit = -(close/open - 1), so negative close return = profit
short_profit = -(short_diff['daily_return'])
print(f"    Avg return if held (for short): {short_profit.mean():.2f}%")
print(f"    % that would have been profitable: {(short_profit > 0).mean()*100:.1f}%")

print("\n--- LONGING BOTTOM 3 LOSERS (next day) ---")
print(f"  Next-day return (close vs open): {bottom_df['daily_return'].mean():.2f}%")
print(f"  % that go UP (good for long): {(bottom_df['daily_return'] > 0).mean()*100:.1f}%")
print(f"  Intraday low vs open (adverse for long): {(bottom_df['low']/bottom_df['open']-1).mean()*100:.2f}%")

# For LONG positions, stop loss triggers if low <= open * (1 - SL%)
print("\n  Stop loss hit rates for LONG positions:")
for sl in [0.5, 1.0, 1.5, 2.0, 3.0]:
    hit_rate = (bottom_df['long_drawdown'] <= -sl).mean() * 100
    print(f"    {sl}% SL: {hit_rate:.1f}% of positions stopped")

# Positions stopped at 0.5% but NOT at 1% for longs
long_diff = bottom_df[(bottom_df['long_drawdown'] <= -0.5) & (bottom_df['long_drawdown'] > -1.0)]
print(f"\n  Positions stopped at 0.5% but NOT at 1% ({len(long_diff)} positions):")
print(f"    Avg return if held: {long_diff['daily_return'].mean():.2f}%")
print(f"    % that would have been profitable: {(long_diff['daily_return'] > 0).mean()*100:.1f}%")

print("\n" + "="*80)
print("CRITICAL ANALYSIS: PnL Breakdown by Stop Loss Level")
print("="*80)

def calculate_strategy_pnl(sl_pct):
    """Calculate total PnL for mean reversion strategy at given SL"""
    
    # LONG bottom losers
    long_pnl = 0
    long_stopped = 0
    for _, row in bottom_df.iterrows():
        drawdown = row['long_drawdown']
        close_return = row['daily_return']
        
        if sl_pct is None:
            long_pnl += close_return
        elif drawdown <= -sl_pct:
            long_pnl += -sl_pct
            long_stopped += 1
        else:
            long_pnl += close_return
    
    # SHORT top winners
    short_pnl = 0
    short_stopped = 0
    for _, row in top_df.iterrows():
        drawdown = row['short_drawdown']
        close_return = -row['daily_return']  # Negative close return = profit for short
        
        if sl_pct is None:
            short_pnl += close_return
        elif drawdown <= -sl_pct:
            short_pnl += -sl_pct
            short_stopped += 1
        else:
            short_pnl += close_return
    
    total_positions = len(bottom_df) + len(top_df)
    total_stopped = long_stopped + short_stopped
    
    return long_pnl, short_pnl, total_stopped, total_positions

print(f"\nSL Level   Long PnL%   Short PnL%   Total PnL%   Stop Rate")
print("-"*65)
for sl in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, None]:
    long_pnl, short_pnl, stopped, total = calculate_strategy_pnl(sl)
    total_pnl = long_pnl + short_pnl
    sl_str = f"{sl}%" if sl else "None"
    stop_rate = stopped / total * 100 if total > 0 else 0
    print(f"  {sl_str:6s}    {long_pnl:8.1f}%   {short_pnl:9.1f}%   {total_pnl:10.1f}%   {stop_rate:5.1f}%")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
