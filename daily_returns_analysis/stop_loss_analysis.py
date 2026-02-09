"""
Stop Loss Mechanism Analysis
Analyzes intraday high/low prices to understand stop loss behavior
for the mean reversion strategy.

Key question: Given daily OHLC data, how reliable is the stop loss detection
when we don't know whether the high or low was reached first?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration (match main backtest)
N = 3
STOP_LOSS_PCT = 0.5
TRADING_FEE = 0.045
ROUND_TRIP_FEE = TRADING_FEE * 2

print("=" * 80)
print("STOP LOSS MECHANISM ANALYSIS")
print("=" * 80)

# ─────────────────────────────────────────────────────────────
# 1. Load hourly data for precise stop loss verification
# ─────────────────────────────────────────────────────────────
print("\nLoading hourly price data for precise analysis...")
hourly_df = pd.read_csv('price_history.csv')
hourly_df['timestamp'] = pd.to_datetime(hourly_df['timestamp'])
hourly_df['date'] = hourly_df['timestamp'].dt.date
hourly_df = hourly_df.sort_values(['coin', 'timestamp'])

print(f"Loaded {len(hourly_df):,} hourly records")

# Also load daily OHLC
ohlc_df = pd.read_csv('daily_ohlc.csv')
ohlc_df['date'] = pd.to_datetime(ohlc_df['date']).dt.date
ohlc_df = ohlc_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['daily_return'])
dates = sorted(ohlc_df['date'].unique())

print(f"Loaded {len(ohlc_df):,} daily OHLC records")
print(f"Date range: {dates[0]} to {dates[-1]}")


# ─────────────────────────────────────────────────────────────
# 2. Hourly-based stop loss check (ground truth)
# ─────────────────────────────────────────────────────────────

def check_stop_loss_hourly(coin, date, is_long, sl_pct, hourly_data):
    """
    Walk through hourly prices to determine:
    1. Was the stop loss actually hit? (and at which hour)
    2. What was the price path? (did it dip then recover, or vice versa)
    
    Returns: (final_return, was_stopped, stop_hour, hourly_prices)
    """
    coin_hours = hourly_data[(hourly_data['coin'] == coin) & (hourly_data['date'] == date)]
    if len(coin_hours) == 0:
        return None, False, None, []

    coin_hours = coin_hours.sort_values('timestamp')
    prices = coin_hours['price'].tolist()
    timestamps = coin_hours['timestamp'].tolist()
    open_price = prices[0]
    close_price = prices[-1]

    if is_long:
        stop_price = open_price * (1 - sl_pct / 100)
        for i, p in enumerate(prices):
            if p <= stop_price:
                return -sl_pct, True, i, prices
        return (close_price / open_price - 1) * 100, False, None, prices
    else:
        stop_price = open_price * (1 + sl_pct / 100)
        for i, p in enumerate(prices):
            if p >= stop_price:
                return -sl_pct, True, i, prices
        return -(close_price / open_price - 1) * 100, False, None, prices


def check_stop_loss_daily(open_price, high_price, low_price, close_price, is_long, sl_pct):
    """
    Check stop loss using daily OHLC only (current method).
    """
    if is_long:
        stop_price = open_price * (1 - sl_pct / 100)
        if low_price <= stop_price:
            return -sl_pct, True
        return (close_price / open_price - 1) * 100, False
    else:
        stop_price = open_price * (1 + sl_pct / 100)
        if high_price >= stop_price:
            return -sl_pct, True
        return -(close_price / open_price - 1) * 100, False


# ─────────────────────────────────────────────────────────────
# 3. Compare daily vs hourly stop loss detection
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("COMPARING DAILY OHLC vs HOURLY STOP LOSS DETECTION")
print(f"Stop Loss: {STOP_LOSS_PCT}%")
print("=" * 80)

# Collect actual traded positions (top/bottom N movers)
comparisons = []
sample_count = 0

print("\nProcessing trading days...", flush=True)
for i in range(1, len(dates)):
    signal_date = dates[i - 1]
    trade_date = dates[i]

    signal_data = ohlc_df[ohlc_df['date'] == signal_date]
    if len(signal_data) < N * 2:
        continue

    top_coins = signal_data.nlargest(N, 'daily_return')['coin'].tolist()
    bottom_coins = signal_data.nsmallest(N, 'daily_return')['coin'].tolist()

    trade_data = ohlc_df[ohlc_df['date'] == trade_date]

    # Mean Reversion: SHORT top, LONG bottom
    for coin in top_coins:
        row = trade_data[trade_data['coin'] == coin]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        daily_ret, daily_stopped = check_stop_loss_daily(
            row['open'], row['high'], row['low'], row['close'],
            is_long=False, sl_pct=STOP_LOSS_PCT
        )
        hourly_ret, hourly_stopped, stop_hour, prices = check_stop_loss_hourly(
            coin, trade_date, is_long=False, sl_pct=STOP_LOSS_PCT, hourly_data=hourly_df
        )
        if hourly_ret is not None:
            comparisons.append({
                'date': trade_date, 'coin': coin, 'direction': 'SHORT',
                'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close'],
                'daily_stopped': daily_stopped, 'daily_return': daily_ret,
                'hourly_stopped': hourly_stopped, 'hourly_return': hourly_ret,
                'stop_hour': stop_hour, 'n_hours': len(prices),
            })

    for coin in bottom_coins:
        row = trade_data[trade_data['coin'] == coin]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        daily_ret, daily_stopped = check_stop_loss_daily(
            row['open'], row['high'], row['low'], row['close'],
            is_long=True, sl_pct=STOP_LOSS_PCT
        )
        hourly_ret, hourly_stopped, stop_hour, prices = check_stop_loss_hourly(
            coin, trade_date, is_long=True, sl_pct=STOP_LOSS_PCT, hourly_data=hourly_df
        )
        if hourly_ret is not None:
            comparisons.append({
                'date': trade_date, 'coin': coin, 'direction': 'LONG',
                'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close'],
                'daily_stopped': daily_stopped, 'daily_return': daily_ret,
                'hourly_stopped': hourly_stopped, 'hourly_return': hourly_ret,
                'stop_hour': stop_hour, 'n_hours': len(prices),
            })

comp_df = pd.DataFrame(comparisons)
print(f"\nTotal positions analyzed: {len(comp_df)}")

# ─────────────────────────────────────────────────────────────
# 4. Results
# ─────────────────────────────────────────────────────────────

# Agreement between daily and hourly
agree = (comp_df['daily_stopped'] == comp_df['hourly_stopped']).sum()
disagree = (comp_df['daily_stopped'] != comp_df['hourly_stopped']).sum()
agreement_pct = agree / len(comp_df) * 100

print(f"\n--- Stop Loss Detection Agreement ---")
print(f"  Agree:    {agree:,} ({agreement_pct:.2f}%)")
print(f"  Disagree: {disagree:,} ({100-agreement_pct:.2f}%)")

# Breakdown of disagreements
daily_yes_hourly_no = comp_df[(comp_df['daily_stopped']) & (~comp_df['hourly_stopped'])]
daily_no_hourly_yes = comp_df[(~comp_df['daily_stopped']) & (comp_df['hourly_stopped'])]

print(f"\n  Daily says STOPPED, Hourly says NOT: {len(daily_yes_hourly_no)}")
print(f"  Daily says NOT, Hourly says STOPPED: {len(daily_no_hourly_yes)}")

# Stop loss hit rates
daily_stop_rate = comp_df['daily_stopped'].mean() * 100
hourly_stop_rate = comp_df['hourly_stopped'].mean() * 100
print(f"\n--- Stop Loss Hit Rates ---")
print(f"  Daily OHLC method:  {daily_stop_rate:.1f}%")
print(f"  Hourly method:      {hourly_stop_rate:.1f}%")

# PnL comparison
daily_total_pnl = (comp_df['daily_return'] - ROUND_TRIP_FEE).sum()
hourly_total_pnl = (comp_df['hourly_return'] - ROUND_TRIP_FEE).sum()
print(f"\n--- Total PnL Comparison ---")
print(f"  Daily OHLC method:  {daily_total_pnl:+.1f}%")
print(f"  Hourly method:      {hourly_total_pnl:+.1f}%")
print(f"  Difference:         {hourly_total_pnl - daily_total_pnl:+.1f}%")

# ─────────────────────────────────────────────────────────────
# 5. When does stop loss trigger within the day?
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STOP LOSS TIMING ANALYSIS (Hourly)")
print("=" * 80)

stopped_positions = comp_df[comp_df['hourly_stopped']]
if len(stopped_positions) > 0:
    print(f"\nPositions that hit stop loss: {len(stopped_positions)}")
    hour_dist = stopped_positions['stop_hour'].value_counts().sort_index()
    
    print(f"\nHour of stop loss trigger (0=open, 23=close):")
    total_stopped = len(stopped_positions)
    cumulative = 0
    for hour in range(24):
        count = hour_dist.get(hour, 0)
        cumulative += count
        pct = count / total_stopped * 100
        cum_pct = cumulative / total_stopped * 100
        bar = '█' * int(pct * 2)
        print(f"  Hour {hour:2d}: {count:5d} ({pct:5.1f}%) cum: {cum_pct:5.1f}%  {bar}")

    print(f"\n  Median stop hour: {stopped_positions['stop_hour'].median():.0f}")
    print(f"  Mean stop hour:   {stopped_positions['stop_hour'].mean():.1f}")
    
    # Stop within first hour vs rest
    first_hour = (stopped_positions['stop_hour'] == 0).sum()
    first_3h = (stopped_positions['stop_hour'] <= 2).sum()
    print(f"\n  Stopped in first hour:    {first_hour} ({first_hour/total_stopped*100:.1f}%)")
    print(f"  Stopped in first 3 hours: {first_3h} ({first_3h/total_stopped*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 6. Edge cases: positions where BOTH high and low would trigger
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("EDGE CASE: BOTH HIGH & LOW EXCEED STOP LOSS THRESHOLDS")
print("=" * 80)
print("(Cases where daily OHLC can't determine order of price movement)")

edge_cases = 0
edge_details = []
for _, row in comp_df.iterrows():
    open_p = row['open']
    high_p = row['high']
    low_p = row['low']
    
    long_stop = open_p * (1 - STOP_LOSS_PCT / 100)
    short_stop = open_p * (1 + STOP_LOSS_PCT / 100)
    
    # For this position's direction, check if the opposite extreme also breached
    if row['direction'] == 'LONG':
        # We check if low triggers stop. But did high also go far enough that
        # a short SL would have triggered too? (indicates wild price swing)
        if low_p <= long_stop and high_p >= short_stop:
            edge_cases += 1
            edge_details.append(row)
    else:
        # SHORT: high triggers stop. But did low also drop enough?
        if high_p >= short_stop and low_p <= long_stop:
            edge_cases += 1
            edge_details.append(row)

print(f"\nPositions where price range exceeds {STOP_LOSS_PCT*2}% from open: {edge_cases}")
print(f"  ({edge_cases/len(comp_df)*100:.1f}% of all positions)")
print(f"  These are volatile days where daily OHLC ordering matters most")

if edge_details:
    edge_df = pd.DataFrame(edge_details)
    print(f"\n  Daily OHLC got it right: {(edge_df['daily_stopped'] == edge_df['hourly_stopped']).sum()}")
    print(f"  Daily OHLC got it wrong: {(edge_df['daily_stopped'] != edge_df['hourly_stopped']).sum()}")

# ─────────────────────────────────────────────────────────────
# 7. Adverse Excursion Analysis
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("MAX ADVERSE EXCURSION (MAE) ANALYSIS")
print("=" * 80)

# For LONG positions: MAE = (low - open) / open
longs = comp_df[comp_df['direction'] == 'LONG'].copy()
longs['mae'] = (longs['low'] / longs['open'] - 1) * 100

# For SHORT positions: MAE = -(high - open) / open
shorts = comp_df[comp_df['direction'] == 'SHORT'].copy()
shorts['mae'] = -(shorts['high'] / shorts['open'] - 1) * 100

all_mae = pd.concat([longs['mae'], shorts['mae']])

print(f"\nMAE Statistics (how much price moves against us before close):")
print(f"  Mean:    {all_mae.mean():.2f}%")
print(f"  Median:  {all_mae.median():.2f}%")
print(f"  Std:     {all_mae.std():.2f}%")
print(f"  10th %:  {all_mae.quantile(0.10):.2f}%")
print(f"  5th %:   {all_mae.quantile(0.05):.2f}%")
print(f"  1st %:   {all_mae.quantile(0.01):.2f}%")
print(f"  Worst:   {all_mae.min():.2f}%")

print(f"\n  % of positions with MAE > 0.5%:  {(all_mae < -0.5).mean()*100:.1f}%")
print(f"  % of positions with MAE > 1.0%:  {(all_mae < -1.0).mean()*100:.1f}%")
print(f"  % of positions with MAE > 2.0%:  {(all_mae < -2.0).mean()*100:.1f}%")
print(f"  % of positions with MAE > 5.0%:  {(all_mae < -5.0).mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────
# 8. Charts
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("GENERATING CHARTS...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Stop Loss Analysis ({STOP_LOSS_PCT}% SL)', fontsize=14, fontweight='bold')

# 1. MAE Distribution
ax = axes[0, 0]
ax.hist(all_mae, bins=100, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
ax.axvline(x=-STOP_LOSS_PCT, color='red', linestyle='--', linewidth=2, label=f'SL = -{STOP_LOSS_PCT}%')
ax.set_xlabel('Max Adverse Excursion (%)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Max Adverse Excursion')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-15, 2)

# 2. Stop Loss Trigger Hour
ax = axes[0, 1]
if len(stopped_positions) > 0:
    hours = stopped_positions['stop_hour'].values
    ax.hist(hours, bins=range(25), color='coral', alpha=0.7, edgecolor='black', linewidth=0.5, align='left')
    ax.set_xlabel('Hour of Day (0=Open)')
    ax.set_ylabel('Count')
    ax.set_title('When Stop Loss Triggers (Hour)')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3, axis='y')

# 3. MAE vs Close Return (scatter)
ax = axes[1, 0]
# For longs
if len(longs) > 0:
    ax.scatter(longs['mae'], longs['hourly_return'], alpha=0.15, s=5, color='blue', label='LONG')
# For shorts
if len(shorts) > 0:
    ax.scatter(shorts['mae'], shorts['hourly_return'], alpha=0.15, s=5, color='red', label='SHORT')
ax.axvline(x=-STOP_LOSS_PCT, color='red', linestyle='--', alpha=0.7, label=f'SL = -{STOP_LOSS_PCT}%')
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.set_xlabel('Max Adverse Excursion (%)')
ax.set_ylabel('Final Return (%)')
ax.set_title('MAE vs Final Return')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(-15, 2)

# 4. Daily vs Hourly PnL comparison
ax = axes[1, 1]
ax.scatter(comp_df['daily_return'], comp_df['hourly_return'], alpha=0.1, s=5, color='green')
# Perfect agreement line
lim = max(abs(comp_df['daily_return'].max()), abs(comp_df['hourly_return'].max()))
ax.plot([-lim, lim], [-lim, lim], 'r-', alpha=0.5, linewidth=1, label='Perfect agreement')
ax.set_xlabel('Daily OHLC Return (%)')
ax.set_ylabel('Hourly Return (%)')
ax.set_title('Daily vs Hourly Stop Loss Return')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stop_loss_mechanism_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Chart saved to: stop_loss_mechanism_analysis.png")

# ─────────────────────────────────────────────────────────────
# 9. Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Stop Loss Mechanism Review:
  Method:     Check if intraday HIGH/LOW breaches stop loss threshold
  SL Level:   {STOP_LOSS_PCT}%
  Agreement:  Daily OHLC vs Hourly = {agreement_pct:.1f}%
  
  Daily OHLC Stop Rate:  {daily_stop_rate:.1f}%
  Hourly Stop Rate:      {hourly_stop_rate:.1f}%
  
  Daily OHLC Total PnL:  {daily_total_pnl:+.1f}%
  Hourly Total PnL:      {hourly_total_pnl:+.1f}%
  
  Conclusion: The daily OHLC-based stop loss detection is {"reliable" if agreement_pct > 98 else "mostly reliable but has some discrepancies"}.
  {"The small difference confirms that checking high/low is a good proxy for hourly stop loss simulation." if agreement_pct > 98 else f"There are {disagree} cases where the methods disagree, which could affect backtest accuracy."}
""")

print("=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
