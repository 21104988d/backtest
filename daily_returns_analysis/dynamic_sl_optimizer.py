import pandas as pd
import time

from config import (
    DAILY_OHLC_FILE,
    DEFAULT_N,
    DYNAMIC_MULTIPLIERS,
    INITIAL_CAPITAL,
    MAX_SL_PCT,
    MIN_ASSETS_PER_DAY,
    MIN_SL_PCT,
    POSITION_SIZE_FIXED,
    ROUND_TRIP_FEE_PCT,
)

N = DEFAULT_N
ROUND_TRIP_FEE = ROUND_TRIP_FEE_PCT

print("Loading data...")
t0 = time.time()
ohlc_df = pd.read_csv(DAILY_OHLC_FILE)
ohlc_df['date'] = pd.to_datetime(ohlc_df['date']).dt.date
ohlc_df = ohlc_df.dropna(subset=['daily_return'])
dates = sorted(ohlc_df['date'].unique())
print(f"Data loaded in {time.time() - t0:.2f}s")

# Use dictionaries for O(1) lookup: (date, coin) -> row
# This avoids expensive loops over DataFrame
ohlc_records = ohlc_df.set_index(['date', 'coin']).to_dict('index')

signal_df = ohlc_df[['date', 'coin', 'daily_return']].copy()
# Group by date for O(1) signal access
signals_by_date = {}
for dt, group in signal_df.groupby('date'):
    min_assets_for_day = max(MIN_ASSETS_PER_DAY, N * 2)
    if len(group) >= min_assets_for_day:
        top_n = group.nlargest(N, 'daily_return')[['coin', 'daily_return']].values.tolist()
        bottom_n = group.nsmallest(N, 'daily_return')[['coin', 'daily_return']].values.tolist()
        signals_by_date[dt] = (top_n, bottom_n)

results = []

def calculate_position_return(row, is_long, sl_pct):
    open_price = row['open']
    high_price = row['high']
    low_price = row['low']
    close_price = row['close']
    
    if is_long:
        stop_loss_price = open_price * (1 - sl_pct / 100)
        if low_price <= stop_loss_price:
            return -sl_pct, True
        else:
            return (close_price / open_price - 1) * 100, False
    else:
        stop_loss_price = open_price * (1 + sl_pct / 100)
        if high_price >= stop_loss_price:
            return -sl_pct, True
        else:
            return -(close_price / open_price - 1) * 100, False

def apply_fee(p_ret):
    return p_ret - ROUND_TRIP_FEE


def add_balance_score(dataframe):
    """Add ranking metrics to compare multipliers from multiple angles."""
    dataframe = dataframe.copy()
    max_drawdown_abs = dataframe['Max Drawdown %'].abs().replace(0, 1e-9)
    dataframe['Return x Win'] = dataframe['MR Return %'] * dataframe['MR Win Rate']
    dataframe['Return x Win / DD'] = dataframe['Return x Win'] / max_drawdown_abs
    dataframe['Expectancy x Win'] = dataframe['Expectancy %'] * dataframe['MR Win Rate']
    dataframe['Rank Return x Win'] = dataframe['Return x Win'].rank(ascending=False, method='min').astype(int)
    dataframe['Rank Return x Win / DD'] = dataframe['Return x Win / DD'].rank(ascending=False, method='min').astype(int)
    dataframe['Rank Expectancy x Win'] = dataframe['Expectancy x Win'].rank(ascending=False, method='min').astype(int)
    return dataframe

print("Simulating multipliers...")
for multiplier in DYNAMIC_MULTIPLIERS:
    mr_equity = INITIAL_CAPITAL
    mr_stops = 0
    mr_trades = 0
    mr_wins = 0
    mr_trade_returns = []
    mr_equity_curve = [INITIAL_CAPITAL]
    
    for i in range(1, len(dates)):
        signal_date = dates[i-1]
        trade_date = dates[i]
        
        if signal_date not in signals_by_date:
            continue
            
        top_n, bottom_n = signals_by_date[signal_date]
        mr_batch_returns = []
        
        for coin, t1_return in top_n:
            key = (trade_date, coin)
            if key in ohlc_records:
                row = ohlc_records[key]
                sl_pct = max(MIN_SL_PCT, min(MAX_SL_PCT, abs(t1_return) * multiplier))
                ret, stopped = calculate_position_return(row, False, sl_pct)
                r_fee = apply_fee(ret)
                mr_batch_returns.append(r_fee)
                mr_trade_returns.append(r_fee)
                mr_trades += 1
                if stopped:
                    mr_stops += 1
                if r_fee > 0:
                    mr_wins += 1

        for coin, t1_return in bottom_n:
            key = (trade_date, coin)
            if key in ohlc_records:
                row = ohlc_records[key]
                sl_pct = max(MIN_SL_PCT, min(MAX_SL_PCT, abs(t1_return) * multiplier))
                ret, stopped = calculate_position_return(row, True, sl_pct)
                r_fee = apply_fee(ret)
                mr_batch_returns.append(r_fee)
                mr_trade_returns.append(r_fee)
                mr_trades += 1
                if stopped:
                    mr_stops += 1
                if r_fee > 0:
                    mr_wins += 1
                
        mr_equity += sum([r / 100 * POSITION_SIZE_FIXED for r in mr_batch_returns])
        mr_equity_curve.append(mr_equity)

    mr_return_pct = (mr_equity / INITIAL_CAPITAL - 1) * 100
    mr_win_rate = (mr_wins / mr_trades * 100) if mr_trades > 0 else 0
    mr_stop_rate = (mr_stops / mr_trades * 100) if mr_trades > 0 else 0
    mr_expectancy = sum(mr_trade_returns) / mr_trades if mr_trades > 0 else 0

    equity_series = pd.Series(mr_equity_curve)
    running_peak = equity_series.cummax()
    drawdown_series = (equity_series - running_peak) / running_peak * 100
    mr_max_drawdown = drawdown_series.min()

    results.append({
        'Multiplier': multiplier,
        'MR Return %': mr_return_pct,
        'MR Win Rate': mr_win_rate,
        'MR Stop Rate': mr_stop_rate,
        'Expectancy %': mr_expectancy,
        'Max Drawdown %': mr_max_drawdown,
    })

df = pd.DataFrame(results)
df = add_balance_score(df)
df = df.sort_values(['Return x Win / DD', 'Expectancy x Win'], ascending=False).reset_index(drop=True)

display_df = df.copy()
display_df['Multiplier'] = display_df['Multiplier'].map(lambda value: f"{value:.2f}x")
for column in [
    'MR Return %',
    'MR Win Rate',
    'MR Stop Rate',
    'Expectancy %',
    'Max Drawdown %',
    'Return x Win',
    'Return x Win / DD',
    'Expectancy x Win',
]:
    display_df[column] = display_df[column].map(lambda value: f"{value:.2f}")

best_return = df.loc[df['MR Return %'].idxmax()]
best_return_win = df.loc[df['Return x Win'].idxmax()]
best_return_win_dd = df.loc[df['Return x Win / DD'].idxmax()]
best_expectancy_win = df.loc[df['Expectancy x Win'].idxmax()]

print("\n=== Dynamic Stop-Loss Optimizer Results ===")
try:
    print(display_df.to_markdown(index=False))
except ImportError:
    print(display_df.to_string(index=False))

print("\n=== Suggested Multiplier Selection ===")
print(
    f"Best by return: {best_return['Multiplier']:.2f}x "
    f"(Return {best_return['MR Return %']:.2f}%, Win Rate {best_return['MR Win Rate']:.2f}%)"
)
print(
    f"Best by return x win: {best_return_win['Multiplier']:.2f}x "
    f"(Score {best_return_win['Return x Win']:.2f})"
)
print(
    f"Best by return x win / drawdown: {best_return_win_dd['Multiplier']:.2f}x "
    f"(Score {best_return_win_dd['Return x Win / DD']:.2f})"
)
print(
    f"Best by expectancy x win: {best_expectancy_win['Multiplier']:.2f}x "
    f"(Score {best_expectancy_win['Expectancy x Win']:.2f})"
)

