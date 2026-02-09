import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
N = 3  # Number of top/bottom assets to trade
INITIAL_CAPITAL = 1000  # Starting capital
POSITION_SIZE_FIXED = 100  # Fixed position size per trade in USD
POSITION_FRACTION = 1/6  # Dynamic position size: 1/6 of capital per position
STOP_LOSS_PCT = 0.5  # Stop loss percentage (positive value, e.g. 0.5 = 0.5%)
TRADING_FEE = 0.045  # Trading fee percentage per trade (one-way)
ROUND_TRIP_FEE = TRADING_FEE * 2  # Entry + Exit fee

print("=" * 80)
print("POSITION SIZING COMPARISON: FIXED $100 vs DYNAMIC 1/6 CAPITAL")
print("=" * 80)
print(f"\nStrategy: Mean Reversion & Trend Following")
print(f"  At T=0, identify top {N} gainers and bottom {N} losers")
print(f"  Mean Reversion: SHORT top {N}, LONG bottom {N}")
print(f"  Trend Following: LONG top {N}, SHORT bottom {N}")
print(f"\nSTOP LOSS: {STOP_LOSS_PCT}% per position")
print(f"TRADING FEE: {TRADING_FEE}% per trade ({ROUND_TRIP_FEE}% round-trip)")
print(f"\nPOSITION SIZING MODES:")
print(f"  FIXED: ${POSITION_SIZE_FIXED} per asset (${POSITION_SIZE_FIXED * N * 2} total per day)")
print(f"  DYNAMIC: 1/6 of current capital per position (full capital deployed)")
print(f"\nInitial Capital: ${INITIAL_CAPITAL:,.0f}")

# Read the daily OHLC data (generated from hourly prices)
print("\nLoading daily OHLC data...")
ohlc_df = pd.read_csv(os.path.join(SCRIPT_DIR, 'daily_ohlc.csv'))
ohlc_df['date'] = pd.to_datetime(ohlc_df['date']).dt.date

print(f"Loaded {len(ohlc_df):,} daily OHLC records")

# Drop any NaN or infinite returns
ohlc_df = ohlc_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['daily_return'])

# Get sorted dates
dates = sorted(ohlc_df['date'].unique())
print(f"Data range: {dates[0]} to {dates[-1]}")
print(f"Total trading days: {len(dates)}")

# Backtest results storage
backtest_results = []

# Track equity for dynamic position sizing
fixed_mr_equity = INITIAL_CAPITAL
fixed_tf_equity = INITIAL_CAPITAL
dynamic_mr_equity = INITIAL_CAPITAL
dynamic_tf_equity = INITIAL_CAPITAL

# For each day, we need:
# 1. At day T-1: identify top N and bottom N performers
# 2. At day T: calculate returns from both strategies using OHLC for stop loss

for i in range(1, len(dates)):
    signal_date = dates[i-1]  # Day we observe (T=0)
    trade_date = dates[i]     # Day we trade (T=1)
    
    # Get returns on signal day (T=0)
    signal_day_data = ohlc_df[ohlc_df['date'] == signal_date].copy()
    
    if len(signal_day_data) < N * 2:
        continue  # Not enough assets
    
    # Identify top N gainers and bottom N losers on signal day
    top_n = signal_day_data.nlargest(N, 'daily_return')[['coin', 'daily_return']].values.tolist()
    bottom_n = signal_day_data.nsmallest(N, 'daily_return')[['coin', 'daily_return']].values.tolist()
    
    top_coins = [x[0] for x in top_n]
    bottom_coins = [x[0] for x in bottom_n]
    
    # Get OHLC data on trade day (T=1)
    trade_day_data = ohlc_df[ohlc_df['date'] == trade_date].copy()
    
    # Helper function to calculate position return with proper stop loss using high/low
    def calculate_position_return(coin, is_long, trade_data, sl_pct):
        """
        Calculate position return with stop loss based on intraday high/low.
        
        Args:
            sl_pct: Positive stop loss percentage (e.g. 0.5 means 0.5%)
        
        For LONG position:
        - Stop loss price = open * (1 - sl_pct/100)
        - Triggered if low <= stop_loss_price
        - If triggered, return = -sl_pct
        
        For SHORT position:
        - Stop loss price = open * (1 + sl_pct/100)
        - Triggered if high >= stop_loss_price
        - If triggered, return = -sl_pct
        
        Note: With daily OHLC, we cannot determine whether the high or low
        was reached first within the day. We conservatively assume stop loss
        triggers whenever the adverse price level was touched.
        """
        coin_data = trade_data[trade_data['coin'] == coin]
        if len(coin_data) == 0:
            return None, False
        
        row = coin_data.iloc[0]
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        
        if is_long:
            # Long position: buy at open, stop loss if price drops
            stop_loss_price = open_price * (1 - sl_pct / 100)
            
            if low_price <= stop_loss_price:
                # Stop loss triggered
                return -sl_pct, True
            else:
                # No stop loss, use close price
                return (close_price / open_price - 1) * 100, False
        else:
            # Short position: sell at open, stop loss if price rises
            stop_loss_price = open_price * (1 + sl_pct / 100)
            
            if high_price >= stop_loss_price:
                # Stop loss triggered
                return -sl_pct, True
            else:
                # No stop loss, use close price (profit when price goes down)
                return -(close_price / open_price - 1) * 100, False
    
    # Apply trading fee (round-trip: entry + exit)
    def apply_fee(position_return, fee):
        """Subtract round-trip trading fee from position return"""
        return position_return - fee
    
    # MEAN REVERSION: Short top, Long bottom
    mr_short_returns = []
    mr_short_stops = 0
    for coin in top_coins:
        ret, stopped = calculate_position_return(coin, is_long=False, trade_data=trade_day_data, sl_pct=STOP_LOSS_PCT)
        if ret is not None:
            ret_with_fee = apply_fee(ret, ROUND_TRIP_FEE)
            mr_short_returns.append(ret_with_fee)
            if stopped:
                mr_short_stops += 1
    
    mr_long_returns = []
    mr_long_stops = 0
    for coin in bottom_coins:
        ret, stopped = calculate_position_return(coin, is_long=True, trade_data=trade_day_data, sl_pct=STOP_LOSS_PCT)
        if ret is not None:
            ret_with_fee = apply_fee(ret, ROUND_TRIP_FEE)
            mr_long_returns.append(ret_with_fee)
            if stopped:
                mr_long_stops += 1
    
    # Calculate PnL in dollars - FIXED position size
    mr_short_pnl_fixed = sum([ret / 100 * POSITION_SIZE_FIXED for ret in mr_short_returns])
    mr_long_pnl_fixed = sum([ret / 100 * POSITION_SIZE_FIXED for ret in mr_long_returns])
    mr_total_pnl_fixed = mr_short_pnl_fixed + mr_long_pnl_fixed
    
    # Calculate PnL in dollars - DYNAMIC position size (1/6 of current capital)
    dynamic_mr_position_size = dynamic_mr_equity * POSITION_FRACTION
    mr_short_pnl_dynamic = sum([ret / 100 * dynamic_mr_position_size for ret in mr_short_returns])
    mr_long_pnl_dynamic = sum([ret / 100 * dynamic_mr_position_size for ret in mr_long_returns])
    mr_total_pnl_dynamic = mr_short_pnl_dynamic + mr_long_pnl_dynamic
    
    mr_short_return = np.mean(mr_short_returns) if mr_short_returns else 0
    mr_long_return = np.mean(mr_long_returns) if mr_long_returns else 0
    mr_combined = (mr_short_return + mr_long_return) / 2
    mr_stops = mr_short_stops + mr_long_stops
    
    # TREND FOLLOWING: Long top, Short bottom
    tf_long_returns = []
    tf_long_stops = 0
    for coin in top_coins:
        ret, stopped = calculate_position_return(coin, is_long=True, trade_data=trade_day_data, sl_pct=STOP_LOSS_PCT)
        if ret is not None:
            ret_with_fee = apply_fee(ret, ROUND_TRIP_FEE)
            tf_long_returns.append(ret_with_fee)
            if stopped:
                tf_long_stops += 1
    
    tf_short_returns = []
    tf_short_stops = 0
    for coin in bottom_coins:
        ret, stopped = calculate_position_return(coin, is_long=False, trade_data=trade_day_data, sl_pct=STOP_LOSS_PCT)
        if ret is not None:
            ret_with_fee = apply_fee(ret, ROUND_TRIP_FEE)
            tf_short_returns.append(ret_with_fee)
            if stopped:
                tf_short_stops += 1
    
    # Calculate PnL in dollars - FIXED position size
    tf_long_pnl_fixed = sum([ret / 100 * POSITION_SIZE_FIXED for ret in tf_long_returns])
    tf_short_pnl_fixed = sum([ret / 100 * POSITION_SIZE_FIXED for ret in tf_short_returns])
    tf_total_pnl_fixed = tf_long_pnl_fixed + tf_short_pnl_fixed
    
    # Calculate PnL in dollars - DYNAMIC position size (1/6 of current capital)
    dynamic_tf_position_size = dynamic_tf_equity * POSITION_FRACTION
    tf_long_pnl_dynamic = sum([ret / 100 * dynamic_tf_position_size for ret in tf_long_returns])
    tf_short_pnl_dynamic = sum([ret / 100 * dynamic_tf_position_size for ret in tf_short_returns])
    tf_total_pnl_dynamic = tf_long_pnl_dynamic + tf_short_pnl_dynamic
    
    tf_long_return = np.mean(tf_long_returns) if tf_long_returns else 0
    tf_short_return = np.mean(tf_short_returns) if tf_short_returns else 0
    tf_combined = (tf_long_return + tf_short_return) / 2
    tf_stops = tf_long_stops + tf_short_stops
    
    # Store individual position returns for detailed analysis
    all_mr_returns = mr_short_returns + mr_long_returns
    all_tf_returns = tf_long_returns + tf_short_returns
    
    # Update equity for dynamic position sizing
    fixed_mr_equity += mr_total_pnl_fixed
    fixed_tf_equity += tf_total_pnl_fixed
    dynamic_mr_equity += mr_total_pnl_dynamic
    dynamic_tf_equity += tf_total_pnl_dynamic
    
    backtest_results.append({
        'signal_date': signal_date,
        'trade_date': trade_date,
        'top_coins': top_coins,
        'bottom_coins': bottom_coins,
        # Mean Reversion
        'mr_short_return': mr_short_return,
        'mr_long_return': mr_long_return,
        'mr_combined': mr_combined,
        'mr_pnl_fixed': mr_total_pnl_fixed,
        'mr_pnl_dynamic': mr_total_pnl_dynamic,
        'mr_equity_fixed': fixed_mr_equity,
        'mr_equity_dynamic': dynamic_mr_equity,
        'mr_stops': mr_stops,
        'mr_all_returns': all_mr_returns,
        # Trend Following
        'tf_long_return': tf_long_return,
        'tf_short_return': tf_short_return,
        'tf_combined': tf_combined,
        'tf_pnl_fixed': tf_total_pnl_fixed,
        'tf_pnl_dynamic': tf_total_pnl_dynamic,
        'tf_equity_fixed': fixed_tf_equity,
        'tf_equity_dynamic': dynamic_tf_equity,
        'tf_stops': tf_stops,
        'tf_all_returns': all_tf_returns
    })

# Convert to DataFrame
results_df = pd.DataFrame(backtest_results)

# Cumulative PnL for fixed position size (already tracked in equity columns)
results_df['mr_cum_pnl_fixed'] = results_df['mr_pnl_fixed'].cumsum()
results_df['tf_cum_pnl_fixed'] = results_df['tf_pnl_fixed'].cumsum()

# For legacy compatibility
results_df['mr_equity'] = results_df['mr_equity_fixed']
results_df['tf_equity'] = results_df['tf_equity_fixed']

# For cumulative return tracking (for charts)
results_df['mr_cum_combined_fixed'] = results_df['mr_equity_fixed'] / INITIAL_CAPITAL
results_df['tf_cum_combined_fixed'] = results_df['tf_equity_fixed'] / INITIAL_CAPITAL
results_df['mr_cum_combined_dynamic'] = results_df['mr_equity_dynamic'] / INITIAL_CAPITAL
results_df['tf_cum_combined_dynamic'] = results_df['tf_equity_dynamic'] / INITIAL_CAPITAL

# Also track leg performance
results_df['mr_cum_short'] = 1 + results_df['mr_short_return'].cumsum() / 100
results_df['mr_cum_long'] = 1 + results_df['mr_long_return'].cumsum() / 100
results_df['tf_cum_long'] = 1 + results_df['tf_long_return'].cumsum() / 100
results_df['tf_cum_short'] = 1 + results_df['tf_short_return'].cumsum() / 100

# Save results to CSV
results_df.to_csv(os.path.join(SCRIPT_DIR, 'strategy_comparison_results.csv'), index=False)

# Flatten all individual position returns for detailed analysis
mr_all_position_returns = []
tf_all_position_returns = []
for _, row in results_df.iterrows():
    mr_all_position_returns.extend(row['mr_all_returns'])
    tf_all_position_returns.extend(row['tf_all_returns'])

mr_all_returns = pd.Series(mr_all_position_returns)
tf_all_returns = pd.Series(tf_all_position_returns)

# Performance Statistics
print("\n" + "=" * 80)
print("PERFORMANCE STATISTICS")
print("=" * 80)

def calculate_detailed_stats(daily_pnl, all_position_returns, name, initial_capital, position_size, n_assets, is_dynamic=False, equity_series=None):
    """Calculate detailed performance statistics"""
    pnl_series = pd.Series(daily_pnl)
    returns_series = pd.Series(all_position_returns)
    
    # Basic stats
    total_pnl = pnl_series.sum()
    
    # For dynamic sizing, use final equity vs initial
    if is_dynamic and equity_series is not None:
        final_equity = equity_series.iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital * 100
    else:
        total_return = total_pnl / initial_capital * 100
    
    # Calculate annualized return
    trading_days = len(pnl_series)
    
    if is_dynamic and equity_series is not None:
        # For dynamic sizing, calculate CAGR
        years = trading_days / 365
        final_equity = equity_series.iloc[-1]
        cagr = ((final_equity / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        annualized_return = cagr
    else:
        daily_pnl_avg = pnl_series.mean()
        annualized_pnl = daily_pnl_avg * 365
        annualized_return = annualized_pnl / initial_capital * 100
    
    # Volatility
    if is_dynamic and equity_series is not None:
        daily_return_pct = equity_series.pct_change().dropna() * 100
    else:
        daily_return_pct = pnl_series / (position_size * n_assets * 2) * 100
    volatility = daily_return_pct.std() * np.sqrt(365)
    
    # Sharpe ratio (assuming 0 risk-free rate)
    sharpe = (daily_return_pct.mean() * 365) / volatility if volatility > 0 else 0
    
    # Max drawdown (based on equity)
    if is_dynamic and equity_series is not None:
        equity = equity_series
    else:
        cum_pnl = pnl_series.cumsum()
        equity = initial_capital + cum_pnl
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()
    
    # Win/Loss statistics (daily)
    daily_wins = (pnl_series > 0).sum()
    daily_losses = (pnl_series < 0).sum()
    daily_breakeven = (pnl_series == 0).sum()
    daily_win_rate = daily_wins / len(pnl_series) * 100
    
    # Win/Loss statistics (per position)
    position_wins = (returns_series > 0).sum()
    position_losses = (returns_series < 0).sum()
    position_breakeven = (returns_series == 0).sum()
    position_win_rate = position_wins / len(returns_series) * 100
    
    # Average profit/loss per position
    winning_returns = returns_series[returns_series > 0]
    losing_returns = returns_series[returns_series < 0]
    
    avg_profit_pct = winning_returns.mean() if len(winning_returns) > 0 else 0
    avg_loss_pct = losing_returns.mean() if len(losing_returns) > 0 else 0
    
    avg_profit_dollar = avg_profit_pct / 100 * position_size
    avg_loss_dollar = avg_loss_pct / 100 * position_size
    
    # Profit factor
    gross_profit = (returns_series[returns_series > 0] / 100 * position_size).sum()
    gross_loss = abs((returns_series[returns_series < 0] / 100 * position_size).sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Expectancy per trade
    expectancy_pct = (position_win_rate/100 * avg_profit_pct) + ((1 - position_win_rate/100) * avg_loss_pct)
    expectancy_dollar = expectancy_pct / 100 * position_size
    
    # Best and worst trades
    best_trade = returns_series.max()
    worst_trade = returns_series.min()
    
    # Consecutive wins/losses
    daily_win_loss = (pnl_series > 0).astype(int)
    
    return {
        'name': name,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'daily_win_rate': daily_win_rate,
        'daily_wins': daily_wins,
        'daily_losses': daily_losses,
        'position_win_rate': position_win_rate,
        'position_wins': position_wins,
        'position_losses': position_losses,
        'avg_profit_pct': avg_profit_pct,
        'avg_loss_pct': avg_loss_pct,
        'avg_profit_dollar': avg_profit_dollar,
        'avg_loss_dollar': avg_loss_dollar,
        'profit_factor': profit_factor,
        'expectancy_pct': expectancy_pct,
        'expectancy_dollar': expectancy_dollar,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'total_trades': len(returns_series),
        'trading_days': trading_days
    }

# Calculate stats for all 4 variants
mr_stats_fixed = calculate_detailed_stats(
    results_df['mr_pnl_fixed'].tolist(), 
    mr_all_position_returns, 
    "Mean Reversion (Fixed $100)",
    INITIAL_CAPITAL, POSITION_SIZE_FIXED, N
)

mr_stats_dynamic = calculate_detailed_stats(
    results_df['mr_pnl_dynamic'].tolist(), 
    mr_all_position_returns, 
    "Mean Reversion (Dynamic 1/6)",
    INITIAL_CAPITAL, POSITION_SIZE_FIXED, N,
    is_dynamic=True, equity_series=results_df['mr_equity_dynamic']
)

tf_stats_fixed = calculate_detailed_stats(
    results_df['tf_pnl_fixed'].tolist(), 
    tf_all_position_returns, 
    "Trend Following (Fixed $100)",
    INITIAL_CAPITAL, POSITION_SIZE_FIXED, N
)

tf_stats_dynamic = calculate_detailed_stats(
    results_df['tf_pnl_dynamic'].tolist(), 
    tf_all_position_returns, 
    "Trend Following (Dynamic 1/6)",
    INITIAL_CAPITAL, POSITION_SIZE_FIXED, N,
    is_dynamic=True, equity_series=results_df['tf_equity_dynamic']
)

def print_strategy_stats(stats):
    print(f"\n{'='*60}")
    print(f" {stats['name'].upper()}")
    print(f"{'='*60}")
    
    print(f"\n--- RETURN METRICS ---")
    print(f"  Total PnL:            ${stats['total_pnl']:+,.2f}")
    print(f"  Total Return:         {stats['total_return']:+.2f}%")
    print(f"  Annualized Return:    {stats['annualized_return']:+.2f}%")
    print(f"  Final Equity:         ${INITIAL_CAPITAL + stats['total_pnl']:,.2f}")
    
    print(f"\n--- RISK METRICS ---")
    print(f"  Volatility (Ann):     {stats['volatility']:.2f}%")
    print(f"  Sharpe Ratio:         {stats['sharpe']:.3f}")
    print(f"  Max Drawdown:         {stats['max_drawdown']:.2f}%")
    
    print(f"\n--- WIN/LOSS ANALYSIS (Daily) ---")
    print(f"  Winning Days:         {stats['daily_wins']} ({stats['daily_win_rate']:.1f}%)")
    print(f"  Losing Days:          {stats['daily_losses']} ({100-stats['daily_win_rate']:.1f}%)")
    
    print(f"\n--- WIN/LOSS ANALYSIS (Per Position) ---")
    print(f"  Total Positions:      {stats['total_trades']}")
    print(f"  Winning Positions:    {stats['position_wins']} ({stats['position_win_rate']:.1f}%)")
    print(f"  Losing Positions:     {stats['position_losses']} ({100-stats['position_win_rate']:.1f}%)")
    
    print(f"\n--- PROFIT/LOSS METRICS ---")
    print(f"  Avg Profit (Win):     {stats['avg_profit_pct']:+.2f}% (${stats['avg_profit_dollar']:+.2f})")
    print(f"  Avg Loss (Loss):      {stats['avg_loss_pct']:+.2f}% (${stats['avg_loss_dollar']:+.2f})")
    print(f"  Best Trade:           {stats['best_trade']:+.2f}%")
    print(f"  Worst Trade:          {stats['worst_trade']:+.2f}%")
    
    print(f"\n--- PERFORMANCE RATIOS ---")
    print(f"  Profit Factor:        {stats['profit_factor']:.2f}")
    print(f"  Expectancy/Trade:     {stats['expectancy_pct']:+.3f}% (${stats['expectancy_dollar']:+.3f})")
    print(f"  Gross Profit:         ${stats['gross_profit']:,.2f}")
    print(f"  Gross Loss:           ${stats['gross_loss']:,.2f}")

# Print all 4 variants
print("\n" + "=" * 80)
print("FIXED POSITION SIZING ($100 per position)")
print("=" * 80)
print_strategy_stats(mr_stats_fixed)
print_strategy_stats(tf_stats_fixed)

print("\n" + "=" * 80)
print("DYNAMIC POSITION SIZING (1/6 of capital per position)")
print("=" * 80)
print_strategy_stats(mr_stats_dynamic)
print_strategy_stats(tf_stats_dynamic)

# Stop loss statistics
total_mr_stops = results_df['mr_stops'].sum()
total_tf_stops = results_df['tf_stops'].sum()
total_positions = len(results_df) * N * 2  # N long + N short per day

# Calculate total fees paid (for fixed sizing)
total_fees_paid_fixed = total_positions * ROUND_TRIP_FEE / 100 * POSITION_SIZE_FIXED

print(f"\n" + "=" * 80)
print("STOP LOSS & FEE STATISTICS (Using OHLC High/Low for Stop Loss)")
print("=" * 80)
print(f"\nStop Loss Level: {STOP_LOSS_PCT}% (triggered based on intraday high/low)")
print(f"Trading Fee: {TRADING_FEE}% per trade ({ROUND_TRIP_FEE}% round-trip)")
print(f"Total Positions Traded: {total_positions}")
print(f"Total Fees Paid (Fixed): ${total_fees_paid_fixed:,.2f}")
print(f"Fee per Position: ${ROUND_TRIP_FEE / 100 * POSITION_SIZE_FIXED:.2f}")
print(f"\nMean Reversion:")
print(f"  Stop Losses Triggered: {total_mr_stops} ({total_mr_stops/total_positions*100:.1f}%)")
print(f"\nTrend Following:")
print(f"  Stop Losses Triggered: {total_tf_stops} ({total_tf_stops/total_positions*100:.1f}%)")

# ============================================================================
# GENERATE CHARTS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING CHARTS...")
print("=" * 80)

# Convert dates for plotting
results_df['trade_date'] = pd.to_datetime(results_df['trade_date'])

# Create figure with multiple subplots - Strategy Comparison
fig = plt.figure(figsize=(18, 22))

# 1. Main Strategy Comparison - Equity Curves (Fixed vs Dynamic)
ax1 = fig.add_subplot(4, 2, 1)
ax1.plot(results_df['trade_date'], results_df['mr_equity_fixed'], label='MR Fixed $100', color='red', linewidth=2)
ax1.plot(results_df['trade_date'], results_df['tf_equity_fixed'], label='TF Fixed $100', color='green', linewidth=2)
ax1.plot(results_df['trade_date'], results_df['mr_equity_dynamic'], label='MR Dynamic 1/6', color='darkred', linewidth=1.5, linestyle='--')
ax1.plot(results_df['trade_date'], results_df['tf_equity_dynamic'], label='TF Dynamic 1/6', color='darkgreen', linewidth=1.5, linestyle='--')
ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.5, label='Initial Capital')
ax1.set_title('Fixed vs Dynamic Position Sizing - Equity Curves', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Portfolio Value ($)')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)
ax1.set_yscale('log')

# 2. Cumulative Returns Comparison
ax2 = fig.add_subplot(4, 2, 2)
ax2.plot(results_df['trade_date'], (results_df['mr_cum_combined_fixed'] - 1) * 100, label='MR Fixed', color='red', linewidth=2)
ax2.plot(results_df['trade_date'], (results_df['tf_cum_combined_fixed'] - 1) * 100, label='TF Fixed', color='green', linewidth=2)
ax2.plot(results_df['trade_date'], (results_df['mr_cum_combined_dynamic'] - 1) * 100, label='MR Dynamic', color='darkred', linewidth=1.5, linestyle='--')
ax2.plot(results_df['trade_date'], (results_df['tf_cum_combined_dynamic'] - 1) * 100, label='TF Dynamic', color='darkgreen', linewidth=1.5, linestyle='--')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_title('Cumulative Returns Comparison (%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Cumulative Return (%)')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 3. Mean Reversion Legs
ax3 = fig.add_subplot(4, 2, 3)
ax3.plot(results_df['trade_date'], (results_df['mr_cum_short'] - 1) * 100, label='Short Winners', color='darkred', linewidth=1.5)
ax3.plot(results_df['trade_date'], (results_df['mr_cum_long'] - 1) * 100, label='Long Losers', color='salmon', linewidth=1.5)
ax3.plot(results_df['trade_date'], (results_df['mr_cum_combined_fixed'] - 1) * 100, label='Combined', color='red', linewidth=2)
ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax3.set_title('Mean Reversion - Leg Performance (%)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Cumulative Return (%)')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# 4. Trend Following Legs
ax4 = fig.add_subplot(4, 2, 4)
ax4.plot(results_df['trade_date'], (results_df['tf_cum_long'] - 1) * 100, label='Long Winners', color='darkgreen', linewidth=1.5)
ax4.plot(results_df['trade_date'], (results_df['tf_cum_short'] - 1) * 100, label='Short Losers', color='lightgreen', linewidth=1.5)
ax4.plot(results_df['trade_date'], (results_df['tf_cum_combined_fixed'] - 1) * 100, label='Combined', color='green', linewidth=2)
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.set_title('Trend Following - Leg Performance (%)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Date')
ax4.set_ylabel('Cumulative Return (%)')
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# 5. Drawdown Comparison (Fixed vs Dynamic)
ax5 = fig.add_subplot(4, 2, 5)
# Mean Reversion drawdown - Fixed
mr_cum_fixed = results_df['mr_cum_combined_fixed']
mr_rolling_max_fixed = mr_cum_fixed.expanding().max()
mr_drawdowns_fixed = (mr_cum_fixed / mr_rolling_max_fixed - 1) * 100
# Mean Reversion drawdown - Dynamic
mr_cum_dynamic = results_df['mr_cum_combined_dynamic']
mr_rolling_max_dynamic = mr_cum_dynamic.expanding().max()
mr_drawdowns_dynamic = (mr_cum_dynamic / mr_rolling_max_dynamic - 1) * 100

ax5.fill_between(results_df['trade_date'], mr_drawdowns_fixed, 0, color='red', alpha=0.3, label='MR Fixed')
ax5.fill_between(results_df['trade_date'], mr_drawdowns_dynamic, 0, color='darkred', alpha=0.2, label='MR Dynamic')
ax5.plot(results_df['trade_date'], mr_drawdowns_fixed, color='red', linewidth=1)
ax5.plot(results_df['trade_date'], mr_drawdowns_dynamic, color='darkred', linewidth=1, linestyle='--')
ax5.set_title('Mean Reversion Drawdown - Fixed vs Dynamic', fontsize=14, fontweight='bold')
ax5.set_xlabel('Date')
ax5.set_ylabel('Drawdown (%)')
ax5.legend(loc='best')
ax5.grid(True, alpha=0.3)
ax5.tick_params(axis='x', rotation=45)

# 6. Daily Returns Distribution Comparison
ax6 = fig.add_subplot(4, 2, 6)
ax6.hist(results_df['mr_combined'], bins=50, color='red', alpha=0.5, edgecolor='black', label=f"MR (mean: {results_df['mr_combined'].mean():.2f}%)")
ax6.hist(results_df['tf_combined'], bins=50, color='green', alpha=0.5, edgecolor='black', label=f"TF (mean: {results_df['tf_combined'].mean():.2f}%)")
ax6.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax6.set_title('Daily Returns Distribution Comparison', fontsize=14, fontweight='bold')
ax6.set_xlabel('Daily Return (%)')
ax6.set_ylabel('Frequency')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Rolling 30-Day Returns Comparison
rolling_window = 30
results_df['mr_rolling_30d'] = results_df['mr_combined'].rolling(window=rolling_window).sum()
results_df['tf_rolling_30d'] = results_df['tf_combined'].rolling(window=rolling_window).sum()

ax7 = fig.add_subplot(4, 2, 7)
ax7.plot(results_df['trade_date'], results_df['mr_rolling_30d'], color='red', linewidth=1.5, label='Mean Reversion')
ax7.plot(results_df['trade_date'], results_df['tf_rolling_30d'], color='green', linewidth=1.5, label='Trend Following')
ax7.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax7.set_title(f'Rolling {rolling_window}-Day Return Comparison (%)', fontsize=14, fontweight='bold')
ax7.set_xlabel('Date')
ax7.set_ylabel('Rolling Return (%)')
ax7.legend()
ax7.grid(True, alpha=0.3)
ax7.tick_params(axis='x', rotation=45)

# 8. Performance Summary Table
ax8 = fig.add_subplot(4, 2, 8)
ax8.axis('off')

summary_data = [
    ['Metric', 'MR Fixed', 'MR Dynamic', 'TF Fixed', 'TF Dynamic'],
    ['Total PnL', f"${mr_stats_fixed['total_pnl']:+,.0f}", f"${mr_stats_dynamic['total_pnl']:+,.0f}", 
     f"${tf_stats_fixed['total_pnl']:+,.0f}", f"${tf_stats_dynamic['total_pnl']:+,.0f}"],
    ['Total Return', f"{mr_stats_fixed['total_return']:+.1f}%", f"{mr_stats_dynamic['total_return']:+.1f}%",
     f"{tf_stats_fixed['total_return']:+.1f}%", f"{tf_stats_dynamic['total_return']:+.1f}%"],
    ['Sharpe Ratio', f"{mr_stats_fixed['sharpe']:.2f}", f"{mr_stats_dynamic['sharpe']:.2f}",
     f"{tf_stats_fixed['sharpe']:.2f}", f"{tf_stats_dynamic['sharpe']:.2f}"],
    ['Max Drawdown', f"{mr_stats_fixed['max_drawdown']:.1f}%", f"{mr_stats_dynamic['max_drawdown']:.1f}%",
     f"{tf_stats_fixed['max_drawdown']:.1f}%", f"{tf_stats_dynamic['max_drawdown']:.1f}%"],
    ['Profit Factor', f"{mr_stats_fixed['profit_factor']:.2f}", f"{mr_stats_dynamic['profit_factor']:.2f}",
     f"{tf_stats_fixed['profit_factor']:.2f}", f"{tf_stats_dynamic['profit_factor']:.2f}"],
]

table = ax8.table(cellText=summary_data[1:], colLabels=summary_data[0], 
                  loc='center', cellLoc='center',
                  colColours=['#4472C4', '#C65911', '#8B0000', '#548235', '#006400'],
                  colWidths=[0.22, 0.19, 0.19, 0.19, 0.19])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)

# Color the header row
for i in range(5):
    table[(0, i)].set_text_props(color='white', fontweight='bold')

ax8.set_title('Performance Summary - Fixed vs Dynamic Position Sizing', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'strategy_comparison_performance.png'), dpi=150, bbox_inches='tight', facecolor='white')
print("\nMain comparison chart saved to: strategy_comparison_performance.png")

# Additional chart: Strategy comparison with buy-and-hold benchmark
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Get BTC as benchmark if available
btc_ohlc = ohlc_df[ohlc_df['coin'] == 'BTC'].copy()
if len(btc_ohlc) > 0:
    btc_ohlc['date'] = pd.to_datetime(btc_ohlc['date'])
    btc_ohlc = btc_ohlc[btc_ohlc['date'].isin(results_df['trade_date'])]
    
    if len(btc_ohlc) > 0:
        btc_ohlc = btc_ohlc.sort_values('date')
        btc_ohlc['cum_return'] = (1 + btc_ohlc['daily_return']/100).cumprod()
        
        # Strategy vs BTC
        ax_comp = axes[0, 0]
        ax_comp.plot(results_df['trade_date'], results_df['mr_cum_combined_fixed'], 
                    label='MR Fixed', color='red', linewidth=2)
        ax_comp.plot(results_df['trade_date'], results_df['mr_cum_combined_dynamic'], 
                    label='MR Dynamic', color='darkred', linewidth=1.5, linestyle='--')
        ax_comp.plot(btc_ohlc['date'], btc_ohlc['cum_return'], 
                    label='BTC Buy & Hold', color='orange', linewidth=2, linestyle=':')
        ax_comp.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax_comp.set_title('Mean Reversion Fixed vs Dynamic vs BTC', fontsize=14, fontweight='bold')
        ax_comp.set_xlabel('Date')
        ax_comp.set_ylabel('Cumulative Return (1 = Initial)')
        ax_comp.legend()
        ax_comp.grid(True, alpha=0.3)
        ax_comp.tick_params(axis='x', rotation=45)
        ax_comp.set_yscale('log')

# Monthly returns comparison
ax_monthly = axes[0, 1]
results_df['year_month'] = results_df['trade_date'].dt.to_period('M')
mr_monthly = results_df.groupby('year_month')['mr_combined'].sum()
tf_monthly = results_df.groupby('year_month')['tf_combined'].sum()

x = np.arange(len(mr_monthly))
width = 0.35
ax_monthly.bar(x - width/2, mr_monthly.values, width, label='Mean Reversion', color='red', alpha=0.7)
ax_monthly.bar(x + width/2, tf_monthly.values, width, label='Trend Following', color='green', alpha=0.7)
ax_monthly.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax_monthly.set_title('Monthly Returns Comparison', fontsize=14, fontweight='bold')
ax_monthly.set_xlabel('Month')
ax_monthly.set_ylabel('Monthly Return (%)')
ax_monthly.legend()
ax_monthly.grid(True, alpha=0.3, axis='y')

# Show only some month labels
tick_positions = list(range(0, len(mr_monthly), max(1, len(mr_monthly)//8)))
tick_labels = [str(mr_monthly.index[i]) for i in tick_positions]
ax_monthly.set_xticks(tick_positions)
ax_monthly.set_xticklabels(tick_labels, rotation=45, ha='right')

# Win rate by strategy over time (rolling)
ax_winrate = axes[1, 0]
window = 60
results_df['mr_win'] = (results_df['mr_combined'] > 0).astype(int)
results_df['tf_win'] = (results_df['tf_combined'] > 0).astype(int)
results_df['mr_rolling_winrate'] = results_df['mr_win'].rolling(window=window).mean() * 100
results_df['tf_rolling_winrate'] = results_df['tf_win'].rolling(window=window).mean() * 100

ax_winrate.plot(results_df['trade_date'], results_df['mr_rolling_winrate'], 
               label='Mean Reversion', color='red', linewidth=1.5)
ax_winrate.plot(results_df['trade_date'], results_df['tf_rolling_winrate'], 
               label='Trend Following', color='green', linewidth=1.5)
ax_winrate.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Line')
ax_winrate.set_title(f'Rolling {window}-Day Win Rate (%)', fontsize=14, fontweight='bold')
ax_winrate.set_xlabel('Date')
ax_winrate.set_ylabel('Win Rate (%)')
ax_winrate.legend()
ax_winrate.grid(True, alpha=0.3)
ax_winrate.tick_params(axis='x', rotation=45)
ax_winrate.set_ylim(30, 70)

# Scatter: MR vs TF daily returns
ax_scatter = axes[1, 1]
ax_scatter.scatter(results_df['mr_combined'], results_df['tf_combined'], alpha=0.3, s=10)
ax_scatter.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax_scatter.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
# Add diagonal line (y = -x, perfect negative correlation)
lim = max(abs(results_df['mr_combined'].max()), abs(results_df['tf_combined'].max()))
ax_scatter.plot([-lim, lim], [lim, -lim], 'r--', alpha=0.5, label='Perfect Neg. Corr.')
ax_scatter.set_title('Mean Reversion vs Trend Following Daily Returns', fontsize=14, fontweight='bold')
ax_scatter.set_xlabel('Mean Reversion Return (%)')
ax_scatter.set_ylabel('Trend Following Return (%)')
ax_scatter.grid(True, alpha=0.3)

# Calculate correlation
corr = results_df['mr_combined'].corr(results_df['tf_combined'])
ax_scatter.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax_scatter.transAxes, 
               fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'strategy_comparison_analysis.png'), dpi=150, bbox_inches='tight', facecolor='white')
print("Analysis chart saved to: strategy_comparison_analysis.png")

print("\n" + "=" * 80)
print("BACKTEST COMPLETE!")
print("=" * 80)
print(f"\nFiles generated:")
print(f"  1. strategy_comparison_results.csv - Daily trade results")
print(f"  2. strategy_comparison_performance.png - Main performance charts")
print(f"  3. strategy_comparison_analysis.png - Additional analysis charts")

# Show some sample trades
print("\n" + "=" * 80)
print("SAMPLE TRADES (First 10 days)")
print("=" * 80)
for _, row in results_df.head(10).iterrows():
    print(f"\nSignal: {row['signal_date']} -> Trade: {row['trade_date']}")
    print(f"  Top coins: {row['top_coins']}, Bottom coins: {row['bottom_coins']}")
    print(f"  Mean Reversion: {row['mr_combined']:+.2f}% (Short top: {row['mr_short_return']:+.2f}%, Long bottom: {row['mr_long_return']:+.2f}%)")
    print(f"  Trend Following: {row['tf_combined']:+.2f}% (Long top: {row['tf_long_return']:+.2f}%, Short bottom: {row['tf_short_return']:+.2f}%)")

# Summary comparison - Fixed vs Dynamic
print("\n" + "=" * 80)
print("POSITION SIZING COMPARISON SUMMARY")
print("=" * 80)
print(f"\n{'Strategy':<30} {'Final Equity':>15} {'Total Return':>15} {'Max DD':>10} {'Sharpe':>10}")
print("-" * 85)
print(f"{'MR Fixed $100':<30} ${mr_stats_fixed['total_pnl']+INITIAL_CAPITAL:>14,.0f} {mr_stats_fixed['total_return']:>+14.1f}% {mr_stats_fixed['max_drawdown']:>+9.1f}% {mr_stats_fixed['sharpe']:>9.2f}")
print(f"{'MR Dynamic 1/6 Capital':<30} ${mr_stats_dynamic['total_pnl']+INITIAL_CAPITAL:>14,.0f} {mr_stats_dynamic['total_return']:>+14.1f}% {mr_stats_dynamic['max_drawdown']:>+9.1f}% {mr_stats_dynamic['sharpe']:>9.2f}")
print(f"{'TF Fixed $100':<30} ${tf_stats_fixed['total_pnl']+INITIAL_CAPITAL:>14,.0f} {tf_stats_fixed['total_return']:>+14.1f}% {tf_stats_fixed['max_drawdown']:>+9.1f}% {tf_stats_fixed['sharpe']:>9.2f}")
print(f"{'TF Dynamic 1/6 Capital':<30} ${tf_stats_dynamic['total_pnl']+INITIAL_CAPITAL:>14,.0f} {tf_stats_dynamic['total_return']:>+14.1f}% {tf_stats_dynamic['max_drawdown']:>+9.1f}% {tf_stats_dynamic['sharpe']:>9.2f}")

print("\n" + "=" * 80)
print("KEY INSIGHT: Dynamic Position Sizing (1/6 Capital)")
print("=" * 80)
print(f"\nDynamic sizing multiplies returns through compounding:")
print(f"  - MR Fixed: {mr_stats_fixed['total_return']:+.1f}%  -->  MR Dynamic: {mr_stats_dynamic['total_return']:+.1f}%  ({mr_stats_dynamic['total_return']/mr_stats_fixed['total_return']:.0f}x)")
print(f"  - TF Fixed: {tf_stats_fixed['total_return']:+.1f}%  -->  TF Dynamic: {tf_stats_dynamic['total_return']:+.1f}%  ({tf_stats_dynamic['total_return']/tf_stats_fixed['total_return']:.0f}x)")
print(f"\nDrawdown increases with dynamic sizing:")
print(f"  - MR Fixed: {mr_stats_fixed['max_drawdown']:.1f}%  -->  MR Dynamic: {mr_stats_dynamic['max_drawdown']:.1f}%")
print(f"  - TF Fixed: {tf_stats_fixed['max_drawdown']:.1f}%  -->  TF Dynamic: {tf_stats_dynamic['max_drawdown']:.1f}%")
print(f"\nSharpe ratios remain similar (risk-adjusted return consistency)")

plt.show()
