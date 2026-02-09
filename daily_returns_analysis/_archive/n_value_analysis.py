"""
N-Value Sensitivity Analysis for Mean Reversion Strategy
Test different numbers of top/bottom movers (N) with 0.5% stop loss
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
STOP_LOSS_PCT = 0.5
INITIAL_CAPITAL = 1000
POSITION_SIZE_FIXED = 100
POSITION_FRACTION = 1/6
TRADING_FEE = 0.045
ROUND_TRIP_FEE = TRADING_FEE * 2

# N values to test
N_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]

print("=" * 80)
print("N-VALUE SENSITIVITY ANALYSIS (Mean Reversion, 0.5% Stop Loss)")
print("=" * 80)
print(f"\nTesting N values: {N_VALUES}")
print(f"Stop Loss: {STOP_LOSS_PCT}%")
print(f"Position Sizing: Fixed ${POSITION_SIZE_FIXED} and Dynamic 1/{int(1/POSITION_FRACTION)} Capital")

# Load OHLC data
print("\nLoading daily OHLC data...")
ohlc_df = pd.read_csv('daily_ohlc.csv')
ohlc_df['date'] = pd.to_datetime(ohlc_df['date']).dt.date
ohlc_df = ohlc_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['daily_return'])

dates = sorted(ohlc_df['date'].unique())
print(f"Data range: {dates[0]} to {dates[-1]}")
print(f"Total trading days: {len(dates)}")

# Pre-group data by date for faster lookup
date_groups = {date: group for date, group in ohlc_df.groupby('date')}
# Count available coins per day
coins_per_day = ohlc_df.groupby('date')['coin'].nunique()
print(f"Avg coins per day: {coins_per_day.mean():.0f}, Min: {coins_per_day.min()}, Max: {coins_per_day.max()}")


def calculate_position_return(row, is_long, sl_pct):
    """Calculate return for a single position using OHLC data"""
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


def run_backtest(n):
    """Run mean reversion backtest with N top/bottom movers"""

    fixed_equity = INITIAL_CAPITAL
    dynamic_equity = INITIAL_CAPITAL

    equity_history_fixed = []
    equity_history_dynamic = []
    all_returns = []
    total_stops = 0
    total_positions = 0
    trading_days = 0

    for i in range(1, len(dates)):
        signal_date = dates[i - 1]
        trade_date = dates[i]

        if signal_date not in date_groups or trade_date not in date_groups:
            continue

        signal_day_data = date_groups[signal_date]
        if len(signal_day_data) < n * 2:
            continue

        top_n = signal_day_data.nlargest(n, 'daily_return')['coin'].tolist()
        bottom_n = signal_day_data.nsmallest(n, 'daily_return')['coin'].tolist()

        trade_day_data = date_groups[trade_date]
        trade_day_dict = {row['coin']: row for _, row in trade_day_data.iterrows()}

        day_returns = []

        # Mean Reversion: SHORT top gainers
        for coin in top_n:
            if coin in trade_day_dict:
                ret, stopped = calculate_position_return(trade_day_dict[coin], is_long=False, sl_pct=STOP_LOSS_PCT)
                day_returns.append(ret - ROUND_TRIP_FEE)
                if stopped:
                    total_stops += 1
                total_positions += 1

        # Mean Reversion: LONG bottom losers
        for coin in bottom_n:
            if coin in trade_day_dict:
                ret, stopped = calculate_position_return(trade_day_dict[coin], is_long=True, sl_pct=STOP_LOSS_PCT)
                day_returns.append(ret - ROUND_TRIP_FEE)
                if stopped:
                    total_stops += 1
                total_positions += 1

        if day_returns:
            trading_days += 1
            all_returns.extend(day_returns)

            # Fixed position sizing
            pnl_fixed = sum(r / 100 * POSITION_SIZE_FIXED for r in day_returns)
            fixed_equity += pnl_fixed

            # Dynamic position sizing (1/N_positions of capital per position)
            dynamic_fraction = POSITION_FRACTION
            pnl_dynamic = sum(r / 100 * dynamic_equity * dynamic_fraction for r in day_returns)
            dynamic_equity += pnl_dynamic

            equity_history_fixed.append({'date': trade_date, 'equity': fixed_equity})
            equity_history_dynamic.append({'date': trade_date, 'equity': dynamic_equity})

    if not all_returns:
        return None

    returns_series = pd.Series(all_returns)
    equity_fixed_series = pd.Series([e['equity'] for e in equity_history_fixed])
    equity_dynamic_series = pd.Series([e['equity'] for e in equity_history_dynamic])

    # Calculate metrics
    total_return_fixed = (fixed_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    total_return_dynamic = (dynamic_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Daily PnL for Sharpe
    daily_pnl = []
    for j in range(len(equity_history_fixed)):
        if j == 0:
            daily_pnl.append(equity_history_fixed[j]['equity'] - INITIAL_CAPITAL)
        else:
            daily_pnl.append(equity_history_fixed[j]['equity'] - equity_history_fixed[j-1]['equity'])
    daily_pnl = pd.Series(daily_pnl)
    daily_ret = daily_pnl / INITIAL_CAPITAL  # normalize by initial capital
    sharpe = (daily_ret.mean() * 365) / (daily_ret.std() * np.sqrt(365)) if daily_ret.std() > 0 else 0

    # Max drawdown
    rolling_max = equity_fixed_series.expanding().max()
    drawdown = (equity_fixed_series - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()

    # Win rate
    win_rate = (returns_series > 0).sum() / len(returns_series) * 100

    # Profit factor
    wins = returns_series[returns_series > 0].sum()
    losses = abs(returns_series[returns_series < 0].sum())
    profit_factor = wins / losses if losses > 0 else float('inf')

    # Average win/loss
    avg_win = returns_series[returns_series > 0].mean() if (returns_series > 0).sum() > 0 else 0
    avg_loss = returns_series[returns_series < 0].mean() if (returns_series < 0).sum() > 0 else 0

    # Per-trade stats
    avg_return = returns_series.mean()
    median_return = returns_series.median()

    stop_rate = total_stops / total_positions * 100 if total_positions > 0 else 0

    return {
        'n': n,
        'total_return_fixed': total_return_fixed,
        'total_return_dynamic': total_return_dynamic,
        'final_equity_fixed': fixed_equity,
        'final_equity_dynamic': dynamic_equity,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_return': avg_return,
        'median_return': median_return,
        'stop_rate': stop_rate,
        'total_positions': total_positions,
        'total_stops': total_stops,
        'trading_days': trading_days,
        'positions_per_day': total_positions / trading_days if trading_days > 0 else 0,
        'equity_fixed': equity_history_fixed,
        'equity_dynamic': equity_history_dynamic,
    }


# Run backtests
print("\n" + "=" * 80)
print("RUNNING BACKTESTS...")
print("=" * 80)

all_results = []
for n in N_VALUES:
    print(f"  Testing N={n} (top/bottom {n} movers)...", end=" ", flush=True)
    result = run_backtest(n)
    if result:
        all_results.append(result)
        print(f"✓ Return: {result['total_return_fixed']:+.1f}%, "
              f"Sharpe: {result['sharpe']:.2f}, "
              f"MaxDD: {result['max_dd']:.1f}%, "
              f"Positions/day: {result['positions_per_day']:.1f}")
    else:
        print("✗ No data")


# Print results table
print("\n" + "=" * 100)
print("RESULTS SUMMARY - MEAN REVERSION WITH 0.5% STOP LOSS")
print("=" * 100)

print(f"\n{'N':>4} {'Pos/Day':>8} {'Return%':>10} {'Sharpe':>8} {'MaxDD%':>8} {'WinRate%':>9} "
      f"{'StopRate%':>10} {'PF':>6} {'AvgWin%':>8} {'AvgLoss%':>9} {'AvgRet%':>8}")
print("-" * 105)
for r in all_results:
    print(f"{r['n']:>4} {r['positions_per_day']:>8.1f} {r['total_return_fixed']:>+9.1f}% "
          f"{r['sharpe']:>8.2f} {r['max_dd']:>+7.1f}% {r['win_rate']:>8.1f}% "
          f"{r['stop_rate']:>9.1f}% {r['profit_factor']:>6.2f} "
          f"{r['avg_win']:>+7.2f}% {r['avg_loss']:>+8.2f}% {r['avg_return']:>+7.3f}%")

print("\n" + "=" * 100)
print("DYNAMIC POSITION SIZING (1/6 Capital)")
print("=" * 100)

print(f"\n{'N':>4} {'Return%':>15} {'Final Equity':>15}")
print("-" * 40)
for r in all_results:
    print(f"{r['n']:>4} {r['total_return_dynamic']:>+14.1f}% ${r['final_equity_dynamic']:>13,.2f}")


# Generate charts
print("\n" + "=" * 80)
print("GENERATING CHARTS...")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Mean Reversion: N-Value Sensitivity Analysis (0.5% Stop Loss)', fontsize=14, fontweight='bold')

n_labels = [str(r['n']) for r in all_results]
x = np.arange(len(n_labels))

# 1. Total Return
ax = axes[0, 0]
returns = [r['total_return_fixed'] for r in all_results]
colors = ['green' if r > 0 else 'red' for r in returns]
ax.bar(x, returns, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xlabel('N (Top/Bottom Movers)')
ax.set_ylabel('Total Return (%)')
ax.set_title('Total Return (Fixed $100)')
ax.set_xticks(x)
ax.set_xticklabels(n_labels)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(returns):
    ax.text(i, v + max(returns)*0.02, f'{v:+.0f}%', ha='center', va='bottom', fontsize=8)

# 2. Sharpe Ratio
ax = axes[0, 1]
sharpes = [r['sharpe'] for r in all_results]
ax.plot(n_labels, sharpes, 'bo-', markersize=8, linewidth=2)
ax.set_xlabel('N (Top/Bottom Movers)')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Sharpe Ratio')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

# 3. Max Drawdown
ax = axes[0, 2]
max_dds = [r['max_dd'] for r in all_results]
ax.plot(n_labels, max_dds, 'rs-', markersize=8, linewidth=2)
ax.set_xlabel('N (Top/Bottom Movers)')
ax.set_ylabel('Max Drawdown (%)')
ax.set_title('Max Drawdown')
ax.grid(True, alpha=0.3)

# 4. Win Rate & Stop Rate
ax = axes[1, 0]
win_rates = [r['win_rate'] for r in all_results]
stop_rates = [r['stop_rate'] for r in all_results]
ax.plot(n_labels, win_rates, 'go-', markersize=8, linewidth=2, label='Win Rate')
ax.plot(n_labels, stop_rates, 'rs-', markersize=8, linewidth=2, label='Stop Rate')
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('N (Top/Bottom Movers)')
ax.set_ylabel('Rate (%)')
ax.set_title('Win Rate & Stop Rate')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Avg Return per Trade
ax = axes[1, 1]
avg_returns = [r['avg_return'] for r in all_results]
ax.bar(x, avg_returns, color=['green' if r > 0 else 'red' for r in avg_returns], alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xlabel('N (Top/Bottom Movers)')
ax.set_ylabel('Avg Return per Trade (%)')
ax.set_title('Average Return per Trade')
ax.set_xticks(x)
ax.set_xticklabels(n_labels)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

# 6. Equity Curves
ax = axes[1, 2]
for r in all_results:
    equity = [e['equity'] for e in r['equity_fixed']]
    ax.plot(equity, label=f'N={r["n"]}', linewidth=1.2)
ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Trading Days')
ax.set_ylabel('Equity ($)')
ax.set_title('Equity Curves (Fixed $100)')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('n_value_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Chart saved to: n_value_analysis.png")

# Save results to CSV
results_data = []
for r in all_results:
    results_data.append({
        'n': r['n'],
        'positions_per_day': r['positions_per_day'],
        'total_return_fixed': r['total_return_fixed'],
        'total_return_dynamic': r['total_return_dynamic'],
        'final_equity_fixed': r['final_equity_fixed'],
        'final_equity_dynamic': r['final_equity_dynamic'],
        'sharpe': r['sharpe'],
        'max_dd': r['max_dd'],
        'win_rate': r['win_rate'],
        'profit_factor': r['profit_factor'],
        'avg_win': r['avg_win'],
        'avg_loss': r['avg_loss'],
        'avg_return': r['avg_return'],
        'stop_rate': r['stop_rate'],
        'total_positions': r['total_positions'],
        'trading_days': r['trading_days'],
    })

pd.DataFrame(results_data).to_csv('n_value_comparison.csv', index=False)
print("Results saved to: n_value_comparison.csv")

# Find optimal N
print("\n" + "=" * 80)
print("OPTIMAL N ANALYSIS")
print("=" * 80)

best_return = max(all_results, key=lambda x: x['total_return_fixed'])
best_sharpe = max(all_results, key=lambda x: x['sharpe'])
best_risk_adj = max(all_results, key=lambda x: x['total_return_fixed'] / abs(x['max_dd']) if x['max_dd'] != 0 else 0)

print(f"\nBest Total Return:       N={best_return['n']} -> {best_return['total_return_fixed']:+.1f}%")
print(f"Best Sharpe Ratio:       N={best_sharpe['n']} -> {best_sharpe['sharpe']:.2f}")
print(f"Best Risk-Adjusted:      N={best_risk_adj['n']} -> Return/MaxDD = {best_risk_adj['total_return_fixed']/abs(best_risk_adj['max_dd']):.2f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
