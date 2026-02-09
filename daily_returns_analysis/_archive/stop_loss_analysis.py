"""
Stop Loss Sensitivity Analysis
Test different stop loss percentages and compare performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
N = 3  # Number of top/bottom assets to trade
INITIAL_CAPITAL = 1000
POSITION_SIZE_FIXED = 100
POSITION_FRACTION = 1/6
TRADING_FEE = 0.045
ROUND_TRIP_FEE = TRADING_FEE * 2

# Stop loss levels to test (including smaller values)
STOP_LOSS_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0, None]  # None = no stop loss

print("=" * 80)
print("STOP LOSS SENSITIVITY ANALYSIS")
print("=" * 80)
print(f"\nTesting Stop Loss Levels: {[f'{sl}%' if sl else 'None' for sl in STOP_LOSS_LEVELS]}")
print(f"Strategies: Mean Reversion & Trend Following")
print(f"Position Sizing: Fixed ${POSITION_SIZE_FIXED} and Dynamic 1/6 Capital")

# Load OHLC data
print("\nLoading daily OHLC data...")
ohlc_df = pd.read_csv('daily_ohlc.csv')
ohlc_df['date'] = pd.to_datetime(ohlc_df['date']).dt.date
ohlc_df = ohlc_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['daily_return'])

dates = sorted(ohlc_df['date'].unique())
print(f"Data range: {dates[0]} to {dates[-1]}")
print(f"Total trading days: {len(dates)}")

def run_backtest(stop_loss_pct):
    """Run backtest with a specific stop loss percentage"""
    
    # Track equity
    fixed_mr_equity = INITIAL_CAPITAL
    fixed_tf_equity = INITIAL_CAPITAL
    dynamic_mr_equity = INITIAL_CAPITAL
    dynamic_tf_equity = INITIAL_CAPITAL
    
    results = []
    mr_all_returns = []
    tf_all_returns = []
    total_mr_stops = 0
    total_tf_stops = 0
    
    for i in range(1, len(dates)):
        signal_date = dates[i-1]
        trade_date = dates[i]
        
        signal_day_data = ohlc_df[ohlc_df['date'] == signal_date].copy()
        if len(signal_day_data) < N * 2:
            continue
        
        top_n = signal_day_data.nlargest(N, 'daily_return')[['coin', 'daily_return']].values.tolist()
        bottom_n = signal_day_data.nsmallest(N, 'daily_return')[['coin', 'daily_return']].values.tolist()
        
        top_coins = [x[0] for x in top_n]
        bottom_coins = [x[0] for x in bottom_n]
        
        trade_day_data = ohlc_df[ohlc_df['date'] == trade_date].copy()
        
        def calculate_position_return(coin, is_long, trade_data, sl_pct):
            coin_data = trade_data[trade_data['coin'] == coin]
            if len(coin_data) == 0:
                return None, False
            
            row = coin_data.iloc[0]
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            if sl_pct is None:
                # No stop loss
                if is_long:
                    return (close_price / open_price - 1) * 100, False
                else:
                    return -(close_price / open_price - 1) * 100, False
            
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
        
        # Mean Reversion: Short top, Long bottom
        mr_short_returns = []
        mr_long_returns = []
        mr_stops = 0
        
        for coin in top_coins:
            ret, stopped = calculate_position_return(coin, is_long=False, trade_data=trade_day_data, sl_pct=stop_loss_pct)
            if ret is not None:
                mr_short_returns.append(ret - ROUND_TRIP_FEE)
                if stopped:
                    mr_stops += 1
        
        for coin in bottom_coins:
            ret, stopped = calculate_position_return(coin, is_long=True, trade_data=trade_day_data, sl_pct=stop_loss_pct)
            if ret is not None:
                mr_long_returns.append(ret - ROUND_TRIP_FEE)
                if stopped:
                    mr_stops += 1
        
        # Trend Following: Long top, Short bottom
        tf_long_returns = []
        tf_short_returns = []
        tf_stops = 0
        
        for coin in top_coins:
            ret, stopped = calculate_position_return(coin, is_long=True, trade_data=trade_day_data, sl_pct=stop_loss_pct)
            if ret is not None:
                tf_long_returns.append(ret - ROUND_TRIP_FEE)
                if stopped:
                    tf_stops += 1
        
        for coin in bottom_coins:
            ret, stopped = calculate_position_return(coin, is_long=False, trade_data=trade_day_data, sl_pct=stop_loss_pct)
            if ret is not None:
                tf_short_returns.append(ret - ROUND_TRIP_FEE)
                if stopped:
                    tf_stops += 1
        
        total_mr_stops += mr_stops
        total_tf_stops += tf_stops
        
        # Calculate PnL
        mr_pnl_fixed = sum([r / 100 * POSITION_SIZE_FIXED for r in mr_short_returns + mr_long_returns])
        tf_pnl_fixed = sum([r / 100 * POSITION_SIZE_FIXED for r in tf_long_returns + tf_short_returns])
        
        mr_pnl_dynamic = sum([r / 100 * dynamic_mr_equity * POSITION_FRACTION for r in mr_short_returns + mr_long_returns])
        tf_pnl_dynamic = sum([r / 100 * dynamic_tf_equity * POSITION_FRACTION for r in tf_long_returns + tf_short_returns])
        
        fixed_mr_equity += mr_pnl_fixed
        fixed_tf_equity += tf_pnl_fixed
        dynamic_mr_equity += mr_pnl_dynamic
        dynamic_tf_equity += tf_pnl_dynamic
        
        mr_all_returns.extend(mr_short_returns + mr_long_returns)
        tf_all_returns.extend(tf_long_returns + tf_short_returns)
        
        results.append({
            'trade_date': trade_date,
            'mr_pnl_fixed': mr_pnl_fixed,
            'tf_pnl_fixed': tf_pnl_fixed,
            'mr_equity_fixed': fixed_mr_equity,
            'tf_equity_fixed': fixed_tf_equity,
            'mr_equity_dynamic': dynamic_mr_equity,
            'tf_equity_dynamic': dynamic_tf_equity,
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    def calc_stats(equity_series, pnl_series, all_returns, name):
        total_pnl = pnl_series.sum()
        total_return = (equity_series.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        
        # Sharpe
        daily_returns = pnl_series / POSITION_SIZE_FIXED / (N * 2) * 100
        sharpe = (daily_returns.mean() * 365) / (daily_returns.std() * np.sqrt(365)) if daily_returns.std() > 0 else 0
        
        # Max drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_dd = drawdown.min()
        
        # Win rate
        returns_series = pd.Series(all_returns)
        win_rate = (returns_series > 0).sum() / len(returns_series) * 100 if len(returns_series) > 0 else 0
        
        # Profit factor
        wins = returns_series[returns_series > 0].sum()
        losses = abs(returns_series[returns_series < 0].sum())
        profit_factor = wins / losses if losses > 0 else float('inf')
        
        # Avg profit/loss
        avg_win = returns_series[returns_series > 0].mean() if (returns_series > 0).sum() > 0 else 0
        avg_loss = returns_series[returns_series < 0].mean() if (returns_series < 0).sum() > 0 else 0
        
        return {
            'total_return': total_return,
            'final_equity': equity_series.iloc[-1],
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
        }
    
    mr_stats_fixed = calc_stats(results_df['mr_equity_fixed'], results_df['mr_pnl_fixed'], mr_all_returns, 'MR Fixed')
    tf_stats_fixed = calc_stats(results_df['tf_equity_fixed'], results_df['tf_pnl_fixed'], tf_all_returns, 'TF Fixed')
    mr_stats_dynamic = calc_stats(results_df['mr_equity_dynamic'], results_df['mr_pnl_fixed'], mr_all_returns, 'MR Dynamic')
    tf_stats_dynamic = calc_stats(results_df['tf_equity_dynamic'], results_df['tf_pnl_fixed'], tf_all_returns, 'TF Dynamic')
    
    total_positions = len(results_df) * N * 2
    
    return {
        'stop_loss': stop_loss_pct,
        'mr_fixed': mr_stats_fixed,
        'tf_fixed': tf_stats_fixed,
        'mr_dynamic': mr_stats_dynamic,
        'tf_dynamic': tf_stats_dynamic,
        'mr_stops_pct': total_mr_stops / total_positions * 100,
        'tf_stops_pct': total_tf_stops / total_positions * 100,
        'results_df': results_df,
    }

# Run backtests for all stop loss levels
print("\n" + "=" * 80)
print("RUNNING BACKTESTS...")
print("=" * 80)

all_results = []
for sl in STOP_LOSS_LEVELS:
    sl_str = f"{sl}%" if sl else "None"
    print(f"  Testing stop loss: {sl_str}...", end=" ")
    result = run_backtest(sl)
    all_results.append(result)
    print(f"✓ MR: {result['mr_fixed']['total_return']:.1f}%, TF: {result['tf_fixed']['total_return']:.1f}%")

# Create summary table
print("\n" + "=" * 80)
print("RESULTS SUMMARY - FIXED POSITION SIZING ($100)")
print("=" * 80)

print(f"\n{'Stop Loss':<10} {'MR Return':>12} {'MR Sharpe':>10} {'MR MaxDD':>10} {'MR WinRate':>10} {'MR Stops':>10}")
print("-" * 65)
for r in all_results:
    sl_str = f"{r['stop_loss']}%" if r['stop_loss'] else "None"
    print(f"{sl_str:<10} {r['mr_fixed']['total_return']:>+11.1f}% {r['mr_fixed']['sharpe']:>10.2f} {r['mr_fixed']['max_dd']:>+9.1f}% {r['mr_fixed']['win_rate']:>9.1f}% {r['mr_stops_pct']:>9.1f}%")

print(f"\n{'Stop Loss':<10} {'TF Return':>12} {'TF Sharpe':>10} {'TF MaxDD':>10} {'TF WinRate':>10} {'TF Stops':>10}")
print("-" * 65)
for r in all_results:
    sl_str = f"{r['stop_loss']}%" if r['stop_loss'] else "None"
    print(f"{sl_str:<10} {r['tf_fixed']['total_return']:>+11.1f}% {r['tf_fixed']['sharpe']:>10.2f} {r['tf_fixed']['max_dd']:>+9.1f}% {r['tf_fixed']['win_rate']:>9.1f}% {r['tf_stops_pct']:>9.1f}%")

print("\n" + "=" * 80)
print("RESULTS SUMMARY - DYNAMIC POSITION SIZING (1/6 Capital)")
print("=" * 80)

print(f"\n{'Stop Loss':<10} {'MR Return':>15} {'MR Final Eq':>15} {'MR MaxDD':>10} {'TF Return':>15} {'TF Final Eq':>15}")
print("-" * 85)
for r in all_results:
    sl_str = f"{r['stop_loss']}%" if r['stop_loss'] else "None"
    print(f"{sl_str:<10} {r['mr_dynamic']['total_return']:>+14.0f}% ${r['mr_dynamic']['final_equity']:>13,.0f} {r['mr_dynamic']['max_dd']:>+9.1f}% {r['tf_dynamic']['total_return']:>+14.0f}% ${r['tf_dynamic']['final_equity']:>13,.0f}")

# Generate charts
print("\n" + "=" * 80)
print("GENERATING CHARTS...")
print("=" * 80)

fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# Prepare data for plotting
sl_labels = [f"{r['stop_loss']}%" if r['stop_loss'] else "None" for r in all_results]
x = np.arange(len(sl_labels))

# 1. Total Return by Stop Loss (Fixed)
ax1 = axes[0, 0]
mr_returns = [r['mr_fixed']['total_return'] for r in all_results]
tf_returns = [r['tf_fixed']['total_return'] for r in all_results]
width = 0.35
ax1.bar(x - width/2, mr_returns, width, label='Mean Reversion', color='red', alpha=0.7)
ax1.bar(x + width/2, tf_returns, width, label='Trend Following', color='green', alpha=0.7)
ax1.set_xlabel('Stop Loss Level')
ax1.set_ylabel('Total Return (%)')
ax1.set_title('Total Return by Stop Loss Level (Fixed $100)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(sl_labels)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 2. Sharpe Ratio by Stop Loss
ax2 = axes[0, 1]
mr_sharpe = [r['mr_fixed']['sharpe'] for r in all_results]
tf_sharpe = [r['tf_fixed']['sharpe'] for r in all_results]
ax2.plot(sl_labels, mr_sharpe, 'ro-', label='Mean Reversion', markersize=8, linewidth=2)
ax2.plot(sl_labels, tf_sharpe, 'gs-', label='Trend Following', markersize=8, linewidth=2)
ax2.set_xlabel('Stop Loss Level')
ax2.set_ylabel('Sharpe Ratio')
ax2.set_title('Sharpe Ratio by Stop Loss Level (Fixed $100)', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Max Drawdown by Stop Loss
ax3 = axes[1, 0]
mr_dd = [r['mr_fixed']['max_dd'] for r in all_results]
tf_dd = [r['tf_fixed']['max_dd'] for r in all_results]
ax3.plot(sl_labels, mr_dd, 'ro-', label='Mean Reversion', markersize=8, linewidth=2)
ax3.plot(sl_labels, tf_dd, 'gs-', label='Trend Following', markersize=8, linewidth=2)
ax3.set_xlabel('Stop Loss Level')
ax3.set_ylabel('Max Drawdown (%)')
ax3.set_title('Max Drawdown by Stop Loss Level (Fixed $100)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.invert_yaxis()  # Lower is better for drawdown

# 4. Win Rate by Stop Loss
ax4 = axes[1, 1]
mr_wr = [r['mr_fixed']['win_rate'] for r in all_results]
tf_wr = [r['tf_fixed']['win_rate'] for r in all_results]
ax4.plot(sl_labels, mr_wr, 'ro-', label='Mean Reversion', markersize=8, linewidth=2)
ax4.plot(sl_labels, tf_wr, 'gs-', label='Trend Following', markersize=8, linewidth=2)
ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Line')
ax4.set_xlabel('Stop Loss Level')
ax4.set_ylabel('Win Rate (%)')
ax4.set_title('Win Rate by Stop Loss Level', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Stop Loss Trigger Rate
ax5 = axes[2, 0]
mr_stops = [r['mr_stops_pct'] for r in all_results]
tf_stops = [r['tf_stops_pct'] for r in all_results]
ax5.bar(x - width/2, mr_stops, width, label='Mean Reversion', color='red', alpha=0.7)
ax5.bar(x + width/2, tf_stops, width, label='Trend Following', color='green', alpha=0.7)
ax5.set_xlabel('Stop Loss Level')
ax5.set_ylabel('Stop Loss Trigger Rate (%)')
ax5.set_title('Stop Loss Trigger Rate by Level', fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(sl_labels)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Risk-Adjusted Return (Return / Max DD)
ax6 = axes[2, 1]
mr_risk_adj = [r['mr_fixed']['total_return'] / abs(r['mr_fixed']['max_dd']) if r['mr_fixed']['max_dd'] != 0 else 0 for r in all_results]
tf_risk_adj = [r['tf_fixed']['total_return'] / abs(r['tf_fixed']['max_dd']) if r['tf_fixed']['max_dd'] != 0 else 0 for r in all_results]
ax6.plot(sl_labels, mr_risk_adj, 'ro-', label='Mean Reversion', markersize=8, linewidth=2)
ax6.plot(sl_labels, tf_risk_adj, 'gs-', label='Trend Following', markersize=8, linewidth=2)
ax6.set_xlabel('Stop Loss Level')
ax6.set_ylabel('Return / Max Drawdown Ratio')
ax6.set_title('Risk-Adjusted Return (Return / |MaxDD|)', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stop_loss_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\nChart saved to: stop_loss_analysis.png")

# Equity curves comparison chart
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

# Select key stop loss levels to compare
key_levels = [0.5, 1.0, 2.0, 5.0, None]
colors = ['blue', 'red', 'green', 'orange', 'purple']

# MR Fixed equity curves
ax_mr = axes2[0, 0]
for sl, color in zip(key_levels, colors):
    result = next((r for r in all_results if r['stop_loss'] == sl), None)
    if result:
        label = f"SL {sl}%" if sl else "No SL"
        ax_mr.plot(result['results_df']['mr_equity_fixed'], label=label, color=color, linewidth=1.5)
ax_mr.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
ax_mr.set_title('Mean Reversion - Equity Curves by Stop Loss (Fixed)', fontweight='bold')
ax_mr.set_xlabel('Trading Days')
ax_mr.set_ylabel('Equity ($)')
ax_mr.legend()
ax_mr.grid(True, alpha=0.3)

# TF Fixed equity curves
ax_tf = axes2[0, 1]
for sl, color in zip(key_levels, colors):
    result = next((r for r in all_results if r['stop_loss'] == sl), None)
    if result:
        label = f"SL {sl}%" if sl else "No SL"
        ax_tf.plot(result['results_df']['tf_equity_fixed'], label=label, color=color, linewidth=1.5)
ax_tf.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
ax_tf.set_title('Trend Following - Equity Curves by Stop Loss (Fixed)', fontweight='bold')
ax_tf.set_xlabel('Trading Days')
ax_tf.set_ylabel('Equity ($)')
ax_tf.legend()
ax_tf.grid(True, alpha=0.3)

# MR Dynamic equity curves (log scale)
ax_mr_dyn = axes2[1, 0]
for sl, color in zip(key_levels, colors):
    result = next((r for r in all_results if r['stop_loss'] == sl), None)
    if result:
        label = f"SL {sl}%" if sl else "No SL"
        ax_mr_dyn.plot(result['results_df']['mr_equity_dynamic'], label=label, color=color, linewidth=1.5)
ax_mr_dyn.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
ax_mr_dyn.set_title('Mean Reversion - Equity Curves by Stop Loss (Dynamic 1/6)', fontweight='bold')
ax_mr_dyn.set_xlabel('Trading Days')
ax_mr_dyn.set_ylabel('Equity ($)')
ax_mr_dyn.set_yscale('log')
ax_mr_dyn.legend()
ax_mr_dyn.grid(True, alpha=0.3)

# TF Dynamic equity curves (log scale)
ax_tf_dyn = axes2[1, 1]
for sl, color in zip(key_levels, colors):
    result = next((r for r in all_results if r['stop_loss'] == sl), None)
    if result:
        label = f"SL {sl}%" if sl else "No SL"
        ax_tf_dyn.plot(result['results_df']['tf_equity_dynamic'], label=label, color=color, linewidth=1.5)
ax_tf_dyn.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
ax_tf_dyn.set_title('Trend Following - Equity Curves by Stop Loss (Dynamic 1/6)', fontweight='bold')
ax_tf_dyn.set_xlabel('Trading Days')
ax_tf_dyn.set_ylabel('Equity ($)')
ax_tf_dyn.set_yscale('log')
ax_tf_dyn.legend()
ax_tf_dyn.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stop_loss_equity_curves.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Equity curves chart saved to: stop_loss_equity_curves.png")

# Find optimal stop loss
print("\n" + "=" * 80)
print("OPTIMAL STOP LOSS ANALYSIS")
print("=" * 80)

# By Sharpe Ratio
best_mr_sharpe = max(all_results, key=lambda x: x['mr_fixed']['sharpe'])
best_tf_sharpe = max(all_results, key=lambda x: x['tf_fixed']['sharpe'])

print(f"\nBest Sharpe Ratio:")
print(f"  Mean Reversion: {best_mr_sharpe['stop_loss']}% SL -> Sharpe: {best_mr_sharpe['mr_fixed']['sharpe']:.2f}")
print(f"  Trend Following: {best_tf_sharpe['stop_loss']}% SL -> Sharpe: {best_tf_sharpe['tf_fixed']['sharpe']:.2f}")

# By Total Return
best_mr_return = max(all_results, key=lambda x: x['mr_fixed']['total_return'])
best_tf_return = max(all_results, key=lambda x: x['tf_fixed']['total_return'])

print(f"\nBest Total Return:")
print(f"  Mean Reversion: {best_mr_return['stop_loss']}% SL -> Return: {best_mr_return['mr_fixed']['total_return']:.1f}%")
print(f"  Trend Following: {best_tf_return['stop_loss']}% SL -> Return: {best_tf_return['tf_fixed']['total_return']:.1f}%")

# By Risk-Adjusted (Return / MaxDD)
best_mr_risk = max(all_results, key=lambda x: x['mr_fixed']['total_return'] / abs(x['mr_fixed']['max_dd']) if x['mr_fixed']['max_dd'] != 0 else 0)
best_tf_risk = max(all_results, key=lambda x: x['tf_fixed']['total_return'] / abs(x['tf_fixed']['max_dd']) if x['tf_fixed']['max_dd'] != 0 else 0)

print(f"\nBest Risk-Adjusted Return (Return/MaxDD):")
print(f"  Mean Reversion: {best_mr_risk['stop_loss']}% SL -> Ratio: {best_mr_risk['mr_fixed']['total_return'] / abs(best_mr_risk['mr_fixed']['max_dd']):.1f}")
print(f"  Trend Following: {best_tf_risk['stop_loss']}% SL -> Ratio: {best_tf_risk['tf_fixed']['total_return'] / abs(best_tf_risk['tf_fixed']['max_dd']):.1f}")

# Save results to CSV
results_data = []
for r in all_results:
    results_data.append({
        'stop_loss': r['stop_loss'] if r['stop_loss'] else 'None',
        'mr_fixed_return': r['mr_fixed']['total_return'],
        'mr_fixed_sharpe': r['mr_fixed']['sharpe'],
        'mr_fixed_maxdd': r['mr_fixed']['max_dd'],
        'mr_fixed_winrate': r['mr_fixed']['win_rate'],
        'mr_stops_pct': r['mr_stops_pct'],
        'tf_fixed_return': r['tf_fixed']['total_return'],
        'tf_fixed_sharpe': r['tf_fixed']['sharpe'],
        'tf_fixed_maxdd': r['tf_fixed']['max_dd'],
        'tf_fixed_winrate': r['tf_fixed']['win_rate'],
        'tf_stops_pct': r['tf_stops_pct'],
        'mr_dynamic_return': r['mr_dynamic']['total_return'],
        'mr_dynamic_equity': r['mr_dynamic']['final_equity'],
        'tf_dynamic_return': r['tf_dynamic']['total_return'],
        'tf_dynamic_equity': r['tf_dynamic']['final_equity'],
    })

pd.DataFrame(results_data).to_csv('stop_loss_comparison.csv', index=False)
print("\nResults saved to: stop_loss_comparison.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

plt.show()
