"""
Generate comprehensive charts for the final backtest results
Plus Alpha/Beta analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gzip
from scipy import stats

print("=" * 80)
print("GENERATING CHARTS & ALPHA/BETA ANALYSIS")
print("=" * 80)

# Load results
trades = pd.read_csv('mean_reversion_results/final_trades.csv')
hourly = pd.read_csv('mean_reversion_results/final_hourly.csv')
trades['entry_hour'] = pd.to_datetime(trades['entry_hour'])
trades['exit_hour'] = pd.to_datetime(trades['exit_hour'])
hourly['hour'] = pd.to_datetime(hourly['hour'])

# Load BTC prices for comparison
with gzip.open("price_history.csv.gz", "rt") as f:
    prices_raw = pd.read_csv(f)
prices_raw["timestamp"] = pd.to_datetime(prices_raw["timestamp"], utc=True)
prices_raw["hour"] = prices_raw["timestamp"].dt.floor("h")
btc_prices = prices_raw[prices_raw['coin'] == 'BTC'].groupby('hour')['price'].last().reset_index()
btc_prices.columns = ['hour', 'btc_price']

# Merge BTC prices with hourly data
hourly = hourly.merge(btc_prices, on='hour', how='left')

STARTING_CAPITAL = 1000

# =============================================================================
# CHART 1: MAIN BACKTEST CHARTS (4 panels)
# =============================================================================
print("\n1. Creating main backtest charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Equity Curve
ax1 = axes[0, 0]
ax1.plot(hourly['hour'], hourly['equity'], linewidth=1.5, color='blue', label='Strategy Equity')
ax1.axhline(STARTING_CAPITAL, color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
ax1.fill_between(hourly['hour'], STARTING_CAPITAL, hourly['equity'], 
                  where=hourly['equity'] >= STARTING_CAPITAL, alpha=0.3, color='green')
ax1.fill_between(hourly['hour'], STARTING_CAPITAL, hourly['equity'], 
                  where=hourly['equity'] < STARTING_CAPITAL, alpha=0.3, color='red')
final_pnl = hourly['equity'].iloc[-1] - STARTING_CAPITAL
ax1.set_title(f'Equity Curve (Final PnL: ${final_pnl:,.0f})')
ax1.set_ylabel('Equity ($)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Panel 2: Drawdown
ax2 = axes[0, 1]
peak = hourly['equity'].cummax()
drawdown = (peak - hourly['equity']) / peak * 100
ax2.fill_between(hourly['hour'], 0, drawdown, color='red', alpha=0.5)
ax2.plot(hourly['hour'], drawdown, color='darkred', linewidth=0.5)
max_dd = drawdown.max()
ax2.axhline(max_dd, color='darkred', linestyle='--', alpha=0.7)
ax2.set_title(f'Drawdown (Max: {max_dd:.1f}%)')
ax2.set_ylabel('Drawdown (%)')
ax2.set_ylim(0, max_dd * 1.1)
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3)

# Panel 3: Number of Positions
ax3 = axes[1, 0]
ax3.fill_between(hourly['hour'], 0, hourly['n_short'], alpha=0.5, color='red', label='SHORT')
ax3.fill_between(hourly['hour'], hourly['n_short'], hourly['n_short'] + hourly['n_long'], 
                  alpha=0.5, color='green', label='LONG')
ax3.set_title('Concurrent Positions')
ax3.set_ylabel('Number of Positions')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Cumulative PnL by Component
ax4 = axes[1, 1]
cum_price = hourly['price_pnl'].cumsum()
cum_hedge = hourly['hedge_pnl'].cumsum()
cum_funding = hourly['funding_pnl'].cumsum()
ax4.plot(hourly['hour'], cum_price, label=f'Price PnL: ${cum_price.iloc[-1]:,.0f}', linewidth=1.5)
ax4.plot(hourly['hour'], cum_hedge, label=f'Hedge PnL: ${cum_hedge.iloc[-1]:,.0f}', linewidth=1.5)
ax4.plot(hourly['hour'], cum_funding, label=f'Funding PnL: ${cum_funding.iloc[-1]:,.0f}', linewidth=1.5)
ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax4.set_title('Cumulative PnL by Component')
ax4.set_ylabel('PnL ($)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mean_reversion_results/final_backtest_charts.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: final_backtest_charts.png")

# =============================================================================
# CHART 2: STRATEGY VS BTC COMPARISON
# =============================================================================
print("\n2. Creating strategy vs BTC comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Normalize BTC to same starting capital
btc_start = hourly['btc_price'].iloc[0]
hourly['btc_equity'] = STARTING_CAPITAL * hourly['btc_price'] / btc_start

# Panel 1: Strategy vs BTC
ax1 = axes[0, 0]
ax1.plot(hourly['hour'], hourly['equity'], label='Strategy', linewidth=1.5, color='blue')
ax1.plot(hourly['hour'], hourly['btc_equity'], label='Buy & Hold BTC', linewidth=1.5, color='orange')
ax1.axhline(STARTING_CAPITAL, color='gray', linestyle='--', alpha=0.5)
strategy_return = (hourly['equity'].iloc[-1] / STARTING_CAPITAL - 1) * 100
btc_return = (hourly['btc_equity'].iloc[-1] / STARTING_CAPITAL - 1) * 100
ax1.set_title(f'Strategy ({strategy_return:.0f}%) vs BTC ({btc_return:.0f}%)')
ax1.set_ylabel('Equity ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Daily Returns Distribution
ax2 = axes[0, 1]
hourly['strategy_return'] = hourly['equity'].pct_change()
hourly['btc_return'] = hourly['btc_equity'].pct_change()
daily_strat = hourly.set_index('hour')['strategy_return'].resample('D').sum().dropna()
daily_btc = hourly.set_index('hour')['btc_return'].resample('D').sum().dropna()
ax2.hist(daily_strat * 100, bins=50, alpha=0.5, label=f'Strategy (σ={daily_strat.std()*100:.2f}%)', color='blue')
ax2.hist(daily_btc * 100, bins=50, alpha=0.5, label=f'BTC (σ={daily_btc.std()*100:.2f}%)', color='orange')
ax2.axvline(0, color='gray', linestyle='--')
ax2.set_title('Daily Returns Distribution')
ax2.set_xlabel('Daily Return (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Rolling Sharpe (30-day)
ax3 = axes[1, 0]
window = 24 * 30  # 30 days
rolling_sharpe = (hourly['strategy_return'].rolling(window).mean() / 
                  hourly['strategy_return'].rolling(window).std() * np.sqrt(24 * 365))
ax3.plot(hourly['hour'], rolling_sharpe, linewidth=1, color='blue')
ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax3.axhline(rolling_sharpe.mean(), color='blue', linestyle='--', alpha=0.5, label=f'Mean: {rolling_sharpe.mean():.2f}')
ax3.set_title('Rolling 30-Day Sharpe Ratio')
ax3.set_ylabel('Sharpe Ratio')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Outperformance vs BTC
ax4 = axes[1, 1]
outperformance = hourly['equity'] - hourly['btc_equity']
ax4.plot(hourly['hour'], outperformance, linewidth=1, color='purple')
ax4.fill_between(hourly['hour'], 0, outperformance, 
                  where=outperformance >= 0, alpha=0.3, color='green')
ax4.fill_between(hourly['hour'], 0, outperformance, 
                  where=outperformance < 0, alpha=0.3, color='red')
ax4.axhline(0, color='gray', linestyle='--')
ax4.set_title(f'Outperformance vs BTC (Final: ${outperformance.iloc[-1]:,.0f})')
ax4.set_ylabel('Outperformance ($)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mean_reversion_results/final_btc_comparison.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: final_btc_comparison.png")

# =============================================================================
# CHART 3: TRADE ANALYSIS
# =============================================================================
print("\n3. Creating trade analysis charts...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Panel 1: PnL Distribution
ax1 = axes[0, 0]
ax1.hist(trades['net_pnl'], bins=50, alpha=0.7, color='blue', edgecolor='black')
ax1.axvline(0, color='red', linestyle='--', linewidth=2)
ax1.axvline(trades['net_pnl'].mean(), color='green', linestyle='--', 
            label=f'Mean: ${trades["net_pnl"].mean():.2f}')
win_rate = (trades['net_pnl'] > 0).mean() * 100
ax1.set_title(f'Trade PnL Distribution (Win Rate: {win_rate:.1f}%)')
ax1.set_xlabel('Net PnL ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: SHORT vs LONG comparison
ax2 = axes[0, 1]
short_trades = trades[trades['direction'] == 'SHORT']
long_trades = trades[trades['direction'] == 'LONG']
categories = ['SHORT', 'LONG']
pnls = [short_trades['net_pnl'].sum(), long_trades['net_pnl'].sum()]
counts = [len(short_trades), len(long_trades)]
win_rates = [(short_trades['net_pnl'] > 0).mean() * 100, (long_trades['net_pnl'] > 0).mean() * 100]

x = np.arange(len(categories))
width = 0.35
bars = ax2.bar(x, pnls, width, color=['red', 'green'], alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels([f'{c}\n({n} trades, {wr:.0f}% WR)' for c, n, wr in zip(categories, counts, win_rates)])
ax2.set_title('Total PnL by Direction')
ax2.set_ylabel('Total PnL ($)')
ax2.axhline(0, color='gray', linestyle='--')
ax2.grid(True, alpha=0.3)

# Panel 3: Hold Time Distribution
ax3 = axes[0, 2]
ax3.hist(trades['hold_hours'], bins=50, alpha=0.7, color='purple', edgecolor='black')
ax3.axvline(trades['hold_hours'].median(), color='red', linestyle='--',
            label=f'Median: {trades["hold_hours"].median():.0f}h')
ax3.set_title('Hold Time Distribution')
ax3.set_xlabel('Hold Time (hours)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: PnL Components per Trade
ax4 = axes[1, 0]
avg_price = trades['price_pnl'].mean()
avg_funding = trades['funding_pnl'].mean()
avg_fees = trades['fees'].mean()
avg_net = trades['net_pnl'].mean()
components = ['Price\nPnL', 'Funding\nPnL', 'Fees', 'Net\nPnL']
values = [avg_price, avg_funding, -avg_fees, avg_net]
colors = ['blue', 'green', 'red', 'purple']
bars = ax4.bar(components, values, color=colors, alpha=0.7)
ax4.axhline(0, color='gray', linestyle='--')
ax4.set_title('Average PnL Components per Trade')
ax4.set_ylabel('PnL ($)')
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2, val, f'${val:.2f}', 
             ha='center', va='bottom' if val >= 0 else 'top')
ax4.grid(True, alpha=0.3)

# Panel 5: Monthly Returns
ax5 = axes[1, 1]
hourly_temp = hourly.set_index('hour')
monthly_returns = hourly_temp['equity'].resample('M').last().pct_change().dropna() * 100
colors = ['green' if r >= 0 else 'red' for r in monthly_returns]
ax5.bar(monthly_returns.index, monthly_returns.values, width=20, color=colors, alpha=0.7)
ax5.axhline(0, color='gray', linestyle='--')
ax5.set_title(f'Monthly Returns (Avg: {monthly_returns.mean():.1f}%)')
ax5.set_ylabel('Return (%)')
ax5.tick_params(axis='x', rotation=45)
ax5.grid(True, alpha=0.3)

# Panel 6: Cumulative Trades & PnL
ax6 = axes[1, 2]
trades_sorted = trades.sort_values('exit_hour')
trades_sorted['cumulative_pnl'] = trades_sorted['net_pnl'].cumsum()
trades_sorted['trade_num'] = range(1, len(trades_sorted) + 1)
ax6.plot(trades_sorted['trade_num'], trades_sorted['cumulative_pnl'], linewidth=1.5, color='blue')
ax6.axhline(0, color='gray', linestyle='--')
ax6.set_title('Cumulative PnL by Trade Number')
ax6.set_xlabel('Trade Number')
ax6.set_ylabel('Cumulative PnL ($)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mean_reversion_results/final_trade_analysis.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: final_trade_analysis.png")

# =============================================================================
# CHART 4: ALPHA/BETA ANALYSIS
# =============================================================================
print("\n4. Running Alpha/Beta Analysis...")

# Prepare daily returns
hourly_clean = hourly.dropna(subset=['strategy_return', 'btc_return'])
daily_data = hourly_clean.set_index('hour').resample('D').agg({
    'strategy_return': 'sum',
    'btc_return': 'sum',
    'equity': 'last',
    'btc_equity': 'last'
}).dropna()

# Run regression: Strategy Return = Alpha + Beta * BTC Return
X = daily_data['btc_return'].values
Y = daily_data['strategy_return'].values

slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

beta = slope
alpha_daily = intercept
alpha_annual = alpha_daily * 365 * 100  # Annualized as percentage
r_squared = r_value ** 2

print(f"\n   ALPHA/BETA RESULTS:")
print(f"   Beta:           {beta:.3f}")
print(f"   Alpha (daily):  {alpha_daily*100:.4f}%")
print(f"   Alpha (annual): {alpha_annual:.2f}%")
print(f"   R-squared:      {r_squared:.3f}")

# Calculate return attribution
total_strategy_return = daily_data['strategy_return'].sum()
total_btc_return = daily_data['btc_return'].sum()
beta_contribution = beta * total_btc_return
alpha_contribution = total_strategy_return - beta_contribution

print(f"\n   RETURN ATTRIBUTION:")
print(f"   Total Strategy Return: {total_strategy_return*100:.1f}%")
print(f"   From Beta (market):    {beta_contribution*100:.1f}%")
print(f"   From Alpha (skill):    {alpha_contribution*100:.1f}%")

# Create Alpha/Beta chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Scatter plot with regression line
ax1 = axes[0, 0]
ax1.scatter(X * 100, Y * 100, alpha=0.5, s=20, color='blue')
x_line = np.linspace(X.min(), X.max(), 100)
y_line = intercept + slope * x_line
ax1.plot(x_line * 100, y_line * 100, color='red', linewidth=2, 
         label=f'y = {alpha_daily*100:.4f} + {beta:.3f}x')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('BTC Daily Return (%)')
ax1.set_ylabel('Strategy Daily Return (%)')
ax1.set_title(f'Alpha/Beta Regression (R² = {r_squared:.3f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Return Attribution
ax2 = axes[0, 1]
categories = ['Total\nReturn', 'From\nBeta', 'From\nAlpha']
values = [total_strategy_return * 100, beta_contribution * 100, alpha_contribution * 100]
colors = ['blue', 'orange', 'green']
bars = ax2.bar(categories, values, color=colors, alpha=0.7)
ax2.axhline(0, color='gray', linestyle='--')
ax2.set_title('Return Attribution')
ax2.set_ylabel('Return (%)')
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}%', 
             ha='center', va='bottom' if val >= 0 else 'top', fontsize=12)
ax2.grid(True, alpha=0.3)

# Panel 3: Rolling Beta
ax3 = axes[1, 0]
window = 30  # 30 days
rolling_data = daily_data.copy()
rolling_betas = []
for i in range(window, len(rolling_data)):
    x_window = rolling_data['btc_return'].iloc[i-window:i].values
    y_window = rolling_data['strategy_return'].iloc[i-window:i].values
    if len(x_window) > 0 and np.std(x_window) > 0:
        b, _, _, _, _ = stats.linregress(x_window, y_window)
        rolling_betas.append(b)
    else:
        rolling_betas.append(np.nan)

rolling_beta_series = pd.Series(rolling_betas, index=rolling_data.index[window:])
ax3.plot(rolling_beta_series.index, rolling_beta_series.values, linewidth=1.5, color='blue')
ax3.axhline(beta, color='red', linestyle='--', label=f'Overall Beta: {beta:.3f}')
ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax3.axhline(1, color='gray', linestyle=':', alpha=0.5)
ax3.set_title('Rolling 30-Day Beta')
ax3.set_ylabel('Beta')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Cumulative Alpha
ax4 = axes[1, 1]
daily_data['alpha_return'] = daily_data['strategy_return'] - beta * daily_data['btc_return']
cumulative_alpha = daily_data['alpha_return'].cumsum() * 100
cumulative_beta_contrib = (beta * daily_data['btc_return']).cumsum() * 100
ax4.plot(daily_data.index, cumulative_alpha, label=f'Alpha: {cumulative_alpha.iloc[-1]:.1f}%', 
         linewidth=1.5, color='green')
ax4.plot(daily_data.index, cumulative_beta_contrib, label=f'Beta: {cumulative_beta_contrib.iloc[-1]:.1f}%',
         linewidth=1.5, color='orange')
ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax4.set_title('Cumulative Return Attribution')
ax4.set_ylabel('Cumulative Return (%)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mean_reversion_results/final_alpha_beta_analysis.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: final_alpha_beta_analysis.png")

# =============================================================================
# CHART 5: HEDGE ANALYSIS
# =============================================================================
print("\n5. Creating hedge analysis charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Hedge Notional Over Time
ax1 = axes[0, 0]
ax1.plot(hourly['hour'], hourly['btc_hedge'], linewidth=1, color='orange', label='BTC Hedge Notional')
ax1.plot(hourly['hour'], hourly['target_hedge'], linewidth=0.5, color='blue', alpha=0.5, label='Target Hedge')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('BTC Hedge Notional Over Time')
ax1.set_ylabel('Notional ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Hedge PnL Contribution
ax2 = axes[0, 1]
cum_hedge_pnl = hourly['hedge_pnl'].cumsum()
ax2.plot(hourly['hour'], cum_hedge_pnl, linewidth=1.5, color='orange')
ax2.fill_between(hourly['hour'], 0, cum_hedge_pnl, alpha=0.3, color='orange')
ax2.axhline(0, color='gray', linestyle='--')
ax2.set_title(f'Cumulative Hedge PnL (Total: ${cum_hedge_pnl.iloc[-1]:,.0f})')
ax2.set_ylabel('Hedge PnL ($)')
ax2.grid(True, alpha=0.3)

# Panel 3: Correlation with BTC
ax3 = axes[1, 0]
window = 24 * 7  # 7-day rolling correlation
rolling_corr = hourly['strategy_return'].rolling(window).corr(hourly['btc_return'])
ax3.plot(hourly['hour'], rolling_corr, linewidth=1, color='purple')
ax3.axhline(rolling_corr.mean(), color='red', linestyle='--', 
            label=f'Mean: {rolling_corr.mean():.3f}')
ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax3.set_title('Rolling 7-Day Correlation with BTC')
ax3.set_ylabel('Correlation')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: PnL Breakdown Bar Chart
ax4 = axes[1, 1]
price_pnl = hourly['price_pnl'].sum()
hedge_pnl = hourly['hedge_pnl'].sum()
funding_pnl = hourly['funding_pnl'].sum()
# Estimate fees
total_fees = trades['fees'].sum()
hedge_fees = (hourly['btc_hedge'].diff().abs().sum() * 0.00045)  # Approx hedge rebalancing fees
position_fees = total_fees

categories = ['Price\nPnL', 'Hedge\nPnL', 'Funding\nPnL', 'Position\nFees', 'Hedge\nFees']
values = [price_pnl, hedge_pnl, funding_pnl, -position_fees, -hedge_fees]
colors = ['blue', 'orange', 'green', 'red', 'darkred']
bars = ax4.bar(categories, values, color=colors, alpha=0.7)
ax4.axhline(0, color='gray', linestyle='--')
ax4.set_title('Complete PnL Breakdown')
ax4.set_ylabel('PnL ($)')
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2, val, f'${val:,.0f}', 
             ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mean_reversion_results/final_hedge_analysis.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: final_hedge_analysis.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

total_pnl = hourly['equity'].iloc[-1] - STARTING_CAPITAL
hourly_returns = hourly['equity'].pct_change().dropna()
sharpe = hourly_returns.mean() / hourly_returns.std() * np.sqrt(24 * 365)

print(f"""
PERFORMANCE METRICS:
────────────────────────────────────────────────────────────
  Total PnL:        ${total_pnl:,.0f} ({total_pnl/STARTING_CAPITAL*100:.0f}% return)
  Max Drawdown:     {drawdown.max():.1f}%
  Sharpe Ratio:     {sharpe:.2f}
  Total Trades:     {len(trades)}
  Win Rate:         {(trades['net_pnl'] > 0).mean()*100:.1f}%

PNL BREAKDOWN:
────────────────────────────────────────────────────────────
  Price PnL:        ${price_pnl:,.0f}
  Hedge PnL:        ${hedge_pnl:,.0f}
  Funding PnL:      ${funding_pnl:,.0f}
  Total Fees:       ${total_fees + hedge_fees:,.0f}

ALPHA/BETA ANALYSIS:
────────────────────────────────────────────────────────────
  Beta:             {beta:.3f}
  Alpha (annual):   {alpha_annual:.1f}%
  R-squared:        {r_squared:.3f}
  
  Return from Beta: {beta_contribution*100:.1f}%
  Return from Alpha:{alpha_contribution*100:.1f}%
  Alpha % of Total: {alpha_contribution/total_strategy_return*100:.1f}%

CHARTS GENERATED:
────────────────────────────────────────────────────────────
  ✅ final_backtest_charts.png
  ✅ final_btc_comparison.png
  ✅ final_trade_analysis.png
  ✅ final_alpha_beta_analysis.png
  ✅ final_hedge_analysis.png
""")
