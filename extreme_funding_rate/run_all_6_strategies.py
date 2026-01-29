"""
Run all 8 strategies with proper price cache and compare results.
Includes visualization of equity curves and performance metrics.
"""
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASEDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASEDIR)

from config import load_config

# Set style for better looking charts
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Load config
config = load_config()
INITIAL_CAPITAL = config['initial_capital']
POSITION_SIZE = config.get('position_size_fixed', 1000)
TRANSACTION_COST = config['transaction_cost']

# Load data
print("Loading data...")
price_cache = pd.read_csv('price_cache_with_beta_history.csv')
price_cache['timestamp'] = pd.to_datetime(price_cache['timestamp'])
price_lookup = dict(zip(
    price_cache['coin'] + '_' + price_cache['timestamp'].astype(str),
    price_cache['price']
))

funding = pd.read_csv('funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed')
funding['hour'] = funding['datetime'].dt.floor('h')

# Get trading hours
hours = sorted(funding['hour'].unique())
# Filter to hours where we have price data
price_hours = set(price_cache['timestamp'].unique())
hours = [h for h in hours if h in price_hours and (h + timedelta(hours=1)) in price_hours]
print(f"Trading hours: {len(hours)} (from {hours[0]} to {hours[-1]})")

def get_price(coin, timestamp):
    key = f"{coin}_{timestamp}"
    return price_lookup.get(key)

def get_top_negative(hour, n=1):
    """Get the single most negative funding coin (no substitution)."""
    hour_data = funding[funding['hour'] == hour].sort_values('funding_rate').head(n)
    return [(r['coin'], r['funding_rate']) for _, r in hour_data.iterrows()]

def get_top_positive(hour, n=1):
    """Get the single most positive funding coin (no substitution)."""
    hour_data = funding[funding['hour'] == hour].sort_values('funding_rate', ascending=False).head(n)
    return [(r['coin'], r['funding_rate']) for _, r in hour_data.iterrows()]

def calculate_beta(coin, current_time, lookback_hours=720):
    """Calculate rolling beta vs BTC."""
    start_time = current_time - timedelta(hours=lookback_hours)
    
    # Get price data
    coin_data = price_cache[(price_cache['coin'] == coin) & 
                            (price_cache['timestamp'] >= start_time) & 
                            (price_cache['timestamp'] < current_time)].sort_values('timestamp')
    btc_data = price_cache[(price_cache['coin'] == 'BTC') & 
                           (price_cache['timestamp'] >= start_time) & 
                           (price_cache['timestamp'] < current_time)].sort_values('timestamp')
    
    if len(coin_data) < 24 or len(btc_data) < 24:
        return 1.0  # Default to delta hedge
    
    # Calculate returns
    coin_ret = coin_data.set_index('timestamp')['price'].pct_change().dropna()
    btc_ret = btc_data.set_index('timestamp')['price'].pct_change().dropna()
    
    common = coin_ret.index.intersection(btc_ret.index)
    if len(common) < 24:
        return 1.0
    
    cov = coin_ret.loc[common].cov(btc_ret.loc[common])
    var = btc_ret.loc[common].var()
    
    if var == 0:
        return 1.0
    
    return cov / var

def calculate_normal_beta(coin, current_time, lookback_hours=720):
    """Calculate rolling beta vs BTC using only normal funding periods.
    
    Normal funding is defined as: -0.01%/8 to +0.01%/8 (i.e., -0.000125 to +0.000125)
    This filters out extreme funding periods for more stable beta estimation.
    """
    start_time = current_time - timedelta(hours=lookback_hours)
    normal_threshold = 0.0001 / 8  # 0.01% / 8 = 0.000125
    
    # Get normal funding hours for this coin
    normal_hours = funding[
        (funding['coin'] == coin) &
        (funding['hour'] >= start_time) &
        (funding['hour'] < current_time) &
        (funding['funding_rate'].abs() <= normal_threshold)
    ]['hour'].unique()
    
    if len(normal_hours) < 24:
        return 1.0  # Not enough normal periods, default to delta
    
    # Get price data only for normal hours
    coin_data = price_cache[
        (price_cache['coin'] == coin) & 
        (price_cache['timestamp'].isin(normal_hours))
    ].sort_values('timestamp')
    
    btc_data = price_cache[
        (price_cache['coin'] == 'BTC') & 
        (price_cache['timestamp'].isin(normal_hours))
    ].sort_values('timestamp')
    
    if len(coin_data) < 24 or len(btc_data) < 24:
        return 1.0
    
    # Calculate returns
    coin_ret = coin_data.set_index('timestamp')['price'].pct_change().dropna()
    btc_ret = btc_data.set_index('timestamp')['price'].pct_change().dropna()
    
    common = coin_ret.index.intersection(btc_ret.index)
    if len(common) < 24:
        return 1.0
    
    cov = coin_ret.loc[common].cov(btc_ret.loc[common])
    var = btc_ret.loc[common].var()
    
    if var == 0:
        return 1.0
    
    return cov / var

def run_strategy(name, get_coins_func, direction, hedge_type):
    """
    Run a strategy.
    direction: 'long' or 'short' on the altcoin
    hedge_type: None, 'delta', 'beta', or 'normal_beta'
    Returns results dict including equity curve for visualization.
    """
    capital = INITIAL_CAPITAL
    trades = []
    equity = [(hours[0], capital)]
    
    for hour in hours[:-1]:  # Exclude last hour (no exit price)
        entry_time = hour
        exit_time = hour + timedelta(hours=1)
        
        # Get the single most extreme coin (no substitution)
        coins = get_coins_func(hour)
        
        # Only trade the most extreme coin - skip hour if no price data
        trade = None
        if coins:
            coin, fr = coins[0]  # Only the most extreme
            entry_price = get_price(coin, entry_time)
            exit_price = get_price(coin, exit_time)
            btc_entry = get_price('BTC', entry_time)
            btc_exit = get_price('BTC', exit_time)
            
            if entry_price and exit_price and btc_entry and btc_exit:
                # Calculate position
                trade_capital = min(POSITION_SIZE, capital)
                
                if hedge_type == 'beta':
                    beta = calculate_beta(coin, entry_time)
                elif hedge_type == 'normal_beta':
                    beta = calculate_normal_beta(coin, entry_time)
                elif hedge_type == 'delta':
                    beta = 1.0
                else:
                    beta = 0  # No hedge
                
                # Split capital between coin and hedge
                if beta > 0:
                    coin_capital = trade_capital / (1 + beta)
                    btc_capital = trade_capital * beta / (1 + beta)
                else:
                    coin_capital = trade_capital
                    btc_capital = 0
                
                # Entry costs
                entry_cost = (coin_capital + btc_capital) * TRANSACTION_COST
                
                # Coin position
                coin_position = (coin_capital - entry_cost/2) / entry_price
                if direction == 'long':
                    coin_pnl = coin_position * (exit_price - entry_price)
                else:  # short
                    coin_pnl = coin_position * (entry_price - exit_price)
                
                # Hedge position (opposite direction)
                if btc_capital > 0:
                    btc_position = (btc_capital - entry_cost/2) / btc_entry
                    if direction == 'long':
                        # Long coin + Short BTC
                        btc_pnl = btc_position * (btc_entry - btc_exit)
                    else:
                        # Short coin + Long BTC  
                        btc_pnl = btc_position * (btc_exit - btc_entry)
                else:
                    btc_pnl = 0
                
                # Exit costs
                exit_value = coin_capital + coin_pnl + btc_capital + btc_pnl
                exit_cost = exit_value * TRANSACTION_COST
                
                total_pnl = coin_pnl + btc_pnl - entry_cost - exit_cost
                
                trade = {
                    'entry_time': entry_time,
                    'coin': coin,
                    'funding_rate': fr,
                    'beta': beta,
                    'pnl': total_pnl
                }
        
        if trade:
            trades.append(trade)
            capital += trade['pnl']
        
        equity.append((exit_time, capital))
    
    # Calculate metrics
    df = pd.DataFrame(trades)
    if df.empty:
        return {'name': name, 'return': None}
    
    final = capital
    total_return = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    wins = len(df[df['pnl'] > 0])
    win_rate = wins / len(df) * 100
    
    # Max drawdown
    eq_df = pd.DataFrame(equity, columns=['time', 'capital'])
    eq_df['peak'] = eq_df['capital'].cummax()
    eq_df['dd'] = (eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100
    max_dd = abs(eq_df['dd'].min())
    
    avg_beta = df['beta'].mean() if 'beta' in df else None
    
    # Sharpe Ratio (annualized)
    # Calculate hourly returns from PnL
    hourly_returns = df['pnl'] / POSITION_SIZE  # Return per trade
    if len(hourly_returns) > 1 and hourly_returns.std() > 0:
        # Annualize: sqrt(8760) for hourly data (8760 hours/year)
        sharpe = (hourly_returns.mean() / hourly_returns.std()) * np.sqrt(8760)
    else:
        sharpe = 0.0
    
    return {
        'name': name,
        'return': total_return,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'trades': len(df),
        'avg_beta': avg_beta,
        'sharpe': sharpe,
        'final_capital': final,
        'equity_curve': eq_df,  # Add equity curve for plotting
        'trades_df': df  # Add trades for analysis
    }

# Run all 8 strategies
print("\n" + "="*70)
print("RUNNING ALL 8 STRATEGIES")
print("="*70)

strategies = [
    ("Mean Reversion (LONG)", get_top_negative, 'long', None),
    ("Mean Reversion + Delta Hedge", get_top_negative, 'long', 'delta'),
    ("Mean Reversion + Beta Hedge", get_top_negative, 'long', 'beta'),
    ("Mean Reversion + Normal Beta", get_top_negative, 'long', 'normal_beta'),
    ("Trend Following (SHORT)", get_top_negative, 'short', None),
    ("Trend Following + Delta Hedge", get_top_negative, 'short', 'delta'),
    ("Trend Following + Beta Hedge", get_top_negative, 'short', 'beta'),
    ("Trend Following + Normal Beta", get_top_negative, 'short', 'normal_beta'),
]

results = []
for name, get_coins, direction, hedge in strategies:
    print(f"\nRunning: {name}...")
    result = run_strategy(name, get_coins, direction, hedge)
    results.append(result)
    if result['return'] is not None:
        print(f"  Return: {result['return']:+.2f}% | Win Rate: {result['win_rate']:.1f}% | Max DD: {result['max_dd']:.2f}% | Sharpe: {result['sharpe']:.2f}")

# Summary
print("\n" + "="*70)
print("FINAL COMPARISON - ALL 8 STRATEGIES")
print("="*70)
print(f"\n{'Strategy':<35} {'Return':>10} {'Win Rate':>10} {'Max DD':>10} {'Sharpe':>10} {'Trades':>8}")
print("-" * 95)

for r in sorted(results, key=lambda x: x.get('return') or -999, reverse=True):
    ret = f"{r['return']:+.2f}%" if r['return'] is not None else 'N/A'
    wr = f"{r['win_rate']:.1f}%" if r.get('win_rate') else 'N/A'
    dd = f"{r['max_dd']:.2f}%" if r.get('max_dd') else 'N/A'
    sharpe = f"{r['sharpe']:.2f}" if r.get('sharpe') is not None else 'N/A'
    trades = str(r.get('trades', 'N/A'))
    print(f"{r['name']:<35} {ret:>10} {wr:>10} {dd:>10} {sharpe:>10} {trades:>8}")

print("="*70)

# Save results
results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['equity_curve', 'trades_df']} for r in results])
results_df.to_csv('strategy_comparison_all8.csv', index=False)
print(f"\n💾 Results saved to strategy_comparison_all8.csv")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_equity_curves(results):
    """Plot equity curves for all strategies."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color palette for different strategy types
    colors = {
        'Mean Reversion (LONG)': '#2ecc71',
        'Mean Reversion + Delta Hedge': '#27ae60',
        'Mean Reversion + Beta Hedge': '#1abc9c',
        'Mean Reversion + Normal Beta': '#16a085',
        'Trend Following (SHORT)': '#e74c3c',
        'Trend Following + Delta Hedge': '#c0392b',
        'Trend Following + Beta Hedge': '#e67e22',
        'Trend Following + Normal Beta': '#d35400',
    }
    
    for r in results:
        if r.get('equity_curve') is not None:
            eq = r['equity_curve']
            color = colors.get(r['name'], '#3498db')
            ax.plot(eq['time'], eq['capital'], label=f"{r['name']} ({r['return']:+.2f}%)", 
                   color=color, linewidth=1.5, alpha=0.8)
    
    ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax.set_xlabel('Date')
    ax.set_ylabel('Capital ($)')
    ax.set_title('Equity Curves - All 8 Strategies\n(3-Month Backtest)')
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('equity_curves.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: equity_curves.png")
    plt.close()

def plot_performance_comparison(results):
    """Create bar charts comparing strategy performance."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Filter valid results
    valid_results = [r for r in results if r.get('return') is not None]
    names = [r['name'].replace(' + ', '\n+').replace('Mean Reversion', 'MR').replace('Trend Following', 'TF') for r in valid_results]
    
    # Colors based on return
    returns = [r['return'] for r in valid_results]
    colors = ['#2ecc71' if ret > 0 else '#e74c3c' for ret in returns]
    
    # 1. Total Return
    ax1 = axes[0, 0]
    bars = ax1.bar(names, returns, color=colors, edgecolor='white', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Total Return')
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, returns):
        height = bar.get_height()
        ax1.annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3 if height >= 0 else -12), textcoords='offset points',
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    # 2. Win Rate
    ax2 = axes[0, 1]
    win_rates = [r['win_rate'] for r in valid_results]
    bars = ax2.bar(names, win_rates, color='#3498db', edgecolor='white', linewidth=0.5)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50%')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate')
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, win_rates):
        height = bar.get_height()
        ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    ax2.set_ylim(0, 100)
    
    # 3. Max Drawdown
    ax3 = axes[1, 0]
    drawdowns = [r['max_dd'] for r in valid_results]
    bars = ax3.bar(names, drawdowns, color='#e74c3c', edgecolor='white', linewidth=0.5)
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.set_title('Maximum Drawdown (Lower is Better)')
    ax3.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, drawdowns):
        height = bar.get_height()
        ax3.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    
    # 4. Sharpe Ratio
    ax4 = axes[1, 1]
    sharpes = [r['sharpe'] for r in valid_results]
    colors_sharpe = ['#2ecc71' if s > 0 else '#e74c3c' for s in sharpes]
    bars = ax4.bar(names, sharpes, color=colors_sharpe, edgecolor='white', linewidth=0.5)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_ylabel('Sharpe Ratio')
    ax4.set_title('Sharpe Ratio (Annualized)')
    ax4.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, sharpes):
        height = bar.get_height()
        ax4.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3 if height >= 0 else -12), textcoords='offset points',
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: performance_comparison.png")
    plt.close()

def plot_drawdown_curves(results):
    """Plot drawdown curves for all strategies."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    for i, r in enumerate(results):
        if r.get('equity_curve') is not None:
            eq = r['equity_curve']
            short_name = r['name'].replace('Mean Reversion', 'MR').replace('Trend Following', 'TF')
            ax.fill_between(eq['time'], eq['dd'], 0, alpha=0.3, color=colors[i])
            ax.plot(eq['time'], eq['dd'], label=short_name, color=colors[i], linewidth=1)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown Over Time - All 8 Strategies')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    ax.set_ylim(top=0)
    
    plt.tight_layout()
    plt.savefig('drawdown_curves.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: drawdown_curves.png")
    plt.close()

def plot_cumulative_pnl_distribution(results):
    """Plot distribution of daily PnL for each strategy type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean Reversion strategies
    ax1 = axes[0]
    mr_strategies = [r for r in results if 'Mean Reversion' in r['name'] and r.get('trades_df') is not None]
    for r in mr_strategies:
        pnl_cumsum = r['trades_df']['pnl'].cumsum()
        short_name = r['name'].replace('Mean Reversion', 'MR')
        ax1.plot(range(len(pnl_cumsum)), pnl_cumsum, label=short_name, linewidth=1.2)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Cumulative PnL ($)')
    ax1.set_title('Mean Reversion Strategies - Cumulative PnL')
    ax1.legend(fontsize=8)
    
    # Trend Following strategies  
    ax2 = axes[1]
    tf_strategies = [r for r in results if 'Trend Following' in r['name'] and r.get('trades_df') is not None]
    for r in tf_strategies:
        pnl_cumsum = r['trades_df']['pnl'].cumsum()
        short_name = r['name'].replace('Trend Following', 'TF')
        ax2.plot(range(len(pnl_cumsum)), pnl_cumsum, label=short_name, linewidth=1.2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Cumulative PnL ($)')
    ax2.set_title('Trend Following Strategies - Cumulative PnL')
    ax2.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('cumulative_pnl.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: cumulative_pnl.png")
    plt.close()

def plot_summary_dashboard(results):
    """Create a summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # 1. Main equity curves (large, top)
    ax_equity = fig.add_subplot(gs[0:2, :2])
    colors_list = ['#2ecc71', '#27ae60', '#1abc9c', '#16a085', '#e74c3c', '#c0392b', '#e67e22', '#d35400']
    for i, r in enumerate(results):
        if r.get('equity_curve') is not None:
            eq = r['equity_curve']
            short_name = r['name'].replace('Mean Reversion', 'MR').replace('Trend Following', 'TF')
            ax_equity.plot(eq['time'], eq['capital'], label=f"{short_name}", 
                          color=colors_list[i % len(colors_list)], linewidth=1.5, alpha=0.8)
    ax_equity.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
    ax_equity.set_title('Equity Curves - All Strategies', fontsize=12, fontweight='bold')
    ax_equity.set_ylabel('Capital ($)')
    ax_equity.legend(loc='upper left', fontsize=7)
    ax_equity.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # 2. Summary table (top right)
    ax_table = fig.add_subplot(gs[0:2, 2])
    ax_table.axis('off')
    
    valid_results = sorted([r for r in results if r.get('return') is not None], 
                          key=lambda x: x['return'], reverse=True)
    table_data = []
    for r in valid_results:
        name = r['name'].replace('Mean Reversion', 'MR').replace('Trend Following', 'TF').replace(' + ', '\n+')
        table_data.append([
            name[:20],
            f"{r['return']:+.1f}%",
            f"{r['win_rate']:.0f}%",
            f"{r['max_dd']:.1f}%",
            f"{r['sharpe']:.1f}"
        ])
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=['Strategy', 'Return', 'Win%', 'MaxDD', 'Sharpe'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # Color the return column
    for i, r in enumerate(valid_results):
        color = '#d4edda' if r['return'] > 0 else '#f8d7da'
        table[(i+1, 1)].set_facecolor(color)
    
    ax_table.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    # 3. Return comparison (bottom left)
    ax_return = fig.add_subplot(gs[2, 0])
    names = [r['name'].replace('Mean Reversion', 'MR').replace('Trend Following', 'TF').replace(' + ', '\n+') for r in valid_results]
    returns = [r['return'] for r in valid_results]
    colors = ['#2ecc71' if ret > 0 else '#e74c3c' for ret in returns]
    ax_return.barh(names, returns, color=colors)
    ax_return.axvline(x=0, color='black', linewidth=0.5)
    ax_return.set_xlabel('Return (%)')
    ax_return.set_title('Returns Ranked', fontsize=10)
    ax_return.tick_params(axis='y', labelsize=7)
    
    # 4. Risk-adjusted (Sharpe) comparison (bottom middle)
    ax_sharpe = fig.add_subplot(gs[2, 1])
    sharpes = [r['sharpe'] for r in valid_results]
    colors_sharpe = ['#2ecc71' if s > 0 else '#e74c3c' for s in sharpes]
    ax_sharpe.barh(names, sharpes, color=colors_sharpe)
    ax_sharpe.axvline(x=0, color='black', linewidth=0.5)
    ax_sharpe.set_xlabel('Sharpe Ratio')
    ax_sharpe.set_title('Sharpe Ratio Ranked', fontsize=10)
    ax_sharpe.tick_params(axis='y', labelsize=7)
    
    # 5. Win rate vs Return scatter (bottom right)
    ax_scatter = fig.add_subplot(gs[2, 2])
    for i, r in enumerate(valid_results):
        color = '#2ecc71' if 'Mean Reversion' in r['name'] else '#e74c3c'
        marker = 'o' if 'Delta' not in r['name'] and 'Beta' not in r['name'] else 's'
        ax_scatter.scatter(r['win_rate'], r['return'], color=color, s=100, marker=marker, alpha=0.7)
        ax_scatter.annotate(r['name'][:10], (r['win_rate'], r['return']), fontsize=6, alpha=0.8)
    ax_scatter.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_scatter.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    ax_scatter.set_xlabel('Win Rate (%)')
    ax_scatter.set_ylabel('Return (%)')
    ax_scatter.set_title('Win Rate vs Return', fontsize=10)
    
    plt.suptitle('Extreme Funding Rate Strategy Backtest - 3 Month Summary', fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig('backtest_dashboard.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: backtest_dashboard.png")
    plt.close()

# Generate all visualizations
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

plot_equity_curves(results)
plot_performance_comparison(results)
plot_drawdown_curves(results)
plot_cumulative_pnl_distribution(results)
plot_summary_dashboard(results)

print("\n✅ All visualizations generated successfully!")
print("   - equity_curves.png")
print("   - performance_comparison.png") 
print("   - drawdown_curves.png")
print("   - cumulative_pnl.png")
print("   - backtest_dashboard.png")
