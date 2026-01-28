"""
Generate equity curve and performance visualizations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime


def load_trades(filepath='backtest_trades.csv'):
    """Load trade data."""
    if not Path(filepath).exists():
        print(f"Error: {filepath} not found. Run backtest first.")
        return None
    
    df = pd.read_csv(filepath)
    if 'entry_time' in df.columns:
        df['entry_time'] = pd.to_datetime(df['entry_time'])
    if 'exit_time' in df.columns:
        df['exit_time'] = pd.to_datetime(df['exit_time'])
    return df


def create_equity_curve(trades_df, initial_capital):
    """Create equity curve from trades."""
    equity_curve = [initial_capital]
    timestamps = [trades_df['entry_time'].iloc[0]]
    
    for idx, trade in trades_df.iterrows():
        equity_curve.append(trade['capital_after'])
        timestamps.append(trade['exit_time'])
    
    return pd.DataFrame({
        'time': timestamps,
        'equity': equity_curve
    })


def plot_equity_performance(trades_df=None, initial_capital=10000, save_path='equity_performance.png'):
    """
    Create comprehensive equity curve visualization.
    
    Args:
        trades_df: DataFrame with trade data (optional, will load from file if None)
        initial_capital: Starting capital
        save_path: Where to save the plot
    """
    # Load trades if not provided
    if trades_df is None:
        trades_df = load_trades()
        if trades_df is None:
            return
    
    # Extract initial capital from first trade if available
    if 'capital_after' in trades_df.columns and len(trades_df) > 0:
        # Back-calculate initial capital
        first_pnl = trades_df.iloc[0]['pnl']
        first_capital_after = trades_df.iloc[0]['capital_after']
        initial_capital = first_capital_after - first_pnl
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. Main Equity Curve (large, top)
    ax1 = fig.add_subplot(gs[0:2, :])
    equity_curve = create_equity_curve(trades_df, initial_capital)
    
    # Plot equity
    ax1.plot(equity_curve['time'], equity_curve['equity'], 
             linewidth=2.5, color='#2E86AB', label='Portfolio Value')
    ax1.fill_between(equity_curve['time'], initial_capital, equity_curve['equity'],
                     where=(equity_curve['equity'] >= initial_capital), 
                     alpha=0.3, color='green', label='Profit')
    ax1.fill_between(equity_curve['time'], initial_capital, equity_curve['equity'],
                     where=(equity_curve['equity'] < initial_capital), 
                     alpha=0.3, color='red', label='Loss')
    
    # Add initial capital line
    ax1.axhline(y=initial_capital, color='black', linestyle='--', 
                linewidth=1.5, alpha=0.7, label=f'Initial Capital (${initial_capital:,.0f})')
    
    # Calculate and display stats on plot
    final_capital = equity_curve['equity'].iloc[-1]
    total_return = (final_capital / initial_capital - 1) * 100
    max_equity = equity_curve['equity'].max()
    min_equity = equity_curve['equity'].min()
    
    # Add text box with key metrics
    textstr = f'Initial: ${initial_capital:,.0f}\n'
    textstr += f'Final: ${final_capital:,.0f}\n'
    textstr += f'Return: {total_return:+.2f}%\n'
    textstr += f'Peak: ${max_equity:,.0f}\n'
    textstr += f'Trough: ${min_equity:,.0f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    # 2. Drawdown Chart
    ax2 = fig.add_subplot(gs[2, :])
    running_max = equity_curve['equity'].expanding().max()
    drawdown = (equity_curve['equity'] - running_max) / running_max * 100
    
    ax2.fill_between(equity_curve['time'], 0, drawdown, 
                     color='red', alpha=0.4)
    ax2.plot(equity_curve['time'], drawdown, color='darkred', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    max_dd = drawdown.min()
    ax2.set_title(f'Drawdown (Max: {max_dd:.2f}%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    # 3. Cumulative Returns
    ax3 = fig.add_subplot(gs[3, 0])
    trades_df['cumulative_return'] = ((trades_df['capital_after'] / initial_capital) - 1) * 100
    ax3.plot(range(len(trades_df)), trades_df['cumulative_return'], 
             linewidth=2, color='#06A77D')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.fill_between(range(len(trades_df)), 0, trades_df['cumulative_return'],
                     where=(trades_df['cumulative_return'] >= 0), alpha=0.3, color='green')
    ax3.fill_between(range(len(trades_df)), 0, trades_df['cumulative_return'],
                     where=(trades_df['cumulative_return'] < 0), alpha=0.3, color='red')
    ax3.set_title('Cumulative Return %', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Trade Number', fontsize=10)
    ax3.set_ylabel('Return (%)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Rolling Sharpe Ratio (30-trade window)
    ax4 = fig.add_subplot(gs[3, 1])
    window = min(30, len(trades_df) // 3)
    if window >= 5:
        returns = trades_df['pnl_pct']
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)  # Annualized
        
        ax4.plot(range(len(rolling_sharpe)), rolling_sharpe, 
                linewidth=2, color='#A23B72')
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax4.axhline(y=1, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Sharpe=1')
        ax4.fill_between(range(len(rolling_sharpe)), 0, rolling_sharpe,
                        where=(rolling_sharpe >= 0), alpha=0.3, color='purple')
        ax4.set_title(f'Rolling Sharpe Ratio ({window}-trade window)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Trade Number', fontsize=10)
        ax4.set_ylabel('Sharpe Ratio', fontsize=10)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor rolling Sharpe', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
    
    # Overall title
    fig.suptitle('Extreme Funding Rate Strategy - Equity Performance', 
                fontsize=18, fontweight='bold', y=0.995)
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Equity performance chart saved to {save_path}")
    plt.close()
    
    return equity_curve


def print_performance_summary(trades_df, initial_capital):
    """Print detailed performance summary."""
    final_capital = trades_df['capital_after'].iloc[-1]
    total_return = (final_capital / initial_capital - 1) * 100
    
    print("\n" + "="*80)
    print("EQUITY PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\n💰 Capital:")
    print(f"  Initial Capital:        ${initial_capital:,.2f}")
    print(f"  Final Capital:          ${final_capital:,.2f}")
    print(f"  Total PnL:              ${final_capital - initial_capital:+,.2f}")
    print(f"  Total Return:           {total_return:+.2f}%")
    
    # Calculate equity stats
    equity_curve = create_equity_curve(trades_df, initial_capital)
    peak = equity_curve['equity'].max()
    trough = equity_curve['equity'].min()
    
    running_max = equity_curve['equity'].expanding().max()
    drawdown = (equity_curve['equity'] - running_max) / running_max * 100
    max_dd = drawdown.min()
    
    print(f"\n📈 Equity Stats:")
    print(f"  Peak Equity:            ${peak:,.2f} ({(peak/initial_capital-1)*100:+.2f}%)")
    print(f"  Trough Equity:          ${trough:,.2f} ({(trough/initial_capital-1)*100:+.2f}%)")
    print(f"  Max Drawdown:           {max_dd:.2f}%")
    
    # Trade stats
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]
    
    print(f"\n🎯 Trade Performance:")
    print(f"  Total Trades:           {len(trades_df)}")
    print(f"  Winning Trades:         {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.1f}%)")
    print(f"  Losing Trades:          {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}%)")
    print(f"  Best Trade:             ${trades_df['pnl'].max():+,.2f} ({trades_df['pnl_pct'].max():+.2f}%)")
    print(f"  Worst Trade:            ${trades_df['pnl'].min():+,.2f} ({trades_df['pnl_pct'].min():+.2f}%)")
    print(f"  Average PnL:            ${trades_df['pnl'].mean():+,.2f}")
    
    print("\n" + "="*80)


def main():
    """Main execution."""
    print("="*80)
    print("GENERATING EQUITY PERFORMANCE VISUALIZATION")
    print("="*80)
    
    # Load trades
    trades_df = load_trades()
    if trades_df is None:
        print("\n⚠️  No trade data found.")
        print("Run the backtest first: python backtest_strategy.py")
        return
    
    # Get initial capital
    if 'capital_after' in trades_df.columns and len(trades_df) > 0:
        first_pnl = trades_df.iloc[0]['pnl']
        first_capital_after = trades_df.iloc[0]['capital_after']
        initial_capital = first_capital_after - first_pnl
    else:
        initial_capital = 10000
    
    # Generate plot
    print(f"\nLoaded {len(trades_df)} trades")
    equity_curve = plot_equity_performance(trades_df, initial_capital)
    
    # Print summary
    print_performance_summary(trades_df, initial_capital)
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
