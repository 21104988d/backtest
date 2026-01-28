"""
Example script showing how to load and visualize the results
after running the backtest.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_results():
    """Load all result files."""
    results = {}
    
    files = {
        'trades': 'backtest_trades.csv',
        'metrics': 'backtest_metrics.csv',
        'funding': 'funding_history.csv',
        'extremes': 'extreme_funding_events.csv'
    }
    
    for key, filename in files.items():
        if Path(filename).exists():
            results[key] = pd.read_csv(filename)
            print(f"✓ Loaded {filename}")
        else:
            print(f"⚠ Missing {filename}")
            results[key] = None
    
    return results


def plot_custom_analysis(trades_df):
    """Create custom analysis plots."""
    
    if trades_df is None or trades_df.empty:
        print("No trade data available")
        return
    
    # Convert datetime columns
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Monthly Returns
    ax1 = fig.add_subplot(gs[0, :2])
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    monthly_returns = trades_df.groupby('month')['pnl'].sum()
    colors = ['green' if x > 0 else 'red' for x in monthly_returns.values]
    monthly_returns.plot(kind='bar', ax=ax1, color=colors, alpha=0.7)
    ax1.set_title('Monthly PnL', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('PnL ($)')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    
    # 2. Top Coins by Trade Count
    ax2 = fig.add_subplot(gs[0, 2])
    top_coins = trades_df['coin'].value_counts().head(10)
    ax2.barh(range(len(top_coins)), top_coins.values, alpha=0.7, color='skyblue')
    ax2.set_yticks(range(len(top_coins)))
    ax2.set_yticklabels(top_coins.index)
    ax2.set_xlabel('Trade Count')
    ax2.set_title('Most Traded Coins', fontsize=10, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Funding Rate Distribution (Traded)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(trades_df['funding_rate'] * 100, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(x=trades_df['funding_rate'].mean() * 100, color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {trades_df["funding_rate"].mean()*100:.4f}%')
    ax3.set_xlabel('Funding Rate (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Funding Rate Distribution (Traded)', fontsize=10, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Win/Loss Streak Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    trades_df['is_win'] = trades_df['pnl'] > 0
    streaks = []
    current_streak = 0
    last_result = None
    
    for is_win in trades_df['is_win']:
        if is_win == last_result or last_result is None:
            current_streak += 1
        else:
            streaks.append((last_result, current_streak))
            current_streak = 1
        last_result = is_win
    streaks.append((last_result, current_streak))
    
    win_streaks = [s[1] for s in streaks if s[0] == True]
    loss_streaks = [s[1] for s in streaks if s[0] == False]
    
    ax4.hist([win_streaks, loss_streaks], bins=20, label=['Win Streaks', 'Loss Streaks'],
             alpha=0.7, color=['green', 'red'])
    ax4.set_xlabel('Streak Length')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Win/Loss Streak Distribution', fontsize=10, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Rolling Win Rate
    ax5 = fig.add_subplot(gs[1, 2])
    window = 50
    rolling_win_rate = trades_df['is_win'].rolling(window=window).mean() * 100
    ax5.plot(rolling_win_rate, linewidth=2, color='purple')
    ax5.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50%')
    ax5.fill_between(range(len(rolling_win_rate)), rolling_win_rate, 50, 
                     where=(rolling_win_rate > 50), alpha=0.3, color='green')
    ax5.fill_between(range(len(rolling_win_rate)), rolling_win_rate, 50, 
                     where=(rolling_win_rate <= 50), alpha=0.3, color='red')
    ax5.set_xlabel('Trade Number')
    ax5.set_ylabel('Win Rate (%)')
    ax5.set_title(f'Rolling Win Rate ({window} trades)', fontsize=10, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. PnL by Coin (Top 15)
    ax6 = fig.add_subplot(gs[2, :])
    coin_pnl = trades_df.groupby('coin')['pnl'].sum().sort_values(ascending=False)
    top_15 = coin_pnl.head(15)
    colors = ['green' if x > 0 else 'red' for x in top_15.values]
    top_15.plot(kind='bar', ax=ax6, color=colors, alpha=0.7)
    ax6.set_title('Top 15 Coins by Total PnL', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Coin')
    ax6.set_ylabel('Total PnL ($)')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax6.grid(True, alpha=0.3, axis='y')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Extreme Funding Rate Strategy - Extended Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('extended_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Extended analysis saved to extended_analysis.png")
    plt.close()


def print_detailed_stats(trades_df, metrics_df):
    """Print detailed statistics."""
    
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    
    if metrics_df is not None and not metrics_df.empty:
        print("\n📊 Overall Metrics:")
        for col in metrics_df.columns:
            value = metrics_df.iloc[0][col]
            if isinstance(value, (int, float)):
                if col.endswith('_pct'):
                    print(f"  {col}: {value:.2f}%")
                elif col in ['sharpe_ratio', 'profit_factor']:
                    print(f"  {col}: {value:.2f}")
                else:
                    print(f"  {col}: ${value:,.2f}" if value > 1000 else f"  {col}: {value:.4f}")
    
    if trades_df is not None and not trades_df.empty:
        print("\n🎯 Trading Patterns:")
        
        # Day of week analysis
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['day_of_week'] = trades_df['entry_time'].dt.day_name()
        trades_df['is_win'] = trades_df['pnl'] > 0
        
        dow_stats = trades_df.groupby('day_of_week').agg({
            'pnl': ['sum', 'mean'],
            'is_win': 'mean'
        })
        
        print("\n  Performance by Day of Week:")
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            if day in dow_stats.index:
                total = dow_stats.loc[day, ('pnl', 'sum')]
                avg = dow_stats.loc[day, ('pnl', 'mean')]
                wr = dow_stats.loc[day, ('is_win', 'mean')] * 100
                print(f"    {day:10s}: Total=${total:7.2f}, Avg=${avg:6.2f}, WR={wr:5.1f}%")
        
        # Hour of day analysis
        trades_df['hour'] = trades_df['entry_time'].dt.hour
        hour_stats = trades_df.groupby('hour').agg({
            'pnl': ['sum', 'mean', 'count'],
            'is_win': 'mean'
        })
        
        print("\n  Best Hours (by total PnL):")
        best_hours = hour_stats.nlargest(5, ('pnl', 'sum'))
        for hour, row in best_hours.iterrows():
            print(f"    Hour {hour:02d}: Total=${row[('pnl', 'sum')]:7.2f}, "
                  f"Trades={int(row[('pnl', 'count')])}, WR={row[('is_win', 'mean')]*100:5.1f}%")
        
        # Funding rate threshold analysis
        print("\n  Performance by Funding Rate Threshold:")
        thresholds = [-0.01, -0.005, -0.001, 0]
        for i in range(len(thresholds)-1):
            lower = thresholds[i]
            upper = thresholds[i+1]
            subset = trades_df[(trades_df['funding_rate'] >= lower) & 
                             (trades_df['funding_rate'] < upper)]
            if len(subset) > 0:
                avg_pnl = subset['pnl'].mean()
                win_rate = (subset['pnl'] > 0).mean() * 100
                print(f"    {lower*100:.2f}% to {upper*100:.2f}%: "
                      f"Trades={len(subset):3d}, Avg=${avg_pnl:6.2f}, WR={win_rate:5.1f}%")


def main():
    """Main function."""
    print("="*80)
    print("LOADING BACKTEST RESULTS FOR CUSTOM ANALYSIS")
    print("="*80)
    
    # Load results
    results = load_results()
    
    if results['trades'] is None:
        print("\n⚠ No trade data found. Run backtest_strategy.py first.")
        return
    
    # Print detailed stats
    print_detailed_stats(results['trades'], results['metrics'])
    
    # Create custom plots
    print("\n" + "="*80)
    print("GENERATING EXTENDED VISUALIZATIONS")
    print("="*80)
    plot_custom_analysis(results['trades'])
    
    print("\n" + "="*80)
    print("Analysis complete! Check extended_analysis.png")
    print("="*80)


if __name__ == "__main__":
    main()
