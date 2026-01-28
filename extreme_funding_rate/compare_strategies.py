"""
Compare Standard vs Delta-Hedged Strategy Performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (16, 12)

def load_results():
    """Load both strategy results"""
    # Standard strategy
    standard_trades = pd.read_csv('backtest_trades.csv')
    standard_metrics = pd.read_csv('backtest_metrics.csv').iloc[0]
    
    # Delta-hedged strategy
    delta_trades = pd.read_csv('backtest_delta_hedged_trades.csv')
    delta_metrics = pd.read_csv('backtest_delta_hedged_metrics.csv').iloc[0]
    
    return standard_trades, standard_metrics, delta_trades, delta_metrics

def create_comparison_table(standard_metrics, delta_metrics):
    """Create side-by-side comparison table"""
    comparison = pd.DataFrame({
        'Standard Strategy': [
            f"${standard_metrics['initial_capital']:,.2f}",
            f"${standard_metrics['final_capital']:,.2f}",
            f"{standard_metrics['total_return_pct']:.2f}%",
            f"${standard_metrics['total_pnl']:,.2f}",
            int(standard_metrics['num_trades']),
            f"{standard_metrics['win_rate_pct']:.2f}%",
            f"${standard_metrics['avg_win']:.2f}",
            f"${standard_metrics['avg_loss']:.2f}",
            f"{standard_metrics['profit_factor']:.2f}",
            f"{standard_metrics['max_drawdown_pct']:.2f}%",
            f"{standard_metrics['sharpe_ratio']:.2f}",
            f"${standard_metrics['avg_pnl_per_trade']:.2f}"
        ],
        'Delta-Hedged Strategy': [
            f"${delta_metrics['initial_capital']:,.2f}",
            f"${delta_metrics['final_capital']:,.2f}",
            f"{delta_metrics['total_return_pct']:.2f}%",
            f"${delta_metrics['total_pnl']:,.2f}",
            int(delta_metrics['num_trades']),
            f"{delta_metrics['win_rate_pct']:.2f}%",
            f"${delta_metrics['avg_win']:.2f}",
            f"${delta_metrics['avg_loss']:.2f}",
            f"{delta_metrics['profit_factor']:.2f}",
            f"{delta_metrics['max_drawdown_pct']:.2f}%",
            f"{delta_metrics['sharpe_ratio']:.2f}",
            f"${delta_metrics['avg_pnl_per_trade']:.2f}"
        ]
    }, index=[
        'Initial Capital',
        'Final Capital',
        'Total Return',
        'Total PnL',
        'Number of Trades',
        'Win Rate',
        'Average Win',
        'Average Loss',
        'Profit Factor',
        'Max Drawdown',
        'Sharpe Ratio',
        'Avg PnL/Trade'
    ])
    
    # Calculate improvement
    improvement = []
    for i, metric in enumerate(comparison.index):
        if metric in ['Initial Capital', 'Number of Trades']:
            improvement.append('-')
        elif metric in ['Total Return', 'Total PnL', 'Final Capital', 'Win Rate', 
                       'Average Win', 'Profit Factor', 'Sharpe Ratio', 'Avg PnL/Trade']:
            std_val = standard_metrics[comparison.index.get_loc(metric)]
            dlt_val = delta_metrics[comparison.index.get_loc(metric)]
            
            if metric == 'Win Rate':
                std_val = standard_metrics['win_rate_pct']
                dlt_val = delta_metrics['win_rate_pct']
            elif metric == 'Average Win':
                std_val = standard_metrics['avg_win']
                dlt_val = delta_metrics['avg_win']
            elif metric == 'Average Loss':
                std_val = standard_metrics['avg_loss']
                dlt_val = delta_metrics['avg_loss']
            elif metric == 'Profit Factor':
                std_val = standard_metrics['profit_factor']
                dlt_val = delta_metrics['profit_factor']
            elif metric == 'Max Drawdown':
                std_val = standard_metrics['max_drawdown_pct']
                dlt_val = delta_metrics['max_drawdown_pct']
            elif metric == 'Sharpe Ratio':
                std_val = standard_metrics['sharpe_ratio']
                dlt_val = delta_metrics['sharpe_ratio']
            elif metric == 'Avg PnL/Trade':
                std_val = standard_metrics['avg_pnl_per_trade']
                dlt_val = delta_metrics['avg_pnl_per_trade']
            elif metric == 'Total Return':
                std_val = standard_metrics['total_return_pct']
                dlt_val = delta_metrics['total_return_pct']
            elif metric == 'Total PnL':
                std_val = standard_metrics['total_pnl']
                dlt_val = delta_metrics['total_pnl']
            elif metric == 'Final Capital':
                std_val = standard_metrics['final_capital']
                dlt_val = delta_metrics['final_capital']
            
            if std_val != 0:
                pct_change = ((dlt_val - std_val) / abs(std_val)) * 100
                if metric == 'Max Drawdown' or metric == 'Average Loss':
                    # For these, lower is better
                    improvement.append(f"{-pct_change:+.1f}% {'✓' if pct_change < 0 else '✗'}")
                else:
                    improvement.append(f"{pct_change:+.1f}% {'✓' if pct_change > 0 else '✗'}")
            else:
                improvement.append('-')
        else:
            improvement.append('-')
    
    comparison['Improvement'] = improvement
    
    return comparison

def plot_comparison(standard_trades, delta_trades, standard_metrics, delta_metrics):
    """Create comprehensive comparison visualization"""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Calculate equity curves
    standard_equity = [10000]
    for pnl in standard_trades['pnl']:
        standard_equity.append(standard_equity[-1] + pnl)
    
    delta_equity = [10000]
    for pnl in delta_trades['pnl']:
        delta_equity.append(delta_equity[-1] + pnl)
    
    # 1. Equity Curves
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(range(len(standard_equity)), standard_equity, label='Standard Strategy', 
             linewidth=2, color='#e74c3c', alpha=0.8)
    ax1.plot(range(len(delta_equity)), delta_equity, label='Delta-Hedged Strategy', 
             linewidth=2, color='#3498db', alpha=0.8)
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_title('Equity Curve Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative Returns
    ax2 = fig.add_subplot(gs[1, 0])
    standard_cumret = ((np.array(standard_equity) / 10000 - 1) * 100)
    delta_cumret = ((np.array(delta_equity) / 10000 - 1) * 100)
    ax2.plot(range(len(standard_cumret)), standard_cumret, label='Standard', 
             linewidth=2, color='#e74c3c', alpha=0.8)
    ax2.plot(range(len(delta_cumret)), delta_cumret, label='Delta-Hedged', 
             linewidth=2, color='#3498db', alpha=0.8)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Cumulative Returns (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Return (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown Comparison
    ax3 = fig.add_subplot(gs[1, 1])
    standard_dd = []
    running_max = 10000
    for val in standard_equity:
        running_max = max(running_max, val)
        dd = ((val - running_max) / running_max) * 100
        standard_dd.append(dd)
    
    delta_dd = []
    running_max = 10000
    for val in delta_equity:
        running_max = max(running_max, val)
        dd = ((val - running_max) / running_max) * 100
        delta_dd.append(dd)
    
    ax3.fill_between(range(len(standard_dd)), standard_dd, 0, alpha=0.3, color='#e74c3c', label='Standard')
    ax3.fill_between(range(len(delta_dd)), delta_dd, 0, alpha=0.3, color='#3498db', label='Delta-Hedged')
    ax3.set_title('Drawdown Comparison (%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Trade Number')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Returns Distribution
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(standard_trades['pnl_pct'], bins=50, alpha=0.5, color='#e74c3c', label='Standard', density=True)
    ax4.hist(delta_trades['pnl_pct'], bins=50, alpha=0.5, color='#3498db', label='Delta-Hedged', density=True)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_title('Return Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Return (%)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Win Rate Comparison
    ax5 = fig.add_subplot(gs[2, 0])
    win_rates = [standard_metrics['win_rate_pct'], delta_metrics['win_rate_pct']]
    colors = ['#e74c3c', '#3498db']
    bars = ax5.bar(['Standard', 'Delta-Hedged'], win_rates, color=colors, alpha=0.7)
    ax5.set_title('Win Rate Comparison', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Win Rate (%)')
    ax5.set_ylim(0, 100)
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Profit Factor Comparison
    ax6 = fig.add_subplot(gs[2, 1])
    pf = [standard_metrics['profit_factor'], delta_metrics['profit_factor']]
    bars = ax6.bar(['Standard', 'Delta-Hedged'], pf, color=colors, alpha=0.7)
    ax6.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Break-even')
    ax6.set_title('Profit Factor Comparison', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Profit Factor')
    for bar, val in zip(bars, pf):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Sharpe Ratio Comparison
    ax7 = fig.add_subplot(gs[2, 2])
    sharpe = [standard_metrics['sharpe_ratio'], delta_metrics['sharpe_ratio']]
    bars = ax7.bar(['Standard', 'Delta-Hedged'], sharpe, color=colors, alpha=0.7)
    ax7.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax7.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Sharpe Ratio')
    for bar, val in zip(bars, sharpe):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Key Metrics Table
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    table_data = [
        ['Metric', 'Standard', 'Delta-Hedged', 'Improvement'],
        ['Total Return', f"{standard_metrics['total_return_pct']:.2f}%", 
         f"{delta_metrics['total_return_pct']:.2f}%",
         f"{delta_metrics['total_return_pct'] - standard_metrics['total_return_pct']:+.2f}%"],
        ['Final Capital', f"${standard_metrics['final_capital']:,.2f}", 
         f"${delta_metrics['final_capital']:,.2f}",
         f"${delta_metrics['final_capital'] - standard_metrics['final_capital']:+,.2f}"],
        ['Max Drawdown', f"{standard_metrics['max_drawdown_pct']:.2f}%", 
         f"{delta_metrics['max_drawdown_pct']:.2f}%",
         f"{delta_metrics['max_drawdown_pct'] - standard_metrics['max_drawdown_pct']:+.2f}%"],
        ['Avg PnL/Trade', f"${standard_metrics['avg_pnl_per_trade']:.2f}", 
         f"${delta_metrics['avg_pnl_per_trade']:.2f}",
         f"${delta_metrics['avg_pnl_per_trade'] - standard_metrics['avg_pnl_per_trade']:+.2f}"]
    ]
    
    table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Strategy Comparison: Standard vs Delta-Hedged', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('strategy_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Comparison chart saved to strategy_comparison.png")

def main():
    print("\n" + "="*80)
    print("STRATEGY COMPARISON ANALYSIS")
    print("="*80 + "\n")
    
    # Check if files exist
    if not Path('backtest_trades.csv').exists():
        print("❌ Error: backtest_trades.csv not found. Run standard backtest first.")
        return
    
    if not Path('backtest_delta_hedged_trades.csv').exists():
        print("❌ Error: backtest_delta_hedged_trades.csv not found. Run delta-hedged backtest first.")
        return
    
    # Load results
    print("Loading results...")
    standard_trades, standard_metrics, delta_trades, delta_metrics = load_results()
    print(f"  Standard: {len(standard_trades)} trades")
    print(f"  Delta-Hedged: {len(delta_trades)} trades\n")
    
    # Create comparison table
    print("="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80 + "\n")
    comparison = create_comparison_table(standard_metrics, delta_metrics)
    print(comparison.to_string())
    print()
    
    # Save comparison
    comparison.to_csv('strategy_comparison.csv')
    print("✓ Comparison table saved to strategy_comparison.csv\n")
    
    # Create visualizations
    print("Generating comparison charts...")
    plot_comparison(standard_trades, delta_trades, standard_metrics, delta_metrics)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    
    return_diff = delta_metrics['total_return_pct'] - standard_metrics['total_return_pct']
    dd_diff = delta_metrics['max_drawdown_pct'] - standard_metrics['max_drawdown_pct']
    sharpe_diff = delta_metrics['sharpe_ratio'] - standard_metrics['sharpe_ratio']
    
    print(f"Return Difference: {return_diff:+.2f}%")
    print(f"Drawdown Difference: {dd_diff:+.2f}%")
    print(f"Sharpe Improvement: {sharpe_diff:+.2f}")
    
    if return_diff > 0:
        print(f"\n✓ Delta-hedged strategy outperformed by {return_diff:.2f}%")
    else:
        print(f"\n✗ Delta-hedged strategy underperformed by {abs(return_diff):.2f}%")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
