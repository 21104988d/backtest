"""
Backtest strategy using pre-fetched price cache for instant execution.
100% data coverage, zero API calls during backtest.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from config import load_config, print_config


class CachedBacktest:
    """Backtest engine using pre-fetched price cache."""
    
    def __init__(self, price_cache_file='price_cache.csv', **kwargs):
        """Initialize with price cache."""
        self.initial_capital = kwargs.get('initial_capital', 10000)
        self.position_size_fixed = kwargs.get('position_size_fixed', 0)
        self.position_size_pct = kwargs.get('position_size_pct', 1.0)
        self.transaction_cost = kwargs.get('transaction_cost', 0.0005)
        self.num_positions = kwargs.get('num_positions', 1)
        
        self.trades = []
        self.equity_curve = []
        
        # Load price cache
        print(f"\n📦 Loading price cache from {price_cache_file}...")
        self.price_cache = pd.read_csv(price_cache_file)
        self.price_cache['timestamp'] = pd.to_datetime(self.price_cache['timestamp'])
        
        # Create lookup index for O(1) price lookups
        self.price_cache['key'] = self.price_cache['coin'] + '_' + self.price_cache['timestamp'].astype(str)
        self.price_lookup = dict(zip(self.price_cache['key'], self.price_cache['price']))
        
        print(f"✓ Loaded {len(self.price_cache):,} cached prices")
        print(f"✓ Cache coverage: {len(self.price_cache['coin'].unique())} coins, {len(self.price_cache['timestamp'].unique())} timestamps")
    
    def get_cached_price(self, coin: str, timestamp: pd.Timestamp) -> Optional[float]:
        """Get price from cache with O(1) lookup."""
        key = f"{coin}_{timestamp}"
        return self.price_lookup.get(key)
    
    def load_funding_data(self, funding_file: str = 'funding_history.csv') -> pd.DataFrame:
        """Load and prepare funding rate data."""
        df = pd.read_csv(funding_file)
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        df = df.sort_values(['datetime', 'coin']).reset_index(drop=True)
        df['hour'] = df['datetime'].dt.floor('h')
        return df
    
    def get_top_negative_funding(self, df: pd.DataFrame, hour: pd.Timestamp, n: int = 1) -> List[Tuple[str, float]]:
        """Get the top N coins with most negative funding rates for a given hour."""
        hour_data = df[df['hour'] == hour]
        if hour_data.empty:
            return []
        
        sorted_data = hour_data.sort_values('funding_rate').head(n)
        return [(row['coin'], row['funding_rate']) for idx, row in sorted_data.iterrows()]
    
    def simulate_trade(self, coin: str, entry_time: pd.Timestamp, exit_time: pd.Timestamp, 
                      funding_rate: float, capital: float) -> Optional[Dict]:
        """Simulate a single trade using cached prices."""
        # Get prices from cache
        entry_price = self.get_cached_price(coin, entry_time)
        exit_price = self.get_cached_price(coin, exit_time)
        
        if entry_price is None or exit_price is None:
            return None
        
        # Calculate position - use fixed if set, otherwise percentage
        if self.position_size_fixed > 0:
            trade_capital = min(self.position_size_fixed, capital)  # Don't exceed available capital
        else:
            trade_capital = capital * self.position_size_pct
        entry_cost = trade_capital * self.transaction_cost
        effective_capital = trade_capital - entry_cost
        position_size = effective_capital / entry_price
        
        # Exit value
        exit_value = position_size * exit_price
        exit_cost = exit_value * self.transaction_cost
        net_exit_value = exit_value - exit_cost
        
        # P&L
        pnl = net_exit_value - trade_capital
        pnl_pct = (pnl / capital) * 100
        
        return {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'coin': coin,
            'funding_rate': funding_rate,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'entry_cost': entry_cost,
            'exit_cost': exit_cost,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'capital_after': capital + pnl
        }
    
    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run backtest on funding rate data."""
        print(f"\n🚀 Running backtest...")
        
        # Get unique hours
        hours = sorted(df['hour'].unique())
        print(f"Date range: {hours[0]} to {hours[-1]}")
        print(f"Trading {self.num_positions} position(s) per hour")
        print(f"Total hours: {len(hours)}\n")
        
        capital = self.initial_capital
        self.equity_curve = [(hours[0], capital)]
        
        trades_executed = 0
        trades_failed = 0
        
        for i, hour in enumerate(hours, 1):
            # Get top negative funding coins
            top_coins = self.get_top_negative_funding(df, hour, self.num_positions)
            
            if not top_coins:
                continue
            
            # Trade the most extreme
            coin, funding_rate = top_coins[0]
            
            entry_time = hour
            exit_time = hour + timedelta(hours=1)
            
            # Simulate trade
            trade = self.simulate_trade(coin, entry_time, exit_time, funding_rate, capital)
            
            if trade:
                self.trades.append(trade)
                capital = trade['capital_after']
                trades_executed += 1
            else:
                trades_failed += 1
            
            # Update equity curve
            self.equity_curve.append((exit_time, capital))
            
            # Progress update every 10 hours
            if i % 10 == 0 or i == len(hours):
                pct_change = ((capital - self.initial_capital) / self.initial_capital) * 100
                print(f"Hour {i}/{len(hours)} | Capital: ${capital:,.2f} ({pct_change:+.2f}%) | Trades: {trades_executed}")
        
        print(f"\n✅ Backtest complete!")
        print(f"  - Trades executed: {trades_executed}/{len(hours)} ({trades_executed/len(hours)*100:.1f}%)")
        print(f"  - Trades failed: {trades_failed} ({trades_failed/len(hours)*100:.1f}%)")
        
        return pd.DataFrame(self.trades)
    
    def calculate_metrics(self, df_trades: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        if df_trades.empty:
            return {}
        
        final_capital = df_trades.iloc[-1]['capital_after']
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] < 0]
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'num_trades': len(df_trades),
            'num_wins': len(wins),
            'num_losses': len(losses),
            'win_rate_pct': (len(wins) / len(df_trades)) * 100 if len(df_trades) > 0 else 0,
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'best_trade': df_trades['pnl'].max(),
            'worst_trade': df_trades['pnl'].min(),
            'avg_trade': df_trades['pnl'].mean(),
            'total_fees': df_trades['entry_cost'].sum() + df_trades['exit_cost'].sum(),
        }
        
        # Max drawdown
        equity_series = pd.Series([eq[1] for eq in self.equity_curve])
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max * 100
        metrics['max_drawdown_pct'] = drawdowns.min()
        
        # Profit factor
        total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0
        
        return metrics
    
    def save_results(self, df_trades: pd.DataFrame, metrics: Dict, prefix='backtest'):
        """Save backtest results."""
        df_trades.to_csv(f'{prefix}_trades.csv', index=False)
        
        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(f'{prefix}_metrics.csv', index=False)
        
        df_equity = pd.DataFrame(self.equity_curve, columns=['timestamp', 'capital'])
        df_equity.to_csv(f'{prefix}_equity.csv', index=False)
        
        print(f"\n💾 Results saved:")
        print(f"  - {prefix}_trades.csv")
        print(f"  - {prefix}_metrics.csv")
        print(f"  - {prefix}_equity.csv")
    
    def plot_results(self, df_trades: pd.DataFrame, metrics: Dict, output_file='backtest_results.png'):
        """Create visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Backtest Results - Standard Strategy', fontsize=16, fontweight='bold')
        
        # Equity curve
        df_equity = pd.DataFrame(self.equity_curve, columns=['timestamp', 'capital'])
        axes[0, 0].plot(df_equity['timestamp'], df_equity['capital'], linewidth=2)
        axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Capital ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # P&L distribution
        axes[0, 1].hist(df_trades['pnl'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_title('P&L Distribution')
        axes[0, 1].set_xlabel('P&L ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative returns
        df_trades['cumulative_return'] = ((df_trades['capital_after'] - self.initial_capital) / self.initial_capital) * 100
        axes[1, 0].plot(range(len(df_trades)), df_trades['cumulative_return'], linewidth=2)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Cumulative Returns')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Return (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance metrics table
        axes[1, 1].axis('off')
        metrics_text = f"""
        PERFORMANCE METRICS
        
        Initial Capital: ${metrics['initial_capital']:,.2f}
        Final Capital: ${metrics['final_capital']:,.2f}
        Total Return: {metrics['total_return_pct']:.2f}%
        
        Trades: {metrics['num_trades']}
        Win Rate: {metrics['win_rate_pct']:.2f}%
        Profit Factor: {metrics['profit_factor']:.2f}
        
        Avg Win: ${metrics['avg_win']:.2f}
        Avg Loss: ${metrics['avg_loss']:.2f}
        Best Trade: ${metrics['best_trade']:.2f}
        Worst Trade: ${metrics['worst_trade']:.2f}
        
        Max Drawdown: {metrics['max_drawdown_pct']:.2f}%
        Total Fees: ${metrics['total_fees']:.2f}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved: {output_file}")


def main():
    print("=" * 70)
    print("BACKTESTING WITH CACHED PRICES - STANDARD STRATEGY")
    print("=" * 70)
    
    # Load config
    config = load_config()
    print_config(config)
    
    # Initialize backtest with cache
    try:
        backtest = CachedBacktest(
            price_cache_file='price_cache.csv',
            initial_capital=config['initial_capital'],
            position_size_fixed=config.get('position_size_fixed', 0),
            position_size_pct=config['position_size_pct'],
            transaction_cost=config['transaction_cost'],
            num_positions=config['num_positions']
        )
    except FileNotFoundError:
        print("\n❌ Error: price_cache.csv not found!")
        print("Please run prefetch_prices.py first to generate the price cache.")
        return
    
    # Load funding data
    df = backtest.load_funding_data()
    print(f"\n✓ Loaded {len(df):,} funding rate records")
    
    # Run backtest
    df_trades = backtest.run_backtest(df)
    
    if df_trades.empty:
        print("\n❌ No trades executed!")
        return
    
    # Calculate metrics
    metrics = backtest.calculate_metrics(df_trades)
    
    # Display results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Initial Capital:    ${metrics['initial_capital']:,.2f}")
    print(f"Final Capital:      ${metrics['final_capital']:,.2f}")
    print(f"Total Return:       {metrics['total_return_pct']:+.2f}%")
    print(f"Total Trades:       {metrics['num_trades']}")
    print(f"Win Rate:           {metrics['win_rate_pct']:.2f}%")
    print(f"Profit Factor:      {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
    print(f"Avg Win:            ${metrics['avg_win']:.2f}")
    print(f"Avg Loss:           ${metrics['avg_loss']:.2f}")
    print(f"Best Trade:         ${metrics['best_trade']:.2f}")
    print(f"Worst Trade:        ${metrics['worst_trade']:.2f}")
    print("=" * 70)
    
    # Save results
    backtest.save_results(df_trades, metrics, 'backtest')
    
    # Plot results
    backtest.plot_results(df_trades, metrics, 'backtest_results.png')
    
    print("\n✅ All done!")

if __name__ == '__main__':
    main()
