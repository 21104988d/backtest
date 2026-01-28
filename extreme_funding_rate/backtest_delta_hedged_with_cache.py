"""
Delta-hedged backtest strategy using pre-fetched price cache.
Long extreme funding coin + Short BTC hedge.
100% data coverage, instant execution.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from config import load_config, print_config

# Set base directory to script location
BASEDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASEDIR)


class DeltaHedgedCachedBacktest:
    """Delta-hedged backtest using price cache."""
    
    def __init__(self, price_cache_file='price_cache.csv', **kwargs):
        """Initialize with price cache."""
        self.initial_capital = kwargs.get('initial_capital', 10000)
        self.position_size_fixed = kwargs.get('position_size_fixed', 1000)  # Fixed USD per trade
        self.position_size_pct = kwargs.get('position_size_pct', 1.0)
        self.transaction_cost = kwargs.get('transaction_cost', 0.0005)
        self.num_positions = kwargs.get('num_positions', 1)
        
        self.trades = []
        self.equity_curve = []
        
        # Load price cache
        print(f"\n📦 Loading price cache from {price_cache_file}...")
        self.price_cache = pd.read_csv(price_cache_file)
        self.price_cache['timestamp'] = pd.to_datetime(self.price_cache['timestamp'], format='mixed')
        
        # Create lookup index
        self.price_cache['key'] = self.price_cache['coin'] + '_' + self.price_cache['timestamp'].astype(str)
        self.price_lookup = dict(zip(self.price_cache['key'], self.price_cache['price']))
        
        print(f"✓ Loaded {len(self.price_cache):,} cached prices")
    
    def get_cached_price(self, coin: str, timestamp: pd.Timestamp) -> Optional[float]:
        """Get price from cache."""
        key = f"{coin}_{timestamp}"
        return self.price_lookup.get(key)
    
    def load_funding_data(self, funding_file: str = 'funding_history.csv') -> pd.DataFrame:
        """Load funding rate data."""
        df = pd.read_csv(funding_file)
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        df = df.sort_values(['datetime', 'coin']).reset_index(drop=True)
        df['hour'] = df['datetime'].dt.floor('h')
        return df
    
    def get_top_negative_funding_with_price(self, df: pd.DataFrame, hour: pd.Timestamp, 
                                             exit_time: pd.Timestamp) -> Optional[Tuple[str, float]]:
        """Get the most negative funding coin that has valid price data."""
        hour_data = df[df['hour'] == hour]
        if hour_data.empty:
            return None
        
        # Try top 10 most negative funding coins until we find one with price data
        candidates = hour_data.nsmallest(10, 'funding_rate')
        
        for _, row in candidates.iterrows():
            coin = row['coin']
            # Check if all required prices exist
            coin_entry = self.get_cached_price(coin, hour)
            coin_exit = self.get_cached_price(coin, exit_time)
            btc_entry = self.get_cached_price('BTC', hour)
            btc_exit = self.get_cached_price('BTC', exit_time)
            
            if all([coin_entry, coin_exit, btc_entry, btc_exit]):
                return (coin, row['funding_rate'])
        
        # No valid coin found
        return None
    
    def get_top_negative_funding(self, df: pd.DataFrame, hour: pd.Timestamp) -> Optional[Tuple[str, float]]:
        """Get the most negative funding coin (legacy, doesn't check prices)."""
        hour_data = df[df['hour'] == hour]
        if hour_data.empty:
            return None
        
        most_negative = hour_data.nsmallest(1, 'funding_rate').iloc[0]
        return (most_negative['coin'], most_negative['funding_rate'])
    
    def simulate_delta_hedged_trade(self, coin: str, entry_time: pd.Timestamp, 
                                   exit_time: pd.Timestamp, funding_rate: float, 
                                   capital: float) -> Optional[Dict]:
        """
        Simulate delta-hedged trade:
        - Long the extreme funding coin
        - Short equal value of BTC
        """
        # Get all 4 prices from cache
        coin_entry = self.get_cached_price(coin, entry_time)
        coin_exit = self.get_cached_price(coin, exit_time)
        btc_entry = self.get_cached_price('BTC', entry_time)
        btc_exit = self.get_cached_price('BTC', exit_time)
        
        if None in [coin_entry, coin_exit, btc_entry, btc_exit]:
            return None
        
        # Calculate returns
        coin_return_pct = ((coin_exit - coin_entry) / coin_entry) * 100
        btc_return_pct = ((btc_exit - btc_entry) / btc_entry) * 100
        
        # Position sizing - use fixed size if set, otherwise percentage
        if self.position_size_fixed > 0:
            position_capital = min(self.position_size_fixed, capital)  # Don't exceed available capital
        else:
            position_capital = capital * self.position_size_pct
        
        # Costs (4 transactions: buy coin, sell coin, short BTC, cover BTC)
        coin_entry_cost = position_capital * self.transaction_cost
        coin_exit_cost = position_capital * self.transaction_cost
        btc_entry_cost = position_capital * self.transaction_cost
        btc_exit_cost = position_capital * self.transaction_cost
        total_costs = coin_entry_cost + coin_exit_cost + btc_entry_cost + btc_exit_cost
        
        # P&L calculation
        coin_pnl = position_capital * (coin_return_pct / 100)  # Long coin P&L
        btc_pnl = position_capital * (-btc_return_pct / 100)   # Short BTC P&L (profit when BTC falls)
        
        total_pnl = coin_pnl + btc_pnl - total_costs
        total_pnl_pct = (total_pnl / capital) * 100
        
        # Delta-hedged return (coin return minus BTC return)
        hedged_return_pct = coin_return_pct - btc_return_pct
        
        return {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'coin': coin,
            'funding_rate': funding_rate,
            'coin_entry_price': coin_entry,
            'coin_exit_price': coin_exit,
            'btc_entry_price': btc_entry,
            'btc_exit_price': btc_exit,
            'position_size': position_capital,
            'coin_return_pct': coin_return_pct,
            'btc_return_pct': btc_return_pct,
            'hedged_return_pct': hedged_return_pct,
            'coin_pnl': coin_pnl,
            'btc_pnl': btc_pnl,
            'total_costs': total_costs,
            'pnl': total_pnl,
            'pnl_pct': total_pnl_pct,
            'capital_after': capital + total_pnl
        }
    
    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run delta-hedged backtest."""
        print(f"\n🚀 Running delta-hedged backtest...")
        
        hours = sorted(df['hour'].unique())
        print(f"Date range: {hours[0]} to {hours[-1]}")
        print(f"Total hours: {len(hours)}\n")
        
        capital = self.initial_capital
        self.equity_curve = [(hours[0], capital)]
        
        trades_executed = 0
        trades_failed = 0
        
        for i, hour in enumerate(hours, 1):
            entry_time = hour
            exit_time = hour + timedelta(hours=1)
            
            # Get most negative funding coin WITH valid price data
            result = self.get_top_negative_funding_with_price(df, hour, exit_time)
            if not result:
                trades_failed += 1
                self.equity_curve.append((exit_time, capital))
                if i % 10 == 0 or i == len(hours):
                    pct_change = ((capital - self.initial_capital) / self.initial_capital) * 100
                    print(f"Hour {i}/{len(hours)} | Capital: ${capital:,.2f} ({pct_change:+.2f}%) | Trades: {trades_executed}")
                continue
            
            coin, funding_rate = result
            
            # Simulate delta-hedged trade (should always succeed since we verified prices)
            trade = self.simulate_delta_hedged_trade(coin, entry_time, exit_time, funding_rate, capital)
            
            if trade:
                self.trades.append(trade)
                capital = trade['capital_after']
                trades_executed += 1
                self.equity_curve.append((exit_time, capital))
            
            # Progress every 10 hours
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
            'win_rate_pct': (len(wins) / len(df_trades)) * 100,
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'best_trade': df_trades['pnl'].max(),
            'worst_trade': df_trades['pnl'].min(),
            'avg_trade': df_trades['pnl'].mean(),
            'total_fees': df_trades['total_costs'].sum(),
            'avg_hedged_return': df_trades['hedged_return_pct'].mean(),
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
    
    def save_results(self, df_trades: pd.DataFrame, metrics: Dict):
        """Save results."""
        df_trades.to_csv('backtest_delta_hedged_trades.csv', index=False)
        pd.DataFrame([metrics]).to_csv('backtest_delta_hedged_metrics.csv', index=False)
        pd.DataFrame(self.equity_curve, columns=['timestamp', 'capital']).to_csv('backtest_delta_hedged_equity.csv', index=False)
        
        print(f"\n💾 Results saved:")
        print(f"  - backtest_delta_hedged_trades.csv")
        print(f"  - backtest_delta_hedged_metrics.csv")
        print(f"  - backtest_delta_hedged_equity.csv")
    
    def plot_results(self, df_trades: pd.DataFrame, metrics: Dict):
        """Create visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Backtest Results - Delta-Hedged Strategy', fontsize=16, fontweight='bold')
        
        # Equity curve
        df_equity = pd.DataFrame(self.equity_curve, columns=['timestamp', 'capital'])
        axes[0, 0].plot(df_equity['timestamp'], df_equity['capital'], linewidth=2, color='green')
        axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Capital ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # P&L distribution
        axes[0, 1].hist(df_trades['pnl'], bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_title('P&L Distribution')
        axes[0, 1].set_xlabel('P&L ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hedged returns
        axes[1, 0].hist(df_trades['hedged_return_pct'], bins=50, edgecolor='black', alpha=0.7, color='blue')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_title('Delta-Hedged Returns Distribution')
        axes[1, 0].set_xlabel('Hedged Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Metrics
        axes[1, 1].axis('off')
        metrics_text = f"""
        DELTA-HEDGED PERFORMANCE
        
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
        Avg Hedged Return: {metrics['avg_hedged_return']:.2f}%
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('backtest_delta_hedged_results.png', dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved: backtest_delta_hedged_results.png")


def main():
    print("=" * 70)
    print("DELTA-HEDGED BACKTEST WITH CACHED PRICES")
    print("=" * 70)
    
    config = load_config()
    print_config(config)
    
    try:
        backtest = DeltaHedgedCachedBacktest(
            price_cache_file='price_cache.csv',
            initial_capital=config['initial_capital'],
            position_size_fixed=config.get('position_size_fixed', 0),
            position_size_pct=config['position_size_pct'],
            transaction_cost=config['transaction_cost'],
            num_positions=config['num_positions']
        )
    except FileNotFoundError:
        print("\n❌ Error: price_cache.csv not found!")
        print("Please run prefetch_prices.py first.")
        return
    
    df = backtest.load_funding_data()
    print(f"\n✓ Loaded {len(df):,} funding rate records")
    
    df_trades = backtest.run_backtest(df)
    
    if df_trades.empty:
        print("\n❌ No trades executed!")
        return
    
    metrics = backtest.calculate_metrics(df_trades)
    
    # Display results
    print("\n" + "=" * 70)
    print("FINAL RESULTS - DELTA-HEDGED STRATEGY")
    print("=" * 70)
    print(f"Initial Capital:    ${metrics['initial_capital']:,.2f}")
    print(f"Final Capital:      ${metrics['final_capital']:,.2f}")
    print(f"Total Return:       {metrics['total_return_pct']:+.2f}%")
    print(f"Total Trades:       {metrics['num_trades']}")
    print(f"Win Rate:           {metrics['win_rate_pct']:.2f}%")
    print(f"Profit Factor:      {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
    print(f"Avg Hedged Return:  {metrics['avg_hedged_return']:+.2f}%")
    print("=" * 70)
    
    backtest.save_results(df_trades, metrics)
    backtest.plot_results(df_trades, metrics)
    
    print("\n✅ All done!")

if __name__ == '__main__':
    main()
