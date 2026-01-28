"""
Backtest strategy: Buy the most extreme negative funding rate coin(s) each hour
and hold for specified period.

Data source: Hyperliquid MAINNET only.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from config import load_config, print_config


class ExtremeFundingBacktest:
    """Backtest engine for extreme negative funding rate strategy."""
    
    def __init__(self, initial_capital: float = 10000, position_size: float = 1.0, 
                 transaction_cost: float = 0.0005, num_positions: int = 1):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital in USD
            position_size: Fraction of capital to use per position (0-1)
            transaction_cost: Transaction cost as fraction (0.05% = 0.0005)
            num_positions: Number of extreme negative funding positions to trade
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        self.num_positions = num_positions
        self.trades = []
        self.equity_curve = []
        
    def load_data(self, funding_file: str = 'funding_history.csv') -> pd.DataFrame:
        """Load and prepare funding rate data."""
        df = pd.read_csv(funding_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(['datetime', 'coin']).reset_index(drop=True)
        df['hour'] = df['datetime'].dt.floor('h')
        return df
    
    def get_top_negative_funding(self, df: pd.DataFrame, hour: pd.Timestamp, 
                                 n: int = 1) -> List[Tuple[str, float]]:
        """
        Get the top N coins with most negative funding rates for a given hour.
        
        Args:
            df: Funding rate dataframe
            hour: Hour timestamp
            n: Number of top coins to return
            
        Returns:
            List of (coin_symbol, funding_rate) tuples
        """
        hour_data = df[df['hour'] == hour]
        
        if hour_data.empty:
            return []
        
        # Sort by funding rate and get top N most negative
        sorted_data = hour_data.sort_values('funding_rate').head(n)
        
        results = []
        for idx, row in sorted_data.iterrows():
            results.append((row['coin'], row['funding_rate']))
        
        return results
    
    def fetch_price_for_trade(self, coin: str, entry_time: pd.Timestamp, 
                             exit_time: pd.Timestamp) -> Tuple[float, float]:
        """
        Fetch entry and exit prices for a trade from Hyperliquid MAINNET.
        Uses exponential backoff retry to ensure 100% data acquisition.
        
        Returns:
            (entry_price, exit_price) or (None, None) if data unavailable after all retries
        """
        import requests
        import time
        
        start_ms = int(entry_time.timestamp() * 1000)
        end_ms = int(exit_time.timestamp() * 1000)
        
        url = "https://api.hyperliquid.xyz/info"  # Mainnet API endpoint
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "1h",
                "startTime": start_ms,
                "endTime": end_ms
            }
        }
        
        # Retry with exponential backoff
        max_retries = 5
        base_delay = 1.0  # Increased from 0.2s to 1.0s
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if isinstance(data, list) and len(data) >= 2:
                    entry_price = float(data[0]['c'])  # Close of first candle
                    exit_price = float(data[-1]['c'])  # Close of last candle
                    
                    # Delay before next request
                    time.sleep(base_delay)
                    return entry_price, exit_price
                
                # Data insufficient, retry
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"Insufficient data for {coin}, retry {attempt+1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)
                        print(f"Rate limit hit for {coin}, retry {attempt+1}/{max_retries} after {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        print(f"Max retries reached for {coin}, rate limit persists")
                else:
                    print(f"HTTP error fetching price for {coin}: {e}")
                    break
            except Exception as e:
                print(f"Error fetching price for {coin}: {e}")
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"Retry {attempt+1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
        
        return None, None
    
    def simulate_trade(self, coin: str, entry_time: pd.Timestamp, 
                      exit_time: pd.Timestamp, funding_rate: float, 
                      capital: float) -> Dict:
        """
        Simulate a single trade.
        
        Returns:
            Dictionary with trade details and results
        """
        # Fetch prices
        entry_price, exit_price = self.fetch_price_for_trade(coin, entry_time, exit_time)
        
        if entry_price is None or exit_price is None:
            return None
        
        # Calculate position size
        trade_capital = capital * self.position_size
        
        # Entry cost
        entry_cost = trade_capital * self.transaction_cost
        effective_capital = trade_capital - entry_cost
        
        # Position size in coins
        position_size = effective_capital / entry_price
        
        # Exit value
        exit_value = position_size * exit_price
        exit_cost = exit_value * self.transaction_cost
        final_value = exit_value - exit_cost
        
        # Calculate PnL
        pnl = final_value - trade_capital
        pnl_pct = (pnl / trade_capital) * 100
        
        # Calculate price return
        price_return_pct = ((exit_price - entry_price) / entry_price) * 100
        
        trade_record = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'coin': coin,
            'funding_rate': funding_rate,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'capital_used': trade_capital,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'price_return_pct': price_return_pct,
            'entry_cost': entry_cost,
            'exit_cost': exit_cost
        }
        
        return trade_record
    
    def run_backtest(self, df: pd.DataFrame, start_date: str = None, 
                    end_date: str = None) -> pd.DataFrame:
        """
        Run the backtest strategy.
        
        Strategy: Every hour, buy the top N coins with most negative funding rates
        and hold for 1 hour.
        
        Returns:
            DataFrame with all trades
        """
        # Filter date range if specified
        if start_date:
            df = df[df['datetime'] >= start_date]
        if end_date:
            df = df[df['datetime'] <= end_date]
        
        # Get unique hours
        hours = sorted(df['hour'].unique())
        
        print(f"Running backtest for {len(hours)} hours...")
        print(f"Date range: {hours[0]} to {hours[-1]}")
        print(f"Trading {self.num_positions} position(s) per hour")
        
        current_capital = self.initial_capital
        self.trades = []
        self.equity_curve = [(hours[0], current_capital)]
        
        for i, hour in enumerate(hours[:-1]):  # Exclude last hour (need next hour for exit)
            # Get top N coins with most negative funding rates
            top_coins = self.get_top_negative_funding(df, hour, self.num_positions)
            
            if not top_coins:
                continue
            
            # Define entry and exit times
            entry_time = hour
            exit_time = hours[i + 1]  # Next hour
            
            # Calculate capital per position
            capital_per_position = current_capital * (self.position_size / self.num_positions)
            
            # Execute trades for each position
            hour_pnl = 0
            successful_trades = 0
            
            for coin, funding_rate in top_coins:
                # Simulate trade
                trade = self.simulate_trade(coin, entry_time, exit_time, 
                                           funding_rate, capital_per_position)
                
                if trade is not None:
                    hour_pnl += trade['pnl']
                    successful_trades += 1
                    
                    # Store trade with updated capital
                    trade['capital_before'] = current_capital
                    trade['capital_after'] = current_capital + hour_pnl
                    trade['position_number'] = successful_trades
                    self.trades.append(trade)
                
                # Rate limiting
                import time
                time.sleep(0.1)
            
            # Update capital after all positions for this hour
            if successful_trades > 0:
                current_capital += hour_pnl
                self.equity_curve.append((exit_time, current_capital))
                
                # Progress update
                if (i + 1) % 20 == 0:
                    print(f"Processed {i+1}/{len(hours)-1} hours... "
                          f"Capital: ${current_capital:.2f} ({((current_capital/self.initial_capital - 1)*100):.2f}%) "
                          f"| Trades: {len(self.trades)}")
        
        trades_df = pd.DataFrame(self.trades)
        
        print(f"\n✓ Backtest complete! Executed {len(self.trades)} trades across {len(hours)-1} hours.")
        
        return trades_df
        
        return trades_df
    
    def calculate_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        if trades_df.empty:
            return {}
        
        total_return = (trades_df['capital_after'].iloc[-1] / self.initial_capital - 1) * 100
        total_pnl = trades_df['pnl'].sum()
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = (abs(winning_trades['pnl'].sum()) / abs(losing_trades['pnl'].sum()) 
                        if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf'))
        
        # Calculate max drawdown
        equity_series = pd.Series([eq[1] for eq in self.equity_curve])
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (annualized, assuming 24*365 trading hours per year)
        returns = trades_df['pnl_pct']
        sharpe = (returns.mean() / returns.std() * np.sqrt(24 * 365)) if returns.std() > 0 else 0
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': trades_df['capital_after'].iloc[-1],
            'total_return_pct': total_return,
            'total_pnl': total_pnl,
            'num_trades': len(trades_df),
            'num_winning': len(winning_trades),
            'num_losing': len(losing_trades),
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe,
            'avg_pnl_per_trade': trades_df['pnl'].mean(),
            'avg_return_per_trade_pct': trades_df['pnl_pct'].mean()
        }
        
        return metrics
    
    def print_results(self, metrics: Dict):
        """Print backtest results."""
        print("\n" + "="*80)
        print("BACKTEST RESULTS - EXTREME NEGATIVE FUNDING STRATEGY")
        print("="*80)
        
        print(f"\n📊 Capital & Returns:")
        print(f"  Initial Capital:     ${metrics['initial_capital']:,.2f}")
        print(f"  Final Capital:       ${metrics['final_capital']:,.2f}")
        print(f"  Total Return:        {metrics['total_return_pct']:.2f}%")
        print(f"  Total PnL:           ${metrics['total_pnl']:,.2f}")
        
        print(f"\n📈 Trade Statistics:")
        print(f"  Total Trades:        {metrics['num_trades']}")
        print(f"  Winning Trades:      {metrics['num_winning']} ({metrics['win_rate_pct']:.2f}%)")
        print(f"  Losing Trades:       {metrics['num_losing']}")
        print(f"  Avg PnL per Trade:   ${metrics['avg_pnl_per_trade']:.2f} ({metrics['avg_return_per_trade_pct']:.4f}%)")
        
        print(f"\n💰 Win/Loss Analysis:")
        print(f"  Average Win:         ${metrics['avg_win']:.2f}")
        print(f"  Average Loss:        ${metrics['avg_loss']:.2f}")
        print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
        
        print(f"\n📉 Risk Metrics:")
        print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        
        print("\n" + "="*80)
    
    def plot_results(self, trades_df: pd.DataFrame, save_file: str = 'backtest_results.png'):
        """Plot backtest results."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Extreme Negative Funding Strategy - Backtest Results', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        equity_df = pd.DataFrame(self.equity_curve, columns=['time', 'equity'])
        axes[0, 0].plot(equity_df['time'], equity_df['equity'], linewidth=2, color='blue')
        axes[0, 0].axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Capital ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. PnL Distribution
        axes[0, 1].hist(trades_df['pnl'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('PnL Distribution')
        axes[0, 1].set_xlabel('PnL ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cumulative PnL
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        axes[1, 0].plot(range(len(trades_df)), trades_df['cumulative_pnl'], linewidth=2, color='green')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Cumulative PnL')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Cumulative PnL ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Returns Distribution
        axes[1, 1].hist(trades_df['pnl_pct'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_title('Return % Distribution')
        axes[1, 1].set_xlabel('Return (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Funding Rate vs Return
        axes[2, 0].scatter(trades_df['funding_rate'] * 100, trades_df['price_return_pct'], 
                          alpha=0.5, s=20)
        axes[2, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[2, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[2, 0].set_title('Funding Rate vs Price Return')
        axes[2, 0].set_xlabel('Funding Rate (%)')
        axes[2, 0].set_ylabel('1h Price Return (%)')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Win Rate by Hour of Day
        trades_df['hour_of_day'] = pd.to_datetime(trades_df['entry_time']).dt.hour
        trades_df['is_win'] = trades_df['pnl'] > 0
        hourly_win_rate = trades_df.groupby('hour_of_day')['is_win'].mean() * 100
        axes[2, 1].bar(hourly_win_rate.index, hourly_win_rate.values, alpha=0.7, color='purple')
        axes[2, 1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
        axes[2, 1].set_title('Win Rate by Hour of Day')
        axes[2, 1].set_xlabel('Hour (UTC)')
        axes[2, 1].set_ylabel('Win Rate (%)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Results plot saved to {save_file}")
        plt.close()


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    
    print("="*80)
    print("BACKTESTING EXTREME NEGATIVE FUNDING STRATEGY")
    print("="*80)
    
    # Print configuration
    print_config(config)
    
    # Initialize backtest engine with config
    backtest = ExtremeFundingBacktest(
        initial_capital=config['initial_capital'],
        position_size=config['position_size_pct'],
        transaction_cost=config['transaction_cost'],
        num_positions=config['num_positions']
    )
    
    # Load data
    print("\nLoading funding rate data...")
    df = backtest.load_data('funding_history.csv')
    print(f"Loaded {len(df):,} funding rate records")
    
    # Run backtest
    print("\nRunning backtest...")
    trades_df = backtest.run_backtest(df)
    
    if trades_df.empty:
        print("No trades executed. Exiting.")
        return
    
    # Save trades
    trades_df.to_csv('backtest_trades.csv', index=False)
    print("✓ Trades saved to backtest_trades.csv")
    
    # Calculate metrics
    metrics = backtest.calculate_metrics(trades_df)
    
    # Print results
    backtest.print_results(metrics)
    
    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('backtest_metrics.csv', index=False)
    print("\n✓ Metrics saved to backtest_metrics.csv")
    
    # Plot results
    print("\nGenerating plots...")
    backtest.plot_results(trades_df)
    
    # Additional analysis
    print("\n" + "="*80)
    print("ADDITIONAL INSIGHTS")
    print("="*80)
    
    # Top performing coins
    print("\nTop 10 Coins by Total PnL:")
    coin_pnl = trades_df.groupby('coin')['pnl'].sum().sort_values(ascending=False).head(10)
    for coin, pnl in coin_pnl.items():
        print(f"  {coin}: ${pnl:.2f}")
    
    # Best and worst trades
    print("\nBest Trade:")
    best_trade = trades_df.loc[trades_df['pnl'].idxmax()]
    print(f"  {best_trade['coin']} on {best_trade['entry_time']}")
    print(f"  PnL: ${best_trade['pnl']:.2f} ({best_trade['pnl_pct']:.2f}%)")
    
    print("\nWorst Trade:")
    worst_trade = trades_df.loc[trades_df['pnl'].idxmin()]
    print(f"  {worst_trade['coin']} on {worst_trade['entry_time']}")
    print(f"  PnL: ${worst_trade['pnl']:.2f} ({worst_trade['pnl_pct']:.2f}%)")
    
    print("\n" + "="*80)
    print("Backtest complete!")
    print("="*80)


if __name__ == "__main__":
    main()
