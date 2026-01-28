"""
Mean Reversion Strategy with BETA Hedge.
- LONG the most extreme negative funding coin
- SHORT BTC with beta-adjusted position size (30-day rolling beta)

Beta = Cov(coin, BTC) / Var(BTC) over lookback period
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASEDIR)

from config import load_config, print_config


class MeanReversionBetaHedgedBacktest:
    """Backtest engine for mean reversion with beta-adjusted BTC hedge."""
    
    def __init__(self, price_cache_file='price_cache.csv', beta_lookback_hours=720, **kwargs):
        """
        Initialize with price cache.
        
        Args:
            price_cache_file: Path to cached prices
            beta_lookback_hours: Hours of lookback for beta calculation (720 = 30 days)
        """
        self.initial_capital = kwargs.get('initial_capital', 10000)
        self.position_size_fixed = kwargs.get('position_size_fixed', 0)
        self.position_size_pct = kwargs.get('position_size_pct', 1.0)
        self.transaction_cost = kwargs.get('transaction_cost', 0.0005)
        self.beta_lookback_hours = beta_lookback_hours
        
        self.trades = []
        self.equity_curve = []
        
        # Load price cache
        print(f"\n📦 Loading price cache from {price_cache_file}...")
        self.price_cache = pd.read_csv(price_cache_file)
        self.price_cache['timestamp'] = pd.to_datetime(self.price_cache['timestamp'])
        
        # Create lookup index
        self.price_cache['key'] = self.price_cache['coin'] + '_' + self.price_cache['timestamp'].astype(str)
        self.price_lookup = dict(zip(self.price_cache['key'], self.price_cache['price']))
        
        # Pre-compute price series for beta calculation
        self._prepare_price_series()
        
        print(f"✓ Loaded {len(self.price_cache):,} cached prices")
        print(f"✓ Beta lookback: {beta_lookback_hours} hours ({beta_lookback_hours/24:.0f} days)")
    
    def _prepare_price_series(self):
        """Prepare price series for efficient beta calculation."""
        # Pivot to get price matrix: rows=timestamps, columns=coins
        self.price_matrix = self.price_cache.pivot(
            index='timestamp', 
            columns='coin', 
            values='price'
        ).sort_index()
        
        # Calculate returns
        self.returns_matrix = self.price_matrix.pct_change()
        
        # Check if BTC is available
        if 'BTC' not in self.returns_matrix.columns:
            print("⚠️ Warning: BTC not in price cache, beta hedging may fail")
    
    def calculate_beta(self, coin: str, current_time: pd.Timestamp) -> Optional[float]:
        """
        Calculate rolling beta of coin vs BTC.
        
        Beta = Cov(coin, BTC) / Var(BTC)
        """
        if 'BTC' not in self.returns_matrix.columns:
            return None
        if coin not in self.returns_matrix.columns:
            return None
        
        # Get lookback window
        start_time = current_time - timedelta(hours=self.beta_lookback_hours)
        
        # Filter returns within window
        mask = (self.returns_matrix.index >= start_time) & (self.returns_matrix.index < current_time)
        window_returns = self.returns_matrix.loc[mask]
        
        if len(window_returns) < 24:  # Need at least 24 hours of data
            return None
        
        coin_returns = window_returns[coin].dropna()
        btc_returns = window_returns['BTC'].dropna()
        
        # Align the series
        common_idx = coin_returns.index.intersection(btc_returns.index)
        if len(common_idx) < 24:
            return None
        
        coin_returns = coin_returns.loc[common_idx]
        btc_returns = btc_returns.loc[common_idx]
        
        # Calculate beta
        covariance = np.cov(coin_returns, btc_returns)[0, 1]
        btc_variance = np.var(btc_returns)
        
        if btc_variance == 0:
            return None
        
        beta = covariance / btc_variance
        
        # Clamp beta to reasonable range
        beta = np.clip(beta, 0.1, 3.0)
        
        return beta
    
    def get_cached_price(self, coin: str, timestamp: pd.Timestamp) -> Optional[float]:
        """Get price from cache."""
        key = f"{coin}_{timestamp}"
        return self.price_lookup.get(key)
    
    def load_funding_data(self, funding_file: str = 'funding_history.csv') -> pd.DataFrame:
        """Load and prepare funding rate data."""
        df = pd.read_csv(funding_file)
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        df = df.sort_values(['datetime', 'coin']).reset_index(drop=True)
        df['hour'] = df['datetime'].dt.floor('h')
        return df
    
    def get_top_negative_funding(self, df: pd.DataFrame, hour: pd.Timestamp, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N coins with most negative funding rates."""
        hour_data = df[df['hour'] == hour]
        if hour_data.empty:
            return []
        
        sorted_data = hour_data.sort_values('funding_rate').head(n)
        return [(row['coin'], row['funding_rate']) for idx, row in sorted_data.iterrows()]
    
    def simulate_trade(self, coin: str, entry_time: pd.Timestamp, exit_time: pd.Timestamp,
                      funding_rate: float, capital: float) -> Optional[Dict]:
        """
        Simulate a beta-hedged trade.
        
        LONG coin + SHORT beta*BTC
        """
        # Get prices
        coin_entry = self.get_cached_price(coin, entry_time)
        coin_exit = self.get_cached_price(coin, exit_time)
        btc_entry = self.get_cached_price('BTC', entry_time)
        btc_exit = self.get_cached_price('BTC', exit_time)
        
        if None in [coin_entry, coin_exit, btc_entry, btc_exit]:
            return None
        
        # Calculate beta
        beta = self.calculate_beta(coin, entry_time)
        if beta is None:
            beta = 1.0  # Fallback to delta hedge if beta unavailable
        
        # Position sizing
        if self.position_size_fixed > 0:
            trade_capital = min(self.position_size_fixed, capital)
        else:
            trade_capital = capital * self.position_size_pct
        
        # Split capital: coin position and beta-adjusted BTC hedge
        coin_capital = trade_capital / (1 + beta)
        btc_capital = trade_capital * beta / (1 + beta)
        
        # Transaction costs
        total_entry_cost = (coin_capital + btc_capital) * self.transaction_cost
        
        # LONG coin position
        coin_position = (coin_capital - total_entry_cost/2) / coin_entry
        coin_exit_value = coin_position * coin_exit
        coin_pnl = coin_exit_value - coin_capital
        
        # SHORT BTC hedge (beta-adjusted)
        btc_position = (btc_capital - total_entry_cost/2) / btc_entry
        btc_pnl = btc_position * (btc_entry - btc_exit)  # Profit if BTC falls
        
        # Total exit value and costs
        total_exit_value = coin_exit_value + (btc_capital + btc_pnl)
        total_exit_cost = total_exit_value * self.transaction_cost
        
        # Net P&L
        total_pnl = (coin_pnl + btc_pnl) - total_entry_cost - total_exit_cost
        pnl_pct = (total_pnl / trade_capital) * 100
        
        # Individual returns
        coin_return = ((coin_exit - coin_entry) / coin_entry) * 100
        btc_return = ((btc_exit - btc_entry) / btc_entry) * 100
        hedged_return = coin_return - beta * btc_return
        
        return {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'coin': coin,
            'funding_rate': funding_rate,
            'beta': beta,
            'coin_entry': coin_entry,
            'coin_exit': coin_exit,
            'btc_entry': btc_entry,
            'btc_exit': btc_exit,
            'coin_capital': coin_capital,
            'btc_capital': btc_capital,
            'coin_pnl': coin_pnl,
            'btc_pnl': btc_pnl,
            'coin_return_pct': coin_return,
            'btc_return_pct': btc_return,
            'hedged_return_pct': hedged_return,
            'total_pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'capital_after': capital + total_pnl
        }
    
    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run mean reversion + beta hedge backtest."""
        print(f"\n🚀 Running MEAN REVERSION + BETA HEDGE backtest...")
        
        hours = sorted(df['hour'].unique())
        print(f"Date range: {hours[0]} to {hours[-1]}")
        print(f"Total hours: {len(hours)}")
        
        capital = self.initial_capital
        self.equity_curve = [(hours[0], capital)]
        
        trades_executed = 0
        trades_failed = 0
        
        for i, hour in enumerate(hours, 1):
            # Get top negative funding coins
            top_coins = self.get_top_negative_funding(df, hour, n=10)
            
            if not top_coins:
                continue
            
            entry_time = hour
            exit_time = hour + timedelta(hours=1)
            
            # Try coins until we find one with valid data
            trade = None
            for coin, funding_rate in top_coins:
                trade = self.simulate_trade(coin, entry_time, exit_time, funding_rate, capital)
                if trade:
                    break
            
            if trade:
                self.trades.append(trade)
                capital = trade['capital_after']
                trades_executed += 1
            else:
                trades_failed += 1
            
            self.equity_curve.append((exit_time, capital))
            
            if i % 10 == 0 or i == len(hours):
                pct = ((capital - self.initial_capital) / self.initial_capital) * 100
                print(f"Hour {i}/{len(hours)} | Capital: ${capital:,.2f} ({pct:+.2f}%) | Trades: {trades_executed}")
        
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
        
        wins = df_trades[df_trades['total_pnl'] > 0]
        losses = df_trades[df_trades['total_pnl'] < 0]
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'num_trades': len(df_trades),
            'num_wins': len(wins),
            'num_losses': len(losses),
            'win_rate_pct': (len(wins) / len(df_trades)) * 100 if len(df_trades) > 0 else 0,
            'avg_beta': df_trades['beta'].mean(),
            'avg_hedged_return': df_trades['hedged_return_pct'].mean(),
        }
        
        # Max drawdown
        equity_series = pd.Series([eq[1] for eq in self.equity_curve])
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max * 100
        metrics['max_drawdown_pct'] = drawdowns.min()
        
        # Profit factor
        total_wins = wins['total_pnl'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['total_pnl'].sum()) if len(losses) > 0 else 0
        metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0
        
        return metrics
    
    def save_results(self, df_trades: pd.DataFrame, metrics: Dict, prefix='backtest_mr_beta'):
        """Save results to CSV."""
        df_trades.to_csv(f'{prefix}_trades.csv', index=False)
        pd.DataFrame([metrics]).to_csv(f'{prefix}_metrics.csv', index=False)
        
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_df.to_csv(f'{prefix}_equity.csv', index=False)
        
        print(f"\n💾 Results saved to {prefix}_*.csv")


def main():
    print("=" * 70)
    print("MEAN REVERSION + BETA HEDGE BACKTEST")
    print("(LONG extreme negative funding + SHORT beta*BTC)")
    print("=" * 70)
    
    config = load_config()
    print_config(config)
    
    # Initialize backtest
    try:
        backtest = MeanReversionBetaHedgedBacktest(
            price_cache_file='price_cache_with_beta_history.csv',
            beta_lookback_hours=720,  # 30 days
            initial_capital=config['initial_capital'],
            position_size_fixed=config.get('position_size_fixed', 0),
            position_size_pct=config['position_size_pct'],
            transaction_cost=config['transaction_cost']
        )
    except FileNotFoundError:
        print("\n❌ Error: price_cache.csv not found!")
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
    print("FINAL RESULTS - MEAN REVERSION + BETA HEDGE")
    print("=" * 70)
    print(f"Initial Capital:    ${metrics['initial_capital']:,.2f}")
    print(f"Final Capital:      ${metrics['final_capital']:,.2f}")
    print(f"Total Return:       {metrics['total_return_pct']:+.2f}%")
    print(f"Total Trades:       {metrics['num_trades']}")
    print(f"Win Rate:           {metrics['win_rate_pct']:.2f}%")
    print(f"Profit Factor:      {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
    print(f"Avg Beta:           {metrics['avg_beta']:.2f}")
    print(f"Avg Hedged Return:  {metrics['avg_hedged_return']:.2f}%")
    print("=" * 70)
    
    # Save results
    backtest.save_results(df_trades, metrics)
    
    print("\n✅ All done!")


if __name__ == '__main__':
    main()
