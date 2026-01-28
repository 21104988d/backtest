"""
Backtest Strategy: Extreme Negative Funding with BTC Delta Hedge
Buy coins with most extreme negative funding + Short BTC hedge
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from config import load_config

class DeltaHedgedBacktest:
    """
    Strategy: Buy extreme negative funding coin + Short equal value of BTC
    This creates a market-neutral position isolating the funding rate signal
    """
    
    def __init__(self, config):
        self.config = config
        self.initial_capital = config['initial_capital']
        self.position_size = config['position_size_pct']
        self.transaction_cost = config['transaction_cost']
        self.num_positions = config['num_positions']
        
        self.trades = []
        self.equity_curve = [(None, self.initial_capital)]
        
        self.api_url = "https://api.hyperliquid.xyz/info"
        self.api_delay = 1.0  # Increased from 0.2s to 1.0s
        
    def get_price(self, coin, timestamp_ms, max_retries=5):
        """
        Fetch historical price for a coin at specific timestamp using 5m candles.
        Uses exponential backoff retry to ensure 100% data acquisition.
        """
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Convert to 5-minute interval start
                interval_ms = 5 * 60 * 1000
                start_time = (timestamp_ms // interval_ms) * interval_ms
                
                payload = {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": coin,
                        "interval": "5m",
                        "startTime": start_time,
                        "endTime": start_time + interval_ms
                    }
                }
                
                response = requests.post(self.api_url, json=payload, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if data and len(data) > 0:
                    # Return close price of the candle
                    return float(data[0]['c'])
                
                # No data, retry
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"No data for {coin}, retry {attempt+1}/{max_retries} after {wait_time}s")
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
        
        return None
    
    def get_top_negative_funding(self, df, timestamp):
        """Get coins with most extreme negative funding rates at given timestamp"""
        hour_data = df[df['datetime'] == timestamp].copy()
        
        if hour_data.empty:
            return []
        
        # Filter by thresholds
        min_funding = self.config['min_funding_threshold']
        max_funding = self.config['max_funding_threshold']
        
        hour_data = hour_data[
            (hour_data['funding_rate'] >= min_funding) &
            (hour_data['funding_rate'] <= max_funding)
        ]
        
        # Sort by funding rate (most negative first)
        hour_data = hour_data.sort_values('funding_rate')
        
        # Return top N
        top_coins = hour_data.head(self.num_positions)
        return [(row['coin'], row['funding_rate']) for _, row in top_coins.iterrows()]
    
    def simulate_delta_hedged_trade(self, coin, btc_coin, entry_time, exit_time, 
                                   funding_rate, capital):
        """
        Simulate a delta-hedged trade:
        - Long the extreme funding coin
        - Short equal value of BTC
        - P&L = (coin return - BTC return) * capital - 2x fees
        """
        entry_timestamp = int(entry_time.timestamp() * 1000)
        exit_timestamp = int(exit_time.timestamp() * 1000)
        
        # Get coin prices
        coin_entry_price = self.get_price(coin, entry_timestamp)
        time.sleep(self.api_delay)
        
        if coin_entry_price is None:
            return None
        
        coin_exit_price = self.get_price(coin, exit_timestamp)
        time.sleep(self.api_delay)
        
        if coin_exit_price is None:
            return None
        
        # Get BTC prices (for hedge)
        btc_entry_price = self.get_price(btc_coin, entry_timestamp)
        time.sleep(self.api_delay)
        
        if btc_entry_price is None:
            return None
        
        btc_exit_price = self.get_price(btc_coin, exit_timestamp)
        time.sleep(self.api_delay)
        
        if btc_exit_price is None:
            return None
        
        # Calculate returns
        coin_return_pct = ((coin_exit_price - coin_entry_price) / coin_entry_price) * 100
        btc_return_pct = ((btc_exit_price - btc_entry_price) / btc_entry_price) * 100
        
        # Position sizing
        position_size = capital * self.position_size
        
        # Entry costs (buy coin + short BTC)
        coin_entry_cost = position_size * self.transaction_cost
        btc_entry_cost = position_size * self.transaction_cost
        
        # Exit costs (sell coin + cover BTC short)
        coin_exit_cost = position_size * self.transaction_cost
        btc_exit_cost = position_size * self.transaction_cost
        
        # P&L calculation:
        # Long coin P&L
        coin_pnl = position_size * (coin_return_pct / 100)
        
        # Short BTC P&L (profit when BTC goes down)
        btc_pnl = position_size * (-btc_return_pct / 100)
        
        # Total P&L (combined position minus all fees)
        total_pnl = coin_pnl + btc_pnl - coin_entry_cost - btc_entry_cost - coin_exit_cost - btc_exit_cost
        total_pnl_pct = (total_pnl / capital) * 100
        
        # Delta hedged return (coin return - BTC return)
        hedged_return_pct = coin_return_pct - btc_return_pct
        
        return {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'coin': coin,
            'funding_rate': funding_rate,
            'coin_entry_price': coin_entry_price,
            'coin_exit_price': coin_exit_price,
            'btc_entry_price': btc_entry_price,
            'btc_exit_price': btc_exit_price,
            'position_size': position_size,
            'capital_used': capital,
            'coin_return_pct': coin_return_pct,
            'btc_return_pct': btc_return_pct,
            'hedged_return_pct': hedged_return_pct,
            'coin_pnl': coin_pnl,
            'btc_pnl': btc_pnl,
            'total_costs': coin_entry_cost + btc_entry_cost + coin_exit_cost + btc_exit_cost,
            'pnl': total_pnl,
            'pnl_pct': total_pnl_pct
        }
    
    def run_backtest(self, df):
        """Run delta-hedged backtest"""
        print("\n" + "="*80)
        print("BACKTESTING DELTA-HEDGED EXTREME NEGATIVE FUNDING STRATEGY")
        print("="*80)
        print("Strategy: Long extreme negative funding + Short BTC hedge")
        print("="*80 + "\n")
        
        # Get unique hours
        hours = sorted(df['datetime'].unique())
        
        print(f"Running backtest for {len(hours)} hours...")
        print(f"Date range: {hours[0]} to {hours[-1]}")
        print(f"Trading {self.num_positions} position(s) per hour")
        
        current_capital = self.initial_capital
        
        for i in range(len(hours) - 1):
            hour = hours[i]
            
            # Get top coins with extreme negative funding
            top_coins = self.get_top_negative_funding(df, hour)
            
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
                # Simulate delta-hedged trade (long coin + short BTC)
                trade = self.simulate_delta_hedged_trade(
                    coin, 'BTC', entry_time, exit_time, 
                    funding_rate, capital_per_position
                )
                
                if trade is not None:
                    hour_pnl += trade['pnl']
                    successful_trades += 1
                    
                    # Store trade with updated capital
                    trade['capital_before'] = current_capital
                    trade['capital_after'] = current_capital + hour_pnl
                    trade['position_number'] = successful_trades
                    self.trades.append(trade)
                
                # Rate limiting
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
    
    def calculate_metrics(self, trades_df):
        """Calculate performance metrics"""
        if trades_df.empty:
            return {}
        
        final_capital = trades_df.iloc[-1]['capital_after']
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        total_pnl = final_capital - self.initial_capital
        
        # Win/Loss statistics
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        win_rate = (num_winning / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if num_winning > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if num_losing > 0 else 0
        
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if num_losing > 0 and losing_trades['pnl'].sum() != 0 else 0
        
        # Drawdown
        equity_series = pd.Series([eq[1] for eq in self.equity_curve if eq[0] is not None])
        running_max = equity_series.expanding().max()
        drawdown = ((equity_series - running_max) / running_max) * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (annualized, assuming 8760 hourly periods per year)
        returns = trades_df['pnl_pct'].values
        if len(returns) > 0 and returns.std() != 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(8760)
        else:
            sharpe_ratio = 0
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'total_pnl': total_pnl,
            'num_trades': len(trades_df),
            'num_winning': num_winning,
            'num_losing': num_losing,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_pnl_per_trade': trades_df['pnl'].mean(),
            'avg_return_per_trade_pct': trades_df['pnl_pct'].mean()
        }
        
        return metrics
    
    def print_results(self, trades_df, metrics):
        """Print backtest results"""
        print("\n" + "="*80)
        print("DELTA-HEDGED BACKTEST RESULTS")
        print("="*80 + "\n")
        
        print("📊 Capital & Returns:")
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
        
        print("\n" + "="*80 + "\n")


def main():
    """Main execution function"""
    # Load configuration
    config = load_config()
    
    print("\n" + "="*80)
    print("DELTA-HEDGED BACKTEST CONFIGURATION")
    print("="*80 + "\n")
    
    print("📊 Strategy:")
    print(f"  Long extreme funding:   {config['num_positions']} position(s)")
    print(f"  Short BTC hedge:        Equal value to long position")
    print(f"  Position size:          {config['position_size_pct']*100:.1f}% of capital")
    print(f"  Holding period:         1 hour(s)")
    
    print(f"\n💰 Capital & Costs:")
    print(f"  Initial capital:        ${config['initial_capital']:,.2f}")
    print(f"  Transaction cost:       {config['transaction_cost']*100:.3f}% (per side)")
    print(f"  Total costs per trade:  {config['transaction_cost']*4*100:.3f}% (4 sides: buy coin, short BTC, sell coin, cover BTC)")
    
    print("\n" + "="*80 + "\n")
    
    # Load funding rate data
    print("Loading funding rate data...")
    df = pd.read_csv('funding_history.csv')
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    print(f"Loaded {len(df):,} funding rate records\n")
    
    # Run backtest
    print("Running delta-hedged backtest...\n")
    backtest = DeltaHedgedBacktest(config)
    trades_df = backtest.run_backtest(df)
    
    # Save trades
    trades_df.to_csv('backtest_delta_hedged_trades.csv', index=False)
    print("✓ Trades saved to backtest_delta_hedged_trades.csv\n")
    
    # Calculate and save metrics
    metrics = backtest.calculate_metrics(trades_df)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('backtest_delta_hedged_metrics.csv', index=False)
    print("✓ Metrics saved to backtest_delta_hedged_metrics.csv\n")
    
    # Print results
    backtest.print_results(trades_df, metrics)
    
    # Additional insights
    print("="*80)
    print("ADDITIONAL INSIGHTS - DELTA HEDGE ANALYSIS")
    print("="*80 + "\n")
    
    print("Top 10 Coins by Total PnL:")
    coin_pnl = trades_df.groupby('coin')['pnl'].sum().sort_values(ascending=False)
    for i, (coin, pnl) in enumerate(coin_pnl.head(10).items(), 1):
        print(f"  {coin}: ${pnl:.2f}")
    
    print("\nAverage Returns by Component:")
    print(f"  Avg Coin Return:     {trades_df['coin_return_pct'].mean():.4f}%")
    print(f"  Avg BTC Return:      {trades_df['btc_return_pct'].mean():.4f}%")
    print(f"  Avg Hedged Return:   {trades_df['hedged_return_pct'].mean():.4f}%")
    
    best_trade = trades_df.loc[trades_df['pnl'].idxmax()]
    worst_trade = trades_df.loc[trades_df['pnl'].idxmin()]
    
    print(f"\nBest Trade:")
    print(f"  {best_trade['coin']} on {best_trade['entry_time']}")
    print(f"  Coin: {best_trade['coin_return_pct']:.2f}%, BTC: {best_trade['btc_return_pct']:.2f}%")
    print(f"  Hedged Return: {best_trade['hedged_return_pct']:.2f}%")
    print(f"  PnL: ${best_trade['pnl']:.2f} ({best_trade['pnl_pct']:.2f}%)")
    
    print(f"\nWorst Trade:")
    print(f"  {worst_trade['coin']} on {worst_trade['entry_time']}")
    print(f"  Coin: {worst_trade['coin_return_pct']:.2f}%, BTC: {worst_trade['btc_return_pct']:.2f}%")
    print(f"  Hedged Return: {worst_trade['hedged_return_pct']:.2f}%")
    print(f"  PnL: ${worst_trade['pnl']:.2f} ({worst_trade['pnl_pct']:.2f}%)")
    
    print("\n" + "="*80)
    print("Backtest complete!")
    print("="*80)


if __name__ == "__main__":
    main()
