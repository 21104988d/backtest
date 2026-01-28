"""
Trend Following + Delta Hedge: SHORT extreme negative funding + LONG BTC hedge.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from config import load_config, print_config

BASEDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASEDIR)


class TrendFollowingDeltaHedgedBacktest:
    """Trend following with delta hedge: SHORT coin + LONG BTC."""
    
    def __init__(self, price_cache_file='price_cache.csv', **kwargs):
        self.initial_capital = kwargs.get('initial_capital', 10000)
        self.position_size_fixed = kwargs.get('position_size_fixed', 1000)
        self.position_size_pct = kwargs.get('position_size_pct', 1.0)
        self.transaction_cost = kwargs.get('transaction_cost', 0.0005)
        
        self.trades = []
        self.equity_curve = []
        
        print(f"\n📦 Loading price cache from {price_cache_file}...")
        self.price_cache = pd.read_csv(price_cache_file)
        self.price_cache['timestamp'] = pd.to_datetime(self.price_cache['timestamp'], format='mixed')
        self.price_cache['key'] = self.price_cache['coin'] + '_' + self.price_cache['timestamp'].astype(str)
        self.price_lookup = dict(zip(self.price_cache['key'], self.price_cache['price']))
        print(f"✓ Loaded {len(self.price_cache):,} cached prices")
    
    def get_cached_price(self, coin: str, timestamp: pd.Timestamp) -> Optional[float]:
        key = f"{coin}_{timestamp}"
        return self.price_lookup.get(key)
    
    def load_funding_data(self, funding_file: str = 'funding_history.csv') -> pd.DataFrame:
        df = pd.read_csv(funding_file)
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        df = df.sort_values(['datetime', 'coin']).reset_index(drop=True)
        df['hour'] = df['datetime'].dt.floor('h')
        return df
    
    def get_top_negative_funding_with_price(self, df: pd.DataFrame, hour: pd.Timestamp, 
                                             exit_time: pd.Timestamp) -> Optional[Tuple[str, float]]:
        hour_data = df[df['hour'] == hour]
        if hour_data.empty:
            return None
        
        candidates = hour_data.nsmallest(10, 'funding_rate')
        
        for _, row in candidates.iterrows():
            coin = row['coin']
            coin_entry = self.get_cached_price(coin, hour)
            coin_exit = self.get_cached_price(coin, exit_time)
            btc_entry = self.get_cached_price('BTC', hour)
            btc_exit = self.get_cached_price('BTC', exit_time)
            
            if all([coin_entry, coin_exit, btc_entry, btc_exit]):
                return (coin, row['funding_rate'])
        
        return None
    
    def simulate_delta_hedged_short_trade(self, coin: str, entry_time: pd.Timestamp, 
                                          exit_time: pd.Timestamp, funding_rate: float, 
                                          capital: float) -> Optional[Dict]:
        """
        Delta-hedged SHORT trade:
        - SHORT the extreme funding coin
        - LONG equal value of BTC as hedge
        """
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
            position_capital = min(self.position_size_fixed, capital)
        else:
            position_capital = capital * self.position_size_pct
        
        # 4 transactions: short coin, cover coin, buy BTC, sell BTC
        total_costs = position_capital * self.transaction_cost * 4
        
        # P&L calculation
        # SHORT coin: profit when price falls (negative return = profit)
        coin_pnl = position_capital * (-coin_return_pct / 100)
        # LONG BTC hedge: profit when BTC rises
        btc_pnl = position_capital * (btc_return_pct / 100)
        
        total_pnl = coin_pnl + btc_pnl - total_costs
        total_pnl_pct = (total_pnl / capital) * 100
        
        # Hedged return (short coin return minus BTC return)
        hedged_return_pct = -coin_return_pct - btc_return_pct
        
        return {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'coin': coin,
            'funding_rate': funding_rate,
            'direction': 'SHORT + LONG BTC',
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
        print(f"\n🚀 Running TREND FOLLOWING + DELTA HEDGE backtest...")
        
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
            
            result = self.get_top_negative_funding_with_price(df, hour, exit_time)
            if not result:
                trades_failed += 1
                self.equity_curve.append((exit_time, capital))
                continue
            
            coin, funding_rate = result
            trade = self.simulate_delta_hedged_short_trade(coin, entry_time, exit_time, funding_rate, capital)
            
            if trade:
                self.trades.append(trade)
                capital = trade['capital_after']
                trades_executed += 1
                self.equity_curve.append((exit_time, capital))
            
            if i % 10 == 0 or i == len(hours):
                pct_change = ((capital - self.initial_capital) / self.initial_capital) * 100
                print(f"Hour {i}/{len(hours)} | Capital: ${capital:,.2f} ({pct_change:+.2f}%) | Trades: {trades_executed}")
        
        print(f"\n✅ Backtest complete!")
        print(f"  - Trades executed: {trades_executed}/{len(hours)} ({trades_executed/len(hours)*100:.1f}%)")
        print(f"  - Trades failed: {trades_failed} ({trades_failed/len(hours)*100:.1f}%)")
        
        return pd.DataFrame(self.trades)
    
    def calculate_metrics(self, df_trades: pd.DataFrame) -> Dict:
        if df_trades.empty:
            return {}
        
        final_capital = df_trades.iloc[-1]['capital_after']
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] < 0]
        
        equity = [self.initial_capital] + df_trades['capital_after'].tolist()
        peak = equity[0]
        max_dd = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'num_trades': len(df_trades),
            'num_wins': len(wins),
            'num_losses': len(losses),
            'win_rate_pct': (len(wins) / len(df_trades)) * 100,
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_dd,
            'total_fees': df_trades['total_costs'].sum(),
            'avg_hedged_return': df_trades['hedged_return_pct'].mean(),
        }
    
    def save_results(self, df_trades: pd.DataFrame, metrics: Dict):
        df_trades.to_csv('backtest_trend_following_delta_trades.csv', index=False)
        pd.DataFrame([metrics]).to_csv('backtest_trend_following_delta_metrics.csv', index=False)
        
        eq_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        eq_df.to_csv('backtest_trend_following_delta_equity.csv', index=False)


def main():
    print("="*70)
    print("TREND FOLLOWING + DELTA HEDGE BACKTEST")
    print("(SHORT extreme negative funding + LONG BTC)")
    print("="*70)
    
    config = load_config()
    print_config(config)
    
    bt = TrendFollowingDeltaHedgedBacktest(
        initial_capital=config['initial_capital'],
        position_size_fixed=config.get('position_size_fixed', 0),
        position_size_pct=config['position_size_pct'],
        transaction_cost=config['transaction_cost']
    )
    
    df = bt.load_funding_data()
    print(f"\n✓ Loaded {len(df):,} funding rate records")
    
    trades = bt.run_backtest(df)
    metrics = bt.calculate_metrics(trades)
    
    print("\n" + "="*70)
    print("FINAL RESULTS - TREND FOLLOWING + DELTA HEDGE")
    print("="*70)
    print(f"Initial Capital:    ${metrics['initial_capital']:,.2f}")
    print(f"Final Capital:      ${metrics['final_capital']:,.2f}")
    print(f"Total Return:       {metrics['total_return_pct']:.2f}%")
    print(f"Total Trades:       {metrics['num_trades']}")
    print(f"Win Rate:           {metrics['win_rate_pct']:.2f}%")
    print(f"Profit Factor:      {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
    print(f"Avg Hedged Return:  {metrics['avg_hedged_return']:.2f}%")
    print("="*70)
    
    bt.save_results(trades, metrics)
    print("\n💾 Results saved to backtest_trend_following_delta_*.csv")


if __name__ == "__main__":
    main()
