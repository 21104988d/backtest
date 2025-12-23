import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

class BacktestEngine:
    def __init__(self, ohlc_data, funding_data, initial_balance=1000):
        self.ohlc_data = ohlc_data
        self.funding_data = funding_data
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = {
            'size_usd': 0, 
            'side': None, 
            'avg_entry_price': 0,
            'stored_rate': None
        }
        self.trades = []
        self.equity_curve = []
        self.max_position_size = 0

    def prepare_data(self, timeframe):
        # Resample OHLC
        ohlc = self.ohlc_data.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        # Resample/Reindex Funding
        # Funding data is hourly. We need to align it with OHLC.
        # Forward fill funding rates to match OHLC timestamps.
        funding = self.funding_data.reindex(ohlc.index, method='ffill')
        
        # Combine
        df = pd.concat([ohlc, funding], axis=1)
        # Fill any remaining NaNs (e.g. if OHLC starts before funding)
        df['funding_rate'] = df['funding_rate'].ffill().fillna(0)
        
        return df

    def run(self, timeframe):
        self.reset()
        df = self.prepare_data(timeframe)
        
        # print(f"Running backtest on {timeframe} timeframe with {len(df)} candles.")
        
        for timestamp, row in df.iterrows():
            self.process_candle(timestamp, row)
            
            # Track max position size
            if self.position['size_usd'] > self.max_position_size:
                self.max_position_size = self.position['size_usd']

            # Calculate unrealized PnL
            unrealized_pnl = 0
            if self.position['size_usd'] > 0:
                current_price = row['close']
                if self.position['side'] == 'long':
                    pnl_btc = self.position['size_usd'] * (1/self.position['avg_entry_price'] - 1/current_price)
                    unrealized_pnl = pnl_btc * current_price
                else:
                    pnl_btc = self.position['size_usd'] * (1/current_price - 1/self.position['avg_entry_price'])
                    unrealized_pnl = pnl_btc * current_price
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.balance + unrealized_pnl,
                'price': row['close'],
                'funding_rate': row['funding_rate'],
                'position_size': self.position['size_usd']
            })

        return pd.DataFrame(self.equity_curve).set_index('timestamp')

    def process_candle(self, timestamp, row):
        rate = row['funding_rate']
        price = row['close']
        
        # Funding payments (every 8 hours: 04, 12, 20 UTC)
        # We check if the current candle *contains* or *is* the funding time.
        # Since we might be on 5m, we check if hour in [4,12,20] and minute is close to 0.
        # Or simpler: if we crossed a funding timestamp.
        # Let's stick to: if timestamp.hour in [4, 12, 20] and timestamp.minute == 0.
        # But if 5m data, we have 04:00, 04:05. 04:00 is the funding time.
        if timestamp.hour in [4, 12, 20] and timestamp.minute == 0:
             self.apply_funding(rate, price)

        # Strategy Logic
        if self.position['size_usd'] == 0:
            if rate != 0:
                if rate < 0:
                    self.open_position('long', 10, price, rate)
                elif rate > 0:
                    self.open_position('short', 10, price, rate)
        else:
            if self.position['side'] == 'long':
                if rate >= 0:
                    self.close_position(price)
                    if rate > 0:
                        self.open_position('short', 10, price, rate)
                else:
                    if rate < self.position['stored_rate']:
                        self.open_position('long', 10, price, rate)
                        self.position['stored_rate'] = rate
            
            elif self.position['side'] == 'short':
                if rate <= 0:
                    self.close_position(price)
                    if rate < 0:
                        self.open_position('long', 10, price, rate)
                else:
                    if rate > self.position['stored_rate']:
                        self.open_position('short', 10, price, rate)
                        self.position['stored_rate'] = rate

    def open_position(self, side, size_usd, price, rate):
        current_size = self.position['size_usd']
        new_size = current_size + size_usd
        
        if current_size > 0:
            current_btc = current_size / self.position['avg_entry_price']
            new_btc = size_usd / price
            total_btc = current_btc + new_btc
            avg_price = new_size / total_btc
        else:
            avg_price = price
            
        self.position['size_usd'] = new_size
        self.position['side'] = side
        self.position['avg_entry_price'] = avg_price
        
        if current_size == 0:
            self.position['stored_rate'] = rate
            
        self.trades.append({
            'type': 'open' if current_size == 0 else 'add',
            'side': side,
            'size': size_usd,
            'price': price,
            'rate': rate
        })

    def close_position(self, price):
        size = self.position['size_usd']
        side = self.position['side']
        entry = self.position['avg_entry_price']
        
        if side == 'long':
            pnl_btc = size * (1/entry - 1/price)
        else:
            pnl_btc = size * (1/price - 1/entry)
            
        pnl_usd = pnl_btc * price
        self.balance += pnl_usd
        
        self.trades.append({
            'type': 'close',
            'side': side,
            'size': size,
            'price': price,
            'pnl': pnl_usd
        })
        
        self.position = {
            'size_usd': 0,
            'side': None,
            'avg_entry_price': 0,
            'stored_rate': None
        }

    def apply_funding(self, rate, price):
        if self.position['size_usd'] > 0:
            value_btc = self.position['size_usd'] / price
            funding_btc = value_btc * rate
            
            if self.position['side'] == 'long':
                funding_pnl_btc = -funding_btc
            else:
                funding_pnl_btc = funding_btc
                
            funding_pnl_usd = funding_pnl_btc * price
            self.balance += funding_pnl_usd

    def get_stats(self):
        equity = pd.DataFrame(self.equity_curve).set_index('timestamp')
        if equity.empty:
            return {}
            
        start_bal = equity['equity'].iloc[0]
        end_bal = equity['equity'].iloc[-1]
        
        # Returns
        total_return = (end_bal - start_bal) / start_bal
        days = (equity.index[-1] - equity.index[0]).days
        if days > 0:
            apr = (1 + total_return) ** (365 / days) - 1
        else:
            apr = 0
            
        # Drawdown
        equity['peak'] = equity['equity'].cummax()
        equity['drawdown'] = (equity['equity'] - equity['peak']) / equity['peak']
        max_dd = equity['drawdown'].min()
        
        # Trades
        closed_trades = [t for t in self.trades if t['type'] == 'close']
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t['pnl'] > 0])
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0
        avg_return_per_trade = np.mean([t['pnl'] for t in closed_trades]) if total_trades > 0 else 0
        
        # Sharpe & Sortino
        # Calculate daily returns for Sharpe
        daily_returns = equity['equity'].resample('D').last().pct_change().dropna()
        if len(daily_returns) > 1:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
            
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 0:
                sortino = daily_returns.mean() / downside_returns.std() * np.sqrt(365)
            else:
                sortino = np.inf
        else:
            sharpe = 0
            sortino = 0
            
        return {
            'return_pct': total_return * 100,
            'apr_pct': apr * 100,
            'max_drawdown_pct': max_dd * 100,
            'total_trades': total_trades,
            'win_rate_pct': win_rate * 100,
            'avg_return_per_trade': avg_return_per_trade,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_position_size': self.max_position_size
        }

if __name__ == "__main__":
    try:
        import os
        base_dir = os.path.dirname(__file__)
        ohlc_path = os.path.join(base_dir, "ohlc_5m.csv")
        funding_path = os.path.join(base_dir, "funding_rates.csv")
        
        df_ohlc = pd.read_csv(ohlc_path)
        df_ohlc['timestamp'] = pd.to_datetime(df_ohlc['timestamp'])
        df_ohlc = df_ohlc.set_index('timestamp').sort_index()
        
        df_funding = pd.read_csv(funding_path)
        df_funding['timestamp'] = pd.to_datetime(df_funding['timestamp'])
        df_funding = df_funding.set_index('timestamp').sort_index()
        # Keep only funding_rate column
        df_funding = df_funding[['funding_rate']]
        
    except Exception as e:
        print(f"Data error: {e}")
        print("Please run fetch_data.py first.")
        exit()

    timeframes = ['5min', '15min', '1h', '4h', '1d']
    results = {}

    print(f"{'Timeframe':<10} | {'APR':<8} | {'Sharpe':<6} | {'Sortino':<7} | {'Max DD':<8} | {'Trades':<6} | {'Win Rate':<8} | {'Avg PnL':<8}")
    print("-" * 90)

    for tf in timeframes:
        engine = BacktestEngine(df_ohlc, df_funding)
        equity = engine.run(tf)
        stats = engine.get_stats()
        results[tf] = stats
        
        print(f"{tf:<10} | {stats['apr_pct']:>7.2f}% | {stats['sharpe_ratio']:>6.2f} | {stats['sortino_ratio']:>7.2f} | {stats['max_drawdown_pct']:>7.2f}% | {stats['total_trades']:>6} | {stats['win_rate_pct']:>7.2f}% | {stats['avg_return_per_trade']:>8.2f}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(equity.index, equity['equity'], label='Equity')
        plt.title(f'Equity Curve - {tf}')
        plt.legend()
        plt.savefig(os.path.join(base_dir, f'equity_{tf}.png'))
        plt.close()
