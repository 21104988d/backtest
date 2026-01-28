"""
Run all 6 strategies with proper price cache and compare results.
"""
import pandas as pd
import numpy as np
import os
from datetime import timedelta

BASEDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASEDIR)

from config import load_config

# Load config
config = load_config()
INITIAL_CAPITAL = config['initial_capital']
POSITION_SIZE = config.get('position_size_fixed', 1000)
TRANSACTION_COST = config['transaction_cost']

# Load data
print("Loading data...")
price_cache = pd.read_csv('price_cache_with_beta_history.csv')
price_cache['timestamp'] = pd.to_datetime(price_cache['timestamp'])
price_lookup = dict(zip(
    price_cache['coin'] + '_' + price_cache['timestamp'].astype(str),
    price_cache['price']
))

funding = pd.read_csv('funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed')
funding['hour'] = funding['datetime'].dt.floor('h')

# Get trading hours
hours = sorted(funding['hour'].unique())
# Filter to hours where we have price data
price_hours = set(price_cache['timestamp'].unique())
hours = [h for h in hours if h in price_hours and (h + timedelta(hours=1)) in price_hours]
print(f"Trading hours: {len(hours)} (from {hours[0]} to {hours[-1]})")

def get_price(coin, timestamp):
    key = f"{coin}_{timestamp}"
    return price_lookup.get(key)

def get_top_negative(hour, n=10):
    """Get top N most negative funding coins."""
    hour_data = funding[funding['hour'] == hour].sort_values('funding_rate').head(n)
    return [(r['coin'], r['funding_rate']) for _, r in hour_data.iterrows()]

def get_top_positive(hour, n=10):
    """Get top N most positive funding coins."""
    hour_data = funding[funding['hour'] == hour].sort_values('funding_rate', ascending=False).head(n)
    return [(r['coin'], r['funding_rate']) for _, r in hour_data.iterrows()]

def calculate_beta(coin, current_time, lookback_hours=720):
    """Calculate rolling beta vs BTC."""
    start_time = current_time - timedelta(hours=lookback_hours)
    
    # Get price data
    coin_data = price_cache[(price_cache['coin'] == coin) & 
                            (price_cache['timestamp'] >= start_time) & 
                            (price_cache['timestamp'] < current_time)].sort_values('timestamp')
    btc_data = price_cache[(price_cache['coin'] == 'BTC') & 
                           (price_cache['timestamp'] >= start_time) & 
                           (price_cache['timestamp'] < current_time)].sort_values('timestamp')
    
    if len(coin_data) < 24 or len(btc_data) < 24:
        return 1.0  # Default to delta hedge
    
    # Calculate returns
    coin_ret = coin_data.set_index('timestamp')['price'].pct_change().dropna()
    btc_ret = btc_data.set_index('timestamp')['price'].pct_change().dropna()
    
    common = coin_ret.index.intersection(btc_ret.index)
    if len(common) < 24:
        return 1.0
    
    cov = coin_ret.loc[common].cov(btc_ret.loc[common])
    var = btc_ret.loc[common].var()
    
    if var == 0:
        return 1.0
    
    return cov / var

def run_strategy(name, get_coins_func, direction, hedge_type):
    """
    Run a strategy.
    direction: 'long' or 'short' on the altcoin
    hedge_type: None, 'delta', or 'beta'
    """
    capital = INITIAL_CAPITAL
    trades = []
    equity = [(hours[0], capital)]
    
    for hour in hours[:-1]:  # Exclude last hour (no exit price)
        entry_time = hour
        exit_time = hour + timedelta(hours=1)
        
        # Get coins to trade
        coins = get_coins_func(hour)
        
        # Find first tradeable coin
        trade = None
        for coin, fr in coins:
            entry_price = get_price(coin, entry_time)
            exit_price = get_price(coin, exit_time)
            btc_entry = get_price('BTC', entry_time)
            btc_exit = get_price('BTC', exit_time)
            
            if entry_price and exit_price and btc_entry and btc_exit:
                # Calculate position
                trade_capital = min(POSITION_SIZE, capital)
                
                if hedge_type == 'beta':
                    beta = calculate_beta(coin, entry_time)
                elif hedge_type == 'delta':
                    beta = 1.0
                else:
                    beta = 0  # No hedge
                
                # Split capital between coin and hedge
                if beta > 0:
                    coin_capital = trade_capital / (1 + beta)
                    btc_capital = trade_capital * beta / (1 + beta)
                else:
                    coin_capital = trade_capital
                    btc_capital = 0
                
                # Entry costs
                entry_cost = (coin_capital + btc_capital) * TRANSACTION_COST
                
                # Coin position
                coin_position = (coin_capital - entry_cost/2) / entry_price
                if direction == 'long':
                    coin_pnl = coin_position * (exit_price - entry_price)
                else:  # short
                    coin_pnl = coin_position * (entry_price - exit_price)
                
                # Hedge position (opposite direction)
                if btc_capital > 0:
                    btc_position = (btc_capital - entry_cost/2) / btc_entry
                    if direction == 'long':
                        # Long coin + Short BTC
                        btc_pnl = btc_position * (btc_entry - btc_exit)
                    else:
                        # Short coin + Long BTC  
                        btc_pnl = btc_position * (btc_exit - btc_entry)
                else:
                    btc_pnl = 0
                
                # Exit costs
                exit_value = coin_capital + coin_pnl + btc_capital + btc_pnl
                exit_cost = exit_value * TRANSACTION_COST
                
                total_pnl = coin_pnl + btc_pnl - entry_cost - exit_cost
                
                trade = {
                    'entry_time': entry_time,
                    'coin': coin,
                    'funding_rate': fr,
                    'beta': beta,
                    'pnl': total_pnl
                }
                break
        
        if trade:
            trades.append(trade)
            capital += trade['pnl']
        
        equity.append((exit_time, capital))
    
    # Calculate metrics
    df = pd.DataFrame(trades)
    if df.empty:
        return {'name': name, 'return': None}
    
    final = capital
    total_return = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    wins = len(df[df['pnl'] > 0])
    win_rate = wins / len(df) * 100
    
    # Max drawdown
    eq_df = pd.DataFrame(equity, columns=['time', 'capital'])
    eq_df['peak'] = eq_df['capital'].cummax()
    eq_df['dd'] = (eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100
    max_dd = abs(eq_df['dd'].min())
    
    avg_beta = df['beta'].mean() if 'beta' in df else None
    
    return {
        'name': name,
        'return': total_return,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'trades': len(df),
        'avg_beta': avg_beta,
        'final_capital': final
    }

# Run all 6 strategies
print("\n" + "="*70)
print("RUNNING ALL 6 STRATEGIES")
print("="*70)

strategies = [
    ("Mean Reversion (LONG)", get_top_negative, 'long', None),
    ("Mean Reversion + Delta Hedge", get_top_negative, 'long', 'delta'),
    ("Mean Reversion + Beta Hedge", get_top_negative, 'long', 'beta'),
    ("Trend Following (SHORT)", get_top_negative, 'short', None),
    ("Trend Following + Delta Hedge", get_top_negative, 'short', 'delta'),
    ("Trend Following + Beta Hedge", get_top_negative, 'short', 'beta'),
]

results = []
for name, get_coins, direction, hedge in strategies:
    print(f"\nRunning: {name}...")
    result = run_strategy(name, get_coins, direction, hedge)
    results.append(result)
    if result['return'] is not None:
        print(f"  Return: {result['return']:+.2f}% | Win Rate: {result['win_rate']:.1f}% | Max DD: {result['max_dd']:.2f}%")

# Summary
print("\n" + "="*70)
print("FINAL COMPARISON - ALL 6 STRATEGIES")
print("="*70)
print(f"\n{'Strategy':<35} {'Return':>10} {'Win Rate':>10} {'Max DD':>10} {'Trades':>8} {'Avg Beta':>10}")
print("-" * 85)

for r in sorted(results, key=lambda x: x.get('return') or -999, reverse=True):
    ret = f"{r['return']:+.2f}%" if r['return'] is not None else 'N/A'
    wr = f"{r['win_rate']:.1f}%" if r.get('win_rate') else 'N/A'
    dd = f"{r['max_dd']:.2f}%" if r.get('max_dd') else 'N/A'
    trades = str(r.get('trades', 'N/A'))
    beta = f"{r['avg_beta']:.2f}" if r.get('avg_beta') else 'N/A'
    print(f"{r['name']:<35} {ret:>10} {wr:>10} {dd:>10} {trades:>8} {beta:>10}")

print("="*70)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('strategy_comparison_all6.csv', index=False)
print(f"\n💾 Results saved to strategy_comparison_all6.csv")
