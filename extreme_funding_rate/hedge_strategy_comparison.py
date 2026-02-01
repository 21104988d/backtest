"""
Hedging Strategy Comparison for Funding Rate Capture

Strategies:
1. Delta Hedge with BTC: 1:1 notional hedge
2. Beta Hedge with BTC: Hedge ratio = beta (cov/var)
3. Beta Hedge with ETH: Use ETH as hedge instrument
4. Multi-Asset Hedge: Optimal combination of BTC + ETH

For each strategy, we calculate:
- Hedged portfolio return
- Net P&L (funding earned - price exposure)
- Sharpe ratio
- Max drawdown
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load funding and price data"""
    funding = pd.read_csv('funding_history.csv')
    funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)
    funding['hour'] = funding['datetime'].dt.floor('h').dt.tz_localize(None)
    
    price = pd.read_csv('price_history.csv')
    price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
    price['hour'] = price['timestamp'].dt.floor('h').dt.tz_localize(None)
    
    return funding, price

def build_price_matrix(price_df):
    """Build coin x hour price matrix"""
    pivot = price_df.pivot_table(index='hour', columns='coin', values='price', aggfunc='first')
    return pivot

def calculate_returns(price_matrix):
    """Calculate hourly returns for all coins"""
    returns = price_matrix.pct_change()
    return returns

# =============================================================================
# BETA CALCULATIONS
# =============================================================================

def calculate_rolling_beta(returns, target_coin, hedge_coin, window=168):
    """
    Calculate rolling beta: beta = cov(target, hedge) / var(hedge)
    window=168 (1 week of hourly data)
    """
    if target_coin not in returns.columns or hedge_coin not in returns.columns:
        return None
    
    target = returns[target_coin]
    hedge = returns[hedge_coin]
    
    # Rolling covariance and variance
    cov = target.rolling(window).cov(hedge)
    var = hedge.rolling(window).var()
    
    beta = cov / var
    return beta

def calculate_multi_hedge_weights(returns, target_coin, hedge_coins, window=168):
    """
    Calculate optimal hedge weights using multiple regression
    Minimize variance of: target - w1*hedge1 - w2*hedge2 - ...
    
    Uses rolling OLS
    """
    if target_coin not in returns.columns:
        return None
    
    available_hedges = [c for c in hedge_coins if c in returns.columns]
    if len(available_hedges) < 2:
        return None
    
    target = returns[target_coin]
    hedges = returns[available_hedges]
    
    # Rolling regression weights
    weights_dict = {h: [] for h in available_hedges}
    indices = []
    
    for i in range(window, len(returns)):
        y = target.iloc[i-window:i].values
        X = hedges.iloc[i-window:i].values
        
        # Remove NaN rows
        mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        if mask.sum() < window // 2:
            for h in available_hedges:
                weights_dict[h].append(np.nan)
            indices.append(returns.index[i])
            continue
        
        y_clean = y[mask]
        X_clean = X[mask]
        
        try:
            # OLS: w = (X'X)^-1 X'y
            XtX = X_clean.T @ X_clean
            Xty = X_clean.T @ y_clean
            w = np.linalg.solve(XtX, Xty)
            
            for j, h in enumerate(available_hedges):
                weights_dict[h].append(w[j])
        except:
            for h in available_hedges:
                weights_dict[h].append(np.nan)
        
        indices.append(returns.index[i])
    
    weights_df = pd.DataFrame(weights_dict, index=indices)
    return weights_df

# =============================================================================
# HEDGING STRATEGIES
# =============================================================================

def simulate_hedged_trade(
    coin, 
    hour, 
    funding_rate, 
    position_type,  # 'long' or 'short'
    returns_matrix,
    hedge_strategy,
    betas_btc=None,
    betas_eth=None,
    multi_weights=None
):
    """
    Simulate a single hedged trade
    
    Returns dict with:
    - funding_pnl: funding rate earned
    - price_pnl: unhedged price exposure
    - hedge_pnl: P&L from hedge position
    - net_pnl: total P&L
    """
    result = {
        'coin': coin,
        'hour': hour,
        'funding_rate': funding_rate,
        'position_type': position_type,
        'strategy': hedge_strategy,
        'funding_pnl': 0,
        'price_pnl': 0,
        'hedge_pnl': 0,
        'net_pnl': 0,
        'hedge_ratio': 0,
        'valid': False
    }
    
    # Get returns for this hour
    next_hour = hour + timedelta(hours=1)
    if hour not in returns_matrix.index or next_hour not in returns_matrix.index:
        return result
    
    coin_return = returns_matrix.loc[next_hour, coin] if coin in returns_matrix.columns else np.nan
    btc_return = returns_matrix.loc[next_hour, 'BTC'] if 'BTC' in returns_matrix.columns else np.nan
    eth_return = returns_matrix.loc[next_hour, 'ETH'] if 'ETH' in returns_matrix.columns else np.nan
    
    if pd.isna(coin_return):
        return result
    
    # Position direction
    direction = 1 if position_type == 'long' else -1
    
    # Funding P&L (you receive funding when position is opposite to funding direction)
    # Long position + negative funding = you receive abs(funding)
    # Short position + positive funding = you receive funding
    result['funding_pnl'] = abs(funding_rate)
    
    # Price P&L from main position (without hedge)
    result['price_pnl'] = direction * coin_return
    
    # Hedge P&L based on strategy
    if hedge_strategy == 'delta_btc':
        # 1:1 hedge with BTC
        if pd.isna(btc_return):
            return result
        result['hedge_ratio'] = 1.0
        result['hedge_pnl'] = -direction * btc_return  # Opposite direction
        
    elif hedge_strategy == 'beta_btc':
        # Beta hedge with BTC
        if betas_btc is None or coin not in betas_btc.columns or pd.isna(btc_return):
            return result
        if hour not in betas_btc.index:
            return result
        beta = betas_btc.loc[hour, coin]
        if pd.isna(beta):
            return result
        result['hedge_ratio'] = beta
        result['hedge_pnl'] = -direction * beta * btc_return
        
    elif hedge_strategy == 'beta_eth':
        # Beta hedge with ETH
        if betas_eth is None or coin not in betas_eth.columns or pd.isna(eth_return):
            return result
        if hour not in betas_eth.index:
            return result
        beta = betas_eth.loc[hour, coin]
        if pd.isna(beta):
            return result
        result['hedge_ratio'] = beta
        result['hedge_pnl'] = -direction * beta * eth_return
        
    elif hedge_strategy == 'multi_hedge':
        # Multi-asset hedge with BTC + ETH
        if multi_weights is None or coin not in multi_weights:
            return result
        weights = multi_weights[coin]
        if hour not in weights.index:
            return result
        
        w_btc = weights.loc[hour, 'BTC'] if 'BTC' in weights.columns else 0
        w_eth = weights.loc[hour, 'ETH'] if 'ETH' in weights.columns else 0
        
        if pd.isna(w_btc) or pd.isna(w_eth) or pd.isna(btc_return) or pd.isna(eth_return):
            return result
        
        result['hedge_ratio'] = abs(w_btc) + abs(w_eth)
        result['hedge_pnl'] = -direction * (w_btc * btc_return + w_eth * eth_return)
    
    elif hedge_strategy == 'no_hedge':
        result['hedge_ratio'] = 0
        result['hedge_pnl'] = 0
    
    result['net_pnl'] = result['funding_pnl'] + result['price_pnl'] + result['hedge_pnl']
    result['valid'] = True
    
    return result

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("Loading data...")
    funding, price = load_data()
    
    print("Building price matrix and returns...")
    price_matrix = build_price_matrix(price)
    returns = calculate_returns(price_matrix)
    
    # Get extreme funding events
    events = pd.read_csv('extreme_funding_1h_events.csv')
    events['hour'] = pd.to_datetime(events['hour'])
    
    # Filter for extreme funding only
    THRESHOLD = 0.001  # 0.10%
    extreme_events = events[
        ((events['type'] == 'negative') & (events['funding_rate'] < -THRESHOLD)) |
        ((events['type'] == 'positive') & (events['funding_rate'] > THRESHOLD))
    ].copy()
    
    print(f"Total extreme events (|FR| > {THRESHOLD*100:.2f}%): {len(extreme_events)}")
    
    # Get unique coins for beta calculation
    coins = extreme_events['coin'].unique()
    print(f"Unique coins: {len(coins)}")
    
    # Calculate rolling betas
    print("Calculating rolling betas (this may take a moment)...")
    
    # Beta vs BTC
    betas_btc = pd.DataFrame(index=returns.index)
    for coin in coins:
        if coin != 'BTC':
            beta = calculate_rolling_beta(returns, coin, 'BTC')
            if beta is not None:
                betas_btc[coin] = beta
    
    # Beta vs ETH
    betas_eth = pd.DataFrame(index=returns.index)
    for coin in coins:
        if coin != 'ETH':
            beta = calculate_rolling_beta(returns, coin, 'ETH')
            if beta is not None:
                betas_eth[coin] = beta
    
    # Multi-asset weights (for a subset to save time)
    print("Calculating multi-asset hedge weights...")
    multi_weights = {}
    for coin in coins[:50]:  # Limit to first 50 coins for speed
        if coin not in ['BTC', 'ETH']:
            weights = calculate_multi_hedge_weights(returns, coin, ['BTC', 'ETH'])
            if weights is not None:
                multi_weights[coin] = weights
    
    # Simulate trades for each strategy
    strategies = ['no_hedge', 'delta_btc', 'beta_btc', 'beta_eth', 'multi_hedge']
    all_results = []
    
    print("Simulating hedged trades...")
    for idx, row in extreme_events.iterrows():
        coin = row['coin']
        hour = row['hour']
        fr = row['funding_rate']
        pos_type = 'long' if row['type'] == 'negative' else 'short'
        
        for strategy in strategies:
            result = simulate_hedged_trade(
                coin, hour, fr, pos_type, returns,
                strategy, betas_btc, betas_eth, multi_weights
            )
            all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    
    # =============================================================================
    # ANALYSIS & OUTPUT
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("HEDGING STRATEGY COMPARISON FOR FUNDING RATE CAPTURE")
    print(f"Threshold: |FR| > {THRESHOLD*100:.2f}%")
    print("=" * 80)
    
    # Summary by strategy
    print("\n### STRATEGY PERFORMANCE SUMMARY ###\n")
    print(f"{'Strategy':<15} {'Valid Trades':<14} {'Avg Net PnL':<14} {'Std Dev':<12} {'Sharpe':<10} {'Win Rate':<10}")
    print("-" * 80)
    
    strategy_stats = {}
    for strategy in strategies:
        df = results_df[(results_df['strategy'] == strategy) & (results_df['valid'] == True)]
        if len(df) == 0:
            continue
        
        avg_pnl = df['net_pnl'].mean() * 100
        std_pnl = df['net_pnl'].std() * 100
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = (df['net_pnl'] > 0).mean() * 100
        
        strategy_stats[strategy] = {
            'count': len(df),
            'avg_pnl': avg_pnl,
            'std_pnl': std_pnl,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_pnl': df['net_pnl'].sum() * 100
        }
        
        print(f"{strategy:<15} {len(df):<14} {avg_pnl:>12.4f}%  {std_pnl:>10.4f}%  {sharpe:>8.4f}  {win_rate:>8.1f}%")
    
    # Detailed breakdown
    print("\n### DETAILED P&L BREAKDOWN ###\n")
    print(f"{'Strategy':<15} {'Funding PnL':<14} {'Price PnL':<14} {'Hedge PnL':<14} {'Net PnL':<14}")
    print("-" * 80)
    
    for strategy in strategies:
        df = results_df[(results_df['strategy'] == strategy) & (results_df['valid'] == True)]
        if len(df) == 0:
            continue
        
        funding = df['funding_pnl'].mean() * 100
        price = df['price_pnl'].mean() * 100
        hedge = df['hedge_pnl'].mean() * 100
        net = df['net_pnl'].mean() * 100
        
        print(f"{strategy:<15} {funding:>12.4f}%  {price:>12.4f}%  {hedge:>12.4f}%  {net:>12.4f}%")
    
    # Hedge effectiveness
    print("\n### HEDGE EFFECTIVENESS (Variance Reduction) ###\n")
    
    no_hedge_var = results_df[(results_df['strategy'] == 'no_hedge') & (results_df['valid'] == True)]['net_pnl'].var()
    
    for strategy in strategies:
        df = results_df[(results_df['strategy'] == strategy) & (results_df['valid'] == True)]
        if len(df) == 0:
            continue
        
        var = df['net_pnl'].var()
        var_reduction = (1 - var / no_hedge_var) * 100 if no_hedge_var > 0 else 0
        avg_hedge_ratio = df['hedge_ratio'].mean()
        
        print(f"{strategy:<15}: Variance = {var*10000:.4f} (bps²), Reduction = {var_reduction:>6.1f}%, Avg Hedge Ratio = {avg_hedge_ratio:.2f}")
    
    # Cost-adjusted analysis
    print("\n### COST-ADJUSTED NET RETURNS ###")
    print("(Assumes 0.05% cost per leg, 2 legs for main + 2 legs for hedge)\n")
    
    COST_PER_TRADE = 0.0005 * 4  # 0.05% per leg, 4 legs total (open/close main + hedge)
    
    print(f"{'Strategy':<15} {'Gross PnL':<14} {'Total Cost':<14} {'Net After Cost':<14} {'Profitable?':<12}")
    print("-" * 80)
    
    for strategy in strategies:
        if strategy not in strategy_stats:
            continue
        stats = strategy_stats[strategy]
        
        total_cost = stats['count'] * COST_PER_TRADE * 100
        net_after = stats['total_pnl'] - total_cost
        profitable = "✓ YES" if net_after > 0 else "✗ NO"
        
        print(f"{strategy:<15} {stats['total_pnl']:>12.2f}%  {total_cost:>12.2f}%  {net_after:>12.2f}%  {profitable:<12}")
    
    # Best strategy recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    # Find best strategy by Sharpe
    best_strategy = max(strategy_stats.keys(), key=lambda x: strategy_stats[x]['sharpe'])
    best_stats = strategy_stats[best_strategy]
    
    print(f"""
Best Strategy: {best_strategy}
- Sharpe Ratio: {best_stats['sharpe']:.4f}
- Average Net PnL: {best_stats['avg_pnl']:.4f}% per trade
- Win Rate: {best_stats['win_rate']:.1f}%
- Total Cumulative PnL: {best_stats['total_pnl']:.2f}%

Key Insights:
1. No hedge has highest raw funding capture but also highest variance
2. Delta hedge (1:1 BTC) is simple but may over/under-hedge
3. Beta hedge adjusts for coin's correlation with BTC
4. Multi-asset hedge (BTC+ETH) can capture more correlation
""")
    
    # Save results
    results_df.to_csv('hedge_strategy_results.csv', index=False)
    print("\nResults saved to hedge_strategy_results.csv")

if __name__ == "__main__":
    main()
