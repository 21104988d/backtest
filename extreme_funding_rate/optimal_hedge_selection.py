"""
OPTIMAL HEDGE PAIR SELECTION

Strategy:
1. SINGLE BEST HEDGE:
   - For each extreme funding asset, find the highest correlated asset among ALL available
   - Ensure net funding rate is positive (we receive more than we pay)
   - Use beta calculation for hedge ratio
   - Evaluate hedged performance

2. MULTI-ASSET HEDGE (ITERATIVE):
   - Start with extreme funding asset
   - Find highest correlated single asset
   - If correlation < threshold (e.g., 0.7), add another asset
   - Repeat until correlation > threshold
   - Record net funding rate (no requirement for positive)
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING
# =============================================================================

print("Loading data...")
funding = pd.read_csv('funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)
funding['hour'] = funding['datetime'].dt.floor('h').dt.tz_localize(None)

price = pd.read_csv('price_history.csv')
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
price['hour'] = price['timestamp'].dt.floor('h').dt.tz_localize(None)

# Build price matrix and returns
print("Building price matrix and returns...")
price_pivot = price.pivot_table(index='hour', columns='coin', values='price', aggfunc='first')
returns = price_pivot.pct_change(fill_method=None)

# Get available price data coverage
price_hours = set(price_pivot.index)
price_coins = set(price_pivot.columns)

# Build funding rate lookup: (hour, coin) -> funding_rate
print("Building funding rate lookup...")
funding_lookup = {}
for _, row in funding.iterrows():
    key = (row['hour'], row['coin'])
    funding_lookup[key] = row['funding_rate']

def get_funding_rate(hour, coin):
    return funding_lookup.get((hour, coin), 0)

# =============================================================================
# GENERATE EXTREME EVENTS FROM RAW FUNDING DATA
# =============================================================================

THRESHOLD = 0.001  # 0.10% funding rate threshold

print(f"\nGenerating extreme funding events from raw data (|FR| > {THRESHOLD*100:.2f}%)...")

# Find extreme funding events from raw funding data
extreme_negative = funding[funding['funding_rate'] < -THRESHOLD].copy()
extreme_negative['type'] = 'negative'

extreme_positive = funding[funding['funding_rate'] > THRESHOLD].copy()
extreme_positive['type'] = 'positive'

all_extreme = pd.concat([extreme_negative, extreme_positive])

# Filter to events that have matching price data
events = all_extreme[
    (all_extreme['hour'].isin(price_hours)) & 
    (all_extreme['coin'].isin(price_coins))
].copy()

# Rename columns to match expected format
events = events.rename(columns={'funding_rate': 'funding_rate'})[['hour', 'coin', 'type', 'funding_rate']]

print(f"Raw extreme events: {len(all_extreme)}")
print(f"  - From funding data date range: {funding['hour'].min()} to {funding['hour'].max()}")
print(f"Events with price data: {len(events)}")
print(f"  - Price data date range: {min(price_hours)} to {max(price_hours)}")

# =============================================================================
# CORRELATION CALCULATION
# =============================================================================

def calculate_rolling_correlation(returns_df, target_coin, other_coin, hour, window=168):
    """
    Calculate rolling correlation between target and other coin
    Uses 168 hours (1 week) of data ending at 'hour'
    """
    if target_coin not in returns_df.columns or other_coin not in returns_df.columns:
        return np.nan
    
    # Get index position
    if hour not in returns_df.index:
        return np.nan
    
    idx = returns_df.index.get_loc(hour)
    if idx < window:
        return np.nan
    
    target_returns = returns_df[target_coin].iloc[idx-window:idx]
    other_returns = returns_df[other_coin].iloc[idx-window:idx]
    
    # Remove NaN
    valid = ~(target_returns.isna() | other_returns.isna())
    if valid.sum() < window // 2:
        return np.nan
    
    return target_returns[valid].corr(other_returns[valid])

def calculate_rolling_beta(returns_df, target_coin, hedge_coin, hour, window=168):
    """
    Calculate rolling beta: beta = cov(target, hedge) / var(hedge)
    """
    if target_coin not in returns_df.columns or hedge_coin not in returns_df.columns:
        return np.nan
    
    if hour not in returns_df.index:
        return np.nan
    
    idx = returns_df.index.get_loc(hour)
    if idx < window:
        return np.nan
    
    target_returns = returns_df[target_coin].iloc[idx-window:idx]
    hedge_returns = returns_df[hedge_coin].iloc[idx-window:idx]
    
    valid = ~(target_returns.isna() | hedge_returns.isna())
    if valid.sum() < window // 2:
        return np.nan
    
    cov = target_returns[valid].cov(hedge_returns[valid])
    var = hedge_returns[valid].var()
    
    if var == 0:
        return np.nan
    
    return cov / var

def find_best_single_hedge(returns_df, target_coin, hour, all_coins, 
                           require_positive_net_funding=True, target_funding=0):
    """
    Find the best single hedge asset for a target coin at a given hour.
    
    If require_positive_net_funding=True:
        - Only consider hedges where net funding (target - hedge) > 0
        - This ensures we receive funding after hedging
    
    Returns: (best_hedge_coin, correlation, beta, hedge_funding, net_funding)
    """
    correlations = []
    
    for coin in all_coins:
        if coin == target_coin:
            continue
        
        corr = calculate_rolling_correlation(returns_df, target_coin, coin, hour)
        if np.isnan(corr):
            continue
        
        hedge_funding = get_funding_rate(hour, coin)
        net_funding = abs(target_funding) - abs(hedge_funding)  # What we net receive
        
        # For negative target funding (we long to receive), we need hedge funding < target funding
        # For positive target funding (we short to receive), we need hedge funding > target funding
        # Simplified: we want abs(target) > abs(hedge) if same sign, or opposite signs
        
        if require_positive_net_funding:
            # Net funding check: after hedging, do we still receive funding?
            # If target_funding < 0 (we long), hedge should ideally have funding >= 0 or less negative
            # If target_funding > 0 (we short), hedge should ideally have funding <= 0 or less positive
            if target_funding < 0:  # We long the target
                # Hedge: we short the hedge coin, so we PAY if hedge_funding < 0
                # Net = receive from target + pay/receive from hedge
                # = |target_funding| + hedge_funding (since we short the hedge)
                actual_net = abs(target_funding) + hedge_funding
            else:  # We short the target
                # Hedge: we long the hedge coin, so we receive if hedge_funding < 0
                actual_net = target_funding - hedge_funding
            
            if actual_net <= 0:
                continue
        else:
            actual_net = net_funding
        
        correlations.append({
            'coin': coin,
            'correlation': corr,
            'hedge_funding': hedge_funding,
            'net_funding': actual_net
        })
    
    if not correlations:
        return None, np.nan, np.nan, np.nan, np.nan
    
    # Sort by correlation (highest first)
    correlations = sorted(correlations, key=lambda x: x['correlation'], reverse=True)
    best = correlations[0]
    
    # Calculate beta
    beta = calculate_rolling_beta(returns_df, target_coin, best['coin'], hour)
    
    return best['coin'], best['correlation'], beta, best['hedge_funding'], best['net_funding']

def find_multi_asset_hedge(returns_df, target_coin, hour, all_coins, 
                           correlation_threshold=0.7, max_assets=50):
    """
    Iteratively add hedge assets until correlation threshold is met.
    No cap on assets - will keep adding until threshold reached or no improvement.
    
    Returns: (list of hedge coins, list of weights, final correlation, net_funding, reached_threshold)
    """
    hedge_coins = []
    weights = []
    
    # Get target returns for the window
    window = 168
    if hour not in returns_df.index:
        return [], [], np.nan, np.nan, False
    
    idx = returns_df.index.get_loc(hour)
    if idx < window:
        return [], [], np.nan, np.nan, False
    
    target_returns = returns_df[target_coin].iloc[idx-window:idx].values
    valid_mask = ~np.isnan(target_returns)
    if valid_mask.sum() < window // 2:
        return [], [], np.nan, np.nan, False
    
    target_returns_clean = target_returns[valid_mask]
    
    # Track residual (what's not explained by hedges)
    residual = target_returns_clean.copy()
    
    available_coins = [c for c in all_coins if c != target_coin and c in returns_df.columns]
    
    reached_threshold = False
    prev_corr = 0
    
    for iteration in range(max_assets):
        best_coin = None
        best_corr = -1
        best_weight = 0
        
        for coin in available_coins:
            if coin in hedge_coins:
                continue
            
            coin_returns = returns_df[coin].iloc[idx-window:idx].values[valid_mask]
            if np.isnan(coin_returns).any():
                continue
            
            # Calculate correlation of residual with this coin
            corr = np.corrcoef(residual, coin_returns)[0, 1]
            if np.isnan(corr):
                continue
            
            if corr > best_corr:
                best_corr = corr
                best_coin = coin
                # Weight = cov(residual, coin) / var(coin)
                best_weight = np.cov(residual, coin_returns)[0, 1] / np.var(coin_returns)
        
        if best_coin is None:
            break
        
        # Check if adding this coin improves correlation meaningfully
        if best_corr < 0.05 and len(hedge_coins) > 0:
            # No more meaningful improvement possible
            break
        
        hedge_coins.append(best_coin)
        weights.append(best_weight)
        
        # Update residual
        coin_returns = returns_df[best_coin].iloc[idx-window:idx].values[valid_mask]
        residual = residual - best_weight * coin_returns
        
        # Calculate current correlation (1 - explained variance)
        hedge_returns = np.zeros_like(target_returns_clean)
        for i, (c, w) in enumerate(zip(hedge_coins, weights)):
            c_returns = returns_df[c].iloc[idx-window:idx].values[valid_mask]
            hedge_returns += w * c_returns
        
        current_corr = np.corrcoef(target_returns_clean, hedge_returns)[0, 1]
        
        # Check if we've reached threshold
        if current_corr >= correlation_threshold:
            reached_threshold = True
            break
        
        # Check if improvement is diminishing (less than 1% improvement)
        if len(hedge_coins) > 3 and (current_corr - prev_corr) < 0.01:
            break
        
        prev_corr = current_corr
    
    # Calculate net funding
    target_funding = get_funding_rate(hour, target_coin)
    hedge_funding_total = sum(abs(w) * get_funding_rate(hour, c) for c, w in zip(hedge_coins, weights))
    net_funding = abs(target_funding) - hedge_funding_total
    
    # Final correlation
    if hedge_coins:
        hedge_returns = np.zeros_like(target_returns_clean)
        for c, w in zip(hedge_coins, weights):
            c_returns = returns_df[c].iloc[idx-window:idx].values[valid_mask]
            hedge_returns += w * c_returns
        final_corr = np.corrcoef(target_returns_clean, hedge_returns)[0, 1]
    else:
        final_corr = np.nan
    
    return hedge_coins, weights, final_corr, net_funding, reached_threshold

# =============================================================================
# TRADING FEE ASSUMPTIONS
# =============================================================================

# Trading fees (per leg, per notional)
MAKER_FEE = 0.00015  # 0.015%
TAKER_FEE = 0.00045  # 0.045%

# For this analysis we assume taker fees for both entry and exit
# (market orders for quick execution at funding times)
# Round-trip fee per leg = 2 * TAKER_FEE

# =============================================================================
# SIMULATE HEDGED TRADE
# =============================================================================

def simulate_single_hedge_trade(target_coin, hedge_coin, beta, hour, position_type, 
                                target_funding, hedge_funding, returns_df):
    """
    Simulate a single-asset hedged trade
    
    Returns GROSS PnL (before fees) and trading fees separately
    """
    result = {
        'target_coin': target_coin,
        'hedge_coin': hedge_coin,
        'hour': hour,
        'position_type': position_type,
        'beta': beta,
        'target_funding': target_funding,
        'hedge_funding': hedge_funding,
        'funding_pnl': 0,
        'price_pnl': 0,
        'hedge_pnl': 0,
        'gross_pnl': 0,  # PnL before fees
        'trading_fee': 0,  # Total trading fees
        'net_pnl': 0,  # PnL after fees (for reference only)
        'valid': False
    }
    
    next_hour = hour + timedelta(hours=1)
    if hour not in returns_df.index or next_hour not in returns_df.index:
        return result
    
    target_ret = returns_df.loc[next_hour, target_coin] if target_coin in returns_df.columns else np.nan
    hedge_ret = returns_df.loc[next_hour, hedge_coin] if hedge_coin in returns_df.columns else np.nan
    
    if pd.isna(target_ret) or pd.isna(hedge_ret) or pd.isna(beta):
        return result
    
    direction = 1 if position_type == 'long' else -1
    
    # Funding P&L
    result['funding_pnl'] = abs(target_funding)
    
    # If we long target, we short hedge (and vice versa)
    # Hedge funding: if we short hedge and hedge_funding < 0, we pay
    # if we short hedge and hedge_funding > 0, we receive
    if direction == 1:  # Long target, short hedge
        result['funding_pnl'] += hedge_funding * beta  # Short position on hedge
    else:  # Short target, long hedge
        result['funding_pnl'] -= hedge_funding * beta  # Long position on hedge
    
    # Price P&L
    result['price_pnl'] = direction * target_ret
    result['hedge_pnl'] = -direction * beta * hedge_ret
    
    # Gross PnL (before fees)
    result['gross_pnl'] = result['funding_pnl'] + result['price_pnl'] + result['hedge_pnl']
    
    # Trading fees:
    # - Target position: 1 unit notional (entry + exit)
    # - Hedge position: |beta| units notional (entry + exit)
    # Round-trip fee = 2 * taker fee (we use taker for both entry and exit)
    target_fee = (2 * TAKER_FEE) * 1.0  # 1 unit for target
    hedge_fee = (2 * TAKER_FEE) * abs(beta)  # beta units for hedge
    result['trading_fee'] = target_fee + hedge_fee
    
    # Net PnL after fees (for reference)
    result['net_pnl'] = result['gross_pnl'] - result['trading_fee']
    result['valid'] = True
    
    return result

def simulate_multi_hedge_trade(target_coin, hedge_coins, weights, hour, position_type,
                               target_funding, returns_df):
    """
    Simulate a multi-asset hedged trade
    
    Returns GROSS PnL (before fees) and trading fees separately
    """
    result = {
        'target_coin': target_coin,
        'hedge_coins': ','.join(hedge_coins),
        'n_hedges': len(hedge_coins),
        'hour': hour,
        'position_type': position_type,
        'weights': ','.join([f'{w:.4f}' for w in weights]),
        'target_funding': target_funding,
        'hedge_funding_total': 0,
        'funding_pnl': 0,
        'price_pnl': 0,
        'hedge_pnl': 0,
        'gross_pnl': 0,  # PnL before fees
        'trading_fee': 0,  # Total trading fees
        'net_pnl': 0,  # PnL after fees (for reference only)
        'valid': False
    }
    
    next_hour = hour + timedelta(hours=1)
    if hour not in returns_df.index or next_hour not in returns_df.index:
        return result
    
    target_ret = returns_df.loc[next_hour, target_coin] if target_coin in returns_df.columns else np.nan
    if pd.isna(target_ret):
        return result
    
    direction = 1 if position_type == 'long' else -1
    
    # Funding P&L from target
    result['funding_pnl'] = abs(target_funding)
    
    # Price P&L from target
    result['price_pnl'] = direction * target_ret
    
    # Trading fee for target position
    result['trading_fee'] = (2 * TAKER_FEE) * 1.0  # 1 unit for target (entry + exit taker)
    
    # Hedge P&L and fees
    total_hedge_notional = 0
    for coin, weight in zip(hedge_coins, weights):
        hedge_ret = returns_df.loc[next_hour, coin] if coin in returns_df.columns else np.nan
        hedge_fr = get_funding_rate(hour, coin)
        
        if pd.isna(hedge_ret):
            return result
        
        result['hedge_pnl'] += -direction * weight * hedge_ret
        result['hedge_funding_total'] += abs(weight) * hedge_fr
        total_hedge_notional += abs(weight)
        
        # Funding from hedge position
        if direction == 1:  # Long target, short hedge
            result['funding_pnl'] += hedge_fr * weight
        else:
            result['funding_pnl'] -= hedge_fr * weight
    
    # Trading fee for hedge positions (taker for both entry and exit)
    result['trading_fee'] += (2 * TAKER_FEE) * total_hedge_notional
    
    # Gross PnL (before fees)
    result['gross_pnl'] = result['funding_pnl'] + result['price_pnl'] + result['hedge_pnl']
    
    # Net PnL after fees (for reference)
    result['net_pnl'] = result['gross_pnl'] - result['trading_fee']
    result['valid'] = True
    
    return result

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

CORRELATION_THRESHOLD = 0.7  # For multi-asset hedge

# Use events already filtered above (all extreme events with price data)
extreme_events = events.copy()

print(f"\nExtreme events for analysis: {len(extreme_events)}")

all_coins = list(returns.columns)
print(f"Total available coins for hedging: {len(all_coins)}")

# =============================================================================
# STRATEGY 1: SINGLE BEST HEDGE
# =============================================================================

print("\n" + "=" * 80)
print("STRATEGY 1: SINGLE BEST CORRELATED HEDGE (with positive net funding)")
print("=" * 80)

single_hedge_results = []

for i, (_, row) in enumerate(extreme_events.iterrows()):
    if i % 100 == 0:
        print(f"Processing {i}/{len(extreme_events)}...")
    
    coin = row['coin']
    hour = row['hour']
    fr = row['funding_rate']
    pos_type = 'long' if row['type'] == 'negative' else 'short'
    
    # Find best hedge
    best_hedge, corr, beta, hedge_fr, net_fr = find_best_single_hedge(
        returns, coin, hour, all_coins, 
        require_positive_net_funding=True, 
        target_funding=fr
    )
    
    if best_hedge is None:
        continue
    
    # Simulate trade
    result = simulate_single_hedge_trade(
        coin, best_hedge, beta, hour, pos_type, fr, hedge_fr, returns
    )
    result['correlation'] = corr
    result['net_funding'] = net_fr
    single_hedge_results.append(result)

single_df = pd.DataFrame(single_hedge_results)
valid_single = single_df[single_df['valid'] == True]

print(f"\nValid trades: {len(valid_single)} / {len(extreme_events)}")

if len(valid_single) > 0:
    print("\n### PERFORMANCE SUMMARY (GROSS - before fees) ###")
    print(f"Average correlation: {valid_single['correlation'].mean():.4f}")
    print(f"Average beta: {valid_single['beta'].mean():.4f}")
    print(f"Average net funding: {valid_single['net_funding'].mean()*100:.4f}%")
    
    print(f"\n{'Metric':<25} {'Value':<15}")
    print("-" * 45)
    print(f"{'Avg Gross PnL':<25} {valid_single['gross_pnl'].mean()*100:.4f}%")
    print(f"{'Std Dev':<25} {valid_single['gross_pnl'].std()*100:.4f}%")
    print(f"{'Sharpe (gross)':<25} {valid_single['gross_pnl'].mean()/valid_single['gross_pnl'].std():.4f}")
    print(f"{'Win Rate (gross)':<25} {(valid_single['gross_pnl'] > 0).mean()*100:.1f}%")
    print(f"{'Total Gross PnL':<25} {valid_single['gross_pnl'].sum()*100:.2f}%")
    
    print(f"\n### TRADING FEES ###")
    print(f"Fee assumptions: Taker={TAKER_FEE*100:.3f}% x2 (entry+exit) per leg")
    print(f"Avg Trading Fee per trade: {valid_single['trading_fee'].mean()*100:.4f}%")
    print(f"Total Trading Fees: {valid_single['trading_fee'].sum()*100:.2f}%")
    print(f"Avg Net PnL (after fees): {valid_single['net_pnl'].mean()*100:.4f}%")
    
    print("\n### P&L BREAKDOWN ###")
    print(f"Avg Funding PnL: {valid_single['funding_pnl'].mean()*100:.4f}%")
    print(f"Avg Price PnL: {valid_single['price_pnl'].mean()*100:.4f}%")
    print(f"Avg Hedge PnL: {valid_single['hedge_pnl'].mean()*100:.4f}%")
    
    # Top hedge pairs
    print("\n### TOP HEDGE PAIRS (by frequency) ###")
    pair_counts = valid_single.groupby(['target_coin', 'hedge_coin']).size().sort_values(ascending=False)
    print(pair_counts.head(20).to_string())

# =============================================================================
# STRATEGY 1B: SINGLE BEST HEDGE (NO NET FUNDING REQUIREMENT)
# =============================================================================

print("\n" + "=" * 80)
print("STRATEGY 1B: SINGLE BEST CORRELATED HEDGE (no funding requirement)")
print("=" * 80)

single_hedge_results_no_req = []

for i, (_, row) in enumerate(extreme_events.iterrows()):
    if i % 100 == 0:
        print(f"Processing {i}/{len(extreme_events)}...")
    
    coin = row['coin']
    hour = row['hour']
    fr = row['funding_rate']
    pos_type = 'long' if row['type'] == 'negative' else 'short'
    
    # Find best hedge without net funding requirement
    best_hedge, corr, beta, hedge_fr, net_fr = find_best_single_hedge(
        returns, coin, hour, all_coins, 
        require_positive_net_funding=False, 
        target_funding=fr
    )
    
    if best_hedge is None:
        continue
    
    result = simulate_single_hedge_trade(
        coin, best_hedge, beta, hour, pos_type, fr, hedge_fr, returns
    )
    result['correlation'] = corr
    result['net_funding'] = net_fr
    single_hedge_results_no_req.append(result)

single_df_no_req = pd.DataFrame(single_hedge_results_no_req)
valid_single_no_req = single_df_no_req[single_df_no_req['valid'] == True]

print(f"\nValid trades: {len(valid_single_no_req)} / {len(extreme_events)}")

if len(valid_single_no_req) > 0:
    print("\n### PERFORMANCE SUMMARY (GROSS - before fees) ###")
    print(f"Average correlation: {valid_single_no_req['correlation'].mean():.4f}")
    print(f"Average beta: {valid_single_no_req['beta'].mean():.4f}")
    
    print(f"\n{'Metric':<25} {'Value':<15}")
    print("-" * 45)
    print(f"{'Avg Gross PnL':<25} {valid_single_no_req['gross_pnl'].mean()*100:.4f}%")
    print(f"{'Std Dev':<25} {valid_single_no_req['gross_pnl'].std()*100:.4f}%")
    print(f"{'Sharpe (gross)':<25} {valid_single_no_req['gross_pnl'].mean()/valid_single_no_req['gross_pnl'].std():.4f}")
    print(f"{'Win Rate (gross)':<25} {(valid_single_no_req['gross_pnl'] > 0).mean()*100:.1f}%")
    print(f"{'Total Gross PnL':<25} {valid_single_no_req['gross_pnl'].sum()*100:.2f}%")
    
    print(f"\n### TRADING FEES ###")
    print(f"Fee assumptions: Taker={TAKER_FEE*100:.3f}% x2 (entry+exit) per leg")
    print(f"Avg Trading Fee per trade: {valid_single_no_req['trading_fee'].mean()*100:.4f}%")
    print(f"Total Trading Fees: {valid_single_no_req['trading_fee'].sum()*100:.2f}%")
    print(f"Avg Net PnL (after fees): {valid_single_no_req['net_pnl'].mean()*100:.4f}%")

# =============================================================================
# STRATEGY 2: MULTI-ASSET HEDGE
# =============================================================================

print("\n" + "=" * 80)
print(f"STRATEGY 2: MULTI-ASSET HEDGE (until correlation > {CORRELATION_THRESHOLD})")
print("=" * 80)

multi_hedge_results = []

for i, (_, row) in enumerate(extreme_events.iterrows()):
    if i % 100 == 0:
        print(f"Processing {i}/{len(extreme_events)}...")
    
    coin = row['coin']
    hour = row['hour']
    fr = row['funding_rate']
    pos_type = 'long' if row['type'] == 'negative' else 'short'
    
    # Find multi-asset hedge (no cap on assets)
    hedge_coins, weights, final_corr, net_fr, reached_threshold = find_multi_asset_hedge(
        returns, coin, hour, all_coins,
        correlation_threshold=CORRELATION_THRESHOLD,
        max_assets=50  # Effectively no cap
    )
    
    if not hedge_coins:
        continue
    
    result = simulate_multi_hedge_trade(
        coin, hedge_coins, weights, hour, pos_type, fr, returns
    )
    result['correlation'] = final_corr
    result['net_funding'] = net_fr
    result['reached_threshold'] = reached_threshold
    multi_hedge_results.append(result)

multi_df = pd.DataFrame(multi_hedge_results)
valid_multi = multi_df[multi_df['valid'] == True]

print(f"\nValid trades: {len(valid_multi)} / {len(extreme_events)}")

if len(valid_multi) > 0:
    # Count how many reached the correlation threshold
    reached_count = valid_multi['reached_threshold'].sum()
    not_reached_count = len(valid_multi) - reached_count
    
    print(f"\n### CORRELATION THRESHOLD ANALYSIS ###")
    print(f"Trades that REACHED corr >= {CORRELATION_THRESHOLD}: {reached_count} ({reached_count/len(valid_multi)*100:.1f}%)")
    print(f"Trades that DID NOT reach threshold: {not_reached_count} ({not_reached_count/len(valid_multi)*100:.1f}%)")
    
    print("\n### PERFORMANCE SUMMARY (GROSS - before fees) ###")
    print(f"Average correlation: {valid_multi['correlation'].mean():.4f}")
    print(f"Average # hedges: {valid_multi['n_hedges'].mean():.2f}")
    print(f"Max # hedges used: {valid_multi['n_hedges'].max()}")
    print(f"Average net funding: {valid_multi['net_funding'].mean()*100:.4f}%")
    
    print(f"\n{'Metric':<25} {'Value':<15}")
    print("-" * 45)
    print(f"{'Avg Gross PnL':<25} {valid_multi['gross_pnl'].mean()*100:.4f}%")
    print(f"{'Std Dev':<25} {valid_multi['gross_pnl'].std()*100:.4f}%")
    print(f"{'Sharpe (gross)':<25} {valid_multi['gross_pnl'].mean()/valid_multi['gross_pnl'].std():.4f}")
    print(f"{'Win Rate (gross)':<25} {(valid_multi['gross_pnl'] > 0).mean()*100:.1f}%")
    print(f"{'Total Gross PnL':<25} {valid_multi['gross_pnl'].sum()*100:.2f}%")
    
    print(f"\n### TRADING FEES ###")
    print(f"Fee assumptions: Taker={TAKER_FEE*100:.3f}% x2 (entry+exit) per leg")
    print(f"Avg Trading Fee per trade: {valid_multi['trading_fee'].mean()*100:.4f}%")
    print(f"Total Trading Fees: {valid_multi['trading_fee'].sum()*100:.2f}%")
    print(f"Avg Net PnL (after fees): {valid_multi['net_pnl'].mean()*100:.4f}%")
    
    print("\n### P&L BREAKDOWN ###")
    print(f"Avg Funding PnL: {valid_multi['funding_pnl'].mean()*100:.4f}%")
    print(f"Avg Price PnL: {valid_multi['price_pnl'].mean()*100:.4f}%")
    print(f"Avg Hedge PnL: {valid_multi['hedge_pnl'].mean()*100:.4f}%")
    
    # Distribution of number of hedge assets
    print("\n### HEDGE ASSET COUNT DISTRIBUTION ###")
    print(valid_multi['n_hedges'].value_counts().sort_index().to_string())

# =============================================================================
# COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("STRATEGY COMPARISON (GROSS PnL - before fees)")
print("=" * 80)

print(f"\n{'Strategy':<35} {'N':<8} {'Avg Gross PnL':<14} {'Sharpe':<10} {'Win Rate':<10} {'Avg Corr':<10}")
print("-" * 95)

if len(valid_single) > 0:
    sharpe = valid_single['gross_pnl'].mean()/valid_single['gross_pnl'].std()
    print(f"{'Single Best (+ net funding req)':<35} {len(valid_single):<8} {valid_single['gross_pnl'].mean()*100:>12.4f}%  {sharpe:>8.4f}  {(valid_single['gross_pnl']>0).mean()*100:>8.1f}%  {valid_single['correlation'].mean():>8.4f}")

if len(valid_single_no_req) > 0:
    sharpe = valid_single_no_req['gross_pnl'].mean()/valid_single_no_req['gross_pnl'].std()
    print(f"{'Single Best (no requirement)':<35} {len(valid_single_no_req):<8} {valid_single_no_req['gross_pnl'].mean()*100:>12.4f}%  {sharpe:>8.4f}  {(valid_single_no_req['gross_pnl']>0).mean()*100:>8.1f}%  {valid_single_no_req['correlation'].mean():>8.4f}")

if len(valid_multi) > 0:
    sharpe = valid_multi['gross_pnl'].mean()/valid_multi['gross_pnl'].std()
    print(f"{'Multi-Asset (corr > 0.7)':<35} {len(valid_multi):<8} {valid_multi['gross_pnl'].mean()*100:>12.4f}%  {sharpe:>8.4f}  {(valid_multi['gross_pnl']>0).mean()*100:>8.1f}%  {valid_multi['correlation'].mean():>8.4f}")

# Trading fee summary
print(f"\n### TRADING FEE SUMMARY ###")
print(f"Fee assumptions: Taker={TAKER_FEE*100:.3f}% x2 (entry+exit) per leg")
print(f"\n{'Strategy':<35} {'Avg Fee/Trade':<15} {'Total Fees':<15}")
print("-" * 70)
if len(valid_single) > 0:
    print(f"{'Single Best (+ net funding req)':<35} {valid_single['trading_fee'].mean()*100:>12.4f}%  {valid_single['trading_fee'].sum()*100:>12.2f}%")
if len(valid_single_no_req) > 0:
    print(f"{'Single Best (no requirement)':<35} {valid_single_no_req['trading_fee'].mean()*100:>12.4f}%  {valid_single_no_req['trading_fee'].sum()*100:>12.2f}%")
if len(valid_multi) > 0:
    print(f"{'Multi-Asset (corr > 0.7)':<35} {valid_multi['trading_fee'].mean()*100:>12.4f}%  {valid_multi['trading_fee'].sum()*100:>12.2f}%")

# =============================================================================
# ANALYSIS: WHY CAN'T WE REACH 0.7 CORRELATION?
# =============================================================================

if len(valid_multi) > 0:
    print("\n" + "=" * 80)
    print("WHY CAN'T WE REACH 0.7 CORRELATION?")
    print("=" * 80)
    
    # Compare trades that reached vs didn't reach threshold
    reached = valid_multi[valid_multi['reached_threshold'] == True]
    not_reached = valid_multi[valid_multi['reached_threshold'] == False]
    
    if len(reached) > 0 and len(not_reached) > 0:
        print(f"\n{'Metric':<25} {'Reached Threshold':<20} {'Did NOT Reach':<20}")
        print("-" * 70)
        print(f"{'Count':<25} {len(reached):<20} {len(not_reached):<20}")
        print(f"{'Avg Correlation':<25} {reached['correlation'].mean():<20.4f} {not_reached['correlation'].mean():<20.4f}")
        print(f"{'Avg # Hedges':<25} {reached['n_hedges'].mean():<20.2f} {not_reached['n_hedges'].mean():<20.2f}")
        print(f"{'Avg Gross PnL':<25} {reached['gross_pnl'].mean()*100:<19.4f}% {not_reached['gross_pnl'].mean()*100:<19.4f}%")
        print(f"{'Win Rate (gross)':<25} {(reached['gross_pnl']>0).mean()*100:<19.1f}% {(not_reached['gross_pnl']>0).mean()*100:<19.1f}%")
        print(f"{'Avg Trading Fee':<25} {reached['trading_fee'].mean()*100:<19.4f}% {not_reached['trading_fee'].mean()*100:<19.4f}%")
        
        print(f"\n** Performance of trades that REACHED threshold (corr >= {CORRELATION_THRESHOLD}): **")
        if len(reached) > 0:
            print(f"   N = {len(reached)}")
            print(f"   Avg Gross PnL = {reached['gross_pnl'].mean()*100:.4f}%")
            print(f"   Sharpe (gross) = {reached['gross_pnl'].mean()/reached['gross_pnl'].std():.4f}")
            print(f"   Win Rate = {(reached['gross_pnl']>0).mean()*100:.1f}%")
            print(f"   Avg Trading Fee = {reached['trading_fee'].mean()*100:.4f}%")
    
    print(f"\n** Analysis: **")
    print(f"The coins with extreme funding rates are often:")
    print(f"1. Small-cap/meme coins with unique price dynamics")
    print(f"2. Coins experiencing idiosyncratic events (listings, news, etc.)")
    print(f"3. These coins don't correlate well with ANY other asset")
    print(f"4. Maximum achievable correlation is limited by coin's idiosyncratic risk")

# =============================================================================
# SAVE RESULTS
# =============================================================================

if len(valid_single) > 0:
    valid_single.to_csv('optimal_single_hedge_results.csv', index=False)
    print("\nSaved: optimal_single_hedge_results.csv")

if len(valid_multi) > 0:
    valid_multi.to_csv('optimal_multi_hedge_results.csv', index=False)
    print("Saved: optimal_multi_hedge_results.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
