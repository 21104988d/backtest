"""
Funding-Neutral Trend Following Strategy Backtest (Optimized)

This strategy captures price alpha from extreme funding rate trend-following signals
while maintaining funding-neutral exposure by hedging with opposite positions on 
low funding rate coins.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PARAMETERS
# =============================================================================

ENTRY_THRESHOLD = 0.0001      # |FR| >= 0.01% for alpha (87.6% APY)
HEDGE_THRESHOLD = 0.000015    # |FR| < 0.0015% for hedge
POSITION_SIZE = 100           # $100 USD per position
TAKER_FEE = 0.00045           # 0.045% per trade

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and merge funding and price data."""
    print("Loading data...")
    
    base_path = Path(__file__).parent.parent
    
    # Load funding data
    funding = pd.read_csv(base_path / 'funding_history.csv')
    funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)
    funding['hour'] = funding['datetime'].dt.floor('h').dt.tz_localize(None)
    
    # Load price data
    price = pd.read_csv(base_path / 'price_history.csv')
    price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
    price['hour'] = price['timestamp'].dt.floor('h').dt.tz_localize(None)
    
    # Merge
    merged = pd.merge(
        funding[['hour', 'coin', 'funding_rate']],
        price[['hour', 'coin', 'price']],
        on=['hour', 'coin'],
        how='inner'
    ).sort_values(['coin', 'hour']).reset_index(drop=True)
    
    # Add next hour price for each coin
    merged['price_next'] = merged.groupby('coin')['price'].shift(-1)
    merged['hour_next'] = merged.groupby('coin')['hour'].shift(-1)
    
    # Filter rows where next hour is consecutive (1 hour later)
    merged = merged[merged['hour_next'] == merged['hour'] + pd.Timedelta(hours=1)]
    merged = merged.dropna(subset=['price_next'])
    
    print(f"  Records with valid next price: {len(merged):,}")
    print(f"  Unique hours: {merged['hour'].nunique():,}")
    print(f"  Unique coins: {merged['coin'].nunique():,}")
    print(f"  Date range: {merged['hour'].min()} to {merged['hour'].max()}")
    
    return merged

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run the funding-neutral trend following backtest.
    """
    print("\nRunning backtest...")
    
    # Pre-compute absolute funding rates
    data = data.copy()
    data['abs_fr'] = data['funding_rate'].abs()
    
    # Get sorted unique hours
    hours = sorted(data['hour'].unique())
    n_hours = len(hours)
    print(f"  Processing {n_hours:,} hours...")
    
    # Position tracking
    alpha_positions = {}  # {coin: {'direction': str, 'entry_hour_idx': int}}
    hedge_positions = {}
    
    # Results
    records = []
    cumulative_pnl = 0.0
    
    # Group by hour for efficient lookup
    hour_groups = {hour: group for hour, group in data.groupby('hour')}
    
    for h_idx, hour in enumerate(hours):
        if hour not in hour_groups:
            continue
            
        # Get data for this hour
        hour_df = hour_groups[hour]
        
        # Build lookup for this hour using vectorized operations
        coin_data = dict(zip(
            hour_df['coin'],
            [{'fr': fr, 'price': p, 'price_next': pn, 'abs_fr': afr} 
             for fr, p, pn, afr in zip(
                 hour_df['funding_rate'].values,
                 hour_df['price'].values, 
                 hour_df['price_next'].values,
                 hour_df['abs_fr'].values
             )]
        ))
        
        available_coins = set(coin_data.keys())
        
        if not available_coins:
            continue
        
        # ---------------------------------------------------------------------
        # STEP 2: Determine target alpha set
        # ---------------------------------------------------------------------
        candidate_alpha = {
            coin for coin in available_coins
            if coin_data[coin]['abs_fr'] >= ENTRY_THRESHOLD
        }
        
        # Sort by |FR| descending
        target_alpha = sorted(
            candidate_alpha,
            key=lambda c: coin_data[c]['abs_fr'],
            reverse=True
        )
        
        # ---------------------------------------------------------------------
        # STEP 3: Determine hedge pool
        # ---------------------------------------------------------------------
        hedge_pool = {
            coin for coin in available_coins
            if coin_data[coin]['abs_fr'] < HEDGE_THRESHOLD
        }
        
        # Sort by |FR| descending
        hedge_pool_sorted = sorted(
            hedge_pool,
            key=lambda c: coin_data[c]['abs_fr'],
            reverse=True
        )
        
        # ---------------------------------------------------------------------
        # STEP 4: Ensure funding neutral
        # ---------------------------------------------------------------------
        funding_to_pay = sum(
            POSITION_SIZE * coin_data[c]['abs_fr'] 
            for c in target_alpha
        )
        
        max_hedge_capacity = sum(
            POSITION_SIZE * coin_data[c]['abs_fr'] 
            for c in hedge_pool_sorted
        )
        
        # Demote weakest alphas if not enough hedge capacity
        target_alpha_set = set(target_alpha)
        
        while max_hedge_capacity < funding_to_pay and target_alpha_set:
            # Find weakest alpha (smallest |FR|, FIFO tie-breaker)
            weakest = None
            weakest_priority = None
            
            for coin in target_alpha_set:
                abs_fr = coin_data[coin]['abs_fr']
                
                if coin in alpha_positions:
                    entry_hour_idx = alpha_positions[coin]['entry_hour_idx']
                else:
                    entry_hour_idx = h_idx
                
                priority = (abs_fr, -entry_hour_idx)
                
                if weakest is None or priority < weakest_priority:
                    weakest = coin
                    weakest_priority = priority
            
            if weakest is None:
                break
            
            target_alpha_set.remove(weakest)
            funding_to_pay -= POSITION_SIZE * coin_data[weakest]['abs_fr']
            
            hedge_pool_sorted.append(weakest)
            hedge_pool_sorted.sort(key=lambda c: coin_data[c]['abs_fr'], reverse=True)
            max_hedge_capacity += POSITION_SIZE * coin_data[weakest]['abs_fr']
        
        target_alpha = list(target_alpha_set)
        
        # ---------------------------------------------------------------------
        # STEP 5: Select minimum hedge coins needed
        # ---------------------------------------------------------------------
        target_hedge = []
        accumulated_receive = 0.0
        
        funding_to_pay = sum(
            POSITION_SIZE * coin_data[c]['abs_fr'] 
            for c in target_alpha
        )
        
        for coin in hedge_pool_sorted:
            if accumulated_receive >= funding_to_pay:
                break
            target_hedge.append(coin)
            accumulated_receive += POSITION_SIZE * coin_data[coin]['abs_fr']
        
        target_hedge_set = set(target_hedge)
        
        # ---------------------------------------------------------------------
        # STEP 6: Close positions no longer needed
        # ---------------------------------------------------------------------
        hourly_fees = 0.0
        positions_closed = 0
        
        for coin in list(alpha_positions.keys()):
            if coin not in target_alpha_set:
                hourly_fees += POSITION_SIZE * TAKER_FEE
                positions_closed += 1
                del alpha_positions[coin]
        
        for coin in list(hedge_positions.keys()):
            if coin not in target_hedge_set:
                hourly_fees += POSITION_SIZE * TAKER_FEE
                positions_closed += 1
                del hedge_positions[coin]
        
        # ---------------------------------------------------------------------
        # STEP 7: Open new positions
        # ---------------------------------------------------------------------
        positions_opened = 0
        
        for coin in target_alpha:
            if coin not in alpha_positions:
                fr = coin_data[coin]['fr']
                direction = 'LONG' if fr > 0 else 'SHORT'
                
                alpha_positions[coin] = {
                    'direction': direction,
                    'entry_hour_idx': h_idx
                }
                
                hourly_fees += POSITION_SIZE * TAKER_FEE
                positions_opened += 1
        
        for coin in target_hedge:
            if coin not in hedge_positions:
                fr = coin_data[coin]['fr']
                direction = 'SHORT' if fr > 0 else 'LONG'
                
                hedge_positions[coin] = {
                    'direction': direction,
                    'entry_hour_idx': h_idx
                }
                
                hourly_fees += POSITION_SIZE * TAKER_FEE
                positions_opened += 1
        
        # ---------------------------------------------------------------------
        # STEP 8: Calculate hourly PnL
        # ---------------------------------------------------------------------
        
        # Alpha price PnL
        alpha_price_pnl = 0.0
        n_alpha_long = 0
        n_alpha_short = 0
        
        for coin, pos in alpha_positions.items():
            if coin not in coin_data:
                continue
            price_h = coin_data[coin]['price']
            price_h1 = coin_data[coin]['price_next']
            price_return = (price_h1 - price_h) / price_h
            
            if pos['direction'] == 'LONG':
                alpha_price_pnl += POSITION_SIZE * price_return
                n_alpha_long += 1
            else:
                alpha_price_pnl += POSITION_SIZE * (-price_return)
                n_alpha_short += 1
        
        # Hedge price PnL
        hedge_price_pnl = 0.0
        n_hedge_long = 0
        n_hedge_short = 0
        
        for coin, pos in hedge_positions.items():
            if coin not in coin_data:
                continue
            price_h = coin_data[coin]['price']
            price_h1 = coin_data[coin]['price_next']
            price_return = (price_h1 - price_h) / price_h
            
            if pos['direction'] == 'LONG':
                hedge_price_pnl += POSITION_SIZE * price_return
                n_hedge_long += 1
            else:
                hedge_price_pnl += POSITION_SIZE * (-price_return)
                n_hedge_short += 1
        
        # Funding PnL
        funding_paid = sum(
            POSITION_SIZE * coin_data[coin]['abs_fr']
            for coin in alpha_positions if coin in coin_data
        )
        
        funding_received = sum(
            POSITION_SIZE * coin_data[coin]['abs_fr']
            for coin in hedge_positions if coin in coin_data
        )
        
        net_funding = funding_received - funding_paid
        
        # Total hourly PnL
        hourly_pnl = alpha_price_pnl + hedge_price_pnl + net_funding - hourly_fees
        cumulative_pnl += hourly_pnl
        
        # ---------------------------------------------------------------------
        # STEP 9: Record
        # ---------------------------------------------------------------------
        n_alpha = len(alpha_positions)
        n_hedge = len(hedge_positions)
        n_total = n_alpha + n_hedge
        
        records.append({
            'hour_idx': h_idx,
            'hour': hour,
            'n_alpha': n_alpha,
            'n_alpha_long': n_alpha_long,
            'n_alpha_short': n_alpha_short,
            'n_hedge': n_hedge,
            'n_hedge_long': n_hedge_long,
            'n_hedge_short': n_hedge_short,
            'n_total': n_total,
            'funding_paid': funding_paid,
            'funding_received': funding_received,
            'net_funding': net_funding,
            'alpha_price_pnl': alpha_price_pnl,
            'hedge_price_pnl': hedge_price_pnl,
            'fees': hourly_fees,
            'positions_opened': positions_opened,
            'positions_closed': positions_closed,
            'hourly_pnl': hourly_pnl,
            'cumulative_pnl': cumulative_pnl,
        })
        
        # Progress indicator
        if (h_idx + 1) % 2000 == 0:
            print(f"    Processed {h_idx + 1:,} / {n_hours:,} hours... (PnL: ${cumulative_pnl:,.2f})")
    
    print(f"  Completed: {len(records):,} hourly records")
    
    return pd.DataFrame(records)

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_summary(records: pd.DataFrame):
    """Print comprehensive summary statistics."""
    
    print("\n" + "=" * 80)
    print("FUNDING-NEUTRAL TREND FOLLOWING BACKTEST RESULTS")
    print("=" * 80)
    
    # Parameters
    print("\nPARAMETERS:")
    print(f"  Entry Threshold: |FR| >= {ENTRY_THRESHOLD*100:.4f}% ({ENTRY_THRESHOLD*100*8760:.1f}% APY)")
    print(f"  Position Size: ${POSITION_SIZE} per coin")
    print(f"  Taker Fee: {TAKER_FEE*100:.3f}%")
    
    # Data period
    print("\nDATA PERIOD:")
    print(f"  Start: {records['hour'].min()}")
    print(f"  End: {records['hour'].max()}")
    print(f"  Total Hours: {len(records):,}")
    
    # -------------------------------------------------------------------------
    # Concurrent Position Statistics
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("CONCURRENT POSITION STATISTICS")
    print("-" * 80)
    
    print("\nAlpha Positions:")
    print(f"  Average: {records['n_alpha'].mean():.1f} per hour")
    print(f"  Min: {records['n_alpha'].min()}")
    print(f"  Max: {records['n_alpha'].max()}")
    print(f"  Avg Long: {records['n_alpha_long'].mean():.1f}")
    print(f"  Avg Short: {records['n_alpha_short'].mean():.1f}")
    
    print("\nHedge Positions:")
    print(f"  Average: {records['n_hedge'].mean():.1f} per hour")
    print(f"  Min: {records['n_hedge'].min()}")
    print(f"  Max: {records['n_hedge'].max()}")
    print(f"  Avg Long: {records['n_hedge_long'].mean():.1f}")
    print(f"  Avg Short: {records['n_hedge_short'].mean():.1f}")
    
    print("\nTotal Concurrent Positions:")
    print(f"  Average: {records['n_total'].mean():.1f} per hour")
    print(f"  Min: {records['n_total'].min()}")
    print(f"  Max: {records['n_total'].max()}")
    print(f"  P50 (Median): {records['n_total'].quantile(0.50):.0f}")
    print(f"  P75: {records['n_total'].quantile(0.75):.0f}")
    print(f"  P95: {records['n_total'].quantile(0.95):.0f}")
    print(f"  P99: {records['n_total'].quantile(0.99):.0f}")
    
    # -------------------------------------------------------------------------
    # Turnover Statistics
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TURNOVER STATISTICS")
    print("-" * 80)
    
    total_opened = records['positions_opened'].sum()
    total_closed = records['positions_closed'].sum()
    
    print(f"\nTotal Positions Opened: {total_opened:,}")
    print(f"Total Positions Closed: {total_closed:,}")
    print(f"Average Turnover per Hour: {(records['positions_opened'] + records['positions_closed']).mean():.2f}")
    
    if total_opened > 0:
        avg_positions = records['n_total'].mean()
        if total_closed > 0:
            avg_holding = (avg_positions * len(records)) / total_closed
            print(f"Estimated Avg Holding Time: {avg_holding:.1f} hours")
    
    # -------------------------------------------------------------------------
    # Funding Analysis
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("FUNDING ANALYSIS")
    print("-" * 80)
    
    total_paid = records['funding_paid'].sum()
    total_received = records['funding_received'].sum()
    total_net = records['net_funding'].sum()
    
    print(f"\nTotal Funding Paid (Alpha): ${total_paid:,.2f}")
    print(f"Total Funding Received (Hedge): ${total_received:,.2f}")
    print(f"Net Funding: ${total_net:,.2f}")
    
    if total_paid > 0:
        hedge_ratio = total_received / total_paid * 100
        print(f"Funding Hedge Ratio: {hedge_ratio:.1f}%")
    
    print(f"\nAvg Hourly Funding Paid: ${records['funding_paid'].mean():.4f}")
    print(f"Avg Hourly Funding Received: ${records['funding_received'].mean():.4f}")
    print(f"Avg Hourly Net Funding: ${records['net_funding'].mean():.4f}")
    
    # -------------------------------------------------------------------------
    # Price PnL Analysis
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PRICE PNL ANALYSIS")
    print("-" * 80)
    
    total_alpha_price = records['alpha_price_pnl'].sum()
    total_hedge_price = records['hedge_price_pnl'].sum()
    total_price = total_alpha_price + total_hedge_price
    
    print(f"\nAlpha Price PnL: ${total_alpha_price:,.2f}")
    print(f"Hedge Price PnL: ${total_hedge_price:,.2f}")
    print(f"Total Price PnL: ${total_price:,.2f}")
    
    print(f"\nAvg Hourly Alpha Price PnL: ${records['alpha_price_pnl'].mean():.4f}")
    print(f"Avg Hourly Hedge Price PnL: ${records['hedge_price_pnl'].mean():.4f}")
    
    # -------------------------------------------------------------------------
    # Fee Analysis
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("FEE ANALYSIS")
    print("-" * 80)
    
    total_fees = records['fees'].sum()
    print(f"\nTotal Fees Paid: ${total_fees:,.2f}")
    print(f"Avg Hourly Fees: ${records['fees'].mean():.4f}")
    
    gross_pnl = total_alpha_price + total_hedge_price + total_net
    if gross_pnl != 0:
        fee_pct = abs(total_fees / gross_pnl * 100)
        print(f"Fees as % of Gross PnL: {fee_pct:.1f}%")
    
    # -------------------------------------------------------------------------
    # Final Results
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("FINAL RESULTS")
    print("-" * 80)
    
    net_pnl = records['cumulative_pnl'].iloc[-1]
    gross_pnl = total_alpha_price + total_hedge_price + total_net
    
    print(f"\nGross PnL (before fees): ${gross_pnl:,.2f}")
    print(f"Net PnL (after fees): ${net_pnl:,.2f}")
    print(f"\nAvg Hourly PnL: ${records['hourly_pnl'].mean():.4f}")
    
    if records['hourly_pnl'].std() > 0:
        sharpe = records['hourly_pnl'].mean() / records['hourly_pnl'].std()
        sharpe_annual = sharpe * np.sqrt(8760)
        print(f"Sharpe Ratio (hourly): {sharpe:.4f}")
        print(f"Sharpe Ratio (annualized): {sharpe_annual:.2f}")
    
    cumulative = records['cumulative_pnl']
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()
    max_dd_pct = (max_dd / running_max.max() * 100) if running_max.max() > 0 else 0
    print(f"Max Drawdown: ${max_dd:,.2f} ({max_dd_pct:.1f}%)")
    
    # -------------------------------------------------------------------------
    # Breakdown Table
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("BREAKDOWN BY COMPONENT")
    print("-" * 80)
    
    print(f"\n{'Component':<20} | {'Total ($)':>14} | {'Per Hour ($)':>14} | {'% of Net':>10}")
    print("-" * 20 + "-+-" + "-" * 14 + "-+-" + "-" * 14 + "-+-" + "-" * 10)
    
    components = [
        ('Alpha Price PnL', total_alpha_price),
        ('Hedge Price PnL', total_hedge_price),
        ('Net Funding', total_net),
        ('Fees', -total_fees),
    ]
    
    for name, value in components:
        per_hour = value / len(records)
        pct = (value / net_pnl * 100) if net_pnl != 0 else 0
        print(f"{name:<20} | {value:>+14,.2f} | {per_hour:>+14.4f} | {pct:>+10.1f}%")
    
    print("-" * 20 + "-+-" + "-" * 14 + "-+-" + "-" * 14 + "-+-" + "-" * 10)
    per_hour_net = net_pnl / len(records)
    print(f"{'NET PNL':<20} | {net_pnl:>+14,.2f} | {per_hour_net:>+14.4f} | {'100.0':>10}%")
    
    print("\n" + "=" * 80)
    
    # -------------------------------------------------------------------------
    # Yearly Breakdown
    # -------------------------------------------------------------------------
    print("\nYEARLY BREAKDOWN:")
    print("-" * 80)
    
    records_copy = records.copy()
    records_copy['year'] = records_copy['hour'].dt.year
    yearly = records_copy.groupby('year').agg({
        'hourly_pnl': ['sum', 'mean', 'count'],
        'alpha_price_pnl': 'sum',
        'hedge_price_pnl': 'sum',
        'net_funding': 'sum',
        'fees': 'sum',
        'n_total': 'mean',
    })
    
    print(f"\n{'Year':<6} | {'Hours':>8} | {'Net PnL ($)':>14} | {'Avg/Hour':>12} | {'Avg Pos':>10}")
    print("-" * 6 + "-+-" + "-" * 8 + "-+-" + "-" * 14 + "-+-" + "-" * 12 + "-+-" + "-" * 10)
    
    for year in yearly.index:
        hours_count = yearly.loc[year, ('hourly_pnl', 'count')]
        total = yearly.loc[year, ('hourly_pnl', 'sum')]
        avg = yearly.loc[year, ('hourly_pnl', 'mean')]
        avg_pos = yearly.loc[year, ('n_total', 'mean')]
        print(f"{year:<6} | {hours_count:>8,.0f} | {total:>+14,.2f} | {avg:>+12.4f} | {avg_pos:>10.1f}")
    
    print("=" * 80)

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Load data
    data = load_data()
    
    # Run backtest
    records = run_backtest(data)
    
    # Print summary
    print_summary(records)
    
    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    records.to_csv(output_dir / 'hourly_records.csv', index=False)
    print(f"\nHourly records saved to: {output_dir / 'hourly_records.csv'}")

if __name__ == '__main__':
    main()
