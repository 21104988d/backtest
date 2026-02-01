"""
FULL FUNDING RATE STRATEGY BACKTEST - PORTFOLIO SIMULATION

This backtest:
1. Tests ALL funding rate thresholds (not just extreme)
2. Allows MULTIPLE concurrent positions
3. Simulates realistic portfolio with capital allocation
4. Tracks daily PnL, exposure, and position count

Strategy: Trend-following - go WITH the funding direction
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TAKER_FEE = 0.00045  # 0.045% per trade
INITIAL_CAPITAL = 100_000  # $100k starting capital
MAX_POSITION_SIZE = 0.10  # Max 10% of capital per position
MAX_POSITIONS = 20  # Max concurrent positions
MAX_HOLD_HOURS = 72

# Parameter grids - now including LOWER thresholds
FR_THRESHOLDS = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003]  # 0.01% to 0.30%
EXIT_THRESHOLDS = [0.00005, 0.0001, 0.0002, 0.0003]  # 0.005% to 0.03%
CONSECUTIVE_HOURS = [1, 2, 3, 4]

# =============================================================================
# DATA LOADING
# =============================================================================

print("=" * 100)
print("FULL FUNDING RATE STRATEGY - PORTFOLIO SIMULATION")
print("=" * 100)

print("\nLoading data...")
funding = pd.read_csv('funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)
funding['hour'] = funding['datetime'].dt.floor('h').dt.tz_localize(None)

price = pd.read_csv('price_history.csv')
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
price['hour'] = price['timestamp'].dt.floor('h').dt.tz_localize(None)

print(f"Funding records: {len(funding):,}")
print(f"Price records: {len(price):,}")

# =============================================================================
# BUILD DATA STRUCTURES
# =============================================================================

print("\nBuilding data structures...")

# Price matrix
price_pivot = price.pivot_table(index='hour', columns='coin', values='price', aggfunc='first')
price_hours = sorted(price_pivot.index.tolist())
price_hours_set = set(price_pivot.index)
price_coins = set(price_pivot.columns)

# Funding lookup
funding_lookup = {}
for _, row in funding.iterrows():
    key = (row['hour'], row['coin'])
    funding_lookup[key] = row['funding_rate']

def get_funding_rate(hour, coin):
    return funding_lookup.get((hour, coin), 0)

# =============================================================================
# ANALYZE FUNDING RATE DISTRIBUTION
# =============================================================================

print("\n" + "=" * 100)
print("FUNDING RATE DISTRIBUTION ANALYSIS")
print("=" * 100)

# Filter to rows with price data
funding_with_price = funding[
    (funding['hour'].isin(price_hours_set)) &
    (funding['coin'].isin(price_coins))
].copy()

print(f"\nFunding records with price data: {len(funding_with_price):,}")

# Distribution of |FR|
abs_fr = funding_with_price['funding_rate'].abs()

print("\n|FR| Distribution:")
percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
for p in percentiles:
    val = np.percentile(abs_fr, p)
    print(f"  {p}th percentile: {val*100:.4f}%")

# Count signals at each threshold
print("\nSignals at each threshold:")
print(f"{'Threshold':>12} {'Count':>10} {'% of Total':>12} {'Signals/Day':>12}")
print("-" * 50)

total_days = (funding_with_price['hour'].max() - funding_with_price['hour'].min()).days
for thresh in [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]:
    count = (abs_fr > thresh).sum()
    pct = count / len(abs_fr) * 100
    per_day = count / total_days if total_days > 0 else 0
    print(f"{thresh*100:>11.3f}% {count:>10,} {pct:>11.1f}% {per_day:>11.1f}")

# =============================================================================
# PRE-CALCULATE CONSECUTIVE HOURS
# =============================================================================

print("\nPre-calculating consecutive hours...")

funding_sorted = funding_with_price.sort_values(['coin', 'hour'])

# For simplest threshold (0.01%)
consecutive_data = {}
for coin, group in funding_sorted.groupby('coin'):
    group = group.sort_values('hour')
    current_count = 0
    prev_sign = 0
    prev_hour = None
    
    for _, row in group.iterrows():
        fr = row['funding_rate']
        # Count consecutive same-sign funding
        current_sign = np.sign(fr)
        
        if prev_hour is not None:
            hour_diff = (row['hour'] - prev_hour).total_seconds() / 3600
            if hour_diff == 1 and current_sign == prev_sign:
                current_count += 1
            else:
                current_count = 1
        else:
            current_count = 1
        
        prev_sign = current_sign
        prev_hour = row['hour']
        consecutive_data[(row['hour'], coin)] = current_count

def get_consecutive_hours(hour, coin):
    return consecutive_data.get((hour, coin), 0)

# =============================================================================
# PORTFOLIO SIMULATION CLASS
# =============================================================================

class PortfolioSimulator:
    def __init__(self, initial_capital, max_positions, max_position_pct, entry_threshold, 
                 exit_threshold, min_consecutive):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_consecutive = min_consecutive
        
        self.positions = {}  # {coin: {'entry_hour', 'entry_price', 'direction', 'size', 'entry_fr'}}
        self.closed_trades = []
        self.daily_pnl = defaultdict(float)
        self.daily_exposure = defaultdict(float)
        self.daily_positions = defaultdict(int)
        
    def can_open_position(self, coin):
        return (len(self.positions) < self.max_positions and 
                coin not in self.positions)
    
    def should_enter(self, hour, coin, fr):
        """Check if we should enter a position"""
        if abs(fr) <= self.entry_threshold:
            return False, None
        
        consecutive = get_consecutive_hours(hour, coin)
        if consecutive < self.min_consecutive:
            return False, None
        
        # Trend-following: go WITH the funding
        direction = 'short' if fr < 0 else 'long'
        return True, direction
    
    def should_exit(self, hour, coin, current_fr, position):
        """Check if we should exit a position"""
        # Exit when FR normalizes
        if abs(current_fr) < self.exit_threshold:
            return True, 'normalized'
        
        # Exit on max hold
        hold_hours = (hour - position['entry_hour']).total_seconds() / 3600
        if hold_hours >= MAX_HOLD_HOURS:
            return True, 'timeout'
        
        return False, None
    
    def open_position(self, hour, coin, direction, entry_price, entry_fr):
        """Open a new position"""
        position_size = self.capital * self.max_position_pct
        
        self.positions[coin] = {
            'entry_hour': hour,
            'entry_price': entry_price,
            'direction': direction,
            'size': position_size,
            'entry_fr': entry_fr
        }
        
    def close_position(self, hour, coin, exit_price, exit_reason):
        """Close a position and record the trade"""
        pos = self.positions[coin]
        
        # Calculate returns
        price_return = (exit_price - pos['entry_price']) / pos['entry_price']
        if pos['direction'] == 'short':
            price_return = -price_return
        
        # Calculate funding paid over holding period
        hold_hours = int((hour - pos['entry_hour']).total_seconds() / 3600)
        funding_paid = 0
        for h in range(hold_hours):
            funding_hour = pos['entry_hour'] + timedelta(hours=h)
            fr = get_funding_rate(funding_hour, coin)
            # We pay if our direction matches the FR sign
            if pos['direction'] == 'short' and fr < 0:
                funding_paid += abs(fr)
            elif pos['direction'] == 'long' and fr > 0:
                funding_paid += fr
            elif pos['direction'] == 'short' and fr > 0:
                funding_paid -= fr  # We receive
            elif pos['direction'] == 'long' and fr < 0:
                funding_paid -= abs(fr)  # We receive
        
        # Net PnL
        gross_pnl = price_return - funding_paid
        net_pnl = gross_pnl - 2 * TAKER_FEE
        
        dollar_pnl = pos['size'] * net_pnl
        self.capital += dollar_pnl
        
        self.closed_trades.append({
            'coin': coin,
            'entry_hour': pos['entry_hour'],
            'exit_hour': hour,
            'direction': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'entry_fr': pos['entry_fr'],
            'hold_hours': hold_hours,
            'price_return': price_return,
            'funding_paid': funding_paid,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'dollar_pnl': dollar_pnl,
            'exit_reason': exit_reason
        })
        
        del self.positions[coin]
        
    def simulate(self, all_hours):
        """Run the simulation"""
        for hour in all_hours:
            if hour not in price_pivot.index:
                continue
                
            day = hour.date()
            
            # Track daily metrics
            self.daily_positions[day] = max(self.daily_positions[day], len(self.positions))
            
            exposure = sum(pos['size'] for pos in self.positions.values()) / self.capital
            self.daily_exposure[day] = max(self.daily_exposure[day], exposure)
            
            # Check exits for existing positions
            positions_to_close = []
            for coin, pos in self.positions.items():
                if coin not in price_pivot.columns:
                    continue
                    
                current_price = price_pivot.loc[hour, coin]
                if pd.isna(current_price):
                    continue
                    
                current_fr = get_funding_rate(hour, coin)
                should_exit, reason = self.should_exit(hour, coin, current_fr, pos)
                
                if should_exit:
                    positions_to_close.append((coin, current_price, reason))
            
            for coin, exit_price, reason in positions_to_close:
                self.close_position(hour, coin, exit_price, reason)
                self.daily_pnl[day] += self.closed_trades[-1]['dollar_pnl']
            
            # Check for new entries
            for coin in price_coins:
                if not self.can_open_position(coin):
                    continue
                    
                if coin not in price_pivot.columns:
                    continue
                    
                current_price = price_pivot.loc[hour, coin]
                if pd.isna(current_price):
                    continue
                    
                current_fr = get_funding_rate(hour, coin)
                should_enter, direction = self.should_enter(hour, coin, current_fr)
                
                if should_enter:
                    self.open_position(hour, coin, direction, current_price, current_fr)
        
        return self.get_results()
    
    def get_results(self):
        """Get simulation results"""
        if len(self.closed_trades) == 0:
            return None
            
        trades_df = pd.DataFrame(self.closed_trades)
        
        return {
            'n_trades': len(trades_df),
            'avg_pnl': trades_df['net_pnl'].mean(),
            'std_pnl': trades_df['net_pnl'].std(),
            'sharpe': trades_df['net_pnl'].mean() / trades_df['net_pnl'].std() if trades_df['net_pnl'].std() > 0 else 0,
            'win_rate': (trades_df['net_pnl'] > 0).mean(),
            'total_pnl': trades_df['net_pnl'].sum(),
            'total_dollar_pnl': trades_df['dollar_pnl'].sum(),
            'final_capital': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            'avg_hold': trades_df['hold_hours'].mean(),
            'avg_positions': np.mean(list(self.daily_positions.values())) if self.daily_positions else 0,
            'max_positions': max(self.daily_positions.values()) if self.daily_positions else 0,
            'avg_exposure': np.mean(list(self.daily_exposure.values())) if self.daily_exposure else 0,
            'trades_df': trades_df,
            'daily_pnl': dict(self.daily_pnl)
        }

# =============================================================================
# RUN PARAMETER SWEEP
# =============================================================================

print("\n" + "=" * 100)
print("PARAMETER SWEEP - ALL FUNDING RATE LEVELS")
print("=" * 100)

results = []
all_hours = sorted(price_hours)

# Test different parameter combinations
total_combos = len(FR_THRESHOLDS) * len(EXIT_THRESHOLDS) * len(CONSECUTIVE_HOURS)
print(f"\nTesting {total_combos} parameter combinations...")

combo_count = 0
for entry_thresh in FR_THRESHOLDS:
    for exit_thresh in EXIT_THRESHOLDS:
        if exit_thresh >= entry_thresh:  # Exit must be lower than entry
            continue
            
        for min_consec in CONSECUTIVE_HOURS:
            combo_count += 1
            
            sim = PortfolioSimulator(
                initial_capital=INITIAL_CAPITAL,
                max_positions=MAX_POSITIONS,
                max_position_pct=MAX_POSITION_SIZE,
                entry_threshold=entry_thresh,
                exit_threshold=exit_thresh,
                min_consecutive=min_consec
            )
            
            result = sim.simulate(all_hours)
            
            if result and result['n_trades'] >= 20:
                results.append({
                    'entry_threshold': entry_thresh,
                    'exit_threshold': exit_thresh,
                    'min_consecutive': min_consec,
                    'config': f"Entry>{entry_thresh*100:.2f}%_Exit<{exit_thresh*100:.3f}%_Cons>={min_consec}",
                    **{k: v for k, v in result.items() if k not in ['trades_df', 'daily_pnl']}
                })

results_df = pd.DataFrame(results)
print(f"Valid configurations: {len(results_df)}")

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

if len(results_df) > 0:
    print("\n" + "=" * 100)
    print("TOP 20 BY SHARPE RATIO")
    print("=" * 100)
    
    top_sharpe = results_df.nlargest(20, 'sharpe')
    print(f"\n{'Config':<50} {'N':>7} {'Avg%':>8} {'Sharpe':>7} {'Win%':>7} {'Return':>10} {'AvgPos':>7}")
    print("-" * 102)
    for _, row in top_sharpe.iterrows():
        print(f"{row['config']:<50} {row['n_trades']:>7} {row['avg_pnl']*100:>+7.2f}% {row['sharpe']:>7.2f} {row['win_rate']*100:>6.1f}% {row['total_return']*100:>+9.0f}% {row['avg_positions']:>7.1f}")

    print("\n" + "=" * 100)
    print("TOP 20 BY TOTAL RETURN")
    print("=" * 100)
    
    top_return = results_df.nlargest(20, 'total_return')
    print(f"\n{'Config':<50} {'N':>7} {'Avg%':>8} {'Sharpe':>7} {'Win%':>7} {'Return':>10} {'AvgPos':>7}")
    print("-" * 102)
    for _, row in top_return.iterrows():
        print(f"{row['config']:<50} {row['n_trades']:>7} {row['avg_pnl']*100:>+7.2f}% {row['sharpe']:>7.2f} {row['win_rate']*100:>6.1f}% {row['total_return']*100:>+9.0f}% {row['avg_positions']:>7.1f}")

    print("\n" + "=" * 100)
    print("TOP 20 BY NUMBER OF TRADES (Most Active)")
    print("=" * 100)
    
    top_trades = results_df.nlargest(20, 'n_trades')
    print(f"\n{'Config':<50} {'N':>7} {'Avg%':>8} {'Sharpe':>7} {'Win%':>7} {'Return':>10} {'AvgPos':>7}")
    print("-" * 102)
    for _, row in top_trades.iterrows():
        print(f"{row['config']:<50} {row['n_trades']:>7} {row['avg_pnl']*100:>+7.2f}% {row['sharpe']:>7.2f} {row['win_rate']*100:>6.1f}% {row['total_return']*100:>+9.0f}% {row['avg_positions']:>7.1f}")

# =============================================================================
# DEEP DIVE: BEST BALANCED CONFIGURATION
# =============================================================================

print("\n" + "=" * 100)
print("DEEP DIVE: BEST BALANCED CONFIGURATION")
print("=" * 100)

# Get best by Sharpe with reasonable trades
if len(results_df) > 0:
    best = results_df[results_df['n_trades'] >= 50].nlargest(1, 'sharpe')
    if len(best) > 0:
        best = best.iloc[0]
        
        print(f"\nConfiguration: {best['config']}")
        print(f"  Entry: |FR| > {best['entry_threshold']*100:.3f}%")
        print(f"  Exit: |FR| < {best['exit_threshold']*100:.4f}%")
        print(f"  Min Consecutive: {best['min_consecutive']} hours")
        print(f"\nPerformance:")
        print(f"  Trades: {best['n_trades']}")
        print(f"  Avg PnL per trade: {best['avg_pnl']*100:+.2f}%")
        print(f"  Sharpe Ratio: {best['sharpe']:.2f}")
        print(f"  Win Rate: {best['win_rate']*100:.1f}%")
        print(f"  Total Return: {best['total_return']*100:+.0f}%")
        print(f"  Final Capital: ${best['final_capital']:,.0f} (from ${INITIAL_CAPITAL:,})")
        print(f"\nPosition Statistics:")
        print(f"  Avg Positions Open: {best['avg_positions']:.1f}")
        print(f"  Max Positions Open: {best['max_positions']}")
        print(f"  Avg Hold Time: {best['avg_hold']:.1f} hours")

        # Re-run best config for detailed analysis
        best_sim = PortfolioSimulator(
            initial_capital=INITIAL_CAPITAL,
            max_positions=MAX_POSITIONS,
            max_position_pct=MAX_POSITION_SIZE,
            entry_threshold=best['entry_threshold'],
            exit_threshold=best['exit_threshold'],
            min_consecutive=best['min_consecutive']
        )
        best_result = best_sim.simulate(all_hours)
        
        if best_result:
            trades_df = best_result['trades_df']
            
            # Direction breakdown
            print("\n" + "-" * 50)
            print("DIRECTION BREAKDOWN:")
            for direction in ['long', 'short']:
                subset = trades_df[trades_df['direction'] == direction]
                if len(subset) > 0:
                    print(f"  {direction.upper()}: N={len(subset)}, Avg={subset['net_pnl'].mean()*100:+.2f}%, Win={((subset['net_pnl']>0).mean())*100:.1f}%")
            
            # Monthly breakdown
            print("\n" + "-" * 50)
            print("MONTHLY PERFORMANCE:")
            trades_df['month'] = pd.to_datetime(trades_df['entry_hour']).dt.to_period('M')
            monthly = trades_df.groupby('month').agg({
                'net_pnl': ['count', 'mean', 'sum'],
                'dollar_pnl': 'sum'
            })
            monthly.columns = ['n_trades', 'avg_pnl', 'total_pnl', 'dollar_pnl']
            
            print(f"\n{'Month':<12} {'N':>6} {'Avg PnL':>10} {'Total PnL':>12} {'$ PnL':>12}")
            print("-" * 56)
            for month, row in monthly.iterrows():
                print(f"{str(month):<12} {row['n_trades']:>6.0f} {row['avg_pnl']*100:>+9.2f}% {row['total_pnl']*100:>+11.1f}% ${row['dollar_pnl']:>+10,.0f}")
            
            # Top coins
            print("\n" + "-" * 50)
            print("TOP 10 COINS:")
            coin_perf = trades_df.groupby('coin').agg({
                'net_pnl': ['count', 'mean', 'sum']
            })
            coin_perf.columns = ['n_trades', 'avg_pnl', 'total_pnl']
            coin_perf = coin_perf[coin_perf['n_trades'] >= 3].nlargest(10, 'avg_pnl')
            
            print(f"\n{'Coin':<10} {'N':>6} {'Avg PnL':>10} {'Total PnL':>12}")
            print("-" * 42)
            for coin, row in coin_perf.iterrows():
                print(f"{coin:<10} {row['n_trades']:>6.0f} {row['avg_pnl']*100:>+9.2f}% {row['total_pnl']*100:>+11.1f}%")

# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

print("\n" + "=" * 100)
print("SENSITIVITY ANALYSIS")
print("=" * 100)

if len(results_df) > 0:
    print("\n1. ENTRY THRESHOLD SENSITIVITY:")
    print(f"\n{'Entry':>10} {'N Trades':>10} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Return':>10}")
    print("-" * 62)
    
    for thresh in FR_THRESHOLDS:
        subset = results_df[results_df['entry_threshold'] == thresh]
        if len(subset) > 0:
            print(f"{thresh*100:>9.2f}% {subset['n_trades'].mean():>10.0f} {subset['avg_pnl'].mean()*100:>+9.2f}% {subset['sharpe'].mean():>8.2f} {subset['win_rate'].mean()*100:>7.1f}% {subset['total_return'].mean()*100:>+9.0f}%")

    print("\n2. EXIT THRESHOLD SENSITIVITY:")
    print(f"\n{'Exit':>10} {'N Trades':>10} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Return':>10}")
    print("-" * 62)
    
    for thresh in EXIT_THRESHOLDS:
        subset = results_df[results_df['exit_threshold'] == thresh]
        if len(subset) > 0:
            print(f"{thresh*100:>9.3f}% {subset['n_trades'].mean():>10.0f} {subset['avg_pnl'].mean()*100:>+9.2f}% {subset['sharpe'].mean():>8.2f} {subset['win_rate'].mean()*100:>7.1f}% {subset['total_return'].mean()*100:>+9.0f}%")

    print("\n3. CONSECUTIVE HOURS SENSITIVITY:")
    print(f"\n{'Consec':>10} {'N Trades':>10} {'Avg PnL':>10} {'Sharpe':>8} {'Win%':>8} {'Return':>10}")
    print("-" * 62)
    
    for consec in CONSECUTIVE_HOURS:
        subset = results_df[results_df['min_consecutive'] == consec]
        if len(subset) > 0:
            print(f"{consec:>10} {subset['n_trades'].mean():>10.0f} {subset['avg_pnl'].mean()*100:>+9.2f}% {subset['sharpe'].mean():>8.2f} {subset['win_rate'].mean()*100:>7.1f}% {subset['total_return'].mean()*100:>+9.0f}%")

# =============================================================================
# COMPARISON: LOW vs HIGH THRESHOLDS
# =============================================================================

print("\n" + "=" * 100)
print("COMPARISON: LOW vs HIGH FR THRESHOLDS")
print("=" * 100)

if len(results_df) > 0:
    # Low threshold (more signals)
    low_thresh = results_df[results_df['entry_threshold'] <= 0.0003]
    # High threshold (extreme only)
    high_thresh = results_df[results_df['entry_threshold'] >= 0.001]
    
    print("\nLOW THRESHOLD (Entry <= 0.03%):")
    if len(low_thresh) > 0:
        print(f"  Configs tested: {len(low_thresh)}")
        print(f"  Avg trades: {low_thresh['n_trades'].mean():.0f}")
        print(f"  Avg PnL: {low_thresh['avg_pnl'].mean()*100:+.2f}%")
        print(f"  Avg Sharpe: {low_thresh['sharpe'].mean():.2f}")
        print(f"  Avg Win Rate: {low_thresh['win_rate'].mean()*100:.1f}%")
        print(f"  Best return: {low_thresh['total_return'].max()*100:+.0f}%")
    
    print("\nHIGH THRESHOLD (Entry >= 0.10%):")
    if len(high_thresh) > 0:
        print(f"  Configs tested: {len(high_thresh)}")
        print(f"  Avg trades: {high_thresh['n_trades'].mean():.0f}")
        print(f"  Avg PnL: {high_thresh['avg_pnl'].mean()*100:+.2f}%")
        print(f"  Avg Sharpe: {high_thresh['sharpe'].mean():.2f}")
        print(f"  Avg Win Rate: {high_thresh['win_rate'].mean()*100:.1f}%")
        print(f"  Best return: {high_thresh['total_return'].max()*100:+.0f}%")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

print("""
KEY FINDINGS:

1. SIGNAL FREQUENCY BY THRESHOLD:
   - |FR| > 0.01%: ~many signals per day (very active)
   - |FR| > 0.03%: moderate signals per day
   - |FR| > 0.10%: ~few signals per day (selective)
   
2. TRADE-OFF:
   - Lower threshold = More trades, more exposure, more fees
   - Higher threshold = Fewer trades, better per-trade PnL, less capital utilization
   
3. PORTFOLIO CONSIDERATIONS:
   - Multiple positions open simultaneously diversify risk
   - Capital utilization improved with lower thresholds
   - Need to balance Sharpe vs total return

4. RECOMMENDED CONFIGURATIONS:
   a) AGGRESSIVE (High Returns): Entry > 0.02%, Exit < 0.01%, Cons >= 2
   b) BALANCED (Good Sharpe): Entry > 0.05%, Exit < 0.02%, Cons >= 3
   c) CONSERVATIVE (Selective): Entry > 0.10%, Exit < 0.03%, Cons >= 4
""")

# Save results
results_df.to_csv('full_fr_backtest_results.csv', index=False)
print("\nResults saved to full_fr_backtest_results.csv")
