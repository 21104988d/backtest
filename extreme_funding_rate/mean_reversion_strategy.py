"""
Mean Reversion Strategy Backtest - Final Version

Strategy Rules:
- Entry: 0.0014% < |FR| < 0.0015%
- Exit: |FR| <= 0.0003%
- Hold: Keep position open if FR > 0.0015% (no exit high)
- Direction: SHORT if FR > 0, LONG if FR < 0 (bet against FR direction)

Capital & Risk:
- Capital: $1,000
- Position Size: $100 per coin
- Max Concurrent: ~88 positions (based on backtest)
- Leverage: Calculated based on actual usage

Benchmark:
- Buy $1,000 BTC and hold for comparison
- Correlation analysis between strategy and BTC
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# =============================================================================
# PARAMETERS
# =============================================================================

CAPITAL = 1_000               # Total capital
POSITION_SIZE = 100           # $100 per position
ENTRY_LOW = 0.000014          # 0.0014%
ENTRY_HIGH = 0.000015         # 0.0015%
EXIT_LOW = 0.000003           # 0.0003%
TAKER_FEE = 0.00045           # 0.045% per trade

print("=" * 80)
print("MEAN REVERSION STRATEGY BACKTEST")
print("=" * 80)
print(f"\nSTRATEGY PARAMETERS:")
print(f"  Entry Range: {ENTRY_LOW*100:.4f}% < |FR| < {ENTRY_HIGH*100:.4f}%")
print(f"  Exit: |FR| <= {EXIT_LOW*100:.4f}%")
print(f"  Direction: SHORT if FR > 0, LONG if FR < 0")
print(f"\nCAPITAL & RISK:")
print(f"  Total Capital: ${CAPITAL:,}")
print(f"  Position Size: ${POSITION_SIZE}")
print(f"  Taker Fee: {TAKER_FEE*100:.3f}%")

# =============================================================================
# DATA LOADING
# =============================================================================

print("\nLoading data...")
base_path = Path(__file__).parent

funding = pd.read_csv(base_path / 'funding_history.csv')
funding['datetime'] = pd.to_datetime(funding['datetime'], format='mixed', utc=True)
funding['hour'] = funding['datetime'].dt.floor('h').dt.tz_localize(None)

price = pd.read_csv(base_path / 'price_history.csv')
price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
price['hour'] = price['timestamp'].dt.floor('h').dt.tz_localize(None)

merged = pd.merge(
    funding[['hour', 'coin', 'funding_rate']],
    price[['hour', 'coin', 'price']],
    on=['hour', 'coin'],
    how='inner'
).sort_values(['hour', 'coin']).reset_index(drop=True)

merged['abs_fr'] = merged['funding_rate'].abs()

print(f"  Records: {len(merged):,}")
print(f"  Hours: {merged['hour'].nunique():,}")
print(f"  Coins: {merged['coin'].nunique()}")
print(f"  Date Range: {merged['hour'].min()} to {merged['hour'].max()}")

# =============================================================================
# BTC BENCHMARK - Buy $1,000 BTC and hold
# =============================================================================

print("\nSetting up BTC benchmark...")
btc_prices = price[price['coin'] == 'BTC'][['hour', 'price']].copy()
btc_prices = btc_prices.drop_duplicates('hour').sort_values('hour').reset_index(drop=True)

# Get first and last BTC price
btc_start_price = btc_prices['price'].iloc[0]
btc_start_date = btc_prices['hour'].iloc[0]
btc_btc_qty = CAPITAL / btc_start_price  # How much BTC we can buy with $1,000

print(f"  BTC Start Price: ${btc_start_price:,.2f}")
print(f"  BTC Quantity: {btc_btc_qty:.6f} BTC")
print(f"  BTC Start Date: {btc_start_date}")

# Create BTC portfolio value over time
btc_benchmark = btc_prices.copy()
btc_benchmark['btc_value'] = btc_btc_qty * btc_benchmark['price']
btc_benchmark['btc_pnl'] = btc_benchmark['btc_value'] - CAPITAL
btc_benchmark['btc_return_pct'] = (btc_benchmark['btc_value'] / CAPITAL - 1) * 100
btc_benchmark = btc_benchmark.set_index('hour')

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

print("\n" + "=" * 80)
print("RUNNING BACKTEST...")
print("=" * 80)

hours = sorted(merged['hour'].unique())
hour_groups = {h: g for h, g in merged.groupby('hour')}

# Position tracking: {coin: {direction, entry_hour, entry_price, funding_accumulated}}
positions = {}

# Results
trades = []
hourly_records = []

# Cumulative tracking
cumulative_pnl = 0.0
cumulative_fees = 0.0
peak_pnl = 0.0
max_drawdown = 0.0

for h_idx, hour in enumerate(hours):
    if hour not in hour_groups:
        continue
    
    df = hour_groups[hour]
    coin_data = dict(zip(
        df['coin'],
        [{'fr': fr, 'abs_fr': afr, 'price': p} 
         for fr, afr, p in zip(df['funding_rate'], df['abs_fr'], df['price'])]
    ))
    
    available_coins = set(coin_data.keys())
    
    hourly_funding = 0.0
    hourly_trades_opened = 0
    hourly_trades_closed = 0
    hourly_fees = 0.0
    hourly_realized_pnl = 0.0
    
    # -----------------------------------------------------------------
    # STEP 1: Check exits - ONLY exit when |FR| <= EXIT_LOW
    # -----------------------------------------------------------------
    for coin in list(positions.keys()):
        if coin not in available_coins:
            continue
        
        data = coin_data[coin]
        pos = positions[coin]
        
        # Exit condition: FR dropped below threshold
        if data['abs_fr'] <= EXIT_LOW:
            exit_price = data['price']
            entry_price = pos['entry_price']
            direction = pos['direction']
            
            # Calculate PnL
            price_return = (exit_price - entry_price) / entry_price
            price_pnl = direction * price_return * POSITION_SIZE
            funding_pnl = pos['funding_accumulated']
            fees = POSITION_SIZE * TAKER_FEE * 2  # Round-trip
            net_pnl = price_pnl + funding_pnl - fees
            
            hold_hours = (hour - pos['entry_hour']).total_seconds() / 3600
            
            trades.append({
                'coin': coin,
                'direction': 'SHORT' if direction == -1 else 'LONG',
                'entry_hour': pos['entry_hour'],
                'exit_hour': hour,
                'hold_hours': hold_hours,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_fr': pos['entry_fr'],
                'exit_fr': data['fr'],
                'price_pnl': price_pnl,
                'funding_pnl': funding_pnl,
                'fees': fees,
                'net_pnl': net_pnl
            })
            
            hourly_realized_pnl += net_pnl
            hourly_fees += fees
            hourly_trades_closed += 1
            
            del positions[coin]
    
    # -----------------------------------------------------------------
    # STEP 2: Update funding for all open positions
    # -----------------------------------------------------------------
    for coin in positions:
        if coin in coin_data:
            # We RECEIVE funding (betting against FR direction)
            fr_received = coin_data[coin]['abs_fr'] * POSITION_SIZE
            positions[coin]['funding_accumulated'] += fr_received
            hourly_funding += fr_received
    
    # -----------------------------------------------------------------
    # STEP 3: Check entries - enter when ENTRY_LOW < |FR| < ENTRY_HIGH
    # -----------------------------------------------------------------
    for coin in available_coins:
        if coin in positions:
            continue
        
        data = coin_data[coin]
        
        # Entry condition: within entry range
        if data['abs_fr'] > ENTRY_LOW and data['abs_fr'] < ENTRY_HIGH:
            # Direction: SHORT if FR > 0, LONG if FR < 0
            direction = -1 if data['fr'] > 0 else 1
            
            positions[coin] = {
                'direction': direction,
                'entry_hour': hour,
                'entry_price': data['price'],
                'entry_fr': data['fr'],
                'funding_accumulated': 0.0
            }
            
            hourly_fees += POSITION_SIZE * TAKER_FEE  # Entry fee
            hourly_trades_opened += 1
    
    # -----------------------------------------------------------------
    # STEP 4: Track hourly metrics
    # -----------------------------------------------------------------
    n_positions = len(positions)
    n_long = sum(1 for p in positions.values() if p['direction'] == 1)
    n_short = sum(1 for p in positions.values() if p['direction'] == -1)
    
    # Calculate unrealized PnL
    unrealized_pnl = 0.0
    for coin, pos in positions.items():
        if coin in coin_data:
            current_price = coin_data[coin]['price']
            entry_price = pos['entry_price']
            direction = pos['direction']
            price_return = (current_price - entry_price) / entry_price
            unrealized_pnl += direction * price_return * POSITION_SIZE
            unrealized_pnl += pos['funding_accumulated']
    
    cumulative_fees += hourly_fees
    cumulative_pnl += hourly_realized_pnl
    
    total_pnl = cumulative_pnl + unrealized_pnl
    
    # Drawdown tracking
    if total_pnl > peak_pnl:
        peak_pnl = total_pnl
    drawdown = peak_pnl - total_pnl
    if drawdown > max_drawdown:
        max_drawdown = drawdown
    
    # Capital usage & leverage
    notional_exposure = n_positions * POSITION_SIZE
    leverage = notional_exposure / CAPITAL if CAPITAL > 0 else 0
    
    hourly_records.append({
        'hour': hour,
        'n_positions': n_positions,
        'n_long': n_long,
        'n_short': n_short,
        'notional_exposure': notional_exposure,
        'leverage': leverage,
        'funding_received': hourly_funding,
        'trades_opened': hourly_trades_opened,
        'trades_closed': hourly_trades_closed,
        'fees': hourly_fees,
        'realized_pnl': hourly_realized_pnl,
        'unrealized_pnl': unrealized_pnl,
        'cumulative_pnl': cumulative_pnl,
        'total_pnl': total_pnl,
        'drawdown': drawdown
    })
    
    if (h_idx + 1) % 5000 == 0:
        print(f"  Processed {h_idx + 1:,} / {len(hours):,} hours... (Total PnL: ${total_pnl:,.0f})")

print(f"\n  Completed: {len(hours):,} hours")

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

trades_df = pd.DataFrame(trades)
hourly_df = pd.DataFrame(hourly_records)

print("\n" + "=" * 80)
print("BACKTEST RESULTS")
print("=" * 80)

# --- Trade Statistics ---
print(f"\n{'─' * 40}")
print("TRADE STATISTICS")
print(f"{'─' * 40}")
print(f"Total Trades Closed: {len(trades_df):,}")
print(f"Positions Still Open: {len(positions)}")
print(f"Unique Coins Traded: {trades_df['coin'].nunique() if len(trades_df) > 0 else 0}")

if len(trades_df) > 0:
    wins = (trades_df['net_pnl'] > 0).sum()
    losses = (trades_df['net_pnl'] <= 0).sum()
    print(f"Winning Trades: {wins:,} ({wins/len(trades_df)*100:.1f}%)")
    print(f"Losing Trades: {losses:,} ({losses/len(trades_df)*100:.1f}%)")
    
    print(f"\n{'─' * 40}")
    print("HOLDING TIME")
    print(f"{'─' * 40}")
    print(f"Average: {trades_df['hold_hours'].mean():.1f} hours ({trades_df['hold_hours'].mean()/24:.1f} days)")
    print(f"Median: {trades_df['hold_hours'].median():.1f} hours")
    print(f"Min: {trades_df['hold_hours'].min():.1f} hours")
    print(f"Max: {trades_df['hold_hours'].max():.1f} hours")

# --- Direction Breakdown ---
if len(trades_df) > 0:
    print(f"\n{'─' * 40}")
    print("DIRECTION BREAKDOWN")
    print(f"{'─' * 40}")
    for direction in ['LONG', 'SHORT']:
        subset = trades_df[trades_df['direction'] == direction]
        if len(subset) > 0:
            wins = (subset['net_pnl'] > 0).sum()
            total_pnl = subset['net_pnl'].sum()
            avg_pnl = subset['net_pnl'].mean()
            print(f"{direction}: {len(subset):,} trades | Win Rate: {wins/len(subset)*100:.1f}% | Total PnL: ${total_pnl:,.2f} | Avg: ${avg_pnl:.2f}")

# --- Position Statistics ---
print(f"\n{'─' * 40}")
print("CONCURRENT POSITIONS")
print(f"{'─' * 40}")
print(f"Average: {hourly_df['n_positions'].mean():.1f}")
print(f"Max: {hourly_df['n_positions'].max()}")
print(f"P50 (Median): {hourly_df['n_positions'].quantile(0.50):.0f}")
print(f"P75: {hourly_df['n_positions'].quantile(0.75):.0f}")
print(f"P95: {hourly_df['n_positions'].quantile(0.95):.0f}")
print(f"P99: {hourly_df['n_positions'].quantile(0.99):.0f}")

# --- Leverage Statistics ---
print(f"\n{'─' * 40}")
print("LEVERAGE & CAPITAL USAGE")
print(f"{'─' * 40}")
print(f"Capital: ${CAPITAL:,}")
print(f"Position Size: ${POSITION_SIZE}")
print(f"Max Notional Exposure: ${hourly_df['notional_exposure'].max():,.0f}")
print(f"Average Leverage: {hourly_df['leverage'].mean():.2f}x")
print(f"Max Leverage: {hourly_df['leverage'].max():.2f}x")
print(f"P95 Leverage: {hourly_df['leverage'].quantile(0.95):.2f}x")

# --- PnL Breakdown ---
print(f"\n{'─' * 40}")
print("PNL BREAKDOWN (Closed Trades)")
print(f"{'─' * 40}")
if len(trades_df) > 0:
    total_price_pnl = trades_df['price_pnl'].sum()
    total_funding = trades_df['funding_pnl'].sum()
    total_fees = trades_df['fees'].sum()
    total_net = trades_df['net_pnl'].sum()
    
    print(f"Price PnL:        ${total_price_pnl:>12,.2f}")
    print(f"Funding Received: ${total_funding:>12,.2f}")
    print(f"Fees Paid:        ${-total_fees:>12,.2f}")
    print(f"{'─' * 30}")
    print(f"Net PnL (Closed): ${total_net:>12,.2f}")

# --- Open Positions Value ---
if positions:
    print(f"\n{'─' * 40}")
    print("OPEN POSITIONS (Unrealized)")
    print(f"{'─' * 40}")
    
    # Get last hour's data
    last_hour = hours[-1]
    last_df = hour_groups[last_hour]
    last_prices = dict(zip(last_df['coin'], last_df['price']))
    
    open_unrealized = 0.0
    open_funding = 0.0
    pending_exit_fees = 0.0
    
    for coin, pos in positions.items():
        if coin in last_prices:
            current_price = last_prices[coin]
            entry_price = pos['entry_price']
            direction = pos['direction']
            price_return = (current_price - entry_price) / entry_price
            open_unrealized += direction * price_return * POSITION_SIZE
            open_funding += pos['funding_accumulated']
            pending_exit_fees += POSITION_SIZE * TAKER_FEE
    
    print(f"Open Positions: {len(positions)}")
    print(f"Unrealized Price PnL: ${open_unrealized:,.2f}")
    print(f"Accumulated Funding: ${open_funding:,.2f}")
    print(f"Pending Exit Fees: ${-pending_exit_fees:,.2f}")
    print(f"Net Unrealized: ${open_unrealized + open_funding - pending_exit_fees:,.2f}")

# --- Total PnL ---
print(f"\n{'─' * 40}")
print("TOTAL PNL SUMMARY")
print(f"{'─' * 40}")
final_total_pnl = hourly_df['total_pnl'].iloc[-1] if len(hourly_df) > 0 else 0
final_cumulative = hourly_df['cumulative_pnl'].iloc[-1] if len(hourly_df) > 0 else 0
final_unrealized = hourly_df['unrealized_pnl'].iloc[-1] if len(hourly_df) > 0 else 0

print(f"Realized PnL (Closed): ${final_cumulative:>12,.2f}")
print(f"Unrealized PnL (Open): ${final_unrealized:>12,.2f}")
print(f"{'─' * 30}")
print(f"TOTAL PNL:             ${final_total_pnl:>12,.2f}")
print(f"\nReturn on Capital: {final_total_pnl/CAPITAL*100:.2f}%")

# --- Risk Metrics ---
print(f"\n{'─' * 40}")
print("RISK METRICS")
print(f"{'─' * 40}")
print(f"Max Drawdown: ${max_drawdown:,.2f}")
print(f"Max Drawdown %: {max_drawdown/CAPITAL*100:.2f}%")

if len(trades_df) > 0:
    avg_trade_pnl = trades_df['net_pnl'].mean()
    std_trade_pnl = trades_df['net_pnl'].std()
    sharpe_per_trade = avg_trade_pnl / std_trade_pnl if std_trade_pnl > 0 else 0
    print(f"Avg PnL per Trade: ${avg_trade_pnl:.2f}")
    print(f"Std PnL per Trade: ${std_trade_pnl:.2f}")
    print(f"Sharpe (per trade): {sharpe_per_trade:.3f}")

# --- Yearly Breakdown ---
if len(trades_df) > 0:
    print(f"\n{'─' * 40}")
    print("YEARLY BREAKDOWN")
    print(f"{'─' * 40}")
    trades_df['year'] = trades_df['exit_hour'].dt.year
    yearly = trades_df.groupby('year').agg({
        'net_pnl': ['sum', 'count', 'mean'],
        'hold_hours': 'mean'
    })
    yearly.columns = ['Net PnL', 'Trades', 'Avg PnL', 'Avg Hold (h)']
    yearly['Win Rate'] = trades_df.groupby('year').apply(lambda x: (x['net_pnl'] > 0).mean() * 100)
    print(yearly.round(2).to_string())

# --- Monthly Returns ---
if len(hourly_df) > 0:
    print(f"\n{'─' * 40}")
    print("MONTHLY RETURNS")
    print(f"{'─' * 40}")
    hourly_df['month'] = hourly_df['hour'].dt.to_period('M')
    monthly = hourly_df.groupby('month').agg({
        'total_pnl': 'last',
        'n_positions': 'mean',
        'leverage': 'max'
    })
    monthly['monthly_return'] = monthly['total_pnl'].diff()
    monthly['monthly_return'].iloc[0] = monthly['total_pnl'].iloc[0]
    monthly = monthly.rename(columns={
        'total_pnl': 'Cumulative PnL',
        'n_positions': 'Avg Positions',
        'leverage': 'Max Leverage',
        'monthly_return': 'Monthly Return'
    })
    print(monthly.round(2).tail(12).to_string())

# =============================================================================
# CAPITAL SENSITIVITY ANALYSIS - Risk at Different Capital Levels
# =============================================================================

print(f"\n{'─' * 40}")
print("CAPITAL SENSITIVITY ANALYSIS")
print(f"{'─' * 40}")

# The strategy's max notional exposure and max drawdown (in $) are fixed
# Only the leverage and MDD% change with capital
max_notional = hourly_df['notional_exposure'].max()
avg_notional = hourly_df['notional_exposure'].mean()

print(f"\nFixed Strategy Metrics (independent of capital):")
print(f"  Max Notional Exposure: ${max_notional:,.0f}")
print(f"  Avg Notional Exposure: ${avg_notional:,.0f}")
print(f"  Max Drawdown ($): ${max_drawdown:,.2f}")
print(f"  Net PnL ($): ${final_total_pnl:,.2f}")

# Test different capital levels
capital_levels = [1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000]

print(f"\n{'Capital':>10} | {'Max Lev':>8} | {'Avg Lev':>8} | {'MDD %':>8} | {'Return %':>10} | {'Risk Rating':>12}")
print("-" * 75)

for cap in capital_levels:
    max_lev = max_notional / cap
    avg_lev = avg_notional / cap
    mdd_pct = (max_drawdown / cap) * 100
    return_pct = (final_total_pnl / cap) * 100
    
    # Risk rating based on MDD%
    if mdd_pct > 100:
        risk = "🔴 LIQUIDATION"
    elif mdd_pct > 50:
        risk = "🟠 Very High"
    elif mdd_pct > 30:
        risk = "🟡 High"
    elif mdd_pct > 20:
        risk = "🟢 Moderate"
    else:
        risk = "✅ Safe"
    
    marker = " <-- Current" if cap == CAPITAL else ""
    print(f"${cap:>9,} | {max_lev:>7.2f}x | {avg_lev:>7.2f}x | {mdd_pct:>7.1f}% | {return_pct:>9.1f}% | {risk}{marker}")

# Find minimum safe capital (MDD < 30%)
min_safe_capital = int(np.ceil(max_drawdown / 0.30 / 100) * 100)  # Round up to nearest $100
min_moderate_capital = int(np.ceil(max_drawdown / 0.50 / 100) * 100)  # MDD < 50%

print(f"\n{'─' * 40}")
print("RECOMMENDED CAPITAL LEVELS")
print(f"{'─' * 40}")
print(f"  For MDD < 30% (Safe): ${min_safe_capital:,} minimum")
print(f"  For MDD < 50% (Moderate): ${min_moderate_capital:,} minimum")
print(f"  For Max Leverage < 1x: ${int(max_notional):,} minimum")

# Calculate optimal capital for different risk tolerances
print(f"\n  At ${min_safe_capital:,} capital:")
print(f"    Max Leverage: {max_notional/min_safe_capital:.2f}x")
print(f"    MDD: {max_drawdown/min_safe_capital*100:.1f}%")
print(f"    Return: {final_total_pnl/min_safe_capital*100:.1f}%")

# =============================================================================
# BTC BENCHMARK COMPARISON & CORRELATION ANALYSIS
# =============================================================================

print(f"\n{'─' * 40}")
print("BTC BENCHMARK COMPARISON")
print(f"{'─' * 40}")

# Merge strategy with BTC benchmark
hourly_df_indexed = hourly_df.set_index('hour')
comparison_df = hourly_df_indexed.join(btc_benchmark[['btc_value', 'btc_pnl', 'btc_return_pct']], how='inner')

# Get final values
if len(comparison_df) > 0:
    strategy_final = comparison_df['total_pnl'].iloc[-1]
    btc_final_pnl = comparison_df['btc_pnl'].iloc[-1]
    btc_final_value = comparison_df['btc_value'].iloc[-1]
    btc_end_price = btc_benchmark['price'].iloc[-1] if 'price' in btc_benchmark.columns else btc_final_value / btc_btc_qty
    
    strategy_return_pct = (strategy_final / CAPITAL) * 100
    btc_return_pct = (btc_final_pnl / CAPITAL) * 100
    
    print(f"\nStarting Capital: ${CAPITAL:,}")
    print(f"\nStrategy Performance:")
    print(f"  Final PnL: ${strategy_final:,.2f}")
    print(f"  Return: {strategy_return_pct:.2f}%")
    print(f"  Final Value: ${CAPITAL + strategy_final:,.2f}")
    
    print(f"\nBTC Buy & Hold:")
    print(f"  BTC Bought: {btc_btc_qty:.6f} BTC @ ${btc_start_price:,.2f}")
    print(f"  Final PnL: ${btc_final_pnl:,.2f}")
    print(f"  Return: {btc_return_pct:.2f}%")
    print(f"  Final Value: ${btc_final_value:,.2f}")
    
    outperformance = strategy_final - btc_final_pnl
    print(f"\nOUTPERFORMANCE vs BTC:")
    print(f"  Absolute: ${outperformance:,.2f}")
    print(f"  Relative: {strategy_return_pct - btc_return_pct:.2f}pp")
    
    # Calculate daily returns for correlation
    comparison_df['strategy_daily_return'] = comparison_df['total_pnl'].diff()
    comparison_df['btc_daily_return'] = comparison_df['btc_pnl'].diff()
    
    # Calculate correlation
    correlation = comparison_df['strategy_daily_return'].corr(comparison_df['btc_daily_return'])
    
    print(f"\n{'─' * 40}")
    print("CORRELATION ANALYSIS")
    print(f"{'─' * 40}")
    print(f"Hourly Return Correlation: {correlation:.4f}")
    
    # Rolling correlation (7-day window = 168 hours)
    comparison_df['rolling_corr'] = comparison_df['strategy_daily_return'].rolling(168).corr(comparison_df['btc_daily_return'])
    
    print(f"Rolling 7-day Correlation:")
    print(f"  Mean: {comparison_df['rolling_corr'].mean():.4f}")
    print(f"  Min: {comparison_df['rolling_corr'].min():.4f}")
    print(f"  Max: {comparison_df['rolling_corr'].max():.4f}")
    
    if abs(correlation) < 0.3:
        corr_interpretation = "WEAK correlation - strategy provides GOOD diversification"
    elif abs(correlation) < 0.6:
        corr_interpretation = "MODERATE correlation - some diversification benefit"
    else:
        corr_interpretation = "STRONG correlation - limited diversification benefit"
    
    print(f"\nInterpretation: {corr_interpretation}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

output_dir = base_path / 'mean_reversion_results'
output_dir.mkdir(exist_ok=True)

trades_df.to_csv(output_dir / 'trades.csv', index=False)
hourly_df.to_csv(output_dir / 'hourly_records.csv', index=False)

print(f"\n{'─' * 40}")
print("OUTPUT FILES")
print(f"{'─' * 40}")
print(f"Trades saved to: {output_dir / 'trades.csv'}")
print(f"Hourly records saved to: {output_dir / 'hourly_records.csv'}")

# =============================================================================
# GENERATE VISUALIZATION GRAPHS
# =============================================================================

print(f"\n{'─' * 40}")
print("GENERATING GRAPHS")
print(f"{'─' * 40}")

# hour column is already datetime
hourly_df['datetime'] = hourly_df['hour']

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Mean Reversion Strategy - ${CAPITAL:,} Capital | Entry: {ENTRY_LOW*100:.4f}%-{ENTRY_HIGH*100:.4f}% | Exit: {EXIT_LOW*100:.4f}%', 
             fontsize=14, fontweight='bold')

# 1. Equity Curve with BTC Comparison
ax1 = axes[0, 0]
ax1.plot(hourly_df['datetime'], hourly_df['total_pnl'], color='blue', linewidth=1.5, label='Mean Reversion Strategy')
# Add BTC benchmark
if len(comparison_df) > 0:
    comparison_df_plot = comparison_df.reset_index()
    ax1.plot(comparison_df_plot['hour'], comparison_df_plot['btc_pnl'], color='orange', linewidth=1.5, label='BTC Buy & Hold', alpha=0.8)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.set_title('Strategy vs BTC Buy & Hold', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('PnL ($)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
ax1.tick_params(axis='x', rotation=45)

# 2. Rolling Concurrent Positions
ax2 = axes[0, 1]
ax2.plot(hourly_df['datetime'], hourly_df['n_positions'], color='purple', linewidth=0.5, alpha=0.7)
# Add rolling average for cleaner view
rolling_positions = hourly_df['n_positions'].rolling(window=168).mean()  # 1-week rolling avg
ax2.plot(hourly_df['datetime'], rolling_positions, color='darkblue', linewidth=2, label='7-day Rolling Avg')
ax2.axhline(y=hourly_df['n_positions'].mean(), color='red', linestyle='--', alpha=0.7, 
            label=f'Mean: {hourly_df["n_positions"].mean():.1f}')
ax2.set_title('Concurrent Positions Over Time', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Number of Positions')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
ax2.tick_params(axis='x', rotation=45)

# 3. Rolling Leverage
ax3 = axes[1, 0]
ax3.plot(hourly_df['datetime'], hourly_df['leverage'], color='orange', linewidth=0.5, alpha=0.7)
rolling_leverage = hourly_df['leverage'].rolling(window=168).mean()
ax3.plot(hourly_df['datetime'], rolling_leverage, color='darkorange', linewidth=2, label='7-day Rolling Avg')
ax3.axhline(y=hourly_df['leverage'].mean(), color='red', linestyle='--', alpha=0.7,
            label=f'Mean: {hourly_df["leverage"].mean():.3f}x')
ax3.axhline(y=hourly_df['leverage'].max(), color='darkred', linestyle=':', alpha=0.5,
            label=f'Max: {hourly_df["leverage"].max():.3f}x')
ax3.set_title('Leverage Over Time', fontsize=12, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Leverage (x)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
ax3.tick_params(axis='x', rotation=45)

# 4. Monthly Returns Bar Chart
ax4 = axes[1, 1]
hourly_df['month'] = hourly_df['datetime'].dt.to_period('M')
monthly_pnl = hourly_df.groupby('month')['total_pnl'].last().diff()
monthly_pnl.iloc[0] = hourly_df.groupby('month')['total_pnl'].last().iloc[0]
colors = ['green' if x >= 0 else 'red' for x in monthly_pnl.values]
months_str = [str(m) for m in monthly_pnl.index]
bars = ax4.bar(months_str, monthly_pnl.values, color=colors, alpha=0.7, edgecolor='black')
ax4.axhline(y=0, color='black', linewidth=0.5)
ax4.set_title('Monthly Returns', fontsize=12, fontweight='bold')
ax4.set_xlabel('Month')
ax4.set_ylabel('Monthly PnL ($)')
ax4.grid(True, alpha=0.3, axis='y')
ax4.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, val in zip(bars, monthly_pnl.values):
    if abs(val) > 100:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'${val:.0f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=7)

plt.tight_layout()
plt.savefig(output_dir / 'backtest_charts.png', dpi=150, bbox_inches='tight')
print(f"Charts saved to: {output_dir / 'backtest_charts.png'}")

# =============================================================================
# ADDITIONAL CHART: Cumulative Fees & Fee Impact
# =============================================================================

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Trading Fee Analysis', fontsize=14, fontweight='bold')

# Calculate cumulative fees from trades
if not trades_df.empty:
    trades_df_sorted = trades_df.sort_values('exit_hour')
    trades_df_sorted['cumulative_fees'] = trades_df_sorted['fees'].cumsum()
    trades_df_sorted['gross_pnl'] = trades_df_sorted['price_pnl'] + trades_df_sorted['funding_pnl']
    trades_df_sorted['cumulative_gross_pnl'] = trades_df_sorted['gross_pnl'].cumsum()
    trades_df_sorted['cumulative_net_pnl'] = trades_df_sorted['net_pnl'].cumsum()
    
    # exit_hour is already datetime
    trades_df_sorted['exit_datetime'] = trades_df_sorted['exit_hour']
    
    # Chart 1: Cumulative Fees Over Time
    ax5 = axes2[0]
    ax5.plot(trades_df_sorted['exit_datetime'], trades_df_sorted['cumulative_fees'], 
             color='red', linewidth=1.5, label='Cumulative Fees')
    ax5.fill_between(trades_df_sorted['exit_datetime'], trades_df_sorted['cumulative_fees'], 
                     alpha=0.3, color='red')
    ax5.set_title('Cumulative Trading Fees', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Total Fees ($)')
    total_fees = trades_df_sorted['cumulative_fees'].iloc[-1]
    ax5.text(0.02, 0.98, f'Total Fees: ${total_fees:,.0f}', transform=ax5.transAxes, 
             fontsize=11, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax5.tick_params(axis='x', rotation=45)
    
    # Chart 2: Gross vs Net PnL
    ax6 = axes2[1]
    ax6.plot(trades_df_sorted['exit_datetime'], trades_df_sorted['cumulative_gross_pnl'], 
             color='blue', linewidth=1.5, label='Gross PnL (before fees)')
    ax6.plot(trades_df_sorted['exit_datetime'], trades_df_sorted['cumulative_net_pnl'], 
             color='green', linewidth=1.5, label='Net PnL (after fees)')
    ax6.fill_between(trades_df_sorted['exit_datetime'], 
                     trades_df_sorted['cumulative_gross_pnl'], 
                     trades_df_sorted['cumulative_net_pnl'],
                     alpha=0.3, color='red', label='Fee Impact')
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax6.set_title('Gross vs Net PnL (Fee Impact)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Cumulative PnL ($)')
    ax6.legend(loc='upper left')
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax6.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(output_dir / 'fee_analysis.png', dpi=150, bbox_inches='tight')
print(f"Fee analysis saved to: {output_dir / 'fee_analysis.png'}")

# =============================================================================
# CHART 3: Trade Distribution Analysis
# =============================================================================

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle('Trade Distribution Analysis', fontsize=14, fontweight='bold')

if not trades_df.empty:
    # Chart 1: PnL Distribution
    ax7 = axes3[0]
    ax7.hist(trades_df['net_pnl'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax7.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax7.axvline(x=trades_df['net_pnl'].mean(), color='green', linestyle='--', linewidth=2,
                label=f'Mean: ${trades_df["net_pnl"].mean():.2f}')
    ax7.set_title('Trade PnL Distribution', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Net PnL per Trade ($)')
    ax7.set_ylabel('Frequency')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Chart 2: Hold Duration Distribution
    ax8 = axes3[1]
    # hold_hours is already a column in trades_df
    hold_hours = trades_df['hold_hours'].values
    ax8.hist(hold_hours, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax8.axvline(x=trades_df['hold_hours'].mean(), color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {trades_df["hold_hours"].mean():.1f}h')
    ax8.set_title('Hold Duration Distribution', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Hold Duration (hours)')
    ax8.set_ylabel('Frequency')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Chart 3: Win/Loss by Direction
    ax9 = axes3[2]
    direction_stats = trades_df.groupby('direction').agg({
        'net_pnl': ['sum', 'count', lambda x: (x > 0).sum()]
    }).round(2)
    direction_stats.columns = ['Total PnL', 'Trades', 'Wins']
    direction_stats['Win Rate'] = (direction_stats['Wins'] / direction_stats['Trades'] * 100).round(1)
    
    x = range(len(direction_stats.index))
    width = 0.35
    bars1 = ax9.bar([i - width/2 for i in x], direction_stats['Total PnL'], width, 
                    label='Total PnL', color=['green' if v > 0 else 'red' for v in direction_stats['Total PnL']])
    ax9.set_ylabel('Total PnL ($)')
    ax9.set_xlabel('Direction')
    ax9.set_xticks(x)
    ax9.set_xticklabels(direction_stats.index)
    ax9.set_title('Performance by Direction', fontsize=12, fontweight='bold')
    
    # Add win rate labels
    for i, (idx, row) in enumerate(direction_stats.iterrows()):
        ax9.text(i, row['Total PnL'] + 100, f'{row["Win Rate"]}% WR\n{int(row["Trades"])} trades', 
                ha='center', fontsize=9, fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'trade_distribution.png', dpi=150, bbox_inches='tight')
print(f"Trade distribution saved to: {output_dir / 'trade_distribution.png'}")

# =============================================================================
# CHART 4: BTC Benchmark & Correlation Analysis
# =============================================================================

if len(comparison_df) > 0:
    fig4, axes4 = plt.subplots(2, 2, figsize=(16, 12))
    fig4.suptitle(f'Strategy vs BTC Benchmark Analysis - ${CAPITAL:,} Capital', fontsize=14, fontweight='bold')
    
    comparison_df_plot = comparison_df.reset_index()
    
    # 1. Portfolio Value Comparison
    ax10 = axes4[0, 0]
    strategy_value = CAPITAL + comparison_df_plot['total_pnl']
    ax10.plot(comparison_df_plot['hour'], strategy_value, color='blue', linewidth=1.5, label='Mean Reversion Strategy')
    ax10.plot(comparison_df_plot['hour'], comparison_df_plot['btc_value'], color='orange', linewidth=1.5, label='BTC Buy & Hold')
    ax10.axhline(y=CAPITAL, color='gray', linestyle='--', alpha=0.5, label=f'Starting Capital (${CAPITAL:,})')
    ax10.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Date')
    ax10.set_ylabel('Portfolio Value ($)')
    ax10.legend(loc='upper left')
    ax10.grid(True, alpha=0.3)
    ax10.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax10.tick_params(axis='x', rotation=45)
    
    # 2. Rolling Correlation
    ax11 = axes4[0, 1]
    ax11.plot(comparison_df_plot['hour'], comparison_df_plot['rolling_corr'], color='purple', linewidth=1)
    ax11.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax11.axhline(y=correlation, color='red', linestyle='--', linewidth=2, label=f'Overall Correlation: {correlation:.3f}')
    ax11.axhspan(-0.3, 0.3, alpha=0.1, color='green', label='Weak Correlation Zone')
    ax11.set_title('Rolling 7-Day Correlation with BTC', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Date')
    ax11.set_ylabel('Correlation')
    ax11.set_ylim(-1, 1)
    ax11.legend(loc='upper right')
    ax11.grid(True, alpha=0.3)
    ax11.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax11.tick_params(axis='x', rotation=45)
    
    # 3. Return Distribution Comparison
    ax12 = axes4[1, 0]
    strategy_returns = comparison_df['strategy_daily_return'].dropna()
    btc_returns = comparison_df['btc_daily_return'].dropna()
    ax12.hist(strategy_returns, bins=50, alpha=0.6, color='blue', label='Strategy', density=True)
    ax12.hist(btc_returns, bins=50, alpha=0.6, color='orange', label='BTC', density=True)
    ax12.axvline(x=strategy_returns.mean(), color='blue', linestyle='--', linewidth=2)
    ax12.axvline(x=btc_returns.mean(), color='orange', linestyle='--', linewidth=2)
    ax12.set_title('Hourly Return Distribution', fontsize=12, fontweight='bold')
    ax12.set_xlabel('Hourly Return ($)')
    ax12.set_ylabel('Density')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    # 4. Scatter Plot: Strategy vs BTC Returns
    ax13 = axes4[1, 1]
    ax13.scatter(btc_returns, strategy_returns, alpha=0.1, s=5, color='purple')
    # Add regression line
    z = np.polyfit(btc_returns.dropna(), strategy_returns.dropna()[:len(btc_returns.dropna())], 1)
    p = np.poly1d(z)
    x_line = np.linspace(btc_returns.min(), btc_returns.max(), 100)
    ax13.plot(x_line, p(x_line), color='red', linewidth=2, label=f'Regression (β={z[0]:.3f})')
    ax13.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax13.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax13.set_title(f'Strategy vs BTC Returns (Correlation: {correlation:.3f})', fontsize=12, fontweight='bold')
    ax13.set_xlabel('BTC Hourly Return ($)')
    ax13.set_ylabel('Strategy Hourly Return ($)')
    ax13.legend()
    ax13.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'btc_comparison.png', dpi=150, bbox_inches='tight')
    print(f"BTC comparison saved to: {output_dir / 'btc_comparison.png'}")

plt.show()

print("\n" + "=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)
