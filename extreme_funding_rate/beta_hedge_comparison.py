"""
BETA HEDGING vs DELTA HEDGING COMPARISON

Delta Hedge: 1:1 hedge ratio ($100 BTC per $100 position)
Beta Hedge:  Use each coin's beta to BTC to determine hedge ratio
"""
import pandas as pd
import numpy as np
import gzip
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt

print("=" * 80)
print("BETA HEDGE vs DELTA HEDGE COMPARISON")
print("=" * 80)
print(f"Run time: {datetime.now()}")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading data...")

with gzip.open("funding_history.csv.gz", "rt") as f:
    funding_raw = pd.read_csv(f)
with gzip.open("price_history.csv.gz", "rt") as f:
    prices_raw = pd.read_csv(f)

funding_raw["timestamp"] = pd.to_datetime(funding_raw["timestamp"], unit="ms", utc=True)
prices_raw["timestamp"] = pd.to_datetime(prices_raw["timestamp"], utc=True)
funding_raw["hour"] = funding_raw["timestamp"].dt.floor("h")
prices_raw["hour"] = prices_raw["timestamp"].dt.floor("h")

prices = prices_raw.groupby(["hour", "coin"])["price"].last().reset_index()
funding = funding_raw.groupby(["hour", "coin"])["funding_rate"].last().reset_index()
data = funding.merge(prices, on=["hour", "coin"], how="inner")
data = data.sort_values(["coin", "hour"]).reset_index(drop=True)

# Create lookups
price_lookup = {(row['hour'], row['coin']): row['price'] for _, row in prices.iterrows()}
funding_lookup = {(row['hour'], row['coin']): row['funding_rate'] for _, row in data.iterrows()}
btc_prices = prices[prices["coin"] == "BTC"].set_index("hour")["price"].to_dict()

print(f"  Data points: {len(data):,}")

# =============================================================================
# CALCULATE BETA FOR EACH COIN
# =============================================================================
print("\nCalculating beta for each coin...")

# Get BTC returns
btc_df = prices[prices['coin'] == 'BTC'].copy()
btc_df = btc_df.sort_values('hour')
btc_df['btc_return'] = btc_df['price'].pct_change()
btc_returns = btc_df.set_index('hour')['btc_return'].to_dict()

# Calculate rolling beta for each coin (using 30-day window)
BETA_WINDOW = 24 * 30  # 30 days of hourly data

coin_betas = {}  # {(coin, hour): beta}
coins = data['coin'].unique()

for coin in coins:
    if coin == 'BTC':
        continue
    
    coin_df = prices[prices['coin'] == coin].copy()
    coin_df = coin_df.sort_values('hour')
    coin_df['return'] = coin_df['price'].pct_change()
    coin_df['btc_return'] = coin_df['hour'].map(btc_returns)
    coin_df = coin_df.dropna()
    
    # Calculate rolling beta
    for i in range(BETA_WINDOW, len(coin_df)):
        window_data = coin_df.iloc[i-BETA_WINDOW:i]
        X = window_data['btc_return'].values
        Y = window_data['return'].values
        
        if len(X) > 10 and np.std(X) > 0:
            slope, _, _, _, _ = stats.linregress(X, Y)
            hour = coin_df.iloc[i]['hour']
            coin_betas[(coin, hour)] = slope

print(f"  Calculated {len(coin_betas):,} coin-hour beta values")

# Default beta if not calculated
DEFAULT_BETA = 1.0

# Analyze beta distribution
beta_values = list(coin_betas.values())
print(f"\n  Beta Distribution:")
print(f"    Mean:   {np.mean(beta_values):.3f}")
print(f"    Median: {np.median(beta_values):.3f}")
print(f"    Std:    {np.std(beta_values):.3f}")
print(f"    Min:    {np.min(beta_values):.3f}")
print(f"    Max:    {np.max(beta_values):.3f}")

# =============================================================================
# PARAMETERS
# =============================================================================
POSITION_SIZE = 100
FEE_RATE = 0.00045
STARTING_CAPITAL = 1000
ENTRY_LOW = 0.000014
ENTRY_HIGH = 0.000015
EXIT_THRESHOLD = 0.000003

# =============================================================================
# BACKTEST FUNCTION
# =============================================================================
def run_backtest(hedge_type="delta"):
    """
    hedge_type: 'delta' (1:1) or 'beta' (beta-adjusted)
    """
    positions = {}
    trades = []
    hourly_records = []
    
    equity = STARTING_CAPITAL
    peak_equity = STARTING_CAPITAL
    max_drawdown = 0
    
    btc_hedge_notional = 0
    prev_btc_price = None
    total_hedge_pnl = 0
    total_funding_pnl = 0
    total_price_pnl = 0
    total_fees = 0
    
    hours = sorted(data["hour"].unique())
    
    for i, hour in enumerate(hours):
        hour_data = data[data["hour"] == hour]
        btc_price = btc_prices.get(hour)
        if btc_price is None:
            continue
        
        hour_price_pnl = 0
        hour_funding_pnl = 0
        hour_hedge_pnl = 0
        hour_fees = 0
        
        # Update current prices
        for coin in list(positions.keys()):
            current_price = price_lookup.get((hour, coin))
            if current_price is not None:
                positions[coin]["current_price"] = current_price
        
        # Collect hourly funding
        for coin, pos in positions.items():
            fr = funding_lookup.get((hour, coin), 0)
            current_notional = POSITION_SIZE * pos["current_price"] / pos["entry_price"]
            
            if pos["direction"] == "SHORT":
                funding = fr * current_notional
            else:
                funding = -fr * current_notional
            
            hour_funding_pnl += funding
            pos["funding_collected"] = pos.get("funding_collected", 0) + funding
        
        total_funding_pnl += hour_funding_pnl
        
        # Process entries and exits
        for _, row in hour_data.iterrows():
            coin = row["coin"]
            if coin == "BTC":
                continue
            
            fr = row["funding_rate"]
            price = row["price"]
            
            # Check exits
            if coin in positions:
                pos = positions[coin]
                if abs(fr) <= EXIT_THRESHOLD and abs(fr) <= abs(pos["entry_fr"]):
                    entry_price = pos["entry_price"]
                    
                    if pos["direction"] == "SHORT":
                        price_pnl = (entry_price - price) / entry_price * POSITION_SIZE
                    else:
                        price_pnl = (price - entry_price) / entry_price * POSITION_SIZE
                    
                    fees = POSITION_SIZE * FEE_RATE * 2
                    
                    trades.append({
                        "coin": coin,
                        "direction": pos["direction"],
                        "entry_hour": pos["entry_hour"],
                        "exit_hour": hour,
                        "hold_hours": (hour - pos["entry_hour"]).total_seconds() / 3600,
                        "price_pnl": price_pnl,
                        "funding_pnl": pos.get("funding_collected", 0),
                        "fees": fees,
                        "net_pnl": price_pnl + pos.get("funding_collected", 0) - fees,
                        "beta": pos.get("beta", 1.0),
                    })
                    
                    hour_price_pnl += price_pnl
                    hour_fees += fees
                    del positions[coin]
            
            # Check entries
            if coin not in positions:
                if ENTRY_LOW <= fr <= ENTRY_HIGH:
                    beta = coin_betas.get((coin, hour), DEFAULT_BETA)
                    positions[coin] = {
                        "entry_hour": hour,
                        "entry_price": price,
                        "current_price": price,
                        "entry_fr": fr,
                        "direction": "SHORT",
                        "funding_collected": 0,
                        "beta": beta,
                    }
                    hour_fees += POSITION_SIZE * FEE_RATE
                
                elif -ENTRY_HIGH <= fr <= -ENTRY_LOW:
                    beta = coin_betas.get((coin, hour), DEFAULT_BETA)
                    positions[coin] = {
                        "entry_hour": hour,
                        "entry_price": price,
                        "current_price": price,
                        "entry_fr": fr,
                        "direction": "LONG",
                        "funding_collected": 0,
                        "beta": beta,
                    }
                    hour_fees += POSITION_SIZE * FEE_RATE
        
        total_price_pnl += hour_price_pnl
        total_fees += hour_fees
        
        # Calculate hedge
        if hedge_type == "beta":
            # Beta hedge: sum of (position_value * beta)
            target_hedge = 0
            for coin, pos in positions.items():
                current_notional = POSITION_SIZE * pos["current_price"] / pos["entry_price"]
                beta = pos.get("beta", DEFAULT_BETA)
                
                if pos["direction"] == "SHORT":
                    target_hedge += current_notional * beta
                else:
                    target_hedge -= current_notional * beta
        else:
            # Delta hedge: 1:1 based on current value
            short_exposure = sum(
                POSITION_SIZE * p["current_price"] / p["entry_price"]
                for p in positions.values() if p["direction"] == "SHORT"
            )
            long_exposure = sum(
                POSITION_SIZE * p["current_price"] / p["entry_price"]
                for p in positions.values() if p["direction"] == "LONG"
            )
            target_hedge = short_exposure - long_exposure
        
        # BTC hedge PnL
        if btc_hedge_notional != 0 and prev_btc_price is not None:
            btc_return = (btc_price - prev_btc_price) / prev_btc_price
            hour_hedge_pnl = btc_hedge_notional * btc_return
            total_hedge_pnl += hour_hedge_pnl
        
        # Rebalance hedge
        hedge_diff = abs(target_hedge - btc_hedge_notional)
        if hedge_diff > 10:
            rebalance_fee = hedge_diff * FEE_RATE
            hour_fees += rebalance_fee
            total_fees += rebalance_fee
            btc_hedge_notional = target_hedge
        
        prev_btc_price = btc_price
        
        # Update equity
        equity += hour_price_pnl + hour_hedge_pnl + hour_funding_pnl - hour_fees
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
        
        n_short = sum(1 for p in positions.values() if p["direction"] == "SHORT")
        n_long = sum(1 for p in positions.values() if p["direction"] == "LONG")
        
        hourly_records.append({
            "hour": hour,
            "equity": equity,
            "n_short": n_short,
            "n_long": n_long,
            "btc_hedge": btc_hedge_notional,
            "target_hedge": target_hedge,
        })
    
    trades_df = pd.DataFrame(trades)
    hourly_df = pd.DataFrame(hourly_records)
    
    total_pnl = equity - STARTING_CAPITAL
    hourly_returns = hourly_df["equity"].pct_change().dropna()
    sharpe = hourly_returns.mean() / hourly_returns.std() * np.sqrt(24 * 365) if hourly_returns.std() > 0 else 0
    
    return {
        "hedge_type": hedge_type,
        "total_pnl": total_pnl,
        "price_pnl": total_price_pnl,
        "hedge_pnl": total_hedge_pnl,
        "funding_pnl": total_funding_pnl,
        "total_fees": total_fees,
        "max_dd": max_drawdown,
        "sharpe": sharpe,
        "n_trades": len(trades_df),
        "trades_df": trades_df,
        "hourly_df": hourly_df,
    }

# =============================================================================
# RUN BACKTESTS
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING BACKTESTS")
print("=" * 80)

print("\n1. Delta Hedge (1:1)...")
delta = run_backtest("delta")
print(f"   → PnL: ${delta['total_pnl']:,.0f}")

print("\n2. Beta Hedge (beta-adjusted)...")
beta_result = run_backtest("beta")
print(f"   → PnL: ${beta_result['total_pnl']:,.0f}")

# =============================================================================
# RESULTS COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS COMPARISON: DELTA vs BETA HEDGING")
print("=" * 80)

print(f"""
                            DELTA HEDGE        BETA HEDGE
                            (1:1 ratio)        (β-adjusted)
────────────────────────────────────────────────────────────────
  Price PnL:                ${delta['price_pnl']:>10,.0f}         ${beta_result['price_pnl']:>10,.0f}
  Hedge PnL:                ${delta['hedge_pnl']:>10,.0f}         ${beta_result['hedge_pnl']:>10,.0f}
  Funding PnL:              ${delta['funding_pnl']:>10,.0f}         ${beta_result['funding_pnl']:>10,.0f}
  Total Fees:               ${delta['total_fees']:>10,.0f}         ${beta_result['total_fees']:>10,.0f}
  ────────────────────────────────────────────────────────────
  NET PnL:                  ${delta['total_pnl']:>10,.0f}         ${beta_result['total_pnl']:>10,.0f}
  
  Max Drawdown:             {delta['max_dd']:>10.1f}%         {beta_result['max_dd']:>10.1f}%
  Sharpe Ratio:             {delta['sharpe']:>10.2f}          {beta_result['sharpe']:>10.2f}
  Trades:                   {delta['n_trades']:>10}          {beta_result['n_trades']:>10}
""")

# Improvement
pnl_diff = beta_result['total_pnl'] - delta['total_pnl']
dd_diff = beta_result['max_dd'] - delta['max_dd']
sharpe_diff = beta_result['sharpe'] - delta['sharpe']

print(f"""
BETA HEDGE IMPROVEMENT:
────────────────────────────────────────────────────────────────
  PnL Change:      ${pnl_diff:+,.0f} ({pnl_diff/delta['total_pnl']*100:+.1f}%)
  Max DD Change:   {dd_diff:+.1f}pp
  Sharpe Change:   {sharpe_diff:+.2f}
""")

# =============================================================================
# GENERATE COMPARISON CHARTS
# =============================================================================
print("Generating comparison charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Equity Curves
ax1 = axes[0, 0]
ax1.plot(delta['hourly_df']['hour'], delta['hourly_df']['equity'], 
         label=f'Delta Hedge: ${delta["total_pnl"]:,.0f}', linewidth=1.5, color='blue')
ax1.plot(beta_result['hourly_df']['hour'], beta_result['hourly_df']['equity'], 
         label=f'Beta Hedge: ${beta_result["total_pnl"]:,.0f}', linewidth=1.5, color='green')
ax1.axhline(STARTING_CAPITAL, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('Equity Curves: Delta vs Beta Hedge')
ax1.set_ylabel('Equity ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Hedge Notional Comparison
ax2 = axes[0, 1]
ax2.plot(delta['hourly_df']['hour'], delta['hourly_df']['btc_hedge'], 
         label='Delta Hedge', linewidth=1, alpha=0.7, color='blue')
ax2.plot(beta_result['hourly_df']['hour'], beta_result['hourly_df']['btc_hedge'], 
         label='Beta Hedge', linewidth=1, alpha=0.7, color='green')
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax2.set_title('BTC Hedge Notional Over Time')
ax2.set_ylabel('Hedge Notional ($)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: PnL Breakdown
ax3 = axes[1, 0]
categories = ['Price\nPnL', 'Hedge\nPnL', 'Funding\nPnL', 'Fees', 'Net\nPnL']
delta_vals = [delta['price_pnl'], delta['hedge_pnl'], delta['funding_pnl'], 
              -delta['total_fees'], delta['total_pnl']]
beta_vals = [beta_result['price_pnl'], beta_result['hedge_pnl'], beta_result['funding_pnl'],
             -beta_result['total_fees'], beta_result['total_pnl']]

x = np.arange(len(categories))
width = 0.35

bars1 = ax3.bar(x - width/2, delta_vals, width, label='Delta Hedge', color='blue', alpha=0.7)
bars2 = ax3.bar(x + width/2, beta_vals, width, label='Beta Hedge', color='green', alpha=0.7)

ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.axhline(0, color='gray', linestyle='--')
ax3.set_title('PnL Breakdown Comparison')
ax3.set_ylabel('PnL ($)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Drawdown Comparison
ax4 = axes[1, 1]
delta_peak = delta['hourly_df']['equity'].cummax()
delta_dd = (delta_peak - delta['hourly_df']['equity']) / delta_peak * 100
beta_peak = beta_result['hourly_df']['equity'].cummax()
beta_dd = (beta_peak - beta_result['hourly_df']['equity']) / beta_peak * 100

ax4.fill_between(delta['hourly_df']['hour'], 0, delta_dd, alpha=0.3, color='blue', label='Delta Hedge')
ax4.fill_between(beta_result['hourly_df']['hour'], 0, beta_dd, alpha=0.3, color='green', label='Beta Hedge')
ax4.plot(delta['hourly_df']['hour'], delta_dd, linewidth=0.5, color='blue')
ax4.plot(beta_result['hourly_df']['hour'], beta_dd, linewidth=0.5, color='green')
ax4.set_title('Drawdown Comparison')
ax4.set_ylabel('Drawdown (%)')
ax4.set_ylim(0, max(delta_dd.max(), beta_dd.max()) * 1.1)
ax4.invert_yaxis()
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mean_reversion_results/beta_vs_delta_hedge.png', dpi=150, bbox_inches='tight')
print("✅ Saved: mean_reversion_results/beta_vs_delta_hedge.png")

# =============================================================================
# ANALYZE BETAS USED
# =============================================================================
print("\n" + "=" * 80)
print("BETA ANALYSIS FROM TRADES")
print("=" * 80)

if len(beta_result['trades_df']) > 0:
    trade_betas = beta_result['trades_df']['beta']
    print(f"""
  Betas used in trades:
    Mean:   {trade_betas.mean():.3f}
    Median: {trade_betas.median():.3f}
    Std:    {trade_betas.std():.3f}
    Min:    {trade_betas.min():.3f}
    Max:    {trade_betas.max():.3f}
    
  If mean beta > 1: Altcoins are more volatile than BTC
  If mean beta < 1: Altcoins are less volatile than BTC
""")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("=" * 80)
print("CONCLUSION")
print("=" * 80)

if beta_result['total_pnl'] > delta['total_pnl']:
    winner = "BETA HEDGE"
    improvement = beta_result['total_pnl'] - delta['total_pnl']
else:
    winner = "DELTA HEDGE"
    improvement = delta['total_pnl'] - beta_result['total_pnl']

print(f"""
  WINNER: {winner}
  
  Delta Hedge (1:1):     ${delta['total_pnl']:,.0f} PnL, {delta['max_dd']:.1f}% MDD, {delta['sharpe']:.2f} Sharpe
  Beta Hedge (β-adj):    ${beta_result['total_pnl']:,.0f} PnL, {beta_result['max_dd']:.1f}% MDD, {beta_result['sharpe']:.2f} Sharpe
  
  Key insight: Beta hedging adjusts for each coin's actual correlation with BTC,
  which can be more/less than 1:1. If altcoins have average β > 1, beta hedge
  uses MORE BTC to hedge. If β < 1, it uses LESS.
""")
