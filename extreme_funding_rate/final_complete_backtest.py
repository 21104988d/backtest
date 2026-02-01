"""
COMPLETE BACKTEST WITH:
  ✅ Hourly funding payments
  ✅ Dynamic delta hedging (hedge current market value, not initial notional)
  ✅ Trading fees
  ✅ Price PnL

Compares STATIC vs DYNAMIC hedging
"""
import pandas as pd
import numpy as np
import gzip
from datetime import datetime

print("=" * 80)
print("COMPLETE BACKTEST: HOURLY FUNDING + DYNAMIC DELTA HEDGE")
print("=" * 80)
print(f"Run time: {datetime.now()}")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

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
print(f"  Unique coins: {data['coin'].nunique()}")
print(f"  Date range: {data['hour'].min()} to {data['hour'].max()}")

# =============================================================================
# PARAMETERS
# =============================================================================
POSITION_SIZE = 100        # Initial position size
FEE_RATE = 0.00045         # 0.045% taker fee
STARTING_CAPITAL = 1000
ENTRY_LOW = 0.000014       # |0.0014%|
ENTRY_HIGH = 0.000015      # |0.0015%|
EXIT_THRESHOLD = 0.000003  # |0.0003%|

print(f"""
PARAMETERS:
  Position Size: ${POSITION_SIZE}
  Fee Rate: {FEE_RATE*100:.3f}%
  Entry: {ENTRY_LOW*100:.4f}% < |FR| < {ENTRY_HIGH*100:.4f}%
  Exit: |FR| <= {EXIT_THRESHOLD*100:.4f}%
""")

# =============================================================================
# BACKTEST FUNCTION
# =============================================================================
def run_backtest(mode="both", hedge_type="dynamic"):
    """
    mode: 'short_only' or 'both'
    hedge_type: 'static' (hedge initial notional) or 'dynamic' (hedge current value)
    """
    positions = {}  # coin -> {entry_hour, entry_price, direction, current_price}
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
        
        # ─────────────────────────────────────────────────────────────────
        # UPDATE CURRENT PRICES FOR ALL POSITIONS
        # ─────────────────────────────────────────────────────────────────
        for coin in list(positions.keys()):
            current_price = price_lookup.get((hour, coin))
            if current_price is not None:
                positions[coin]["current_price"] = current_price
        
        # ─────────────────────────────────────────────────────────────────
        # COLLECT HOURLY FUNDING FOR ALL OPEN POSITIONS
        # ─────────────────────────────────────────────────────────────────
        for coin, pos in positions.items():
            fr = funding_lookup.get((hour, coin), 0)
            
            # Funding is based on CURRENT notional value
            current_notional = POSITION_SIZE * pos["current_price"] / pos["entry_price"]
            
            if pos["direction"] == "SHORT":
                # SHORT receives FR when FR > 0, pays when FR < 0
                funding = fr * current_notional
            else:  # LONG
                # LONG pays FR when FR > 0, receives when FR < 0
                funding = -fr * current_notional
            
            hour_funding_pnl += funding
            pos["funding_collected"] = pos.get("funding_collected", 0) + funding
        
        total_funding_pnl += hour_funding_pnl
        
        # ─────────────────────────────────────────────────────────────────
        # PROCESS ENTRIES AND EXITS
        # ─────────────────────────────────────────────────────────────────
        for _, row in hour_data.iterrows():
            coin = row["coin"]
            if coin == "BTC":
                continue
            
            fr = row["funding_rate"]
            price = row["price"]
            
            # CHECK EXITS
            if coin in positions:
                pos = positions[coin]
                if abs(fr) <= EXIT_THRESHOLD and abs(fr) <= abs(pos["entry_fr"]):
                    entry_price = pos["entry_price"]
                    
                    if pos["direction"] == "SHORT":
                        price_pnl = (entry_price - price) / entry_price * POSITION_SIZE
                    else:  # LONG
                        price_pnl = (price - entry_price) / entry_price * POSITION_SIZE
                    
                    fees = POSITION_SIZE * FEE_RATE * 2  # Entry + exit
                    
                    trades.append({
                        "coin": coin,
                        "direction": pos["direction"],
                        "entry_hour": pos["entry_hour"],
                        "exit_hour": hour,
                        "hold_hours": (hour - pos["entry_hour"]).total_seconds() / 3600,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "entry_fr": pos["entry_fr"],
                        "exit_fr": fr,
                        "price_pnl": price_pnl,
                        "funding_pnl": pos.get("funding_collected", 0),
                        "fees": fees,
                        "net_pnl": price_pnl + pos.get("funding_collected", 0) - fees,
                    })
                    
                    hour_price_pnl += price_pnl
                    hour_fees += fees
                    del positions[coin]
            
            # CHECK ENTRIES
            if coin not in positions:
                # SHORT: positive FR in range
                if ENTRY_LOW <= fr <= ENTRY_HIGH:
                    positions[coin] = {
                        "entry_hour": hour,
                        "entry_price": price,
                        "current_price": price,
                        "entry_fr": fr,
                        "direction": "SHORT",
                        "funding_collected": 0,
                    }
                    hour_fees += POSITION_SIZE * FEE_RATE  # Entry fee
                
                # LONG: negative FR in range (only in 'both' mode)
                elif mode == "both" and -ENTRY_HIGH <= fr <= -ENTRY_LOW:
                    positions[coin] = {
                        "entry_hour": hour,
                        "entry_price": price,
                        "current_price": price,
                        "entry_fr": fr,
                        "direction": "LONG",
                        "funding_collected": 0,
                    }
                    hour_fees += POSITION_SIZE * FEE_RATE  # Entry fee
        
        total_price_pnl += hour_price_pnl
        total_fees += hour_fees
        
        # ─────────────────────────────────────────────────────────────────
        # DYNAMIC DELTA HEDGE
        # ─────────────────────────────────────────────────────────────────
        # Calculate current exposure
        if hedge_type == "dynamic":
            # Hedge based on CURRENT market value of positions
            short_exposure = 0
            long_exposure = 0
            for coin, pos in positions.items():
                current_notional = POSITION_SIZE * pos["current_price"] / pos["entry_price"]
                if pos["direction"] == "SHORT":
                    short_exposure += current_notional
                else:
                    long_exposure += current_notional
            target_hedge = short_exposure - long_exposure
        else:
            # Static: hedge based on number of positions × initial size
            n_short = sum(1 for p in positions.values() if p["direction"] == "SHORT")
            n_long = sum(1 for p in positions.values() if p["direction"] == "LONG")
            target_hedge = (n_short - n_long) * POSITION_SIZE
        
        # BTC hedge PnL (from previous hour's hedge)
        if btc_hedge_notional != 0 and prev_btc_price is not None:
            btc_return = (btc_price - prev_btc_price) / prev_btc_price
            hour_hedge_pnl = btc_hedge_notional * btc_return
            total_hedge_pnl += hour_hedge_pnl
        
        # Rebalance hedge if needed
        hedge_diff = abs(target_hedge - btc_hedge_notional)
        if hedge_diff > 10:  # Rebalance if diff > $10
            rebalance_fee = hedge_diff * FEE_RATE
            hour_fees += rebalance_fee
            total_fees += rebalance_fee
            btc_hedge_notional = target_hedge
        
        prev_btc_price = btc_price
        
        # ─────────────────────────────────────────────────────────────────
        # UPDATE EQUITY
        # ─────────────────────────────────────────────────────────────────
        equity += hour_price_pnl + hour_hedge_pnl + hour_funding_pnl - hour_fees
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate current exposure for logging
        n_short = sum(1 for p in positions.values() if p["direction"] == "SHORT")
        n_long = sum(1 for p in positions.values() if p["direction"] == "LONG")
        
        hourly_records.append({
            "hour": hour,
            "equity": equity,
            "n_short": n_short,
            "n_long": n_long,
            "btc_hedge": btc_hedge_notional,
            "target_hedge": target_hedge,
            "funding_pnl": hour_funding_pnl,
            "price_pnl": hour_price_pnl,
            "hedge_pnl": hour_hedge_pnl,
        })
    
    trades_df = pd.DataFrame(trades)
    hourly_df = pd.DataFrame(hourly_records)
    
    # Calculate metrics
    total_pnl = equity - STARTING_CAPITAL
    hourly_returns = hourly_df["equity"].pct_change().dropna()
    sharpe = hourly_returns.mean() / hourly_returns.std() * np.sqrt(24 * 365) if hourly_returns.std() > 0 else 0
    
    return {
        "mode": mode,
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
print("=" * 80)
print("RUNNING BACKTESTS")
print("=" * 80)

print("\n1. BOTH directions + STATIC hedge...")
static = run_backtest("both", "static")
print(f"   → PnL: ${static['total_pnl']:,.0f}")

print("\n2. BOTH directions + DYNAMIC hedge...")
dynamic = run_backtest("both", "dynamic")
print(f"   → PnL: ${dynamic['total_pnl']:,.0f}")

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS COMPARISON: STATIC vs DYNAMIC HEDGING")
print("=" * 80)

print(f"""
                            STATIC HEDGE       DYNAMIC HEDGE
                            (Initial $100)     (Current Value)
────────────────────────────────────────────────────────────────
  Price PnL:                ${static['price_pnl']:>10,.0f}         ${dynamic['price_pnl']:>10,.0f}
  Hedge PnL:                ${static['hedge_pnl']:>10,.0f}         ${dynamic['hedge_pnl']:>10,.0f}
  Funding PnL:              ${static['funding_pnl']:>10,.0f}         ${dynamic['funding_pnl']:>10,.0f}
  Total Fees:               ${static['total_fees']:>10,.0f}         ${dynamic['total_fees']:>10,.0f}
  ────────────────────────────────────────────────────────────
  NET PnL:                  ${static['total_pnl']:>10,.0f}         ${dynamic['total_pnl']:>10,.0f}
  
  Max Drawdown:             {static['max_dd']:>10.1f}%         {dynamic['max_dd']:>10.1f}%
  Sharpe Ratio:             {static['sharpe']:>10.2f}          {dynamic['sharpe']:>10.2f}
  Trades:                   {static['n_trades']:>10}          {dynamic['n_trades']:>10}
""")

# =============================================================================
# EXPLAIN THE HEDGING LOGIC
# =============================================================================
print("=" * 80)
print("HEDGING LOGIC EXPLAINED")
print("=" * 80)

print("""
EXAMPLE: Your question about coin A and B

TIME 0:
  - SHORT $100 coin A (entry price = $10)
  - BTC hedge = $100 LONG

TIME 1: Coin A drops 10% (price = $9)
  - SHORT position current value = $100 × ($9/$10) = $90
  
  STATIC:  Keep BTC hedge at $100 (based on initial notional)
  DYNAMIC: Reduce BTC hedge to $90 (match current exposure)

TIME 2: Add SHORT $100 coin B
  - Coin A exposure = $90 (current value)
  - Coin B exposure = $100 (new position)
  
  STATIC:  BTC hedge = 2 × $100 = $200
  DYNAMIC: BTC hedge = $90 + $100 = $190

DYNAMIC hedging:
  ✓ More accurate delta neutral
  ✓ Less basis risk
  ✗ More rebalancing = more fees
  
STATIC hedging:
  ✓ Fewer rebalancing trades
  ✓ Lower fees
  ✗ Can drift from delta neutral
""")

# =============================================================================
# DETAILED BREAKDOWN
# =============================================================================
print("=" * 80)
print("DETAILED PNL BREAKDOWN (DYNAMIC HEDGE)")
print("=" * 80)

d = dynamic
print(f"""
  GROSS PROFITS:
    Price PnL:      ${d['price_pnl']:>10,.2f}  (altcoin mean-reversion)
    Hedge PnL:      ${d['hedge_pnl']:>10,.2f}  (BTC delta hedge)
    Funding PnL:    ${d['funding_pnl']:>10,.2f}  (hourly funding payments)
    ─────────────────────────────────
    Gross:          ${d['price_pnl'] + d['hedge_pnl'] + d['funding_pnl']:>10,.2f}
  
  COSTS:
    Trading Fees:   ${d['total_fees']:>10,.2f}  (entry/exit + hedge rebalancing)
    ─────────────────────────────────
    NET PnL:        ${d['total_pnl']:>10,.2f}

  RETURN ON CAPITAL:
    Starting:       ${STARTING_CAPITAL:>10,.0f}
    Final:          ${STARTING_CAPITAL + d['total_pnl']:>10,.0f}
    Return:         {100 * d['total_pnl'] / STARTING_CAPITAL:>10.0f}%
""")

# Save results
dynamic['trades_df'].to_csv("mean_reversion_results/final_trades.csv", index=False)
dynamic['hourly_df'].to_csv("mean_reversion_results/final_hourly.csv", index=False)
print("\n✅ Saved: mean_reversion_results/final_trades.csv")
print("✅ Saved: mean_reversion_results/final_hourly.csv")
