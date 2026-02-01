# Funding-Neutral Trend Following Strategy

## Overview

This strategy captures **price alpha** from extreme funding rate trend-following signals while maintaining **funding-neutral exposure** by hedging with opposite positions on low funding rate coins.

---

## Strategy Logic

### Core Principle

| Group | Threshold | Direction | Funding | Purpose |
|-------|-----------|-----------|---------|---------|
| **Alpha** | \|FR\| ≥ 0.0015% | WITH funding (trend follow) | **PAY** | Capture price alpha |
| **Hedge** | \|FR\| < 0.0015% | AGAINST funding | **RECEIVE** | Offset funding cost |

### Funding Rate Mechanics

| Position | FR > 0 (Positive) | FR < 0 (Negative) |
|----------|-------------------|-------------------|
| **LONG** | PAY | RECEIVE |
| **SHORT** | RECEIVE | PAY |

### Position Direction Rules

| Group | Condition | Direction | Funding Impact |
|-------|-----------|-----------|----------------|
| Alpha | FR > +0.0015% | **LONG** | PAY |
| Alpha | FR < -0.0015% | **SHORT** | PAY |
| Hedge | 0 < FR < +0.0015% | **SHORT** | RECEIVE |
| Hedge | -0.0015% < FR < 0 | **LONG** | RECEIVE |

---

## Parameters

```python
ENTRY_THRESHOLD = 0.000015    # |FR| >= 0.0015% for alpha (13.1% APY)
POSITION_SIZE = 100           # $100 USD per position
TAKER_FEE = 0.00045           # 0.045% per trade
```

---

## Algorithm (Per Hour)

### Step 1: Get Current Data

```
For hour H:
- FR[coin] = funding rate at hour H
- Price_H[coin] = price at hour H
- Price_H1[coin] = price at hour H+1
- available_coins = coins with valid FR, Price_H, and Price_H1
```

### Step 2: Determine Target Alpha Set

```
candidate_alpha = {coin | |FR[coin]| >= ENTRY_THRESHOLD}
Sort by |FR| descending (strongest signals first)
```

### Step 3: Determine Hedge Pool

```
hedge_pool = {coin | |FR[coin]| < ENTRY_THRESHOLD}
Sort by |FR| descending (highest hedge value first)
```

### Step 4: Ensure Funding Neutral

```
funding_to_pay = sum(POSITION_SIZE × |FR[c]|) for alpha coins
max_hedge_capacity = sum(POSITION_SIZE × |FR[c]|) for hedge pool

WHILE max_hedge_capacity < funding_to_pay AND alpha not empty:
    # Demote weakest alpha (smallest |FR|, FIFO tie-breaker)
    weakest = alpha coin with min(|FR|, -entry_hour)
    Move weakest from alpha to hedge pool
    Recalculate funding_to_pay and max_hedge_capacity
```

### Step 5: Select Minimum Hedge Coins

```
target_hedge = []
accumulated = 0

FOR coin in hedge_pool (sorted by |FR| desc):
    target_hedge.append(coin)
    accumulated += POSITION_SIZE × |FR[coin]|
    IF accumulated >= funding_to_pay:
        BREAK
```

### Step 6: Close Positions No Longer Needed

```
FOR each existing alpha position:
    IF coin not in target_alpha:
        Close position (pay TAKER_FEE)

FOR each existing hedge position:
    IF coin not in target_hedge:
        Close position (pay TAKER_FEE)
```

### Step 7: Open New Positions

```
FOR coin in target_alpha not in alpha_positions:
    direction = LONG if FR > 0 else SHORT
    Open position (pay TAKER_FEE)

FOR coin in target_hedge not in hedge_positions:
    direction = SHORT if FR > 0 else LONG  # Opposite to receive funding
    Open position (pay TAKER_FEE)
```

### Step 8: Calculate Hourly PnL

```
# Price PnL
FOR each position:
    price_return = (Price_H1 - Price_H) / Price_H
    pnl = POSITION_SIZE × price_return × (1 if LONG else -1)

# Funding PnL (settled at H+1, based on FR at H)
funding_paid = sum(POSITION_SIZE × |FR|) for alpha positions
funding_received = sum(POSITION_SIZE × |FR|) for hedge positions
net_funding = funding_received - funding_paid

# Total
hourly_pnl = alpha_price_pnl + hedge_price_pnl + net_funding - fees
```

---

## Position Tracking

```python
alpha_positions = {
    'BTC': {
        'direction': 'LONG',      # or 'SHORT'
        'entry_price': 50000.0,
        'entry_hour': 1000,       # For FIFO tie-breaker
    },
    ...
}

hedge_positions = {
    'SOL': {
        'direction': 'SHORT',
        'entry_price': 100.0,
        'entry_hour': 1005,
    },
    ...
}
```

---

## Tie-Breaker Rule (FIFO)

When demoting alpha positions and multiple coins have the same |FR|:

```python
priority = (|FR[coin]|, -entry_hour)
# Lower |FR| = demote first
# Same |FR|: older entry (smaller entry_hour) = demote first (FIFO)
```

---

## Edge Cases

| Scenario | Handling |
|----------|----------|
| No coins meet alpha threshold | No positions, PnL = 0 |
| Not enough hedge capacity | Demote weakest alphas until balanced |
| Coin missing price at H+1 | Exclude from available coins |
| Same \|FR\| tie | FIFO (oldest entry demoted first) |
| New coin (no entry_hour) | Treat as newest (entry_hour = H) |

---

## Output Metrics

### Hourly Record

```python
{
    'hour': H,
    'timestamp': datetime,
    
    # Position counts
    'n_alpha': int,
    'n_alpha_long': int,
    'n_alpha_short': int,
    'n_hedge': int,
    'n_hedge_long': int,
    'n_hedge_short': int,
    'n_total': int,              # Concurrent positions
    
    # Funding
    'funding_paid': float,
    'funding_received': float,
    'net_funding': float,
    
    # Price PnL
    'alpha_price_pnl': float,
    'hedge_price_pnl': float,
    
    # Fees & Turnover
    'fees': float,
    'positions_opened': int,
    'positions_closed': int,
    
    # Net
    'hourly_pnl': float,
    'cumulative_pnl': float,
}
```

### Summary Statistics

```
════════════════════════════════════════════════════════════════
FUNDING-NEUTRAL TREND FOLLOWING BACKTEST RESULTS
════════════════════════════════════════════════════════════════

PARAMETERS:
  Entry Threshold: |FR| >= 0.0015% (13.1% APY)
  Position Size: $100 per coin
  Taker Fee: 0.045%

DATA PERIOD:
  Start: YYYY-MM-DD HH:MM
  End: YYYY-MM-DD HH:MM
  Total Hours: XX,XXX

────────────────────────────────────────────────────────────────
CONCURRENT POSITION STATISTICS
────────────────────────────────────────────────────────────────

Alpha Positions:
  Average: X.X per hour
  Min: X
  Max: X
  Distribution: [histogram or percentiles]

Hedge Positions:
  Average: X.X per hour
  Min: X
  Max: X

Total Concurrent Positions:
  Average: X.X per hour
  Min: X
  Max: X
  P50 (Median): X
  P95: X
  P99: X

────────────────────────────────────────────────────────────────
TURNOVER STATISTICS
────────────────────────────────────────────────────────────────

Total Positions Opened: X,XXX
Total Positions Closed: X,XXX
Average Turnover per Hour: X.X
Average Position Holding Time: X.X hours

────────────────────────────────────────────────────────────────
FUNDING ANALYSIS
────────────────────────────────────────────────────────────────

Total Funding Paid (Alpha): $X,XXX.XX
Total Funding Received (Hedge): $X,XXX.XX
Net Funding: $X.XX
Funding Hedge Ratio: XX.X%

────────────────────────────────────────────────────────────────
PRICE PNL ANALYSIS
────────────────────────────────────────────────────────────────

Alpha Price PnL: $X,XXX.XX
Hedge Price PnL: $X,XXX.XX
Total Price PnL: $X,XXX.XX

────────────────────────────────────────────────────────────────
FEE ANALYSIS
────────────────────────────────────────────────────────────────

Total Fees Paid: $X,XXX.XX
Fees as % of Gross PnL: XX.X%

────────────────────────────────────────────────────────────────
FINAL RESULTS
────────────────────────────────────────────────────────────────

                    | Total ($)    | Per Hour ($) | % of Net
--------------------|--------------|--------------|----------
Alpha Price PnL     | +X,XXX.XX    | +X.XXXX      | +XX.X%
Hedge Price PnL     | +/-X,XXX.XX  | +/-X.XXXX    | +/-XX.X%
Net Funding         | ~0.XX        | ~0.XXXX      | ~0.X%
Fees                | -X,XXX.XX    | -X.XXXX      | -XX.X%
--------------------|--------------|--------------|----------
NET PNL             | +X,XXX.XX    | +X.XXXX      | 100%

Sharpe Ratio (hourly): X.XXX
Max Drawdown: $X,XXX.XX (XX.X%)

════════════════════════════════════════════════════════════════
```

---

## Files

```
funding_neutral_strategy/
├── README.md                    # This file
├── backtest.py                  # Main backtest engine
├── results/
│   ├── hourly_records.csv       # Detailed hourly data
│   └── summary.txt              # Summary statistics
```

---

## Expected Results (Hypothesis)

| Component | Expected | Rationale |
|-----------|----------|-----------|
| Alpha Price PnL | **Positive** | Trend following works (proven in previous backtest) |
| Hedge Price PnL | ~Zero | No directional signal, random |
| Net Funding | ~Zero | Hedged by design |
| Fees | **Negative** | Cost of trading |
| **Net PnL** | **Positive** | Alpha price - Fees |

---

## Verification Checks

1. **Funding Neutral**: `|net_funding| / funding_paid < 5%`
2. **No Overlap**: No coin in both alpha and hedge simultaneously
3. **Fee Calculation**: `fees = (opens + closes) × $100 × 0.045%`
4. **Continuous Hours**: No gaps in hourly records
