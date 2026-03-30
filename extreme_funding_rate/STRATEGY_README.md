# Hyperliquid Funding Rate Mean-Reversion Strategy

## Overview

A collection of backtests that exploit funding-rate dynamics on Hyperliquid
perpetual futures.

### Strategy A — Narrow-Band Mean-Reversion (original)
Enters SHORT when FR sits in a tight positive band (0.0014 %–0.0015 %) and
exits when FR normalises to ≤ 0.0003 %.  BTC delta-hedge keeps the portfolio
market-neutral.

### Strategy B — Bottom-to-Top Extreme Negative FR (new, config-driven)
Each hour, all coins are ranked by funding rate **ascending** (most negative =
bottom → least negative = top).  The `NUM_POSITIONS` coins with the most
extreme negative rates that fall inside the configured threshold range are
selected for a **LONG** entry.

**Key insight** — extreme negative FR signals a heavily-shorted asset:
- LONG positions *receive* funding (shorts pay longs when FR < 0).
- Mean-reversion predicts price will rise as over-extended shorts are covered.

The position is held for `HOLDING_PERIOD_HOURS` and closed regardless of FR
level (time-based exit), with optional stop-loss and take-profit guards.

---

## Strategy A — Final Backtest Results

| Metric | Value |
|--------|-------|
| **Net PnL** | $6,385 (639% return) |
| **Max Drawdown** | 53.1% |
| **Sharpe Ratio** | 1.46 |
| **Total Trades** | 2,406 |
| **Win Rate** | 49.0% |
| **Period** | May 2023 - Jan 2026 |

### PnL Breakdown

| Component | Amount | Description |
|-----------|--------|-------------|
| Price PnL | $4,799 | Altcoin mean-reversion profit |
| Hedge PnL | $1,804 | BTC delta hedge profit |
| Funding PnL | $387 | Hourly funding payments |
| Total Fees | -$605 | Trading + rebalancing fees |
| **Net Total** | **$6,385** | |

### Alpha/Beta Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Beta | 0.786 | Less market exposure than 1:1 |
| Alpha (Annual) | 63.6% | Skill-based return |
| R-squared | 0.188 | Low correlation with BTC |
| Return from Alpha | 57.5% | Majority from skill |
| Return from Beta | 42.5% | From market exposure |

---

## Strategy A — Rules

### Entry Conditions

```
IF 0.0014% < FR < 0.0015%:   → OPEN SHORT (expect FR to drop, price to fall)
IF -0.0015% < FR < -0.0014%: → OPEN LONG (expect FR to rise, price to rise)
```

### Exit Conditions

```
IF |FR| <= 0.0003% AND |FR| <= |entry_FR|:
    → CLOSE POSITION (FR has normalized)
```

### Delta Hedge

```
BTC_HEDGE = (SHORT_exposure - LONG_exposure)

Example:
  - 3 SHORT positions @ $100 = $300 SHORT exposure
  - 1 LONG position @ $100 = $100 LONG exposure
  - BTC_HEDGE = $300 - $100 = $200 LONG BTC
```

**Dynamic Rebalancing**: Hedge is adjusted hourly based on current market value of positions.

### Parameters (hardcoded in `final_complete_backtest.py`)

| Parameter | Value |
|-----------|-------|
| Position Size | $100 per trade |
| Entry Threshold | 0.0014% < \|FR\| < 0.0015% |
| Exit Threshold | \|FR\| ≤ 0.0003% |
| Fee Rate | 0.045% (taker) |
| Funding Interval | Hourly (Hyperliquid) |

---

## Strategy B — Bottom-to-Top Selection Rules

### Selection (each hour)

```
1. Collect all coins with MIN_FUNDING_THRESHOLD ≤ FR ≤ MAX_FUNDING_THRESHOLD
2. Sort ascending: most negative first  (bottom → top)
3. Take the bottom NUM_POSITIONS coins  (most extreme negative)
4. OPEN LONG on each selected coin
```

### Exit Conditions

```
CLOSE LONG after HOLDING_PERIOD_HOURS hours
  (or earlier if STOP_LOSS_PCT / TAKE_PROFIT_PCT triggers)
```

### Parameters (all driven by `.env` / `config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_POSITIONS` | 1 | Max concurrent LONG positions |
| `HOLDING_PERIOD_HOURS` | 1 | Time-based exit (hours) |
| `POSITION_SIZE_FIXED` | $1,000 | USD per position |
| `TRANSACTION_COST` | 0.045% | Taker fee per side |
| `MIN_FUNDING_THRESHOLD` | -0.1% | Minimum FR accepted |
| `MAX_FUNDING_THRESHOLD` | 0 % | Maximum FR accepted (only negative) |
| `STOP_LOSS_PCT` | 0 | Stop-loss (0 = disabled) |
| `TAKE_PROFIT_PCT` | 0 | Take-profit (0 = disabled) |

---

## Key Findings

### 1. Price Mean-Reversion is the Main Profit Driver

The strategy profits primarily from **price movement**, not funding payments:
- Price PnL: $4,799 (75% of gross)
- Funding PnL: $387 (6% of gross)

The funding rate is a **signal** that predicts price direction, not the source of profit.

### 2. Delta Hedge Outperforms Beta Hedge

| Hedge Type | Net PnL | Max DD | Sharpe |
|------------|---------|--------|--------|
| **Delta (1:1)** | **$6,385** | **53.1%** | **1.46** |
| Beta (β-adjusted) | $6,071 | 61.5% | 1.34 |

Beta hedging uses more BTC (avg β=1.45) which leads to over-hedging and higher fees.

### 3. Narrow Entry Threshold is Optimal

Testing different entry ranges showed the tight 0.0014%-0.0015% band performs best:
- Captures "reversion zone" where FR is elevated but not extreme
- Avoids volatile periods where FR can spike further
- Higher Sharpe ratio than wider bands

### 4. Hold Through FR Spikes

When FR spikes above 0.0015% after entry:
- **Exit on spike**: Loses money (wrong direction)
- **Hold through**: More profitable (wait for reversion)

---

## Files

### Core Scripts

| File | Description |
|------|-------------|
| `backtest.py` | **Strategy B** — config-driven, bottom-to-top extreme negative FR |
| `final_complete_backtest.py` | Strategy A — hourly funding + dynamic hedge comparison |
| `generate_charts.py` | Generate all analysis charts |
| `beta_hedge_comparison.py` | Compare delta vs beta hedging |
| `fetch_data.py` | Fetch funding + price data from Hyperliquid |
| `fetch_funding_data.py` | Funding-rate-only fetcher |
| `fetch_price_data.py` | Price-only fetcher |
| `config.py` | Config loader (reads `.env`) |

### Data
| File | Description |
|------|-------------|
| `funding_history.csv.gz` | Historical funding rates |
| `price_history.csv.gz` | Historical prices |

### Results
| File | Description |
|------|-------------|
| `mean_reversion_results/bottom_top_trades.csv` | Strategy B trade details |
| `mean_reversion_results/bottom_top_hourly.csv` | Strategy B hourly equity |
| `mean_reversion_results/final_trades.csv` | Strategy A trade details |
| `mean_reversion_results/final_hourly.csv` | Strategy A hourly equity |
| `mean_reversion_results/final_*.png` | Performance charts |

---

## Running the Strategy

```bash
# 1. Fetch latest data
python fetch_data.py

# 2. Run Strategy B (bottom-to-top, config-driven)
python backtest.py

# 3. Run Strategy A (original narrow-band)
python final_complete_backtest.py

# 4. Generate charts
python generate_charts.py

# 5. Compare hedge types
python beta_hedge_comparison.py
```

---

## Risk Considerations

1. **Max Drawdown**: 53.1% is significant - requires strong risk tolerance
2. **Concentration Risk**: Can hold 80+ concurrent positions
3. **Execution Risk**: Assumes fills at market prices
4. **Funding Rate Changes**: Strategy may underperform if FR dynamics change
5. **Liquidation Risk**: Requires sufficient margin for positions + hedge

---

## Summary

**Strategy A** is a **market-neutral mean-reversion strategy** that:
- Captures price moves predicted by funding rate signals
- Uses BTC delta hedge to reduce market exposure
- Generates 63.6% annualised alpha with 57.5% of returns from skill
- Achieved 639% return over 2.7 years with 1.46 Sharpe ratio

**Strategy B** is a **bottom-to-top extreme negative FR strategy** that:
- Sorts all coins from most extreme negative FR (bottom) to least (top)
- Takes LONG positions on the most extreme negative FR coins each hour
- Collects funding payments as shorts pay longs
- Exits after a fixed holding period (configurable via `.env`)
