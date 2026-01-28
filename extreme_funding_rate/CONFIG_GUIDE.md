# Configuration Guide - .env File

## Overview

The `.env` file allows you to customize all backtesting parameters without editing code. Copy `.env.example` to `.env` and modify the values to suit your testing needs.

## Quick Start

```bash
# Copy example configuration
cp .env.example .env

# Edit with your preferred settings
nano .env  # or use any text editor

# Run backtest with your config
python backtest_strategy.py
```

## Configuration Parameters

### 📊 Data Fetching

**DAYS_BACK**
- How many days of historical funding data to fetch
- Default: `7`
- Example: `DAYS_BACK=30` for one month of data
- More data = longer fetch time but more comprehensive backtest

**FETCH_DELAY**
- Delay between API calls (in seconds) to respect rate limits
- Default: `0.2`
- Increase if you encounter rate limit errors

---

### 💰 Capital & Costs

**INITIAL_CAPITAL**
- Starting capital in USD
- Default: `10000`
- Example: `INITIAL_CAPITAL=50000` to start with $50,000

**POSITION_SIZE_PCT**
- Fraction of capital to allocate to trades
- Range: `0.0` to `1.0`
- Default: `1.0` (100% of capital)
- Examples:
  - `0.5` = Use 50% of capital, keep 50% in reserve
  - `1.0` = Use 100% of capital (fully invested)
  - `2.0` = Use 200% (leverage, if available)

**TRANSACTION_COST**
- Trading fees as a fraction
- Default: `0.0005` (0.05%)
- Examples:
  - `0.001` = 0.1% fee
  - `0.0005` = 0.05% fee (typical maker fee)
  - `0.0` = No fees (unrealistic)

---

### 🎯 Strategy Configuration

**NUM_POSITIONS** ⭐ **NEW**
- Number of extreme negative funding pairs to trade simultaneously
- Default: `1` (trade only #1 most extreme)
- Examples:
  - `1` = Trade only the most extreme negative funding coin
  - `3` = Trade top 3 most extreme coins
  - `5` = Trade top 5 most extreme coins

**Impact:**
- `NUM_POSITIONS=1`: 100% capital in 1 coin per hour
- `NUM_POSITIONS=3`: ~33% capital in each of 3 coins per hour (if POSITION_SIZE_PCT=1.0)
- `NUM_POSITIONS=5`: ~20% capital in each of 5 coins per hour

**HOLDING_PERIOD_HOURS**
- How long to hold each position
- Default: `1` (1 hour)
- Examples:
  - `1` = Exit after 1 hour
  - `4` = Exit after 4 hours
  - `24` = Exit after 24 hours (daily rebalance)

---

### ⚠️ Risk Management (Optional)

**STOP_LOSS_PCT**
- Automatic exit if position loses this percentage
- Default: `0` (disabled)
- Examples:
  - `0` = No stop loss
  - `0.02` = Exit if down 2%
  - `0.05` = Exit if down 5%

**TAKE_PROFIT_PCT**
- Automatic exit if position gains this percentage
- Default: `0` (disabled)
- Examples:
  - `0` = No take profit
  - `0.05` = Exit if up 5%
  - `0.10` = Exit if up 10%

---

### 🔍 Filters (Optional)

**MIN_FUNDING_THRESHOLD**
- Only trade if funding rate is more negative than this
- Default: `-0.001` (-0.1%)
- Examples:
  - `-0.001` = Only trade if funding < -0.1%
  - `-0.005` = Only trade if funding < -0.5% (more selective)
  - `-1.0` = Disable filter (trade all negative funding)

**MAX_FUNDING_THRESHOLD**
- Only trade if funding rate is less positive than this
- Default: `0` (only negative funding)
- Examples:
  - `0` = Only trade negative funding rates
  - `-0.0001` = Must be at least -0.01% negative
  - `1.0` = Disable filter

---

## Example Configurations

### Conservative (Low Risk)
```bash
INITIAL_CAPITAL=10000
POSITION_SIZE_PCT=0.5         # Only use 50% of capital
NUM_POSITIONS=3                # Diversify across 3 positions
STOP_LOSS_PCT=0.02             # 2% stop loss
MIN_FUNDING_THRESHOLD=-0.005   # Only very extreme funding
```

### Aggressive (High Risk/Reward)
```bash
INITIAL_CAPITAL=10000
POSITION_SIZE_PCT=1.0          # Use all capital
NUM_POSITIONS=1                # Concentrate in 1 position
STOP_LOSS_PCT=0                # No stop loss
MIN_FUNDING_THRESHOLD=-1.0     # Trade all negative funding
```

### Diversified
```bash
INITIAL_CAPITAL=50000
POSITION_SIZE_PCT=1.0
NUM_POSITIONS=5                # Trade top 5 extremes
HOLDING_PERIOD_HOURS=1
STOP_LOSS_PCT=0.03             # 3% stop loss
TAKE_PROFIT_PCT=0.05           # 5% take profit
```

### Research Mode (More Data)
```bash
DAYS_BACK=30                   # Fetch 30 days
INITIAL_CAPITAL=10000
POSITION_SIZE_PCT=1.0
NUM_POSITIONS=1
TRANSACTION_COST=0             # Zero fees for pure strategy test
```

---

## How Position Size Works

**Formula:**
```
Capital per position = INITIAL_CAPITAL × POSITION_SIZE_PCT / NUM_POSITIONS
```

**Examples:**

1. **Single position (default)**
   ```
   INITIAL_CAPITAL=10000
   POSITION_SIZE_PCT=1.0
   NUM_POSITIONS=1
   → $10,000 in 1 coin
   ```

2. **Three positions**
   ```
   INITIAL_CAPITAL=10000
   POSITION_SIZE_PCT=1.0
   NUM_POSITIONS=3
   → $3,333 in each of 3 coins
   ```

3. **Five positions with 80% allocation**
   ```
   INITIAL_CAPITAL=10000
   POSITION_SIZE_PCT=0.8
   NUM_POSITIONS=5
   → $1,600 in each of 5 coins
   → $2,000 kept in reserve (20%)
   ```

---

## Testing Different Strategies

### Test 1: Single Position
```bash
NUM_POSITIONS=1
python backtest_strategy.py
```

### Test 2: Multiple Positions
```bash
NUM_POSITIONS=3
python backtest_strategy.py
```

### Test 3: Compare Results
```bash
# Run both and compare metrics in backtest_metrics.csv
```

---

## Viewing Results

After each backtest:

1. **View configuration used:**
   ```bash
   python config.py
   ```

2. **Generate equity curve:**
   ```bash
   python plot_equity.py
   ```

3. **Check files:**
   - `backtest_trades.csv` - All trades
   - `backtest_metrics.csv` - Performance summary
   - `equity_performance.png` - Visual equity curve

---

## Tips

1. **Start conservative:** Test with `NUM_POSITIONS=3` for diversification
2. **Adjust incrementally:** Change one parameter at a time
3. **More data:** Increase `DAYS_BACK` for more robust results
4. **Compare:** Run multiple configs and compare results
5. **Document:** Keep notes on which configs work best

---

## Troubleshooting

**Problem:** Backtest runs too slow
- **Solution:** Decrease `DAYS_BACK` or `NUM_POSITIONS`

**Problem:** API rate limit errors
- **Solution:** Increase `FETCH_DELAY`

**Problem:** Not enough trades
- **Solution:** Decrease `MIN_FUNDING_THRESHOLD` or increase `DAYS_BACK`

**Problem:** Too many losing trades
- **Solution:** Increase `MIN_FUNDING_THRESHOLD` (be more selective)

---

## Advanced: Optimization

To find optimal parameters, systematically test different configurations:

```bash
# Test different position counts
for n in 1 2 3 5; do
  sed -i "s/NUM_POSITIONS=.*/NUM_POSITIONS=$n/" .env
  python backtest_strategy.py
  mv backtest_metrics.csv results_${n}_positions.csv
done
```

Then compare the results files to find the best configuration.
