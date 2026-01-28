# Quick Start Guide - Extreme Funding Rate Strategy

## Overview
This project analyzes extreme funding rates on **Hyperliquid MAINNET** and backtests a mean-reversion strategy that buys coins with the most negative funding rates.

**Note:** Only mainnet data is used (no testnet).

## Quick Start (3 Steps)

### 1. Install Dependencies
```bash
cd /workspaces/backtest/extreme_funding_rate
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python run_all.py
```

This will automatically:
- Fetch 30 days of funding rate data from Hyperliquid (all pairs)
- Identify extreme funding events per hour
- Backtest the trading strategy
- Generate performance reports and visualizations

### 3. Review Results
Check these files:
- `backtest_results.png` - Visual charts
- `backtest_metrics.csv` - Performance summary
- `backtest_trades.csv` - All trade details

## Individual Scripts

If you prefer to run steps individually:

### Step 1: Fetch Data
```bash
python fetch_funding_data.py
```
Creates: `funding_history.csv`

### Step 2: Analyze Extremes
```bash
python analyze_extreme_funding.py
```
Creates: `extreme_funding_events.csv`, `extreme_funding_performance.csv`

### Step 3: Backtest Strategy
```bash
python backtest_strategy.py
```
Creates: `backtest_trades.csv`, `backtest_metrics.csv`, `backtest_results.png`

## Strategy Details

**Entry Rule:** Every hour, buy the coin with the most negative funding rate
**Exit Rule:** Close position after exactly 1 hour
**Position Size:** 100% of capital
**Transaction Cost:** 0.05% per trade

## Customization

Edit parameters in `backtest_strategy.py`:
```python
backtest = ExtremeFundingBacktest(
    initial_capital=10000,      # Starting capital
    position_size=1.0,          # 100% of capital
    transaction_cost=0.0005     # 0.05% fees
)
```

## Expected Runtime

- **Fetch Data**: 5-10 minutes (depends on API)
- **Analyze Extremes**: < 1 minute
- **Backtest**: 10-30 minutes (fetches price data for each trade)

## Troubleshooting

**Issue:** API rate limit errors
**Solution:** The scripts include delays. If errors persist, increase sleep times in the code.

**Issue:** No data returned
**Solution:** Check your internet connection and Hyperliquid API status.

**Issue:** Import errors
**Solution:** Ensure all dependencies are installed: `pip install -r requirements.txt`

## Next Steps

After reviewing results:
1. Analyze which coins perform best
2. Experiment with different parameters
3. Try filtering by funding rate threshold
4. Test multiple position strategy
5. Add risk management rules

Refer to the full [README.md](README.md) for detailed documentation.
