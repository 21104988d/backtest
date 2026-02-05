# Daily Returns Analysis - Mean Reversion Backtest

## Overview

This project backtests mean reversion and trend following strategies on cryptocurrency daily returns using proper OHLC (Open-High-Low-Close) data for accurate stop loss simulation.

## Strategy Description

### Signal Generation (Day T-1)
- Rank all coins by their daily return
- Identify **Top 3 gainers** (biggest winners)
- Identify **Bottom 3 losers** (biggest losers)

### Trading Execution (Day T)

**Mean Reversion Strategy:**
- SHORT the top 3 gainers (bet they will reverse down)
- LONG the bottom 3 losers (bet they will reverse up)

**Trend Following Strategy:**
- LONG the top 3 gainers (bet momentum continues)
- SHORT the bottom 3 losers (bet momentum continues)

### Position Management
- Enter at market **OPEN** price
- Exit at market **CLOSE** price
- Stop loss checked against intraday **HIGH/LOW**:
  - LONG positions: stop if LOW ≤ Open × (1 - SL%)
  - SHORT positions: stop if HIGH ≥ Open × (1 + SL%)

## Key Finding: Why Tighter Stop Loss = Better Performance

This is counterintuitive but makes sense for crypto:

### The Data Shows:
| Stop Loss | Stop Rate | Total Return | Win Rate |
|-----------|-----------|--------------|----------|
| None      | 0%        | -318.7%      | 50.4%    |
| 5.0%      | 36.4%     | +3,075.7%    | 36.0%    |
| 3.0%      | 52.1%     | +3,797.9%    | 30.3%    |
| 2.0%      | 61.8%     | +4,466.7%    | 26.6%    |
| 1.0%      | 73.4%     | +5,238.5%    | 21.6%    |
| 0.5%      | 79.2%     | +5,840.6%    | 18.9%    |

### Why This Works:

1. **Crypto is extremely volatile**: Average intraday drawdown is 3.5-4%
2. **Without stop loss, the strategy loses money**: Mean reversion doesn't work long-term on extreme movers
3. **Stop loss converts a losing strategy into a winning one** by:
   - Capping losses at a small fixed amount (e.g., 0.5%)
   - Letting the ~20% of non-stopped winners run (average +7% gain)

4. **Tighter SL beats looser SL because**:
   - 0.5% SL: Lose 79.2% × 0.5% = 39.6% from stopped positions
   - 1.0% SL: Lose 73.4% × 1.0% = 73.4% from stopped positions
   - The extra 6% positions stopped at 0.5% had avg +4% return (lost opportunity = 0.24%)
   - But saving 34% on stopped losses >> 0.24% missed profit

### Mathematical Explanation

```
Expected Return = (1 - Stop_Rate) × Avg_Winner_Return - Stop_Rate × SL%

At 0.5% SL: 20.8% × 7.0% - 79.2% × 0.5% = 1.46% - 0.40% = +1.06% per trade
At 1.0% SL: 26.6% × 6.5% - 73.4% × 1.0% = 1.73% - 0.73% = +1.00% per trade
```

The tighter stop loss wins because the reduced loss per stopped position more than compensates for the slightly lower win rate.

## Position Sizing Comparison

### Fixed Position ($100 per trade)
- Simple, consistent exposure
- Better during volatile periods
- Lower drawdown

### Dynamic Position (1/6 of capital per trade)
- Compounds gains faster
- Higher peak returns
- But also higher drawdowns

### Results (at 0.5% SL):
| Strategy | Fixed $100 | Dynamic 1/6 |
|----------|------------|-------------|
| Mean Reversion | +532.8% | +573.3% |
| Trend Following | +465.4% | +532.7% |

## Files

| File | Description |
|------|-------------|
| `fetch_ohlc_data.py` | Generates daily OHLC from hourly price data |
| `daily_ohlc.csv` | Daily OHLC data for all coins |
| `mean_reversion_backtest.py` | Main backtest with fixed vs dynamic comparison |
| `stop_loss_analysis.py` | Stop loss sensitivity analysis |
| `verify_stop_loss.py` | Verification of stop loss logic |
| `verify_strategy_positions.py` | Analysis of actual traded positions |

## Configuration

```python
N = 3                    # Number of top/bottom coins to trade
INITIAL_CAPITAL = 1000   # Starting capital ($)
POSITION_SIZE_FIXED = 100 # Fixed position size ($)
POSITION_FRACTION = 1/6  # Dynamic position as fraction of capital
TRADING_FEE = 0.045      # Fee per trade (%)
```

## Data

- **Source**: Binance hourly OHLC data
- **Period**: ~1000 days
- **Coins**: 200+ cryptocurrencies
- **Records**: 101,591 daily OHLC entries

## Usage

```bash
# Generate OHLC data from hourly prices
python fetch_ohlc_data.py

# Run main backtest
python mean_reversion_backtest.py

# Run stop loss analysis
python stop_loss_analysis.py
```

## Key Takeaways

1. **Stop loss is essential** - Without it, both strategies lose money
2. **Tighter is better** for crypto - 0.5% SL outperforms 1%, 2%, 5%
3. **Mean reversion slightly beats trend following** when using tight stop loss
4. **Dynamic sizing compounds better** but with higher risk
5. **~80% of positions hit stop loss** in crypto due to extreme volatility

## Warning

This is a backtest only. Real trading considerations:
- Slippage may prevent getting exact stop loss fills
- Funding rates on perpetual contracts
- Liquidity constraints on smaller coins
- Transaction costs may vary
- Past performance ≠ future results
