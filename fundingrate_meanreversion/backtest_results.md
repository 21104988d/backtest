# Backtest Results: Funding Rate Mean Reversion

## Strategy Overview
- **Open**: Long if Funding < 0, Short if Funding > 0.
- **Add**: Scale in if Funding Rate becomes more extreme (smaller for Long, larger for Short).
- **Close**: Close when Funding Rate reverts to 0 (or crosses 0).

## Data
- **Source**: Deribit BTC-PERPETUAL
- **Period**: ~30 Days (Nov 22, 2025 - Dec 23, 2025)
- **Resolution**: Hourly data

## Performance by Timeframe

| Timeframe | Return | Max Drawdown | Trades | Win Rate | Max Position Size |
|-----------|--------|--------------|--------|----------|-------------------|
| **1h**    | 1.15%  | -1.34%       | 16     | 50.00%   | 230 USD           |
| **4h**    | 0.39%  | -0.33%       | 13     | 46.15%   | 70 USD            |
| **8h**    | 0.25%  | -0.23%       | 12     | 58.33%   | 50 USD            |
| **1d**    | 0.10%  | -0.26%       | 9      | 55.56%   | 50 USD            |

## Observations
1. **Higher Frequency Works Better**: The 1h timeframe yielded the highest return (1.15%). This suggests that funding rate anomalies are often short-lived and best captured with higher frequency checks.
2. **Scaling In**: The 1h timeframe also had the largest position size (230 USD), indicating it successfully scaled into positions during extreme funding events.
3. **Drawdown**: The 1h strategy had higher drawdown, likely due to holding larger positions during adverse price moves before the funding rate reverted.
4. **Win Rate**: Win rates are around 50%, but the strategy relies on capturing funding payments and mean reversion price moves.

## Files
- `fetch_data.py`: Script to fetch funding rate data from Deribit.
- `backtest_engine.py`: The backtest logic and reporting.
- `equity_*.png`: Equity curve plots for each timeframe.
