# Backtest Results: Multi-Timeframe Analysis

## Strategy Overview
- **Logic**: Mean Reversion based on Funding Rate.
- **Execution**:
    - **Open**: Long if Funding < 0, Short if Funding > 0.
    - **Add**: Scale in if Funding Rate becomes more extreme.
    - **Close**: Close when Funding Rate reverts to 0.
- **Data**:
    - **Price**: 5-minute OHLC from Deribit (BTC-PERPETUAL).
    - **Funding**: Hourly Funding Rates from Deribit.
    - **Period**: ~30 Days (Nov 24, 2025 - Dec 24, 2025).

## Performance Metrics

| Timeframe | APR | Sharpe | Sortino | Max Drawdown | Trades | Win Rate | Avg PnL |
|-----------|-----|--------|---------|--------------|--------|----------|---------|
| **5min**  | 5.05% | 1.15 | 1.54 | -1.37% | 16 | 43.75% | 0.20 USD |
| **15min** | 7.67% | 1.66 | 2.30 | -1.37% | 16 | 50.00% | 0.34 USD |
| **1h**    | **9.86%** | 2.14 | 3.00 | -1.34% | 16 | 31.25% | **0.46 USD** |
| **4h**    | 3.68% | 2.16 | 2.96 | -0.45% | 13 | 53.85% | 0.23 USD |
| **1d**    | 1.53% | **2.59** | **4.08** | **-0.10%** | 9 | **66.67%** | 0.13 USD |

## Data Limitation Note
**Important**: The Deribit API only provides historical funding rate data at **1-hour intervals**.
- For the **5min** and **15min** backtests, the funding rate is **forward-filled** from the last available hourly update.
- This means the strategy "sees" the same funding rate for the entire hour.
- As a result, the **number of trades (16)** is identical for 5m, 15m, and 1h timeframes, because the entry/exit signals (based on funding rate changes) only occur on the hour.
- The difference in performance is solely due to the **execution price** (e.g., entering at 10:05 vs 10:15 vs 11:00) and the granularity of the equity curve.
- Real-time execution would likely see more intra-hour funding rate fluctuations, potentially leading to more trades and different results.

## Analysis
1.  **Timeframe Sensitivity**:
    - The **1h timeframe** yielded the highest APR (9.86%) and Average PnL per trade.
    - The **5min** and **15min** timeframes reacted faster to funding rate changes (executing ~5-15 mins after the hour change vs 1 hour later for the 1h timeframe). Interestingly, this faster reaction resulted in *lower* returns in this specific period, possibly because waiting for the hourly candle close filtered out some noise or allowed the price to move favorably before entry.
    - **Daily (1d)** timeframe had the lowest drawdown and highest Sharpe/Sortino ratios, but significantly lower total return. It trades much less frequently.

2.  **Risk/Reward**:
    - **Sharpe Ratio** generally increases with timeframe, indicating better risk-adjusted returns for slower strategies (likely due to lower volatility/drawdown).
    - **Max Drawdown** is highest in lower timeframes (~1.37%), reflecting the higher volatility of 5m/15m equity curves.

3.  **Trade Frequency**:
    - 5m, 15m, and 1h all had 16 trades. This confirms that the trading signals are driven by the *hourly* funding rate updates. The difference in performance is purely due to the *timing* of execution (5 mins past hour vs 60 mins past hour) and the granularity of the equity curve.

## Conclusion
While the **1h timeframe** performed best in terms of raw return, the **1d timeframe** offers the best risk-adjusted return (Sharpe 2.59). The **5min/15min** versions suffer from "over-reacting" or perhaps entering too early before the mean reversion move begins.

## Files
- `ohlc_5m.csv`: 5-minute price data.
- `funding_rates.csv`: Hourly funding rate data.
- `backtest_engine.py`: Updated engine with multi-timeframe support and advanced metrics.
- `equity_*.png`: Equity curves for each timeframe.
