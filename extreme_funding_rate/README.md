# Extreme Funding Rate Strategy Analysis & Backtest

This project analyzes extreme funding rates on **Hyperliquid MAINNET** and backtests a mean-reversion trading strategy based on negative funding rate extremes.

**Note:** This project exclusively uses Hyperliquid mainnet data. Testnet data is not included.

## Strategy Overview

**Hypothesis:** Coins with extremely negative funding rates represent oversold conditions, and prices tend to revert higher in the subsequent hour.

**Strategy:** 
- Every hour, identify the coin with the most negative funding rate
- Enter a long position on that coin
- Hold for exactly 1 hour
- Close the position and repeat

## Project Structure

```
extreme_funding_rate/
├── fetch_funding_data.py          # Step 1: Fetch hourly funding rate data from Hyperliquid
├── analyze_extreme_funding.py     # Step 2-4: Identify extremes and analyze performance
├── backtest_strategy.py           # Step 5: Backtest the trading strategy
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Fetch Funding Rate Data

Fetch historical hourly funding rates for all trading pairs on Hyperliquid:

```bash
python fetch_funding_data.py
```

**Output:** `funding_history.csv` containing timestamp, coin, funding_rate, and premium data.

**Note:** This script fetches 30 days of historical data by default. Adjust the `days_back` parameter in the script to fetch more or less data.

### Step 2-4: Analyze Extreme Funding Events

Identify extreme funding rates per hour and optionally calculate performance:

```bash
python analyze_extreme_funding.py
```

**Outputs:**
- `extreme_funding_events.csv` - Top 5 most positive and negative funding rates per hour
- `extreme_funding_performance.csv` - Performance metrics 1 hour after extreme events (if calculated)

**Features:**
- Identifies top 5 most negative and positive funding rates each hour
- Ranks extremes by severity
- Calculates 1-hour forward returns (optional)
- Provides statistical analysis of performance by extreme type

### Step 5: Backtest the Strategy

Run the complete backtest of the extreme negative funding strategy:

```bash
python backtest_strategy.py
```

**Outputs:**
- `backtest_trades.csv` - Detailed record of every trade
- `backtest_metrics.csv` - Summary performance metrics
- `backtest_results.png` - Comprehensive visualization of results

**Backtest Configuration:**
- Initial Capital: $10,000
- Position Size: 100% of capital per trade
- Transaction Costs: 0.05% per trade (entry + exit)
- Holding Period: 1 hour

## Key Metrics

The backtest calculates the following performance metrics:

- **Total Return %**: Overall portfolio return
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return metric
- **Average PnL per Trade**: Mean profit/loss per trade

## Results Visualization

The backtest generates 6 plots:

1. **Equity Curve** - Capital over time
2. **PnL Distribution** - Histogram of trade outcomes
3. **Cumulative PnL** - Running total of profits/losses
4. **Return % Distribution** - Histogram of percentage returns
5. **Funding Rate vs Return** - Scatter plot showing relationship
6. **Win Rate by Hour** - Performance by time of day

## Data Files

| File | Description |
|------|-------------|
| `hyperliquid_mainnet_funding.csv` | Current mainnet funding snapshot (reference) |
| `funding_history.csv` | Historical funding rate data from Hyperliquid mainnet |
| `extreme_funding_events.csv` | Top 5 extremes per hour |
| `extreme_funding_performance.csv` | 1-hour forward performance |
| `backtest_trades.csv` | All executed trades with details |
| `backtest_metrics.csv` | Summary statistics |
| `backtest_results.png` | Visualization charts |

## API Information

This project uses the **Hyperliquid MAINNET public API**:

**Endpoints:**
- `POST https://api.hyperliquid.xyz/info` with `type: "meta"` - Get all trading pairs (mainnet)
- `POST https://api.hyperliquid.xyz/info` with `type: "fundingHistory"` - Get funding history (mainnet)
- `POST https://api.hyperliquid.xyz/info` with `type: "candleSnapshot"` - Get OHLC data (mainnet)

**Environment:** Mainnet only (not testnet)

**Rate Limits:** The scripts include sleep delays to respect API rate limits.

## Important Notes

1. **Historical Data Availability**: Hyperliquid's funding rate history may be limited. The default setting fetches 30 days.

2. **Price Data**: The backtest fetches hourly candle data for each trade. This can be slow for large datasets.

3. **Transaction Costs**: The default 0.05% transaction cost is an estimate. Adjust based on your actual trading fees.

4. **Slippage**: This backtest does not account for slippage, which can be significant during extreme funding events.

5. **Market Conditions**: Past performance does not guarantee future results. Extreme funding rates may behave differently in various market regimes.

## Customization

You can modify the following parameters:

**In `backtest_strategy.py`:**
- `initial_capital`: Starting capital (default: $10,000)
- `position_size`: Fraction of capital per trade (default: 1.0 = 100%)
- `transaction_cost`: Trading fees (default: 0.0005 = 0.05%)

**In `fetch_funding_data.py`:**
- `days_back`: Historical data window (default: 30 days)

**In `analyze_extreme_funding.py`:**
- Number of extreme events per hour (default: top 5 positive + top 5 negative)
- `hours_forward`: Performance measurement window (default: 1 hour)

## Future Enhancements

Potential improvements to consider:

1. **Multiple Position Strategy**: Hold positions in top N most negative funding rate coins
2. **Dynamic Position Sizing**: Allocate more capital to stronger signals
3. **Filtering**: Only trade when funding rate exceeds a threshold
4. **Stop Loss / Take Profit**: Add risk management rules
5. **Funding Rate Normalization**: Account for typical funding rate ranges per coin
6. **Volume Filters**: Only trade coins with sufficient liquidity
7. **Time-of-Day Filters**: Trade only during high-probability hours

## License

This project is for educational and research purposes only. Use at your own risk.

## Disclaimer

**This is not financial advice.** Trading cryptocurrencies carries substantial risk. This backtest uses historical data and does not account for all real-world trading conditions. Always do your own research and never risk more than you can afford to lose.
