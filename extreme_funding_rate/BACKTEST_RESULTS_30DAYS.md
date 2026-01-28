# 30-Day Backtest Results - Extreme Negative Funding Strategy

## Executive Summary

**Strategy**: Buy coins with the most extreme negative funding rate each hour, hold for 1 hour

**Period**: December 29, 2025 - January 28, 2026 (30 days)

**Configuration**:
- Initial Capital: $10,000
- Position Size: 100% of capital
- Number of Positions: 1 (trade only the #1 most extreme negative funding coin)
- Transaction Cost: 0.05%
- Holding Period: 1 hour

## ⚠️ Performance Results

### Overall Performance
- **Final Capital**: $1,513.65
- **Total Return**: **-84.86%**
- **Total PnL**: **-$8,486.35**
- **Max Drawdown**: **-87.16%**

### Trade Statistics
- **Total Trades**: 558
- **Winning Trades**: 242 (43.37%)
- **Losing Trades**: 316 (56.63%)
- **Average PnL per Trade**: -$15.21 (-0.30%)

### Win/Loss Analysis
- **Average Win**: $64.40
- **Average Loss**: $-76.18
- **Profit Factor**: 0.65 (indicates losses exceed wins)
- **Sharpe Ratio**: -10.17 (very poor risk-adjusted returns)

## Top Performing Coins (by Total PnL)

| Rank | Coin | Total PnL |
|------|------|-----------|
| 1 | IP | $241.83 |
| 2 | CELO | $184.85 |
| 3 | BABY | $100.78 |
| 4 | MELANIA | $54.16 |
| 5 | FOGO | $54.11 |

## Worst Performing Coins (by Total PnL)

| Rank | Coin | Total PnL |
|------|------|-----------|
| 1 | ZORA | -$57.88 |
| 2 | RESOLV | -$23.77 |
| 3 | VVV | -$23.42 |
| 4 | BLAST | -$16.80 |
| 5 | ATOM | -$11.09 |

## Notable Trades

### Best Trade
- **Coin**: BERA
- **Date**: January 14, 2026 10:00 AM
- **PnL**: **+$512.23 (+8.59%)**

### Worst Trade
- **Coin**: TST
- **Date**: December 31, 2025 05:00 AM
- **PnL**: **-$1,410.92 (-16.88%)**

## Risk Analysis

### Drawdown Profile
- **Peak Equity**: $10,149.55 (+1.50%) - reached early in backtest
- **Trough Equity**: $1,302.92 (-86.97%)
- **Recovery**: Did not recover from drawdown

### Capital Progression
- After 20 hours: $9,119.25 (-8.81%)
- After 100 hours: $5,147.42 (-48.53%)
- After 200 hours: $4,859.31 (-51.41%)
- After 400 hours: $5,699.92 (-43.00%)
- After 600 hours: $1,797.67 (-82.02%)
- Final (719 hours): $1,513.65 (-84.86%)

## Technical Issues Encountered

- **API Rate Limiting**: Multiple 429 errors for coins like BLAST, BABY, 0G, SKR, KAITO, BERA, IP
- **Missing Trades**: Some hours had no trades due to price fetch failures
- **Data Quality**: 112 unique coins in 30-day dataset (out of 228 available)

## Strategy Insights

### Why This Strategy Lost Money

1. **Mean Reversion Assumption Failed**: Coins with extreme negative funding rates often continued falling
2. **High Volatility**: Selected coins had high downside volatility (average loss -$76 vs average win $64)
3. **No Stop Loss**: 100% position size with no risk management led to large losses
4. **Short Hold Time**: 1-hour holding period may not allow sufficient recovery time
5. **Profit Factor < 1**: More money lost per losing trade than gained per winning trade

### Data Anomalies

- **SKR Dominance**: Last 150+ trades heavily concentrated in SKR coin
- **Price Gaps**: Some coins showed extreme price movements (TST -16.88% in 1 hour)
- **API Issues**: Rate limiting caused missing data for several coins

## Recommendations

### Strategy Improvements
1. **Add Stop Loss**: Limit downside to 5-10% per trade
2. **Reduce Position Size**: Use 20-50% of capital instead of 100%
3. **Filter Extreme Moves**: Exclude coins with funding rates below -0.5%
4. **Extend Hold Time**: Try 4-8 hour holding periods for mean reversion
5. **Multiple Positions**: Trade top 3 coins to diversify risk
6. **Volume Filter**: Only trade coins with sufficient liquidity

### Risk Management
1. **Max Drawdown Limit**: Stop trading if equity drops below 50%
2. **Daily Loss Limit**: Halt after 3 consecutive losing trades
3. **Position Sizing**: Use Kelly Criterion or fixed fractional sizing
4. **Coin Whitelist**: Exclude newly listed or low-volume coins

## Files Generated

1. `funding_history.csv` - 55,460 funding rate records across 112 coins
2. `backtest_trades.csv` - 558 individual trade records
3. `backtest_metrics.csv` - Summary statistics
4. `backtest_results.png` - Performance charts
5. `equity_performance.png` - Equity curve visualization

## Conclusion

The extreme negative funding rate strategy **did not perform well** over the 30-day backtest period with a -84.86% total return. The strategy assumption that extreme negative funding rates signal oversold conditions suitable for mean reversion was **not validated** in this dataset.

**Key Takeaway**: Extreme negative funding rates may indicate fundamental issues or strong downtrend momentum rather than temporary oversold conditions. Additional filters, risk management, and position sizing are essential for this strategy to be viable.

---

*Generated: January 28, 2026*
*Backtest Engine: Hyperliquid Mainnet API*
*Python Version: 3.12.1*
