# MAG7 vs Hyperliquid Basis Arb — Backtest Results

**Parameters**: z_window=20, entry_z=2.0, exit_z=0.5, fee=4.5bps taker


## Per-Ticker Results

| Ticker | Bars | Trades | Total Return | Ann. Sharpe | Max Drawdown | Win Rate |
|--------|------|--------|-------------|-------------|--------------|----------|
| AAPL | 86 | 12 | +4.5% | +2.99 | -0.9% | 14% |
| AMZN | 89 | 10 | +15.0% | +2.23 | -1.1% | 12% |
| GOOGL | 89 | 12 | +7.6% | +2.78 | -1.2% | 12% |
| META | 87 | 9 | +11.3% | +2.08 | -0.9% | 8% |
| MSFT | 88 | 9 | +15.2% | +2.60 | -0.3% | 12% |
| NVDA | 93 | 10 | +8.1% | +2.82 | -1.0% | 9% |
| TSLA | 92 | 11 | +5.0% | +1.56 | -2.3% | 10% |

## Portfolio Summary

| Total Return | Ann. Sharpe | Max Drawdown |
|-------------|-------------|--------------|
| +8.5% | +3.98 | -0.5% |

## Strategy Notes

- **Basis** = HL perp close / Yahoo Finance close − 1
- **Signal**: Rolling z-score of basis. Short HL when rich, long HL when cheap.
- **Only HL leg traded** (not executing actual stock buys/sells).
- **Fee**: Taker fee on each side of trade (entry + exit).
- **Limitation**: HL stock perps were launched ~Sept 2024; limited history.