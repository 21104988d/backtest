# Stock vs vntl:MAG7 Arbitrage Backtest

Uses HL daily closes for 7 stocks and live `vntl:MAG7` daily candles.

**Overlap bars:** 109  |  **Range:** 2025-12-11 -> 2026-03-29  \n
**Params:** beta_window=30, z_window=20, entry_z=2.0, exit_z=0.5, fee=4.50 bps


## Per-stock

| Stock | Trades | Net Return | Ann Sharpe | Max DD |
|---|---:|---:|---:|---:|
| AAPL | 6 | -1.71% | -0.26 | -9.16% |
| MSFT | 10 | -3.48% | -0.64 | -7.94% |
| NVDA | 6 | +0.38% | +0.13 | -6.62% |
| GOOGL | 9 | +5.12% | +1.10 | -3.53% |
| AMZN | 5 | -13.57% | -1.55 | -16.77% |
| META | 8 | +2.16% | +0.37 | -7.90% |
| TSLA | 9 | +13.00% | +2.42 | -3.86% |

## Portfolio

- Net return: **+0.27%**
- Annualized Sharpe: **+0.12**
- Max drawdown: **-4.43%**