# MAG7 Basis Arb — Fee & Spread Stress Test Report

**Strategy params**: z_window=20, entry_z=2.0, exit_z=0.5


## Fee + Spread Grid (Portfolio Avg Sharpe)

| Fee (bps/side) \ Spread RT (bps) | 0 | 5 | 10 | 15 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 2.58 | 2.42 | 2.25 | 2.07 | 1.89 | 1.51 | 1.12 | 0.73 |
| 0.5 | 2.55 | 2.39 | 2.22 | 2.04 | 1.85 | 1.47 | 1.08 | 0.70 |
| 1.0 | 2.52 | 2.35 | 2.18 | 2.00 | 1.82 | 1.43 | 1.04 | 0.66 |
| 2.0 | 2.45 | 2.29 | 2.11 | 1.93 | 1.74 | 1.36 | 0.97 | 0.58 |
| 3.0 | 2.39 | 2.22 | 2.04 | 1.85 | 1.66 | 1.28 | 0.89 | 0.51 |
| 4.5 | 2.29 | 2.11 | 1.93 | 1.74 | 1.55 | 1.16 | 0.77 | 0.40 |
| 6.0 | 2.18 | 2.00 | 1.82 | 1.63 | 1.43 | 1.04 | 0.66 | 0.29 |
| 8.0 | 2.04 | 1.85 | 1.66 | 1.47 | 1.28 | 0.89 | 0.51 | 0.15 |
| 10.0 | 1.89 | 1.70 | 1.51 | 1.32 | 1.12 | 0.73 | 0.36 | 0.01 |

## Interpretation

- Green cells (Sharpe > 1.0): Strategy profitable after this cost level.
- Yellow cells (0–1.0): Marginal profitability.
- Red cells (< 0): Strategy unprofitable at this cost level.

### Trading Costs on Hyperliquid (Reference)
| Fee Type | Rate | Notes |
|----------|------|-------|
| Taker fee (cross-margin) | 4.5 bps | Standard retail tier |
| Maker fee | −1.1 bps | Rebate for limit orders |
| Stock perp bid-ask spread | ~10–30 bps | Varies by liquidity and session |
| Typical round-trip cost (taker) | ~19–39 bps | 2×4.5 fee + spread |

### Breakeven Analysis

- See heatmap for detailed breakeven visualization.