# MAG7 vs Hyperliquid Basis Arbitrage

Statistical arbitrage between MAG7 stocks (Yahoo Finance real prices) and their
wrapped equity tokens on Hyperliquid's spot DEX (`xyz:AAPL`, `xyz:MSFT`, etc.).

## Strategy Overview

**Instrument**: MAG7 stocks listed as HIP-1 spot tokens on Hyperliquid  
**Tickers**: AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA  
**HL symbols**: `xyz:AAPL`, `xyz:MSFT`, `xyz:NVDA`, `xyz:GOOGL`, `xyz:AMZN`, `xyz:META`, `xyz:TSLA`

**Signal**: Rolling z-score of `basis = HL_price / Yahoo_price - 1`
- Short HL token when z ≥ 2.0 (HL rich vs real stock)
- Long HL token when z ≤ -2.0 (HL cheap vs real stock)
- Exit when |z| ≤ 0.5

## Key Finding: Margin Mode

> `xyz:` MAG7 tokens are **HIP-1 spot tokens**, NOT perpetual futures.  
> **No leverage, no cross-margin, no funding rates.**  
> You can buy (long) or sell existing holdings. Direct shorting on HL is not possible.

For full arb (both legs): use HL spot for long side + stock broker for short side.

## Backtest Results (Nov 2024 – Mar 2026, daily, z_window=20)

| Ticker | Sharpe | Total Return | Max Drawdown |
|--------|--------|-------------|--------------|
| AAPL   | 2.99   | +4.5%       | -0.9%        |
| AMZN   | 2.23   | +15.0%      | -1.1%        |
| GOOGL  | 2.78   | +7.6%       | -1.2%        |
| META   | 2.08   | +11.3%      | -0.9%        |
| MSFT   | 2.60   | +15.2%      | -0.3%        |
| NVDA   | 2.82   | +8.1%       | -1.0%        |
| TSLA   | 1.56   | +5.0%       | -2.3%        |
| **Portfolio** | **3.98** | **+8.5%** | **-0.5%** |

_Note: Short history (~86-93 bars per ticker). Results are indicative only._

## Files

| Script | Purpose |
|--------|---------|
| `fetch_data.py` | Fetch Yahoo Finance + HL daily data, align, compute basis |
| `check_hl_margin.py` | Check HL cross-margin capabilities for MAG7 tokens |
| `backtest_basis_arb.py` | Run basis arb backtest |
| `stress_test_fees.py` | Fee + bid-ask spread stress test heatmap |
| `run_pipeline.py` | Run full pipeline end-to-end |

## Quick Start

```bash
# Full pipeline (fetch + margin check + backtest + stress test)
python run_pipeline.py --start 2024-09-01

# Individual steps
python fetch_data.py --start 2024-09-01
python check_hl_margin.py
python backtest_basis_arb.py --z-window 20 --entry-z 2.0 --exit-z 0.5 --fee 0.00045
python stress_test_fees.py
```

## Trading Fee Reference

| Fee Type | Rate | Notes |
|----------|------|-------|
| HL spot taker | ~5–10 bps | Varies by volume tier |
| HL spot maker | ~2–5 bps | Limit orders |
| Stock broker commission | ~1–5 bps | Interactive Brokers, Alpaca etc. |
| Stock bid-ask spread | 5–30 bps | Round-trip, varies by stock/session |

## Stress Test Summary

At HL taker + 10 bps spread (typical), all tickers maintain Sharpe > 1.5 except TSLA.  
MSFT and NVDA are most robust to spread costs.

See `data/stress_fee_spread_heatmap.png` for full sensitivity analysis.
