# MSTR-BTC Stat-Arb Strategy Specification (For A New Clean Repo)

## Goal

Build a robust, fee-aware statistical arbitrage system that trades mean reversion between `xyz:MSTR` and `BTC` on Hyperliquid without lookahead bias.

## What The AI Should Build

Implement a clean repository with these modules:

- `src/data/fetch_hyperliquid.py`: candle and metadata fetching
- `src/strategy/rolling_beta_stat_arb.py`: core signal and execution logic
- `src/backtest/engine.py`: PnL, fees, drawdown, and performance metrics
- `src/analysis/compare_timeframes.py`: compare `1h`, `4h`, `1d`
- `src/analysis/grid_beta_windows.py`: robustness grid for 4h beta windows
- `reports/`: generated markdown/csv/png artifacts

## Market and Data Requirements

- Exchange: Hyperliquid
- Symbols: `xyz:MSTR` and `BTC`
- Endpoint: `info` API with `candleSnapshot`
- Required intervals: `1h`, `4h`, `1d`
- Align both legs strictly by timestamp intersection
- Drop/flag incomplete bars and stale candles

## Strategy Logic

1. Compute close-to-close returns for both legs.
2. Compute rolling hedge beta using OLS on returns over a past-only window:
  - At bar `t`, estimate beta from `[t-window, t)` only.
3. Compute hedged return:
  - `r_spread(t) = r_mstr(t) - beta(t) * r_btc(t)`
4. Build spread and rolling z-score (`z_window = 30`).
5. Position rules:
  - Enter short spread when `z >= 2`
  - Enter long spread when `z <= -2`
  - Exit any position when `|z| <= 0.5`
6. Apply fee drag on turnover each bar:
  - `fee_cost = abs(pos_t - pos_{t-1}) * taker_fee_rate`

## Baseline Production Parameters

- Timeframe: `4h`
- Beta window: `60` bars (~10 days)
- Z-score window: `30`
- Taker fee: `0.00045`

## Robustness Selection Rule

Run 4h beta-window grid on `[40, 60, 90, 120]` and split strategy returns into `70%` in-sample and `30%` out-of-sample.

Select the production window by maximizing OOS Sharpe with penalties for:

- IS->OOS Sharpe decay
- OOS max drawdown magnitude
- Drawdown instability gap between IS and OOS

Current best from latest run: `60` bars.

## Risk Management Requirements

- Pause new entries when rolling 30-day strategy drawdown breaches `-10%`
- Reduce exposure when rolling leg-correlation falls below `0.5`
- Enforce gross exposure caps and per-trade risk limits
- Reject trading if data alignment/freshness checks fail

## Recalibration Policy

- Recompute rolling beta every new `4h` bar
- Re-run beta-window robustness grid weekly
- Change window only when challenger improves OOS Sharpe without materially worse OOS drawdown

## Required Outputs

- `reports/timeframe_comparison.md`
- `reports/timeframe_comparison.png`
- `reports/grid_4h_beta_windows_oos.csv`
- `reports/grid_4h_beta_windows_oos.md`
- `reports/latest_metrics.json`

## Acceptance Criteria

Implementation is complete only if all checks pass:

- No lookahead in beta or signal generation
- Fee-aware gross/net metrics available
- Reproducible CLI pipeline from clean environment
- 1h/4h/1d comparison generated with charts and markdown
- 4h grid robustness ranking generated for `[40, 60, 90, 120]`
- Explicit final recommendation on production timeframe and beta window
