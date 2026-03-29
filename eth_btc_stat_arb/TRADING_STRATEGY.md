# ETH-BTC Statistical Arbitrage Strategy

## Objective
Trade mean reversion between `ETH` and `BTC` on Hyperliquid with a rolling-beta hedge, no-lookahead signal construction, and fee-aware PnL.

## Baseline Configuration
- Interval: `4h`
- Asset leg: `ETH`
- Hedge leg: `BTC`
- Rolling beta window: `60` bars
- Z-score window: `30` bars
- Entry thresholds: `z >= 2` short spread, `z <= -2` long spread
- Exit threshold: `|z| <= 0.5`
- Taker fee: `0.00045`

## Pipeline
1. Build local cache for `1h`, `4h`, `1d` data.
2. Run backtests for `1h`, `4h`, and `1d`.
3. Compare timeframe curves and risk metrics.
4. Run 4h beta-window robustness grid on `[40, 60, 90, 120]`.
5. Run z-threshold calibration and choose robust entry/exit.

## Commands
```bash
cd /Users/leeisaackaiyui/Desktop/backtest/eth_btc_stat_arb

/Users/leeisaackaiyui/Desktop/backtest/.venv/bin/python prepare_price_cache.py --asset ETH --btc BTC --intervals "1h,4h,1d" --start-ms 0 --end-ms $(date +%s000) --cache-dir data --refresh-cache

/Users/leeisaackaiyui/Desktop/backtest/.venv/bin/python rolling_beta_stat_arb.py --asset ETH --btc BTC --interval 1h --start-ms 0 --end-ms $(date +%s000) --beta-window 60 --taker-fee-rate 0.00045 --cache-dir data --no-network
/Users/leeisaackaiyui/Desktop/backtest/.venv/bin/python rolling_beta_stat_arb.py --asset ETH --btc BTC --interval 4h --start-ms 0 --end-ms $(date +%s000) --beta-window 60 --taker-fee-rate 0.00045 --cache-dir data --no-network
/Users/leeisaackaiyui/Desktop/backtest/.venv/bin/python rolling_beta_stat_arb.py --asset ETH --btc BTC --interval 1d --start-ms 0 --end-ms $(date +%s000) --beta-window 60 --taker-fee-rate 0.00045 --cache-dir data --no-network

/Users/leeisaackaiyui/Desktop/backtest/.venv/bin/python compare_timeframes_fee.py --asset ETH --btc BTC

/Users/leeisaackaiyui/Desktop/backtest/.venv/bin/python grid_4h_beta_windows.py --asset ETH --btc BTC
```

## Expected Outputs
- `eth_1h_rolling_beta_metrics.json`, `eth_4h_rolling_beta_metrics.json`, `eth_1d_rolling_beta_metrics.json`
- `eth_1d_vs_4h_evaluation.md`
- `eth_1d_vs_4h_fee_comparison.png`
- `grid_4h_beta_windows_oos.md`
- `grid_4h_beta_windows_oos.csv`
