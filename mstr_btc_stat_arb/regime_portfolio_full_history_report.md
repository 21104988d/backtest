# Regime Portfolio Evaluation (Full History)

- Period: 2022-01-04 to 2026-03-20
- Regime logic: leakage-safe rolling volatility quantiles from past data only
- Vol window: 30, quantile lookback: 252, min history: 126
- Strategy params: beta_window=120, z_entry=2.25, z_exit=0.75, z_window=30

## Portfolio Performance
- ret_high_low_only: Return=84.63%, Sharpe=0.7942, MaxDD=-35.98%, Active=18.84%
- ret_baseline: Return=80.79%, Sharpe=0.7237, MaxDD=-35.98%, Active=24.91%
- ret_regime_weighted: Return=55.16%, Sharpe=0.6520, MaxDD=-35.98%, Active=22.35%
- ret_high_only: Return=39.27%, Sharpe=0.5367, MaxDD=-35.98%, Active=7.95%
- ret_low_only: Return=32.57%, Sharpe=0.6476, MaxDD=-26.54%, Active=10.89%

## Regime Contribution (Baseline Returns)
- high: bars=331, mean_ret=0.00135, hit_rate=12.39%, Return=39.27%, Sharpe=0.9585, MaxDD=-35.98%
- mid: bars=222, mean_ret=0.00048, hit_rate=7.21%, Return=7.62%, Sharpe=0.5300, MaxDD=-17.63%
- low: bars=348, mean_ret=0.00093, hit_rate=15.23%, Return=32.57%, Sharpe=1.1283, MaxDD=-26.54%

## Artifacts
- Series CSV: regime_portfolio_full_history_series.csv
- Performance CSV: regime_portfolio_full_history_performance.csv
- Regime stats CSV: regime_portfolio_regime_stats.csv
- Equity chart: regime_portfolio_full_history_equity.png
- Non-normalized price+position chart: regime_portfolio_price_with_positions.png
