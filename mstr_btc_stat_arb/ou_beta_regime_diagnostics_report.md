# Rolling OU/Beta Regime Diagnostics

- Pair: MSTR vs BTC-USD
- Period: 2022-01-04 to 2026-03-20
- Beta window: 40
- OU lookback bars: 120

## Summary by Regime
- high: bars=200, beta_mean=1.4020, beta_std=0.2384, kappa_mean=0.06824, hl_median=10.06
- mid: bars=331, beta_mean=1.3538, beta_std=0.2304, kappa_mean=0.07409, hl_median=9.34
- low: bars=402, beta_mean=1.1567, beta_std=0.2222, kappa_mean=0.04559, hl_median=18.60
- unknown: bars=123, beta_mean=1.3280, beta_std=0.2573, kappa_mean=0.03675, hl_median=18.80

- Time-series plot: ou_beta_regime_timeseries.png
- Beta distribution: beta_distribution_by_regime.png
- OU kappa distribution: ou_kappa_distribution_by_regime.png
- OU half-life distribution: ou_half_life_distribution_by_regime.png
- Series CSV: ou_beta_regime_diagnostics_series.csv
- JSON: ou_beta_regime_diagnostics_summary.json
