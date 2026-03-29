# OU Kappa SD Hypothesis Evaluation

Hypothesis tested:
Lower OU kappa variability (standard deviation) is the reason a regime performs better.

## Data and method
- Pair: MSTR vs BTC-USD (Yahoo)
- Strategy returns: from `optimize_oos_20_regime.py` best-parameter baseline
- OU kappa: from rolling diagnostics series in `ou_beta_regime_diagnostics_series.csv`
- Compared:
  - Regime-level kappa SD vs regime strategy performance
  - Monthly relationship between kappa SD and next-month return

## Key results

### 1) Regime-level comparison (OOS 2024+)
- high: kappa_sd=0.0311, ret=+5.54%, Sharpe=0.484
- mid: kappa_sd=0.0475, ret=-3.43%, Sharpe=0.086
- low: kappa_sd=0.0420, ret=-18.14%, Sharpe=-1.037

Interpretation:
- This slice supports a directional story where lower kappa SD (high regime) aligns with better performance.
- But it is not sufficient as a sole explanation because:
  - In full-sample, low regime has low kappa SD too (0.0351) but still underperforms (ret=-6.72%).
  - Mid regime has highest kappa SD in both full and OOS, but performance is not consistently worst in full sample.

### 2) Time-series test (monthly)
- corr(kappa_sd, next_month_ret), FULL: +0.0138 (near zero)
- corr(kappa_sd, next_month_ret), OOS 2024+: +0.0124 (near zero)

Interpretation:
- Predictive relationship between kappa SD and next-month return is effectively zero.
- If kappa SD were a dominant driver, this correlation should be materially negative (or at least non-trivial in magnitude).

## Conclusion
- Current evidence does not support "low OU kappa SD" as the main reason for better strategy performance.
- It may be a contextual marker in some windows/regimes, but not a robust standalone driver.
- More likely: performance depends on a combination of factors (mean-reversion strength level, signal quality, and regime-specific spread behavior), not only kappa stability.

## Produced artifacts
- `ou_kappa_sd_hypothesis_regime_table.csv`
- `ou_kappa_sd_hypothesis_monthly_table.csv`
- `ou_kappa_sd_hypothesis_bucket_table.csv`
- `ou_kappa_sd_hypothesis_summary.json`
