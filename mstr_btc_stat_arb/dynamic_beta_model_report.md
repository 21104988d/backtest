# Dynamic Beta Model (Rolling-40 Anchor)

- Pair: MSTR vs BTC-USD
- Period: 2022-01-03 to 2026-03-20
- Anchor rolling beta window: 40
- Evaluated rows: 744

## Overall
- Corr(asset,BTC)=0.7279, Corr(hedged_true,BTC)=0.0273, Corr(hedged_model,BTC)=0.0256
- Corr reduction true=0.7006, model=0.7023
- Beta MAE=0.0630, RMSE=0.0862

## Time-Slope Coefficient Stats
- mean=-0.025697, std=0.029561, p10=-0.071346, p50=-0.015872, p90=0.003752

## Regime Metrics
- high: bars=185, corr_true=0.0789, corr_model=0.0859, beta_mae=0.0656
- mid: bars=183, corr_true=-0.0595, corr_model=-0.0634, beta_mae=0.0605
- low: bars=376, corr_true=0.0139, corr_model=-0.0006, beta_mae=0.0629
- unknown: bars=0, corr_true=nan, corr_model=nan, beta_mae=nan

- Artifacts: dynamic_beta_model_series.csv, dynamic_beta_model_summary.json, dynamic_beta_model_beta_chart.png, dynamic_beta_model_slope_chart.png
