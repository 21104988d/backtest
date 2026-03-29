# Regime-Specific Parameter Map (IS Fit, OOS Test)

- Split date (80/20 chronological): 2025-05-16
- Workflow: fit params per regime on IS only, freeze map, test on OOS

## Unknown Regime
- Definition: bars with insufficient past rolling-vol history for q33/q66 thresholds
- Unknown bars total: 123
- Unknown bars IS: 123
- Unknown bars OOS: 0
- Unknown policy in mapping: no_trade

## Regime Shares
- IS: {'mid': 0.3127962085308057, 'high': 0.283175355450237, 'low': 0.25829383886255924, 'unknown': 0.1457345971563981}
- OOS: {'low': 0.7452830188679245, 'mid': 0.25471698113207547}

## Short-vs-Long Volatility State Shares
- IS: {'neutral': 0.4158767772511848, 'compression': 0.235781990521327, 'expansion': 0.20734597156398105, 'unknown': 0.14099526066350712}
- OOS: {'compression': 0.5235849056603774, 'neutral': 0.38207547169811323, 'expansion': 0.09433962264150944}

## IS-Fitted Parameter Map
- high: bw=60, z_entry=1.75, z_exit=1.0 | IS Sharpe=0.5181, IS Return=9.11%, IS MDD=-38.04%, bars=239
- mid: bw=120, z_entry=2.0, z_exit=1.0 | IS Sharpe=2.9051, IS Return=111.92%, IS MDD=-19.37%, bars=264
- low: bw=60, z_entry=2.25, z_exit=0.5 | IS Sharpe=1.7421, IS Return=41.36%, IS MDD=-24.15%, bars=218

## Performance Comparison
- Mapped IS: Return=226.88%, Sharpe=1.3681, MDD=-42.97%
- Mapped OOS: Return=-27.19%, Sharpe=-1.4190, MDD=-29.08%
- Global IS: Return=124.82%, Sharpe=1.0780, MDD=-35.98%
- Global OOS: Return=-19.58%, Sharpe=-1.0346, MDD=-23.14%

- Series CSV: regime_param_map_oos20_series.csv
- JSON: regime_param_map_oos20_summary.json
