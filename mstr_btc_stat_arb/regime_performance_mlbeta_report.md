# Regime Performance: ML Beta vs Rolling40

- Pair: MSTR vs BTC-USD
- Period: 2022-01-04 to 2026-03-20
- OOS view starts at: 2024-01-01

## Overall
- All bars roll40: Return=65.30%, Sharpe=0.6001, MDD=-53.13%
- All bars mlbeta: Return=68.88%, Sharpe=0.6165, MDD=-53.13%
- OOS bars roll40: Return=-7.26%, Sharpe=0.1334, MDD=-34.56%
- OOS bars mlbeta: Return=-6.23%, Sharpe=0.1458, MDD=-35.05%

## OOS Regime Breakdown
- high: bars=185 | roll40 ret=47.01%, sh=1.455 | mlbeta ret=44.48%, sh=1.410
- mid: bars=180 | roll40 ret=-16.65%, sh=-0.871 | mlbeta ret=-14.66%, sh=-0.751
- low: bars=191 | roll40 ret=-24.31%, sh=-1.324 | mlbeta ret=-23.95%, sh=-1.298
- unknown: bars=0 | roll40 ret=nan%, sh=nan | mlbeta ret=nan%, sh=nan

- Series CSV: regime_performance_mlbeta_series.csv
- JSON: regime_performance_mlbeta_summary.json
- Chart: regime_performance_mlbeta_oos_return_by_regime.png
- Chart: regime_performance_mlbeta_oos_sharpe_by_regime.png
