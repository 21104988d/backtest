# OOS Comparison: ML Predicted Beta vs Rolling-40 Beta

Method:
- Same signal logic (spread z-score entries/exits)
- Compare hedge engine only: rolling40 beta vs ML-predicted dynamic beta
- Same purged walk-forward OOS folds

- Fold count: 9
- Rolling40 OOS: Return=-7.26%, Sharpe=0.1334, MDD=-34.56%
- ML beta OOS: Return=-6.23%, Sharpe=0.1458, MDD=-35.05%
- Delta (ML - Rolling40): Return=1.03%, Sharpe=0.0124, MDD=-0.49%
- ML better fold ratio: 66.67%


- Fold CSV: oos_dynamic_beta_vs_roll40_folds.csv
- JSON summary: oos_dynamic_beta_vs_roll40_summary.json
