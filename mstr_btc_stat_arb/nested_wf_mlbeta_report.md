# Nested WF: ML Beta vs Rolling-40

- Nested selection on inner folds for each hedge mode separately
- OOS evaluation on purged outer folds

- Rolling40 nested OOS: Return=-7.26%, Sharpe=0.1334, MDD=-34.56%
- ML-beta nested OOS: Return=-6.23%, Sharpe=0.1458, MDD=-35.05%
- Delta (ML - Rolling40): Return=1.03%, Sharpe=0.0124, MDD=-0.49%
- ML better fold ratio: 66.67%

- Fold CSV: nested_wf_mlbeta_folds.csv
- JSON summary: nested_wf_mlbeta_summary.json
