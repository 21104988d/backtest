# Nested Walk-Forward Robustness Report

Method:
- Purged (embargoed) outer walk-forward folds for final OOS evaluation
- Nested inner folds to select global parameters with stability-penalized objective
- Regularized regime map constrained near selected global params

- Fold count: 9
- Embargo bars: 10
- Candidate count: 36

## Aggregate OOS Metrics
- nested_global_oos: Return=-11.50%, Sharpe=0.0608, MDD=-35.04%, Active=32.37%
- regularized_map_oos: Return=-11.50%, Sharpe=0.0608, MDD=-35.04%, Active=32.37%
- naive_is_best_oos: Return=-14.57%, Sharpe=-0.0228, MDD=-35.98%, Active=26.44%

## Robustness Gates (regularized map)
- Median OOS return: 3.45% (pass=True)
- Non-negative fold ratio: 55.56% (pass=False)
- Worst fold MDD: -35.04% (pass=False)
- Overall gate pass: False

- Fold CSV: nested_wf_regularized_folds.csv
- JSON summary: nested_wf_regularized_summary.json
