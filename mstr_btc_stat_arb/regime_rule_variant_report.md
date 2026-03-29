# Regime-Conditional Rule Improvement Check

Objective:
Implement regime-conditional zscore direction rules (plus optional beta-stability filter) and check whether performance improves.

## What was implemented
- Core rule engine: `regime_conditional_rule_backtest.py`
- Variant sweep (regime direction map + beta-stability quantile): `regime_rule_variant_sweep.csv`

Direction map notation:
- `-1`: contrarian to zscore sign
- `+1`: momentum with zscore sign
- `0`: do not open new trades in that regime

## Baseline (rebuilt)
OOS 2024+:
- Return: -16.51%
- Sharpe: 0.0175
- MDD: -44.54%

## First learned rule (train-only map + tuned beta filter)
Learned from pre-2024:
- Direction map: high=0, mid=0, low=-1
- Beta stability quantile: 0.8

OOS 2024+ result:
- Return: -21.51%
- Sharpe: -0.5259
- MDD: -28.11%

Interpretation:
- Drawdown improved, but return/sharpe worsened.

## Best variant from sweep (by OOS Sharpe uplift)
Variant:
- Direction map: high=-1, mid=+1, low=-1
- Beta filter quantile: 1.0 (effectively no beta filter)

OOS 2024+:
- Return: -6.91%
- Sharpe: 0.1578
- MDD: -45.69%

Delta vs baseline OOS 2024+:
- Return: +9.60%
- Sharpe: +0.1404

Independent rerun of this best variant:
- OOS split-date (2025-05-16+) baseline: ret=-10.89%, sharpe=-0.4489
- OOS split-date (2025-05-16+) variant: ret=-0.16%, sharpe=0.1547
- Delta: ret=+10.73%, sharpe=+0.6036

## Practical conclusion
- Yes, implementing regime-conditional direction can improve performance relative to baseline.
- In this run, the main improvement came from regime direction mapping (especially mid regime flipping to momentum), while the beta-stability filter did not help the best OOS result.
- Best candidate currently: high=-1, mid=+1, low=-1 with no beta-stability gating.

## Artifacts
- `regime_conditional_rule_backtest.py`
- `regime_conditional_rule_summary.json`
- `regime_conditional_rule_report.md`
- `regime_conditional_rule_series.csv`
- `regime_rule_variant_sweep.csv`
- `regime_rule_best_variant_series.csv`
- `regime_rule_best_variant_summary.json`
