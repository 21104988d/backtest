# Higher OU Kappa -> Higher Return? (By Regime)

Method:
- Data: Yahoo MSTR/BTC strategy + leak-safe regimes + rolling OU kappa
- Tests per regime: correlation(kappa, forward return), and Q4-high-kappa minus Q1-low-kappa forward-return spread
- Samples: FULL and OOS_2024+

## Verdict Table
- FULL | high | corr_fwd_1d=0.0131 | corr_fwd_5d=0.0072 | q4-q1_1d=-0.00004 | q4-q1_5d=0.00797 | verdict=supported
- FULL | mid | corr_fwd_1d=-0.0310 | corr_fwd_5d=-0.0936 | q4-q1_1d=-0.00053 | q4-q1_5d=-0.00974 | verdict=not_supported
- FULL | low | corr_fwd_1d=-0.0847 | corr_fwd_5d=-0.1438 | q4-q1_1d=-0.00390 | q4-q1_5d=-0.01886 | verdict=not_supported
- OOS_2024+ | high | corr_fwd_1d=-0.0136 | corr_fwd_5d=0.0045 | q4-q1_1d=-0.00358 | q4-q1_5d=-0.00265 | verdict=not_supported
- OOS_2024+ | mid | corr_fwd_1d=-0.0292 | corr_fwd_5d=-0.0828 | q4-q1_1d=-0.00142 | q4-q1_5d=-0.00104 | verdict=not_supported
- OOS_2024+ | low | corr_fwd_1d=-0.0199 | corr_fwd_5d=-0.0920 | q4-q1_1d=-0.00208 | q4-q1_5d=-0.01897 | verdict=not_supported

Artifacts:
- ou_kappa_level_hypothesis_by_regime.csv
- ou_kappa_level_hypothesis_summary.json
