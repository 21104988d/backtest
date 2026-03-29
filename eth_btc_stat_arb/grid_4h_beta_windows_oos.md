# 4H Beta Window Grid (OOS Robustness)

Split: 70% in-sample / 30% out-of-sample on 4H strategy net returns.
Ranking objective: maximize OOS Sharpe with penalties for OOS drawdown and IS->OOS instability.

| beta_window | net_return_full | net_sharpe_full | max_drawdown_full | is_sharpe | oos_sharpe | is_mdd | oos_mdd | sharpe_decay_is_minus_oos | drawdown_stability_gap | robust_score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | -74.4008% | -1.8472 | -75.4826% | -1.7835 | -2.0178 | -62.0965% | -36.4446% | 0.2343 | 25.6520% | -3.1204 |
| 90 | -74.4118% | -1.8564 | -75.4150% | -1.7880 | -2.0400 | -61.6988% | -36.3451% | 0.2520 | 25.3537% | -3.1465 |
| 40 | -76.1440% | -1.9401 | -77.0179% | -1.8782 | -2.1092 | -63.7738% | -37.0490% | 0.2310 | 26.7248% | -3.2329 |
| 120 | -74.7570% | -1.8721 | -75.6758% | -1.7684 | -2.1483 | -61.1117% | -37.4510% | 0.3799 | 23.6607% | -3.3239 |

## Selected Robust Window: 60
- OOS Sharpe: -2.0178
- OOS Max Drawdown: -36.4446%
- IS->OOS Sharpe Decay: 0.2343
- Drawdown Stability Gap: 25.6520%
