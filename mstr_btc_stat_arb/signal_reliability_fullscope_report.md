# Signal Reliability Full-Scope Test

Includes FULL and OOS_2024_PLUS, by regime and horizon.

## Best abs-correlation by scope/regime
- FULL | high | h=3d | corr=0.1147 | p=0.0340 | n=331
- FULL | mid | h=10d | corr=-0.1291 | p=0.0592 | n=214
- FULL | low | h=5d | corr=-0.2954 | p=0.0000 | n=348
- OOS_2024_PLUS | high | h=1d | corr=0.1113 | p=0.0905 | n=218
- OOS_2024_PLUS | mid | h=5d | corr=0.3863 | p=0.0000 | n=137
- OOS_2024_PLUS | low | h=5d | corr=-0.4369 | p=0.0000 | n=196

- Significant correlation rows (p<0.05): 18
- Significant entry-edge rows (CI excludes 0): 4

Files:
- signal_reliability_fullscope_corr.csv
- signal_reliability_fullscope_edge.csv
- signal_reliability_fullscope_summary.json
