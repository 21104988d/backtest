# Rolling Beta Diagnostics

- Pair: MSTR vs BTC-USD
- Period: 2022-01-03 to 2026-03-20
- Rows: 1057
- Beta windows evaluated: [40, 60, 90, 120]

## Summary (across beta windows)
- bw=40: beta_mean=1.2832, beta_std=0.2533, break_rate=5.62%, raw_corr=0.7482, hedged_corr=0.0390, reduction=0.7092
- bw=60: beta_mean=1.2798, beta_std=0.2053, break_rate=4.82%, raw_corr=0.7482, hedged_corr=0.0824, reduction=0.6657
- bw=90: beta_mean=1.2862, beta_std=0.1789, break_rate=4.87%, raw_corr=0.7482, hedged_corr=0.1426, reduction=0.6055
- bw=120: beta_mean=1.2877, beta_std=0.1512, break_rate=4.28%, raw_corr=0.7482, hedged_corr=0.2269, reduction=0.5213

- Best window by corr reduction: 40
- Artifacts: rolling_beta_diag_summary.csv, rolling_beta_diag_summary.json, rolling_beta_diag_bw*.png, rolling_beta_diag_bw*_series.csv
