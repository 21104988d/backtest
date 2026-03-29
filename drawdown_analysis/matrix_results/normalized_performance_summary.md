| Run | Period | Mode | Initial Capital | Ending Capital | PnL (USD) | Return % | Max Drawdown % | Sharpe (365d) | Cycle Opens | Cycle Closes | Win Rate % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025Y_strict | 2025-01-01 to 2025-12-31 | strict | 100000.0 | 100000.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0.0 |
| 2025Y_nearest | 2025-01-01 to 2025-12-31 | nearest | 100000.0 | 282674.5884 | 182674.5884 | 182.6746 | -48.8474 | 1.530741 | 57 | 46 | 100.0 |

Notes:
- This pass is restricted to the period where option-data-driven cycle activity was observed in the Deribit runs.
- Win Rate is computed as recovered cycle closes / total cycle closes.
- Strict mode remains constrained by exact strike matching (`K = O(T-1)`), resulting in no fills in this period.
