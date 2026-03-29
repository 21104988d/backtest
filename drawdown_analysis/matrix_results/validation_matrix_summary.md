| Run | Start | End | Mode | Cycle Opens | Cycle Closes | Added Shorts | Max Active Cycles | Final Equity (USD) | Equity Rows | Trade Rows |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024H1_strict | 2024-01-01 | 2024-06-30 | strict | 0 | 0 | 0 | 0 | 0.00 | 180 | 0 |
| 2024H1_nearest | 2024-01-01 | 2024-06-30 | nearest | 0 | 0 | 0 | 0 | 0.00 | 180 | 0 |
| 2024H2_strict | 2024-07-01 | 2024-12-31 | strict | 0 | 0 | 0 | 0 | 0.00 | 182 | 0 |
| 2024H2_nearest | 2024-07-01 | 2024-12-31 | nearest | 0 | 0 | 0 | 0 | 0.00 | 182 | 0 |
| 2025Y_strict | 2025-01-01 | 2025-12-31 | strict | 0 | 0 | 0 | 0 | 0.00 | 363 | 0 |
| 2025Y_nearest | 2025-01-01 | 2025-12-31 | nearest | 57 | 46 | 273 | 11 | 182674.59 | 363 | 376 |

Notes:
- This matrix is generated from Deribit historical data using `deribit_option_calendar_backtest.py`.
- In strict mode (`K = O(T-1)` exact match), no trades were opened in these windows due sparse exact-strike listing matches in historical chains.
- `Final Equity` starts from a zero-cash accounting baseline in this prototype, so compare modes by relative behavior and trade activity, not the percentage return field.
