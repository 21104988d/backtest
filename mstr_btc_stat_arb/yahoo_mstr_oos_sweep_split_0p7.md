# Yahoo OOS Parameter Sweep (MSTR vs BTC-USD)

- Period: 2022-01-03 to 2026-03-20
- Rows aligned: 1057
- Split: 70% IS / 30% OOS (split date 2024-12-11)
- Z window: 30
- Fee rate: 0.00045
- Search combinations: 64

## Top by OOS Sharpe
1. bw=90, z_entry=1.75, z_exit=1.0 | OOS Sharpe=0.4779, OOS Return=10.00%, OOS MDD=-19.52% | IS Sharpe=0.3132, IS Return=7.00%
2. bw=90, z_entry=2.0, z_exit=1.0 | OOS Sharpe=0.4688, OOS Return=9.26%, OOS MDD=-20.24% | IS Sharpe=0.5861, IS Return=40.52%
3. bw=120, z_entry=1.75, z_exit=1.0 | OOS Sharpe=0.4452, OOS Return=8.86%, OOS MDD=-19.65% | IS Sharpe=0.4441, IS Return=22.44%
4. bw=120, z_entry=2.0, z_exit=1.0 | OOS Sharpe=0.4302, OOS Return=8.05%, OOS MDD=-20.11% | IS Sharpe=0.7594, IS Return=62.00%
5. bw=40, z_entry=1.75, z_exit=1.0 | OOS Sharpe=0.4270, OOS Return=8.24%, OOS MDD=-21.58% | IS Sharpe=0.2367, IS Return=-2.05%

## Best IS Model And Its OOS
- bw=120, z_entry=2.25, z_exit=0.75
- IS: Sharpe=1.1438, Return=121.24%, MDD=-35.98%
- OOS: Sharpe=-0.6979, Return=-18.28%, MDD=-23.14%

- Full CSV: yahoo_mstr_oos_sweep_split_0p7.csv
- Full JSON summary: yahoo_mstr_oos_sweep_split_0p7.json
