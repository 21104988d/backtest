# Yahoo OOS Parameter Sweep (MSTR vs BTC-USD)

- Period: 2022-01-03 to 2026-03-20
- Rows aligned: 1057
- Split: 80% IS / 20% OOS (split date 2025-05-15)
- Z window: 30
- Fee rate: 0.00045
- Search combinations: 64

## Top by OOS Sharpe
1. bw=120, z_entry=1.75, z_exit=1.0 | OOS Sharpe=-0.4991, OOS Return=-11.07%, OOS MDD=-17.73% | IS Sharpe=0.6033, IS Return=49.87%
2. bw=90, z_entry=1.75, z_exit=1.0 | OOS Sharpe=-0.5078, OOS Return=-10.93%, OOS MDD=-17.97% | IS Sharpe=0.4858, IS Return=32.15%
3. bw=40, z_entry=1.75, z_exit=1.0 | OOS Sharpe=-0.6060, OOS Return=-12.86%, OOS MDD=-19.43% | IS Sharpe=0.4178, IS Return=21.67%
4. bw=60, z_entry=1.75, z_exit=1.0 | OOS Sharpe=-0.6530, OOS Return=-13.83%, OOS MDD=-18.33% | IS Sharpe=0.3704, IS Return=14.93%
5. bw=120, z_entry=2.25, z_exit=1.0 | OOS Sharpe=-0.6740, OOS Return=-13.04%, OOS MDD=-18.22% | IS Sharpe=0.9231, IS Return=93.84%

## Best IS Model And Its OOS
- bw=120, z_entry=2.25, z_exit=0.75
- IS: Sharpe=1.0786, Return=124.82%, MDD=-35.98%
- OOS: Sharpe=-1.0370, Return=-19.58%, MDD=-23.14%

- Full CSV: yahoo_mstr_oos_sweep_split_0p8.csv
- Full JSON summary: yahoo_mstr_oos_sweep_split_0p8.json
