# Yahoo OOS Parameter Sweep (MSTR vs BTC-USD)

- Period: 2022-01-03 to 2026-03-20
- Rows aligned: 1057
- Split method: time_cutoff_date
- Split date: 2025-01-02 (IS uses < split date, OOS uses >= split date)
- Z window: 30
- Fee rate: 0.00045
- Search combinations: 64

## Top by OOS Sharpe
1. bw=60, z_entry=2.0, z_exit=0.5 | OOS Sharpe=0.3385, OOS Return=16.71%, OOS MDD=-24.53% | IS Sharpe=0.3458, IS Return=8.08%
2. bw=120, z_entry=2.0, z_exit=0.5 | OOS Sharpe=0.3268, OOS Return=16.33%, OOS MDD=-23.50% | IS Sharpe=0.5518, IS Return=36.94%
3. bw=90, z_entry=2.0, z_exit=0.5 | OOS Sharpe=0.3229, OOS Return=16.11%, OOS MDD=-23.59% | IS Sharpe=0.4442, IS Return=21.81%
4. bw=40, z_entry=2.0, z_exit=0.5 | OOS Sharpe=0.3173, OOS Return=15.77%, OOS MDD=-26.06% | IS Sharpe=0.3759, IS Return=11.92%
5. bw=90, z_entry=2.0, z_exit=1.0 | OOS Sharpe=0.3101, OOS Return=4.07%, OOS MDD=-20.24% | IS Sharpe=0.6281, IS Return=47.54%
6. bw=120, z_entry=2.0, z_exit=1.0 | OOS Sharpe=0.3009, OOS Return=3.80%, OOS MDD=-20.11% | IS Sharpe=0.7902, IS Return=68.64%
7. bw=60, z_entry=2.0, z_exit=0.75 | OOS Sharpe=0.2671, OOS Return=14.61%, OOS MDD=-23.76% | IS Sharpe=0.4129, IS Return=18.55%
8. bw=90, z_entry=2.0, z_exit=0.75 | OOS Sharpe=0.2671, OOS Return=14.64%, OOS MDD=-22.16% | IS Sharpe=0.5364, IS Return=34.97%

## Best IS Model And Its OOS
- bw=120, z_entry=2.25, z_exit=0.75
- IS: Sharpe=1.1330, Return=121.24%, MDD=-35.98%
- OOS: Sharpe=-0.7139, Return=-18.28%, MDD=-23.14%

- Full CSV: yahoo_mstr_oos_sweep_split_date_20250101.csv
- Full JSON summary: yahoo_mstr_oos_sweep_split_date_20250101.json
