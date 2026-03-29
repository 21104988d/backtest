# OOS 20% Regime Parameter Sweep

- Period: 2022-01-01 to 2026-03-23
- Split: first 80% IS, last 19% OOS (chronological)
- Rows evaluated: 1620

## Top Configurations By OOS Sharpe
1. mode=low_only, bw=90, z_entry=1.75, z_exit=1.0, vol_window=40, qlb=252, min_hist=84 | OOS Sharpe=0.8340, OOS Return=10.39%, OOS MDD=-11.25% | IS Sharpe=-0.2678, IS Return=-15.50%
2. mode=high_low_only, bw=90, z_entry=1.75, z_exit=1.0, vol_window=40, qlb=252, min_hist=84 | OOS Sharpe=0.8340, OOS Return=10.39%, OOS MDD=-11.25% | IS Sharpe=0.0641, IS Return=-11.19%
3. mode=low_only, bw=90, z_entry=1.75, z_exit=1.0, vol_window=40, qlb=252, min_hist=126 | OOS Sharpe=0.8340, OOS Return=10.39%, OOS MDD=-11.25% | IS Sharpe=-0.2678, IS Return=-15.50%
4. mode=high_low_only, bw=90, z_entry=1.75, z_exit=1.0, vol_window=40, qlb=252, min_hist=126 | OOS Sharpe=0.8340, OOS Return=10.39%, OOS MDD=-11.25% | IS Sharpe=0.1603, IS Return=-2.12%
5. mode=low_only, bw=120, z_entry=1.75, z_exit=1.0, vol_window=40, qlb=252, min_hist=84 | OOS Sharpe=0.7990, OOS Return=10.01%, OOS MDD=-11.84% | IS Sharpe=-0.1655, IS Return=-9.85%
6. mode=low_only, bw=120, z_entry=1.75, z_exit=1.0, vol_window=40, qlb=252, min_hist=126 | OOS Sharpe=0.7990, OOS Return=10.01%, OOS MDD=-11.84% | IS Sharpe=-0.1655, IS Return=-9.85%
7. mode=low_only, bw=90, z_entry=2.0, z_exit=1.0, vol_window=40, qlb=252, min_hist=84 | OOS Sharpe=0.7785, OOS Return=9.49%, OOS MDD=-11.25% | IS Sharpe=-0.0235, IS Return=-4.57%
8. mode=high_low_only, bw=90, z_entry=2.0, z_exit=1.0, vol_window=40, qlb=252, min_hist=84 | OOS Sharpe=0.7785, OOS Return=9.49%, OOS MDD=-11.25% | IS Sharpe=0.0505, IS Return=-10.69%
9. mode=low_only, bw=90, z_entry=2.0, z_exit=1.0, vol_window=40, qlb=252, min_hist=126 | OOS Sharpe=0.7785, OOS Return=9.49%, OOS MDD=-11.25% | IS Sharpe=-0.0235, IS Return=-4.57%
10. mode=high_low_only, bw=90, z_entry=2.0, z_exit=1.0, vol_window=40, qlb=252, min_hist=126 | OOS Sharpe=0.7785, OOS Return=9.49%, OOS MDD=-11.25% | IS Sharpe=0.1531, IS Return=-1.56%
11. mode=low_only, bw=120, z_entry=2.0, z_exit=1.0, vol_window=40, qlb=252, min_hist=84 | OOS Sharpe=0.7695, OOS Return=9.47%, OOS MDD=-11.84% | IS Sharpe=-0.0801, IS Return=-5.95%
12. mode=low_only, bw=120, z_entry=2.0, z_exit=1.0, vol_window=40, qlb=252, min_hist=126 | OOS Sharpe=0.7695, OOS Return=9.47%, OOS MDD=-11.84% | IS Sharpe=-0.0801, IS Return=-5.95%

- Full CSV: oos20_regime_parameter_sweep.csv
- JSON summary: oos20_regime_parameter_sweep_summary.json
