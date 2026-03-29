# Walk-Forward OOS Validation (MSTR vs BTC-USD)

Leakage control:
- Each fold selects parameters using only data strictly before OOS start.
- Chosen parameters are then applied to the immediately following OOS window.
- Folds are non-overlapping in OOS.

- Period: 2022-01-03 to 2026-03-20
- Walk-forward start requested: 2024-01-01
- Walk-forward start used: 2024-01-02
- OOS window size: 1 months
- Selection objective: ou_sharpe
- OU half-life filter (for ou_sharpe): [4.0, 90.0]
- OU min kappa (for ou_sharpe): 0.005
- OU lookback bars (for ou_sharpe): 252
- Fold count: 26
- Positive OOS folds: 9/26 (34.62%)
- Average OOS Sharpe across folds: -0.0930
- Combined OOS Return: -14.57%
- Combined OOS Sharpe: -0.0228
- Combined OOS Max Drawdown: -35.98%
- Avg OU-valid candidates per fold: 34.46

## Most Selected Parameter Sets
1. bw=120|ze=2.25|zx=0.75, selected 26 folds

## Fold Results
1. OOS 2024-01-02 to 2024-02-01 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=-3.4752, OOS Return=-7.38%, OOS MDD=-12.53% | OU hl(IS)=20.55, kappa(IS)=0.0337
2. OOS 2024-02-02 to 2024-03-01 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=nan, OOS Return=0.00%, OOS MDD=0.00% | OU hl(IS)=23.14, kappa(IS)=0.0300
3. OOS 2024-03-04 to 2024-04-03 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=-1.5083, OOS Return=-15.87%, OOS MDD=-35.98% | OU hl(IS)=45.94, kappa(IS)=0.0151
4. OOS 2024-04-04 to 2024-05-03 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=nan, OOS Return=0.00%, OOS MDD=0.00% | OU hl(IS)=nan, kappa(IS)=nan
5. OOS 2024-05-06 to 2024-06-05 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=1.6679, OOS Return=4.14%, OOS MDD=-7.84% | OU hl(IS)=107.57, kappa(IS)=0.0064
6. OOS 2024-06-06 to 2024-07-05 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=4.7275, OOS Return=11.25%, OOS MDD=-2.65% | OU hl(IS)=197.68, kappa(IS)=0.0035
7. OOS 2024-07-08 to 2024-08-07 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=nan, OOS Return=0.00%, OOS MDD=0.00% | OU hl(IS)=285.93, kappa(IS)=0.0024
8. OOS 2024-08-08 to 2024-09-06 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=nan, OOS Return=0.00%, OOS MDD=0.00% | OU hl(IS)=228.45, kappa(IS)=0.0030
9. OOS 2024-09-09 to 2024-10-08 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=-4.0580, OOS Return=-10.52%, OOS MDD=-15.86% | OU hl(IS)=138.99, kappa(IS)=0.0050
10. OOS 2024-10-09 to 2024-11-08 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=1.7175, OOS Return=6.54%, OOS MDD=-11.14% | OU hl(IS)=235.68, kappa(IS)=0.0029
11. OOS 2024-11-11 to 2024-12-10 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=4.1599, OOS Return=21.47%, OOS MDD=-0.04% | OU hl(IS)=171.40, kappa(IS)=0.0040
12. OOS 2024-12-11 to 2025-01-10 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=nan, OOS Return=0.00%, OOS MDD=0.00% | OU hl(IS)=106.53, kappa(IS)=0.0065
13. OOS 2025-01-13 to 2025-02-12 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=nan, OOS Return=0.00%, OOS MDD=0.00% | OU hl(IS)=53.12, kappa(IS)=0.0130
14. OOS 2025-02-13 to 2025-03-12 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=nan, OOS Return=0.00%, OOS MDD=0.00% | OU hl(IS)=33.86, kappa(IS)=0.0205
15. OOS 2025-03-13 to 2025-04-11 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=0.8479, OOS Return=1.62%, OOS MDD=-11.14% | OU hl(IS)=37.31, kappa(IS)=0.0186
16. OOS 2025-04-14 to 2025-05-13 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=nan, OOS Return=0.00%, OOS MDD=0.00% | OU hl(IS)=55.53, kappa(IS)=0.0125
17. OOS 2025-05-14 to 2025-06-13 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=1.1619, OOS Return=2.17%, OOS MDD=-7.04% | OU hl(IS)=38.72, kappa(IS)=0.0179
18. OOS 2025-06-16 to 2025-07-15 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=nan, OOS Return=0.00%, OOS MDD=0.00% | OU hl(IS)=33.68, kappa(IS)=0.0206
19. OOS 2025-07-16 to 2025-08-15 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=-4.4487, OOS Return=-12.97%, OOS MDD=-15.12% | OU hl(IS)=30.36, kappa(IS)=0.0228
20. OOS 2025-08-18 to 2025-09-17 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=2.1936, OOS Return=3.13%, OOS MDD=-3.13% | OU hl(IS)=19.79, kappa(IS)=0.0350
21. OOS 2025-09-18 to 2025-10-17 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=-0.2837, OOS Return=-0.45%, OOS MDD=-2.73% | OU hl(IS)=11.48, kappa(IS)=0.0604
22. OOS 2025-10-20 to 2025-11-19 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=0.2749, OOS Return=0.25%, OOS MDD=-6.47% | OU hl(IS)=11.21, kappa(IS)=0.0618
23. OOS 2025-11-20 to 2025-12-19 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=nan, OOS Return=0.00%, OOS MDD=0.00% | OU hl(IS)=42.12, kappa(IS)=0.0165
24. OOS 2025-12-22 to 2026-01-21 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=2.7690, OOS Return=4.84%, OOS MDD=-3.95% | OU hl(IS)=222.52, kappa(IS)=0.0031
25. OOS 2026-01-22 to 2026-02-20 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=-4.4872, OOS Return=-13.95%, OOS MDD=-14.67% | OU hl(IS)=266.28, kappa(IS)=0.0026
26. OOS 2026-02-23 to 2026-03-20 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=-2.7462, OOS Return=-2.59%, OOS MDD=-2.57% | OU hl(IS)=124.13, kappa(IS)=0.0056

- Fold CSV: wf_mstr_20240101_1m_ou_sharpe_folds.csv
- Full JSON: wf_mstr_20240101_1m_ou_sharpe_summary.json
- Chart: wf_mstr_20240101_1m_ou_sharpe_oos_return_by_fold.png
- Chart: wf_mstr_20240101_1m_ou_sharpe_oos_sharpe_by_fold.png
- Chart: wf_mstr_20240101_1m_ou_sharpe_combined_oos_equity.png
