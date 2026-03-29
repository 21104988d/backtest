# Walk-Forward OOS Validation (MSTR vs BTC-USD)

Leakage control:
- Each fold selects parameters using only data strictly before OOS start.
- Chosen parameters are then applied to the immediately following OOS window.
- Folds are non-overlapping in OOS.

- Period: 2022-01-03 to 2026-03-20
- Walk-forward start requested: 2024-01-01
- Walk-forward start used: 2024-01-02
- OOS window size: 3 months
- Selection objective: is_sharpe
- OU half-life filter (for ou_sharpe): [2.0, 120.0]
- Fold count: 9
- Positive OOS folds: 5/9 (55.56%)
- Average OOS Sharpe across folds: 0.1396
- Combined OOS Return: -14.57%
- Combined OOS Sharpe: -0.0228
- Combined OOS Max Drawdown: -35.98%
- Avg OU-valid candidates per fold: 0.00

## Most Selected Parameter Sets
1. bw=120|ze=2.25|zx=0.75, selected 9 folds

## Fold Results
1. OOS 2024-01-02 to 2024-04-01 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=-1.4077, OOS Return=-22.05%, OOS MDD=-35.98% | OU hl(IS)=54.97, kappa(IS)=0.0126
2. OOS 2024-04-02 to 2024-07-01 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=2.5208, OOS Return=15.86%, OOS MDD=-7.84% | OU hl(IS)=1605.47, kappa(IS)=0.0004
3. OOS 2024-07-02 to 2024-10-01 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=1.3462, OOS Return=3.61%, OOS MDD=-2.53% | OU hl(IS)=386.89, kappa(IS)=0.0018
4. OOS 2024-10-02 to 2024-12-31 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=1.2309, OOS Return=11.72%, OOS MDD=-22.31% | OU hl(IS)=1661.13, kappa(IS)=0.0004
5. OOS 2025-01-02 to 2025-04-01 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=0.5166, OOS Return=1.62%, OOS MDD=-11.14% | OU hl(IS)=975.43, kappa(IS)=0.0007
6. OOS 2025-04-02 to 2025-07-01 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=0.7018, OOS Return=2.17%, OOS MDD=-7.04% | OU hl(IS)=3542.10, kappa(IS)=0.0002
7. OOS 2025-07-02 to 2025-10-01 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=-1.7811, OOS Return=-10.25%, OOS MDD=-16.24% | OU hl(IS)=732.78, kappa(IS)=0.0009
8. OOS 2025-10-02 to 2025-12-31 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=0.0494, OOS Return=-0.20%, OOS MDD=-8.25% | OU hl(IS)=427.95, kappa(IS)=0.0016
9. OOS 2026-01-02 to 2026-03-20 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=-1.9203, OOS Return=-12.13%, OOS MDD=-17.07% | OU hl(IS)=315.32, kappa(IS)=0.0022

- Fold CSV: wf_mstr_20240101_3m_is_sharpe_folds.csv
- Full JSON: wf_mstr_20240101_3m_is_sharpe_summary.json
