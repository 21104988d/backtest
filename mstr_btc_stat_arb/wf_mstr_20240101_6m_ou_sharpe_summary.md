# Walk-Forward OOS Validation (MSTR vs BTC-USD)

Leakage control:
- Each fold selects parameters using only data strictly before OOS start.
- Chosen parameters are then applied to the immediately following OOS window.
- Folds are non-overlapping in OOS.

- Period: 2022-01-03 to 2026-03-20
- Walk-forward start requested: 2024-01-01
- Walk-forward start used: 2024-01-02
- OOS window size: 6 months
- Selection objective: ou_sharpe
- OU half-life filter (for ou_sharpe): [4.0, 90.0]
- OU min kappa (for ou_sharpe): 0.005
- OU lookback bars (for ou_sharpe): 252
- Fold count: 4
- Positive OOS folds: 2/4 (50.00%)
- Average OOS Sharpe across folds: 0.1146
- Combined OOS Return: -2.79%
- Combined OOS Sharpe: 0.1670
- Combined OOS Max Drawdown: -35.98%
- Avg OU-valid candidates per fold: 48.00

## Most Selected Parameter Sets
1. bw=120|ze=2.25|zx=0.75, selected 4 folds

## Fold Results
1. OOS 2024-01-02 to 2024-07-01 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=-0.1662, OOS Return=-9.69%, OOS MDD=-35.98% | OU hl(IS)=20.55, kappa(IS)=0.0337
2. OOS 2024-07-02 to 2024-12-31 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=1.0596, OOS Return=15.75%, OOS MDD=-24.76% | OU hl(IS)=123.51, kappa(IS)=0.0056
3. OOS 2025-01-02 to 2025-07-01 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=0.6049, OOS Return=3.82%, OOS MDD=-11.14% | OU hl(IS)=74.05, kappa(IS)=0.0094
4. OOS 2025-07-02 to 2025-12-31 | bw=120, z_entry=2.25, z_exit=0.75 | OOS Sharpe=-1.0398, OOS Return=-10.43%, OOS MDD=-17.56% | OU hl(IS)=30.73, kappa(IS)=0.0226

- Fold CSV: wf_mstr_20240101_6m_ou_sharpe_folds.csv
- Full JSON: wf_mstr_20240101_6m_ou_sharpe_summary.json
- Chart: wf_mstr_20240101_6m_ou_sharpe_oos_return_by_fold.png
- Chart: wf_mstr_20240101_6m_ou_sharpe_oos_sharpe_by_fold.png
- Chart: wf_mstr_20240101_6m_ou_sharpe_combined_oos_equity.png
