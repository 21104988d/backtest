# Daily vs 4H Same-Period Strategy Check

- Period: 2025-12-02 to 2026-03-22
- Goal: test whether the same strategy behavior is unique to 4H

## 4H Reference (Hyperliquid)
- Net Return=49.63%, Sharpe=4.4170, MDD=-6.16%
- Params: bw=60, z_window=30, z_entry=2.0, z_exit=0.5, fee=0.00045

## Daily (Yahoo) Same Params
- Net Return=-12.61%, Sharpe=-1.5504, MDD=-17.60%, Active=48.00%
- Params: bw=60, z_window=30, z_entry=2.0, z_exit=0.5, fee=0.00045

## Daily (Yahoo) Alt Thresholds
- Net Return=-12.61%, Sharpe=-1.5504, MDD=-17.60%, Active=48.00%
- Params: bw=60, z_window=30, z_entry=1.75, z_exit=0.5, fee=0.00045

- JSON: daily_vs_4h_same_period_summary.json
