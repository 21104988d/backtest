# Trading Rules And OOS20 Optimization

## Objective
Store the strategy rules in one place, enforce a chronological 80/20 split (first 80% IS, last 20% OOS), and optimize parameters against OOS performance.

## Data And Split Policy
- Asset pair: `MSTR` vs `BTC-USD`
- Data source: Yahoo daily close
- Period: `2022-01-01` to `2026-03-23`
- Split rule: chronological, first 80% in-sample, last 20% out-of-sample
- Effective split date from run: `2025-05-16`

## Strategy Core Rules
1. Compute returns and rolling hedge beta:
- `a_ret = MSTR return`
- `b_ret = BTC return`
- `rolling_beta` estimated on past-only window (`beta_window`)
- `hedged_ret = a_ret - rolling_beta * b_ret`

2. Build spread and z-score:
- Static log-price fit on full aligned series:
  - `spread = log(asset) - (a + b*log(btc))`
- Rolling z-score on spread over `z_window`

3. Position rules:
- Enter short spread (`-1`) when `z >= z_entry`
- Enter long spread (`+1`) when `z <= -z_entry`
- Exit to flat (`0`) when `|z| <= z_exit`

4. Fee model:
- Turnover-based fee on position change
- `fee_rate = 0.00045`

## Volatility Regime Rules (Leakage-Safe)
1. `rolling_vol`:
- Rolling std of `hedged_ret` over `vol_window`

2. High / Mid / Low regime labels:
- At each bar, compute `q33` and `q66` from past rolling vol only
- Past window length: `quantile_lookback`
- Need at least `min_history` bars before assigning regimes
- Regimes:
  - `high`: rolling_vol >= q66
  - `low`: rolling_vol <= q33
  - `mid`: q33 < rolling_vol < q66
  - `unknown`: not enough past history

3. Portfolio mode definitions:
- `baseline`: always trade
- `high_only`: trade only in high-vol regime
- `low_only`: trade only in low-vol regime
- `high_low_only`: trade in high or low, skip mid
- `regime_weighted`: high=1.0x, mid=0.5x, low=0.2x, unknown=0.0x

## Optimization Search Space (Run)
- `beta_window`: [60, 90, 120]
- `z_entry`: [1.75, 2.0, 2.25]
- `z_exit`: [0.5, 0.75, 1.0], with `z_exit < z_entry`
- `z_window`: 30
- `vol_window`: [20, 30, 40]
- `quantile_lookback`: [126, 252]
- `min_history`: [84, 126]
- `portfolio_mode`: [baseline, high_only, low_only, high_low_only, regime_weighted]
- Selection target: highest OOS Sharpe, then OOS Return

## Best OOS20 Result (Current Run)
From `oos20_regime_parameter_sweep_summary.json`:
- Mode: `low_only`
- `beta_window = 90`
- `z_entry = 1.75`
- `z_exit = 1.0`
- `z_window = 30`
- `vol_window = 40`
- `quantile_lookback = 252`
- `min_history = 84`
- OOS metrics:
  - OOS Return: `+10.39%`
  - OOS Sharpe(365): `0.8340`
  - OOS Max Drawdown: `-11.25%`
  - OOS Active Ratio: `33.02%`
- IS metrics (same config):
  - IS Return: `-15.50%`
  - IS Sharpe: `-0.2678`

## Interpretation
- This best OOS result is strong on the selected OOS block but weak in IS, so it may be regime-specific.
- The next validation step should be walk-forward OOS (multiple rolling splits), not one single 80/20 split.

## Active Regime-Conditional Trading Rule (Best Variant)

This section applies the best-performing regime-direction variant found in the follow-up diagnostics.

### Parameter Set
- `beta_window = 90`
- `z_window = 30`
- `z_entry = 1.75`
- `z_exit = 1.0`
- `fee_rate = 0.00045`

### Entry/Exit Logic
1. Compute zscore of spread as before.
2. When flat, open only if `|z| >= 1.75`.
3. Exit any open position when `|z| <= 1.0`.

### Regime Direction Map
Use regime-specific direction mapping from z-score sign:
- `high` regime: contrarian (`-1`)
- `mid` regime: momentum (`+1`)
- `low` regime: contrarian (`-1`)

Practical interpretation:
- If `z > 0` (positive spread side signal):
  - `high/low`: short spread
  - `mid`: long spread
- If `z < 0` (negative spread side signal):
  - `high/low`: long spread
  - `mid`: short spread

### Beta-Stability Gate
- Best variant used `beta_stability_quantile = 1.0`.
- This is effectively no additional beta-stability gate in production rule.

### Best-Variant Validation Snapshot
From `regime_rule_best_variant_summary.json`:
- OOS 2024+ baseline: ret `-16.51%`, sharpe `0.0175`
- OOS 2024+ best variant: ret `-6.91%`, sharpe `0.1578`
- Delta: return `+9.60%`, sharpe `+0.1404`

- OOS split-date baseline: ret `-10.89%`, sharpe `-0.4489`
- OOS split-date best variant: ret `-0.16%`, sharpe `0.1547`
- Delta: return `+10.73%`, sharpe `+0.6036`

## Files Produced
- Full sweep table: `oos20_regime_parameter_sweep.csv`
- Summary JSON: `oos20_regime_parameter_sweep_summary.json`
- Top-ranked markdown: `oos20_regime_parameter_sweep.md`
- Optimizer script: `optimize_oos_20_regime.py`

## Reproduce Command
```bash
cd /Users/leeisaackaiyui/Desktop/backtest/mstr_btc_stat_arb
/Users/leeisaackaiyui/Desktop/backtest/.venv/bin/python -u optimize_oos_20_regime.py
```
