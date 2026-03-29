# MSTR Strategy Circumstance Analysis

## Scope
- Sources: Yahoo daily (walk-forward OOS) and Hyperliquid 4H.
- Objective: identify circumstances where strategy profits vs loses.
- Leakage control: all fold-level guardrail decisions use train-only information before OOS window.

## Baseline Context
- Yahoo OOS 1m: Return=-14.57%, Sharpe=-0.0228, MaxDD=-35.98%
- Yahoo OOS 3m: Return=-14.57%, Sharpe=-0.0228, MaxDD=-35.98%
- Yahoo OOS 6m: Return=-2.79%, Sharpe=0.1670, MaxDD=-35.98%
- Hyperliquid 4H baseline: Return=49.63%, Sharpe=1.8019, MaxDD=-6.16%

## Profitable Circumstances (Yahoo OOS)
- horizon=6m, ou=slow, vol=high: avg_ret=0.00375, hit_rate=8.33%, count=120
- horizon=1m, ou=slow, vol=high: avg_ret=0.00375, hit_rate=8.33%, count=120
- horizon=3m, ou=slow, vol=high: avg_ret=0.00375, hit_rate=8.33%, count=120
- horizon=1m, ou=slow, vol=low: avg_ret=0.00197, hit_rate=8.33%, count=48
- horizon=3m, ou=slow, vol=low: avg_ret=0.00197, hit_rate=8.33%, count=48
- horizon=6m, ou=medium, vol=high: avg_ret=0.00153, hit_rate=5.94%, count=101
- horizon=3m, ou=medium, vol=high: avg_ret=0.00153, hit_rate=5.94%, count=101
- horizon=1m, ou=medium, vol=high: avg_ret=0.00153, hit_rate=5.94%, count=101
- horizon=3m, ou=medium, vol=mid: avg_ret=0.00049, hit_rate=14.49%, count=69
- horizon=6m, ou=medium, vol=mid: avg_ret=0.00049, hit_rate=14.49%, count=69

## Losing Circumstances (Yahoo OOS)
- horizon=1m, ou=medium, vol=low: avg_ret=-0.00407, hit_rate=16.67%, count=48
- horizon=3m, ou=medium, vol=low: avg_ret=-0.00407, hit_rate=16.67%, count=48
- horizon=6m, ou=medium, vol=low: avg_ret=-0.00407, hit_rate=16.67%, count=48
- horizon=1m, ou=slow, vol=mid: avg_ret=-0.00303, hit_rate=14.61%, count=89
- horizon=3m, ou=slow, vol=mid: avg_ret=-0.00303, hit_rate=14.61%, count=89
- horizon=6m, ou=slow, vol=mid: avg_ret=-0.00144, hit_rate=8.82%, count=68
- horizon=6m, ou=slow, vol=low: avg_ret=-0.00002, hit_rate=0.00%, count=19
- horizon=1m, ou=fast, vol=low: avg_ret=0.00029, hit_rate=10.53%, count=57
- horizon=3m, ou=fast, vol=low: avg_ret=0.00029, hit_rate=10.53%, count=57
- horizon=6m, ou=fast, vol=low: avg_ret=0.00029, hit_rate=10.53%, count=57

## Guardrail Counterfactuals
- Horizon 1m:
  - ret_guard_ou: Return=-2.35%, Sharpe=0.0286, MaxDD=-17.56%
  - ret_guard_ou_corr: Return=-14.86%, Sharpe=-0.4352, MaxDD=-26.39%
  - ret_guard_ou_corr_vol: Return=-26.10%, Sharpe=-1.2891, MaxDD=-31.05%
  - ret_size_ou_soft: Return=-4.48%, Sharpe=0.0017, MaxDD=-17.56%
  - ret_size_ou_corr_soft: Return=-12.33%, Sharpe=-0.1962, MaxDD=-21.64%
  - ret_size_adaptive: Return=-6.95%, Sharpe=-0.0415, MaxDD=-26.96%
- Horizon 3m:
  - ret_guard_ou: Return=-2.35%, Sharpe=0.0286, MaxDD=-17.56%
  - ret_guard_ou_corr: Return=-14.86%, Sharpe=-0.4352, MaxDD=-26.39%
  - ret_guard_ou_corr_vol: Return=-26.10%, Sharpe=-1.2891, MaxDD=-31.05%
  - ret_size_ou_soft: Return=-4.48%, Sharpe=0.0017, MaxDD=-17.56%
  - ret_size_ou_corr_soft: Return=-12.33%, Sharpe=-0.1962, MaxDD=-21.64%
  - ret_size_adaptive: Return=-7.18%, Sharpe=-0.0485, MaxDD=-26.96%
- Horizon 6m:
  - ret_guard_ou: Return=-2.35%, Sharpe=0.0301, MaxDD=-17.56%
  - ret_guard_ou_corr: Return=-14.86%, Sharpe=-0.4579, MaxDD=-26.39%
  - ret_guard_ou_corr_vol: Return=-26.10%, Sharpe=-1.3569, MaxDD=-31.05%
  - ret_size_ou_soft: Return=-0.36%, Sharpe=0.1164, MaxDD=-17.56%
  - ret_size_ou_corr_soft: Return=-7.39%, Sharpe=-0.0716, MaxDD=-21.64%
  - ret_size_adaptive: Return=2.99%, Sharpe=0.2096, MaxDD=-26.96%

## Robustness Rule
- Acceptance criterion: improve at least 2 of 3 horizons on OOS Sharpe or OOS drawdown.
- ret_guard_ou: improved_horizons=3/3
- ret_guard_ou_corr: improved_horizons=3/3
- ret_guard_ou_corr_vol: improved_horizons=3/3
- ret_size_ou_soft: improved_horizons=3/3
- ret_size_ou_corr_soft: improved_horizons=3/3
- ret_size_adaptive: improved_horizons=3/3
- Best candidate by criterion: ret_guard_ou

## Artifacts
- Yahoo diagnostics: circumstance_yahoo_oos_diagnostics.csv
- Hyperliquid diagnostics: circumstance_hyperliquid_4h_diagnostics.csv
- Yahoo regime summary: circumstance_yahoo_regime_summary.csv
- Hyperliquid regime summary: circumstance_hyperliquid_regime_summary.csv
- Chart: circumstance_yahoo_3m_ou_vol_heatmap.png
- Chart: circumstance_oos_horizon_guardrail_returns.png
- Chart: circumstance_equity_overlay_1m.png
- Chart: circumstance_equity_overlay_3m.png
- Chart: circumstance_equity_overlay_6m.png
