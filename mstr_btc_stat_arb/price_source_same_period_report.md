# Price Source Comparison (Same Period)

- Period: 2025-12-02 to 2026-03-22
- Compare Hyperliquid (1D and 4H aggregated to daily) vs Yahoo daily

## MSTR: Hyperliquid 1D vs Yahoo Daily
- Price corr=0.9965, mean abs pct diff=0.55%, MAE=0.8169
- Return corr=0.9632, mean return diff=0.0809%, RMSE diff=1.6019%
- Cum return HL1D=-20.89% vs Yahoo=-23.99%

## MSTR: Hyperliquid 4H->Daily vs Yahoo Daily
- Return corr=0.9636, mean return diff=0.0836%, RMSE diff=1.5881%
- Cum return HL4H(daily-close)=-17.63% vs Yahoo=-21.03%

## BTC: Hyperliquid 1D vs Yahoo Daily
- Return corr=0.9985, mean return diff=0.0158%, RMSE diff=0.1600%

- Chart: price_source_mstr_same_period.png
- Chart: price_source_mstr_return_gap_same_period.png
- JSON: price_source_same_period_summary.json
