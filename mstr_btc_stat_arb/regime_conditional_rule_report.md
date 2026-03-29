# Regime-Conditional Rule Backtest

Learned rule:
- Direction map: {'high': 0, 'mid': 0, 'low': -1}
- Beta stability quantile (train only): 0.8
- Beta stability threshold: 0.015687

## full
- Baseline: ret=17.82%, sharpe=0.3509, mdd=-45.71%
- Regime rule: ret=14.27%, sharpe=0.3155, mdd=-29.62%
- Delta: ret=-3.56%, sharpe=-0.0354

## oos_2024_plus
- Baseline: ret=-16.51%, sharpe=0.0175, mdd=-44.54%
- Regime rule: ret=-21.51%, sharpe=-0.5259, mdd=-28.11%
- Delta: ret=-5.00%, sharpe=-0.5433

## oos_split_date
- Baseline: ret=-10.89%, sharpe=-0.4489, mdd=-17.97%
- Regime rule: ret=-5.69%, sharpe=-0.2009, mdd=-17.60%
- Delta: ret=5.20%, sharpe=0.2480
