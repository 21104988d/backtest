# Low-Regime-Only Strict Filter Results

Rule:
- Trade only when vol_regime == low
- Force flat position for high/mid/unknown bars

- FULL: baseline(ret=17.82%, sh=0.3509, mdd=-45.71%) | low_only(ret=20.79%, sh=0.3982, mdd=-28.18%) | delta(ret=2.97%, sh=0.0473)
- IS_PRE_2024: baseline(ret=41.12%, sh=0.8063, mdd=-36.83%) | low_only(ret=42.99%, sh=1.3103, mdd=-14.87%) | delta(ret=1.87%, sh=0.5041)
- IS_SPLIT_DATE: baseline(ret=32.21%, sh=0.4860, mdd=-45.71%) | low_only(ret=19.74%, sh=0.4676, mdd=-28.18%) | delta(ret=-12.48%, sh=-0.0183)
- OOS_2024_PLUS: baseline(ret=-16.51%, sh=0.0175, mdd=-44.54%) | low_only(ret=-15.53%, sh=-0.3352, mdd=-28.18%) | delta(ret=0.98%, sh=-0.3526)
- OOS_SPLIT_DATE: baseline(ret=-10.89%, sh=-0.4489, mdd=-17.97%) | low_only(ret=0.88%, sh=0.1960, mdd=-12.09%) | delta(ret=11.76%, sh=0.6449)
