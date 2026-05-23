# Monte Carlo Liquidation Risk

Ruin is defined as path equity touching the floor at any point within the horizon.
The key liquidation metric requested is probability of touching 0 equity.

## Dataset Stats
- full_history: days=948, mean daily pnl=4.9175, std=19.0012, worst day=-19.6000, observed min equity=978.4405
- nonoverlap_oos: days=692, mean daily pnl=6.1553, std=20.9349, worst day=-19.6000, observed min equity=1020.0111

## Monte Carlo Results (Key)
- full_history | block | 252d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=1550.7018 | p50=2238.4134
- full_history | block | 365d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=1976.9789 | p50=2795.3511
- full_history | block | 730d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=3418.4474 | p50=4610.0801
- full_history | block | 1095d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=4954.9404 | p50=6418.5786
- full_history | iid | 252d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=1557.2984 | p50=2234.7837
- full_history | iid | 365d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=1984.7533 | p50=2789.2116
- full_history | iid | 730d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=3420.3416 | p50=4578.6932
- full_history | iid | 1095d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=4940.6289 | p50=6378.6748
- nonoverlap_oos | block | 252d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=1805.2380 | p50=2541.4642
- nonoverlap_oos | block | 365d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=2336.9840 | p50=3236.1484
- nonoverlap_oos | block | 730d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=4192.2290 | p50=5491.0490
- nonoverlap_oos | block | 1095d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=6136.1252 | p50=7733.8432
- nonoverlap_oos | iid | 252d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=1793.2508 | p50=2546.6819
- nonoverlap_oos | iid | 365d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=2350.3809 | p50=3241.6721
- nonoverlap_oos | iid | 730d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=4205.3139 | p50=5487.8862
- nonoverlap_oos | iid | 1095d: P(hit<=0)=0.0000% | P(hit<=10% init)=0.0000% | P(final<=0)=0.0000% | Final equity p01=6149.2759 | p50=7732.6340

## Notes
- Floors tested for touch probability: 0%, 10%, 25%, 50% of initial capital.
- IID bootstrap resamples single days independently.
- Block bootstrap preserves short-run clustering (volatility/regime streaks).

## Artifact
- Visual summary: monte_carlo_liquidation_risk.png