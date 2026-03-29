# Kalman Beta Stability Filter Evaluation

Compared methods:
- baseline: rolling-beta hedge + default z-score entries
- kalman: Kalman beta hedge + default z-score entries
- kalman_stable: Kalman beta hedge + only enter when beta stability condition passes

Selected parameters (trained on IS_PRE_2024):
- kalman_q=0.00023189068, kalman_r=0.0014610053
- stability_quantile=0.80
- thresholds: velocity<=0.0115921, uncertainty<=0.151295, innovation_z<=1.09666
- stable rate full sample=45.27%

- FULL: baseline(ret=17.82%, sh=0.3509, mdd=-45.71%) | kalman(ret=1.45%, sh=0.2465, mdd=-43.70%) | kalman_stable(ret=64.95%, sh=0.7231, mdd=-24.59%)
- IS_PRE_2024: baseline(ret=41.12%, sh=0.8063, mdd=-36.83%) | kalman(ret=17.27%, sh=0.4826, mdd=-38.64%) | kalman_stable(ret=43.64%, sh=0.9898, mdd=-20.00%)
- IS_SPLIT_DATE: baseline(ret=32.21%, sh=0.4860, mdd=-45.71%) | kalman(ret=11.94%, sh=0.3464, mdd=-43.70%) | kalman_stable(ret=79.92%, sh=0.9459, mdd=-24.59%)
- OOS_2024_PLUS: baseline(ret=-16.51%, sh=0.0175, mdd=-44.54%) | kalman(ret=-13.49%, sh=0.0621, mdd=-42.94%) | kalman_stable(ret=14.83%, sh=0.4583, mdd=-24.59%)
- OOS_SPLIT_DATE: baseline(ret=-10.89%, sh=-0.4489, mdd=-17.97%) | kalman(ret=-9.37%, sh=-0.3569, mdd=-18.44%) | kalman_stable(ret=-8.33%, sh=-0.6769, mdd=-16.75%)
- RECENT_2026: baseline(ret=-11.26%, sh=-1.8042, mdd=-17.60%) | kalman(ret=-12.18%, sh=-2.0456, mdd=-17.99%) | kalman_stable(ret=-14.50%, sh=-3.0355, mdd=-16.75%)
