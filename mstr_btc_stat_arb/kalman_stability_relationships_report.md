# Kalman Stability Relationship Checks

## FULL
- n=1055
- corr(stable, |z|)=-0.151654
- corr(stable, |spread|)=-0.054724
- corr(stable, mr_score_next)=-0.024752
- corr(stable, mr_hit_next)=0.016816
- corr(stable, ret_base_next)=-0.000571
- stable vs unstable |z|=0.972431 vs 1.188926
- stable vs unstable MR-hit=0.4927 vs 0.4758
- stable vs unstable next baseline ret=0.00043553 vs 0.00046366

## OOS_2024_PLUS
- n=555
- corr(stable, |z|)=-0.219055
- corr(stable, |spread|)=0.012049
- corr(stable, mr_score_next)=-0.012105
- corr(stable, mr_hit_next)=-0.005074
- corr(stable, ret_base_next)=-0.019393
- stable vs unstable |z|=0.930687 vs 1.256341
- stable vs unstable MR-hit=0.4672 vs 0.4724
- stable vs unstable next baseline ret=-0.00055614 vs 0.00049477

## OOS_SPLIT_2025_05_16
- n=211
- corr(stable, |z|)=-0.115276
- corr(stable, |spread|)=0.000474
- corr(stable, mr_score_next)=-0.017043
- corr(stable, mr_hit_next)=0.017345
- corr(stable, ret_base_next)=-0.070835
- stable vs unstable |z|=1.109244 vs 1.294086
- stable vs unstable MR-hit=0.4590 vs 0.4400
- stable vs unstable next baseline ret=-0.00235320 vs 0.00030617

## RECENT_2026
- n=53
- corr(stable, |z|)=-0.053381
- corr(stable, |spread|)=-0.792533
- corr(stable, mr_score_next)=-0.088425
- corr(stable, mr_hit_next)=-0.024355
- corr(stable, ret_base_next)=-0.134094
- stable vs unstable |z|=1.141086 vs 1.234073
- stable vs unstable MR-hit=0.5357 vs 0.5600
- stable vs unstable next baseline ret=-0.00485117 vs 0.00083781

## Regime MR-Hit Delta
- high: n=331, stable_mr_hit=0.5500, unstable_mr_hit=0.5088, delta=0.0412
- mid: n=221, stable_mr_hit=0.4296, unstable_mr_hit=0.5814, delta=-0.1518
- low: n=348, stable_mr_hit=0.4800, unstable_mr_hit=0.4619, delta=0.0181
- unknown: n=155, stable_mr_hit=0.5088, unstable_mr_hit=0.3571, delta=0.1516
