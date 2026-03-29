# Hybrid Rule Evaluation

Hybrid rule:
- high regime: baseline
- low regime: variant
- mid/unknown regime: baseline

## FULL
- ALL: bars=1056, baseline(ret=17.82%, sh=0.3509), variant(ret=43.22%, sh=0.5011), hybrid(ret=30.01%, sh=0.4234), hybrid-baseline(ret=12.19%, sh=0.0726)
- high: bars=331, baseline(ret=18.77%, sh=0.6089), variant(ret=21.28%, sh=0.6478), hybrid(ret=18.77%, sh=0.6089), hybrid-baseline(ret=0.00%, sh=0.0000)
- low: bars=348, baseline(ret=22.41%, sh=0.7731), variant(ret=35.07%, sh=1.0640), hybrid(ret=35.07%, sh=1.0640), hybrid-baseline(ret=12.66%, sh=0.2910)
- mid: bars=222, baseline(ret=0.70%, sh=0.2231), variant(ret=-12.57%, sh=-0.3651), hybrid(ret=0.70%, sh=0.2231), hybrid-baseline(ret=0.00%, sh=0.0000)
- unknown: bars=155, baseline(ret=-19.52%, sh=-0.8564), variant(ret=0.00%, sh=nan), hybrid(ret=-19.52%, sh=-0.8564), hybrid-baseline(ret=0.00%, sh=0.0000)

## IS_PRE_2024
- ALL: bars=500, baseline(ret=41.12%, sh=0.8063), variant(ret=53.86%, sh=1.1529), hybrid(ret=41.12%, sh=0.8063), hybrid-baseline(ret=0.00%, sh=0.0000)
- high: bars=113, baseline(ret=29.41%, sh=2.4192), variant(ret=19.58%, sh=1.8908), hybrid(ret=29.41%, sh=2.4192), hybrid-baseline(ret=0.00%, sh=0.0000)
- low: bars=152, baseline(ret=22.22%, sh=1.5469), variant(ret=22.22%, sh=1.5469), hybrid(ret=22.22%, sh=1.5469), hybrid-baseline(ret=0.00%, sh=0.0000)
- mid: bars=80, baseline(ret=10.86%, sh=1.2073), variant(ret=5.27%, sh=0.7255), hybrid(ret=10.86%, sh=1.2073), hybrid-baseline(ret=0.00%, sh=0.0000)
- unknown: bars=155, baseline(ret=-19.52%, sh=-0.8564), variant(ret=0.00%, sh=nan), hybrid(ret=-19.52%, sh=-0.8564), hybrid-baseline(ret=0.00%, sh=0.0000)

## IS_SPLIT_DATE
- ALL: bars=844, baseline(ret=32.21%, sh=0.4860), variant(ret=43.45%, sh=0.5676), hybrid(ret=32.21%, sh=0.4860), hybrid-baseline(ret=0.00%, sh=0.0000)
- high: bars=318, baseline(ret=39.42%, sh=0.9288), variant(ret=42.37%, sh=0.9742), hybrid(ret=39.42%, sh=0.9288), hybrid-baseline(ret=0.00%, sh=0.0000)
- low: bars=189, baseline(ret=15.95%, sh=0.9166), variant(ret=15.95%, sh=0.9166), hybrid(ret=15.95%, sh=0.9166), hybrid-baseline(ret=0.00%, sh=0.0000)
- mid: bars=182, baseline(ret=1.62%, sh=0.2849), variant(ret=-13.10%, sh=-0.4547), hybrid(ret=1.62%, sh=0.2849), hybrid-baseline(ret=0.00%, sh=0.0000)
- unknown: bars=155, baseline(ret=-19.52%, sh=-0.8564), variant(ret=0.00%, sh=nan), hybrid(ret=-19.52%, sh=-0.8564), hybrid-baseline(ret=0.00%, sh=0.0000)

## OOS_2024_PLUS
- ALL: bars=556, baseline(ret=-16.51%, sh=0.0175), variant(ret=-6.91%, sh=0.1578), hybrid(ret=-7.88%, sh=0.1441), hybrid-baseline(ret=8.63%, sh=0.1267)
- high: bars=218, baseline(ret=-8.23%, sh=0.1287), variant(ret=1.42%, sh=0.3701), hybrid(ret=-8.23%, sh=0.1287), hybrid-baseline(ret=0.00%, sh=0.0000)
- low: bars=196, baseline(ret=0.16%, sh=0.1842), variant(ret=10.52%, sh=0.6951), hybrid(ret=10.52%, sh=0.6951), hybrid-baseline(ret=10.36%, sh=0.5109)
- mid: bars=142, baseline(ret=-9.17%, sh=-0.5903), variant(ret=-16.95%, sh=-1.2873), hybrid(ret=-9.17%, sh=-0.5903), hybrid-baseline(ret=0.00%, sh=0.0000)

## OOS_SPLIT_DATE
- ALL: bars=212, baseline(ret=-10.89%, sh=-0.4489), variant(ret=-0.16%, sh=0.1547), hybrid(ret=-1.67%, sh=0.0739), hybrid-baseline(ret=9.22%, sh=0.5228)
- high: bars=13, baseline(ret=-14.82%, sh=-6.3176), variant(ret=-14.82%, sh=-6.3176), hybrid(ret=-14.82%, sh=-6.3176), hybrid-baseline(ret=0.00%, sh=0.0000)
- low: bars=159, baseline(ret=5.57%, sh=0.5638), variant(ret=16.49%, sh=1.3158), hybrid(ret=16.49%, sh=1.3158), hybrid-baseline(ret=10.92%, sh=0.7520)
- mid: bars=40, baseline(ret=-0.91%, sh=-0.2879), variant(ret=0.62%, sh=0.3656), hybrid(ret=-0.91%, sh=-0.2879), hybrid-baseline(ret=0.00%, sh=0.0000)
