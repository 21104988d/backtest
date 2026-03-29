# Beta, Spread-Z, OU Interaction Diagnostics

Question tested:
- Is mean reversion in zscore mainly coming from beta changes, and therefore less monetizable in traded returns?

Produced files:
- beta_ou_z_relationships_correlations.csv
- beta_ou_z_relationships_buckets.csv
- beta_ou_z_mechanism_checks.csv
- beta_ou_z_structural_relationships.csv
- beta_ou_z_relationships_summary.json

## Mechanism Check Snapshot
- FULL | high | corr(betaAdj, zReversion1d)=-0.1265 | corr(|deltaBeta|, zReversion1d)=-0.0265 | corr(betaAdj, fwdRet1d)=-0.0076
- FULL | mid | corr(betaAdj, zReversion1d)=-0.0598 | corr(|deltaBeta|, zReversion1d)=0.0492 | corr(betaAdj, fwdRet1d)=-0.0422
- FULL | low | corr(betaAdj, zReversion1d)=0.0268 | corr(|deltaBeta|, zReversion1d)=-0.0217 | corr(betaAdj, fwdRet1d)=0.1531
- OOS_2024+ | high | corr(betaAdj, zReversion1d)=-0.1281 | corr(|deltaBeta|, zReversion1d)=-0.0065 | corr(betaAdj, fwdRet1d)=-0.0224
- OOS_2024+ | mid | corr(betaAdj, zReversion1d)=-0.0771 | corr(|deltaBeta|, zReversion1d)=0.0434 | corr(betaAdj, fwdRet1d)=0.0167
- OOS_2024+ | low | corr(betaAdj, zReversion1d)=-0.0531 | corr(|deltaBeta|, zReversion1d)=-0.0385 | corr(betaAdj, fwdRet1d)=0.1947
