# Hypothesis Proof Report (N=5, fixed 1.68)

## Verdict
- Conclusion: **SUPPORTED_WITH_CAVEAT** (core 5/5, relative 0/2, overall 5/7)
- Interpretation: this does not prove certainty, but indicates whether forward consistency is strong enough for deployment confidence.

## Core Evidence
- Overlap (33 windows): pass rate 81.82%, median test return 36.73%.
- Non-overlap (11 windows): pass rate 90.91%, median test return 35.78%, worst window 5.97%.
- 1.68 vs 1.5 median-return delta: overlap 0.78%, non-overlap 0.27%.
- 1.68 beat-rate vs 1.5 by window: overlap 36.36%, non-overlap 36.36%.
- Non-overlap high-cost stress test return: 85.23%.
- Caveat: the strategy can pass gates and still lose head-to-head in many windows; monitor this in live tracking.

## Gate Checklist
- Core checks:
- Selected in non-overlap objective: True
- Non-overlap pass rate >= 80%: True
- Non-overlap positive-window ratio >= 90%: True
- Non-overlap worst window return > 0%: True
- Non-overlap high-cost stress return > 0%: True
- Relative checks vs 1.5:
- Beat-rate >= 50%: False
- Median return delta >= 0%: False

## Output Artifacts
- Summary CSV: hypothesis_proof_summary.csv
- Dashboard: hypothesis_proof_visual.png