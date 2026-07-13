# Post-Model-Smoke 004 Audit: Independent-Receipt Boundary Miss

## Verdict

`REPAIR_SAFETY_MARGIN_REQUIRED`. Lattice repair solves all layer-8 rows, but the
independent receipt finds one layer-4 norm just outside tolerance. Mechanics
remains unauthorized.

## Results

- The repairer reports 12/12 rows passed at every layer.
- Layer 8 uses lattice repair on nine rows, at most five exact pairs; independent
  maxima become norm error `7.79e-6` and span projection `0.009895`, both valid.
- Across all layers, span projection now passes.
- Independent layer-4 paired norm error is `1.033804e-5`, exceeding `1e-5` by
  `3.38e-7`, despite the repairer's device-local comparison stopping inside.
- No probability, choice, supplied-target metric, label, or correctness was
  stored.

## Repair

Use a geometry-only safety objective of 0.95 for both live ratios. Rows above
95% of either tolerance enter lattice search, and search stops only below that
margin. Scientific gates remain exactly 1e-5 and 0.01 and are independently
recomputed. Re-anchor before smoke 005.
