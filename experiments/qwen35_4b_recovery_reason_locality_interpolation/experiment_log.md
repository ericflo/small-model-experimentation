# Experiment Log

## 2026-07-12 — intake and preregistration

- Direct parent stopped at its registered locality gate: action-only drift
  0.098, reason drift 0.303, with 85.0% versus 91.7% trained-family recovery.
- Chose action-anchored contrast interpolation over scaling the full reason
  delta, because weakening the known-local action signal would answer a less
  useful question.
- Froze lambdas 0.10/0.18/0.24/0.30 from the endpoint-implied locality frontier
  near 0.25; no scaled checkpoint had been merged or evaluated.
- Added a disjoint locality-confirmation block, hard validity/transition gates,
  action-endpoint comparison, transfer feasibility receipts, and no-fallback
  selection.
- Menagerie and both transfer blocks remain unexposed.
