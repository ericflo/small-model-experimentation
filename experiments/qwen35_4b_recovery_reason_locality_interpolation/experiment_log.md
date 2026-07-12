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

## 2026-07-12 — frozen run

- All four mixtures passed locality; drift was 0.100/0.104/0.111/0.121 and
  entropy/varentropy remained inside the registered bounds. Full reason
  reproduced the 0.303 failure.
- λ=.18 peaked at 58/60 recovery (96.7%), versus 29/60 base, 49/60 happy,
  51/60 action, and 55/60 full reason.
- The selector admitted no candidate. Every point exceeded base+2pp invalid
  turns and missed the 60% immediate rejected-patch change gate. Run stopped
  before confirmation, transfer, scaffold/sampling controls, and Menagerie.
- Post-stop forensics: all 24 invalid steps exhausted exactly 256 answer tokens
  inside a long JSON patch payload; all had already closed thinking. All 30
  rejected cases changed the patch within two turns and solved; 20 used
  INSPECT→PATCH and 10 PATCH→VERIFY.
- Next strategy: a new λ=.18 harness experiment with realistic tool-payload
  capacity under matched compute and changed-patch-within-two retention.
