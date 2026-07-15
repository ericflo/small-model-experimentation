# Adversarial Design Review

A training-free re-adjudication built on the pipeline that passed full
multi-lens reviews in its two predecessors; the review surface here is the
corrected promotion logic, the inherited pins, and the re-adjudication framing.

- Gate-shopping audit: both predecessor failures remain recorded with their
  seeds sealed (78,144 consumed; 78,145 sealed); the correction mechanism was
  queued in the program backlog, in its exact form, before this experiment
  opened, from two independent receipts showing the protocol kind tied at the
  parent ceiling (8/8/8 twice); the new gate uses a fresh task seed so no
  graded item is reused; the corrected bar adds a fail-closed outcome
  (`GATE_UNDETECTABLE`) and weakens nothing else.
- Corrected bar implementation: verified line-by-line against the
  preregistration; the integer ceiling `(2n+2)//3` equals ⌈2n/3⌉ exactly for
  every relevant n; detectability (either control ≥ 9/10) excludes a kind and
  reports it; retention bands and the route-abstention sanity check are
  byte-inherited from the predecessor logic.
- Pins: all four composite trees and weights are constants recomputed by the
  smoke path (full tree manifests re-hashed); no TODO-pins exist anywhere in
  this experiment.
- Verification battery: 46 unit tests including detectability exclusion at the
  ceiling, ⌈2/3⌉ breadth on three detectable kinds, GATE_UNDETECTABLE
  fail-closed, tie-fails, all retention bands, and recovery-writer schema
  parity; full smoke (inherited-source shas, composite tree re-hash, gate
  --check regeneration, compile sweep) exits green.

**Verdict:** `PASS_EXPENSIVE_RUN`.
