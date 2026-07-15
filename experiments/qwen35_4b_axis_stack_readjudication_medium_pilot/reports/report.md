# Axis Stack Re-adjudication Report

## Summary

Training-free re-adjudication of the published axis-stack composites on a fresh instrument (gate seed 88,016) with the measured ceiling-tie flaw corrected prospectively, and a conditional MEDIUM-tier pilot at sealed seed 78,146. Both predecessor failures remain recorded with their seeds sealed. No model event has run.

## Research Program Fit

The axis install replicated across two parents and was blocked once by the quick-tier aggregate comparison and once by a single breadth check whose protocol kind tied at the parent ceiling in both experiments — a measured instrument flaw, queued for exactly this correction in the program backlog before this experiment opened.

## Method

Three inherited composites (weight- and tree-pinned), one fresh two-instrument gate with the detectability-corrected breadth bar (undetectable kinds excluded and reported; two-thirds of detectable kinds required; GATE_UNDETECTABLE fails closed; retention bands unchanged), then the conditional medium pilot (candidate aggregate strictly above base and both controls; every-family-versus-base recorded as the goal gate).

## Results

- Gate (seed 88,016; all arms weight-authenticated): all four kinds detectable; required wins 3.
- Axis holdout of 40: candidate 22, parent 15, replay_squared 18. Per-kind candidate/parent/squared: explore 7/3/6 (win), hygiene 7/5/5 (win), protocol 7/7/5 (tie — third consecutive event), tracefix 1/0/2 (loss).
- Retention of 104: 65/98/5 vs 61/92/12 vs 66/91/13 (correct/parsed/caps) — bands all passed; best termination of the event.
- Two kind wins < 3: NOT_PROMOTED; seed 78,146 permanently sealed; the medium pilot never ran.

## Controls

replay_parent is the baseline; replay_squared the exposure-matched control from the stack trial; the fresh task seed removes any reuse of graded items.

## Oracle Versus Deployable Evidence

Executable truth grades outputs only; `benchmarks/` remains unread.

## Next Stage

None. Closed per contract. The queued successor is axis corpus v2: keep hygiene/explore, replace the redundant protocol block, redesign trace-repair from this line's own raw failure outputs.

## Artifact Manifest

All arms are inherited published composites with pinned checksums; the frozen gate and receipts are tracked in-repo.
