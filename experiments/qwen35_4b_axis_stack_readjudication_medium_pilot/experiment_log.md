# Axis Stack Re-adjudication Experiment Log

## 2026-07-15 — Model-free design freeze

- Opened after the stack trial closed green, blocked by one of ten checks: a
  kind-breadth bar where the protocol kind tied at the parent ceiling for the
  second consecutive experiment and one control produced a 7/10 explore fluke.
  The detectability correction and this re-adjudication were queued in the
  program backlog before this experiment opened.
- Training-free: the three published composites are inherited with recomputed
  weight and tree pins; the fresh gate at seed 88,016 uses the corrected bar
  (undetectable kinds excluded and reported; two-thirds of detectable kinds
  required; GATE_UNDETECTABLE fails closed); retention bands unchanged.
- The conditional pilot is preregistered at the MEDIUM tier, sealed seed
  78,146. Both prior failures remain recorded; their seeds remain sealed.
- No model, GPU, or benchmark event has run.

## 2026-07-15 — Corrected gate event: not promoted; experiment closed

- The gate event ran from freeze commit `3fc5fb6c`: three authenticated engine
  runs over the fresh 144-row input at seed 88,016.
- All four kinds detectable (no control ceiling); required wins 3. Candidate
  axis 22/40 vs parent 15 and squared 18 (third consecutive axis-total win);
  kind wins: explore and hygiene (2 of 3 required); protocol tied the parent
  for the third consecutive event; tracefix lost (1/10). Retention bands all
  passed with the best termination of the event (caps 5 vs 12/13).
- NOT_PROMOTED under the corrected bar; seed 78,146 permanently sealed; the
  medium pilot never ran. The three-replication mechanism map is recorded:
  hygiene/explore/termination install; tracefix/protocol do not.
