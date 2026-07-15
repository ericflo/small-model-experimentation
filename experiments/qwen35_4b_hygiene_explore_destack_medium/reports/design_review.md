# Adversarial Design Review

A minimal-delta experiment on twice-reviewed machinery; the review surface is
the fresh data and the two-kind promotion logic.

- Truth audit: an independent stdlib verifier (no generator import) re-derived
  all 100 fresh rows (80 training + 20 gate axis rows) from prompt text alone —
  exhaustive exactly-budget walk enumeration for explore (uniqueness proven),
  record/decoy/narration cross-checks for hygiene — with five planted negative
  controls each firing correctly. Zero wrong answers; zero narration
  inconsistencies; zero training/gate overlap; no closed-axis lesson kinds
  anywhere in either file.
- Generator: byte-identical to the v2 file whose lesson logic two prior
  independent verifiers re-derived; the co-location oversampling is active
  (0.63 training / 0.75 gate among injected rows).
- Promotion logic: the two-kind corrected bar (both must strictly win; ties
  fail; detectability exclusion; fail-closed GATE_UNDETECTABLE) with the
  unconditional recovery flags; 50 unit tests including the new two-kind cases.
- Pins: parent tree/weights/adapter recomputed on disk; seeds
  77119/55121/55/88018/78148 grep-fresh; TODO-pins fail closed by direct
  invocation; overlap receipts span all five predecessor gates.
- Interference-law compliance: this is dose two on the clean designed_fresh
  lineage; the measured interference bound is dose three.

**Verdict:** `PASS_EXPENSIVE_RUN`.
