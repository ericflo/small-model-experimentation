# Adversarial Design Review

A single-variable vehicle cell on the most-reviewed pipeline in the program.

- The one behavioral delta (the trainer's `--model-path`) is guarded by an
  AST-comparison test proving `encode_row`, the loss, and the sampler are
  byte-unchanged against the reviewed predecessor, so every exposure receipt
  remains comparable; the design receipt's lifecycle contracts additionally
  require rank 64 / alpha 128 / seed 58 / the model-path pins and FORBID any
  warm-start token in the trial and merge scripts.
- The external merger natively supports `--base-model` with a local composite
  (verified against the peft/transformers sources); no merger fork was needed;
  scale 128/64 = 2.0 and the 128-module checks are unchanged.
- Corpus inherited byte-identical (twice-verified; regenerates from the copied
  generator); exposure exact (MILP optimal, zero deltas, zero skips); gate
  rows at fresh seed 88,021 with zero overlap against all eight predecessor
  gates; the explore and hygiene gate rows were re-derived inline (20/20), and
  the tracefix/protocol rows carry the generator's in-generation executable
  audits verified across four prior independent reviews.
- The ordered verdict partition (instability precedence, then supported, then
  refuted) is unit-tested across its boundaries with the install-preserved
  flag; 60 tests green; smoke green; TODO-pins fail closed; seeds
  55124/58/88021 grep-fresh.

**Verdict:** `PASS_EXPENSIVE_RUN`.
