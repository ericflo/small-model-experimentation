# Second Adversarial Compute Review

Scope: the exact-exposure compute contract for the two training arms.

- The two-block MILP (fillerC 80 + control 240, disjoint, control − filler =
  treatment vector) solved to zero gap in 13.2 s; both 240-row variable blocks
  match exactly at forward 134,606 / loss-bearing targets 58,396 / loss mass ×5
  67,928, and full arms at 1,347,403 / 567,653 / 622,729 with zero deltas and
  zero encoder-skipped rows (max row well under the 4,096 window).
- All stream artifacts regenerate byte-identically under `--check`, including a
  deterministic MILP re-solve; 1,280 shared rows are position-aligned and
  byte-identical across the two arm files.
- The trainer encoder is the same code path in measurement, validation, and
  training; train-time binds the trainer bytes to the receipt's encoder hash.
- Training geometry frozen: one event per arm, control first, 190 updates,
  LR 1e-5, rank 32 alpha 64, think/close weights 0.2/0.2, seed 52, warm start
  from the pinned `designed_fresh` adapter continued in place; the published-arm
  hash pins fail closed once each receipt exists.
- The per-row loss-normalization caveat on the mass axis carries over from the
  predecessor and is recorded in the prereg's interpretation limits.

**Verdict:** `PASS_CONTROL_TRAINING`.
