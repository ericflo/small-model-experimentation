# Second Adversarial Compute Review

Scope: the exact-exposure compute contract that training relies on, re-verified
independently of the authors of the matching code.

- The MILP selection (80/80/240 with per-row disjointness and control-minus-filler
  equality) algebraically forces the three 240-row variable blocks equal on all
  three axes; recomputation from raw manifest indices reproduces forward 142,211,
  nonzero targets 63,739, and loss mass ×5 72,755 per block, and arm totals
  1,356,964 / 576,718 / 631,326 with zero deltas.
- All three stream files byte-reconstruct from manifest indices plus the slot
  permutation; exactly 1,280 byte-identical lines at identical positions; zero
  duplicate replay lines; zero partial collisions in variable slots.
- The trainer encoder (`encode_row`, weights 0.2/0.2, max length 4,096) is the same
  code path in measurement, validation, and training; `encoder_sha256` matches; and
  train-time now binds the trainer bytes to the receipt (review fix 6).
- Zero-skip is fail-closed at validation (max row 3,193 < 4,096) and re-enforced at
  train time from the trainer log (1,520 examples, 0 skipped required).
- Training geometry is frozen: 1 epoch, batch 1, accumulation 8, 190 updates, LR
  1e-5, seed 51, warm start from the pinned parent adapter (`bb59d3bd...`), one
  event per arm, control first.
- Caveat carried into interpretation: per-row loss normalization means matched
  total mass does not linearly control gradient magnitude; row counts, update
  counts, and the other two axes are matched simultaneously.

**Verdict:** `PASS_CONTROL_TRAINING`.
