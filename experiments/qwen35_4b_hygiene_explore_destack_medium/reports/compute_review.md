# Second Adversarial Compute Review

- Two-block MILP exact in 0.65 s: both 240-row blocks at forward 139,986 /
  targets 58,961 / mass ×5 67,773; arms at 1,367,212 / 574,619 / 629,207 with
  zero deltas, zero skips, 1,280 aligned shared rows; all --check modes
  regenerate byte-identically; trainer bytes bound to the receipt.
- Training frozen: control first, 1,520 rows, 190 updates, LR 1e-5, rank 32
  alpha 64, think/close 0.2/0.2, seed 55, warm start from the pinned
  designed_fresh adapter continued in place; published-arm pins fail closed.

**Verdict:** `PASS_CONTROL_TRAINING`.
