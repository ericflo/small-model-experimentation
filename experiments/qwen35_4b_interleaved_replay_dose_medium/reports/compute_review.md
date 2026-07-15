# Second Adversarial Compute Review

- Two-block MILP exact in 4.6 s: both 240-row blocks at forward 147,792 /
  targets 63,001 / mass ×5 71,525; arms at 1,373,106 / 579,624 / 633,716 with
  zero deltas, zero skips, 1,280 aligned shared rows; all --check modes
  regenerate byte-identically; trainer bytes bound to the receipt.
- Training frozen: control first, 1,520 rows, 190 updates, LR 1e-5, rank 32
  alpha 64, think/close 0.2/0.2, seed 56, warm start from the pinned
  replay_clean adapter continued in place; published-arm pins fail closed.

**Verdict:** `PASS_CONTROL_TRAINING`.
