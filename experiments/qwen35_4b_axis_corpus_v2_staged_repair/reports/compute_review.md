# Second Adversarial Compute Review

- Two-block MILP solved exactly in 2.8 s; both 240-row blocks at forward
  134,708 / targets 58,660 / mass ×5 68,780; arms at 1,357,677 / 566,115 /
  621,987 with zero deltas, zero skips, 1,280 aligned shared rows; all
  --check modes regenerate byte-identically; trainer bytes bound to the
  receipt's encoder hash.
- Training frozen: control first, 1,520 rows, 190 updates, LR 1e-5, rank 32
  alpha 64, think/close 0.2/0.2, seed 54, warm start from the pinned
  axis_on_replay adapter continued in place; published-arm pins fail closed.

**Verdict:** `PASS_CONTROL_TRAINING`.
