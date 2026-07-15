# Second Adversarial Compute Review

- Standard exact-exposure stream solved and receipted (1,520 rows/arm, zero
  skips, encoder bound); only the candidate stream trains — the matched
  control block is recorded for bookkeeping.
- Training frozen: one event, 190 updates, LR 1e-5, rank 32 alpha 64,
  think/close 0.2/0.2, seed 57, warm start from the pinned designed_fresh
  adapter continued in place.

**Verdict:** `PASS_CONTROL_TRAINING`.
