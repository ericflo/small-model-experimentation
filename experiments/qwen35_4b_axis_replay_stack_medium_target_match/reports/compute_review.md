# Second Adversarial Compute Review

- Two-block MILP solved to zero gap in 0.26 s; both 240-row variable blocks
  match exactly at forward 157,152 / loss-bearing targets 61,443 / loss mass ×5
  71,559; full arms at 1,374,565 / 567,865 / 623,777 with zero deltas, zero
  encoder-skipped rows, 1,280 position-aligned shared rows.
- All stream artifacts regenerate byte-identically under --check, including a
  deterministic MILP re-solve; the trainer encoder is byte-bound to the receipt
  at train time.
- Training geometry frozen: control first, one event per arm, 190 updates,
  LR 1e-5, rank 32 alpha 64, think/close 0.2/0.2, seed 53, warm start from the
  pinned replay_repeat adapter (recomputed hashes) continued in place;
  published-arm pins fail closed once receipts exist.
- The per-row loss-normalization caveat on the mass axis carries over and is
  recorded in the preregistration.

**Verdict:** `PASS_CONTROL_TRAINING`.
