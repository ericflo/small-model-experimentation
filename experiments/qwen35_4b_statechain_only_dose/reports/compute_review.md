# Compute Review

Second adversarial review, scoped to the training events (control first;
its committed receipt gates the candidate).

- Exposure receipt verified at freeze: exact zero-delta MILP on all three
  axes (per-arm forward 1,368,815; nonzero targets 574,630; loss-mass×5
  628,314), 1,280 position-aligned shared rows byte-identical across
  arms, zero encoder-skipped rows, trainer bytes bound (train_think.py
  e0eca2a2…, encode_row byte-identical to the reviewed reference).
- Vehicle: FRESH rank-32/alpha-64 adapter from the authenticated
  hygiene_explore composite via `--model-path` (tree recomputed against
  9eb653d7… before training), no warm-start token anywhere, one epoch
  over 1,520 rows (190 updates), LR 1e-5 cosine warmup 0.03, think/close
  0.2/0.2 answer 1.0, max length 4,096, training seed 67;
  `train_trial.py` refuses the candidate until the control receipt is
  committed and pushed; PUBLISHED_ARM_HASHES pins fail closed on None.
- Both stages require clean pushed green main with the stream receipt and
  this review byte-identical at HEAD; train loss is never capability
  evidence.

**Verdict:** `PASS_CONTROL_TRAINING`.
