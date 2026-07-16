# Compute Review

Second adversarial review, scoped to the training events (control first;
its committed receipt gates the candidate).

- Exposure receipt verified at freeze: exact zero-delta MILP on all three
  axes at the enlarged stream (per-arm forward 1,878,709; nonzero targets
  771,405; loss-mass×5 867,281; 2,280 rows/arm; 285 optimizer updates),
  1,280 position-aligned shared core rows byte-identical across arms,
  zero encoder-skipped rows, trainer bytes bound (encode_row
  byte-identical to the reviewed reference); the control block's
  575-row multiplicity-2 repetition is solver-proven minimal, disclosed,
  and direction-of-bias stated.
- Vehicle: FRESH rank-32/alpha-64 adapter from the authenticated
  hygiene_explore composite via `--model-path` (tree recomputed against
  9eb653d7… before training), no warm-start token anywhere, one epoch,
  batch 1 accumulation 8, LR 1e-5 cosine warmup 0.03, think/close 0.2/0.2
  answer 1.0, max length 4,096, training seed 71; `train_trial.py`
  refuses the candidate until the control receipt is committed and
  pushed; PUBLISHED_ARM_HASHES pins fail closed on None.
- Both stages require clean pushed green main with the stream receipt and
  this review byte-identical at HEAD; train loss is never capability
  evidence.

**Verdict:** `PASS_CONTROL_TRAINING`.
