# Compute Review

Second adversarial review, scoped to the training events (both arms train
under one frozen recipe; the control trains first and its receipt gates
the candidate).

- Exposure receipt verified at freeze: exact zero-delta MILP on all three
  axes (per-arm forward 1,393,242; nonzero targets 584,414; loss-mass×5
  640,286), 1,280 position-aligned shared core rows byte-identical across
  arms, zero encoder-skipped rows, trainer bytes bound
  (sha256(train_think.py) = e0eca2a2… in the receipt; encode_row
  byte-identical to the reviewed reference).
- Vehicle verified by the design review's contract and seal-safety lenses:
  FRESH rank-32/alpha-64 adapter from the hygiene_explore composite via
  `--model-path` (tree recomputed against 9eb653d7… before training), no
  warm-start token anywhere in the trial or merge scripts, one epoch over
  1,520 rows, batch 1 accumulation 8 (190 updates), LR 1e-5 cosine warmup
  0.03, think/close 0.2/0.2 answer 1.0, max length 4,096, training seed
  61; `train_trial.py` refuses the candidate until the control's receipt
  is committed and pushed; PUBLISHED_ARM_HASHES pins fail closed on None.
- Both stages require clean pushed green main with the stream receipt and
  this review byte-identical at HEAD; train loss is never capability
  evidence; no gate or benchmark artifact is touched by training.

**Verdict:** `PASS_CONTROL_TRAINING`.
