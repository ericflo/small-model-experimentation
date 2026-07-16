# Compute Review

Scoped to the training events (control first; its committed receipt
gates the candidate).

- Exposure receipt verified at freeze: exact zero-delta MILP (per-arm
  forward 1,359,192; nonzero 567,805; mass×5 621,517), 1,280
  position-aligned shared rows, zero encoder skips, trainer bytes bound
  (train_think.py e0eca2a2…, encode_row byte-identical).
- Vehicle: fresh rank-32/alpha-64 adapters from the authenticated
  zero_root composite via --model-path (tree recomputed against
  414f5829… pre-training), no warm start, training seed 79;
  train_trial refuses the candidate until the control receipt is
  committed; pins fail closed on None.

**Verdict:** `PASS_CONTROL_TRAINING`.
