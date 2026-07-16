# Compute Review

Scoped to the training events (control first; its committed receipt
gates the candidate).

- Exposure receipt verified at freeze: exact zero-delta MILP on all
  three axes (per-arm forward 1,411,833; nonzero 591,024; mass×5
  644,424), 1,280 position-aligned shared rows, zero encoder skips,
  trainer bytes bound (train_think.py e0eca2a2…, encode_row
  byte-identical to the reviewed reference).
- Vehicle: fresh rank-32/alpha-64 adapters from the authenticated
  zero-root composite via --model-path (tree recomputed against
  414f5829… pre-training), no warm start, the proven statechain recipe
  at training seed 73; train_trial refuses the candidate until the
  control receipt is committed; pins fail closed on None.

**Verdict:** `PASS_CONTROL_TRAINING`.
