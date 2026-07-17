# Compute Review

Scoped to the training events (control first; its committed receipt
gates the candidate).

- Exposure receipt verified at freeze: exact zero-delta MILP (per-arm
  forward 1,436,178; nonzero 572,724; mass×5 629,552), 1,280
  position-aligned shared rows, zero encoder skips, trainer bytes bound
  (encode_row byte-identical to the reviewed reference).
- Vehicle: fresh rank-32/alpha-64 adapters from the authenticated
  zero_root composite via --model-path (tree recomputed against
  414f5829… pre-training), no warm start, training seed 83, one kind at
  full concentration (160 rows u_enum_repair) per the dilution rule;
  train_trial refuses the candidate until the control receipt is
  committed; pins fail closed on None.

**Verdict:** `PASS_CONTROL_TRAINING`.
