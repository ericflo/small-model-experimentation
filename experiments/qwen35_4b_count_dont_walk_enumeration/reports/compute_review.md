# Compute Review

Scoped to the training events (control first; its committed receipt
gates the candidate). Backed by the three-lens pre-GPU adversarial
review recorded in experiment_log.md (2026-07-16, commit 1492fbea): the
pins lens live-verified every item below rather than reading it.

- Exposure receipt verified at freeze: exact zero-delta MILP (per-arm
  forward 1,438,010; nonzero 564,379; mass×5 621,239), 1,280
  position-aligned shared rows, zero encoder skips, trainer bytes bound
  (token_receipt.encoder_sha256 recomputed equal to sha(train_think.py);
  enforced constants at train_trial.py:65-67).
- Vehicle: fresh rank-32/alpha-64 adapters from the authenticated
  zero_root composite via --model-path (full 9GB tree+weights hashes
  independently recomputed pre-review: tree 414f5829…, weights
  6e9aad25…), no warm start, training seed 85 (84 taken; substitution
  recorded in the design receipt with a log errata on the receipt's
  wording), one kind at full concentration (160 rows u_count_walk) per
  the dilution rule; both arm streams sha-verified (replay_ctl7
  94e8259e…, count_walk 71291542…); train_trial refuses the candidate
  until the control receipt is committed and re-validates the published
  prerequisite arm; pins fail closed on None (probed).

**Verdict:** `PASS_CONTROL_TRAINING`.
