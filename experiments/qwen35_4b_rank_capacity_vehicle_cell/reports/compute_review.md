# Second Adversarial Compute Review

- Exposure exact at slot seed 55,124: variable block equals the notional
  control on all three axes (forward 129,814 / targets 55,929 / mass ×5
  65,637); per-arm totals 1,351,265 / 566,044 / 621,352; 1,520 rows, zero
  skips; encoder hash bound to the per-experiment trainer.
- Training frozen: ONE event, fresh rank-64/alpha-128 adapter (no warm start)
  on the pinned clean-parent composite via `--model-path` (full-weights hash
  preflight + tokenizer-file hashes), 190 updates, LR 1e-5, think/close
  0.2/0.2, seed 58; the published-arm pin fails closed.

**Verdict:** `PASS_CONTROL_TRAINING`.
