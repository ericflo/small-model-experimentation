# Compute Review

Scoped to the single training event (stage 8: one replay adapter on the
count_walk composite parent). Backed by the three-lens adversarial
workflow recorded in experiment_log.md (build + review + adversarial
verify at design commit e61915e2); both confirmed MAJORs were fixed
pre-freeze and re-verified directly (the aggregate tie guard reads the
demonstrated 1-ulp flip pair as BOUNDED; the standalone reproduction
path passes sibling-free while in-cell tampers still fail hard).

- Recipe mirrors the chain's proven replay-refresh stages byte-for-byte
  in hyperparameters: fresh rank-32/alpha-64 adapter via the vendored
  train_think.py (sha e0eca2a2…, byte-identical) with --model-path on
  the authenticated count_walk composite (tree d5fdc55c…, weights
  ddd7bc4b… — full model.safetensors hash checked pre-training),
  epochs 1, lr 1e-5, batch 1, grad-accum 8, max-length 4096, w_think
  0.2, w_close 0.2, seed 86. Training pool sft_blend.jsonl sha verified
  25a9595f… (2,240 rows, zero skips, max forward 3,193 < 4,096).
- Parent authentication is fail-closed before training against the
  in-cell sha-pinned merge receipt (sibling original a verification aid
  only); pins fail closed on None; PUBLISHED_ARM_HASHES TODO slot is in
  pre-fill state. No warm start; the count_walk composite is the
  explicit merged parent, not a runtime adapter.

**Verdict:** `PASS_CONTROL_TRAINING`.
