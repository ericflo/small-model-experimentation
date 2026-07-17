# Compute Review

Scoped to the single training event (stage 9: one state-tracking
adapter on the count_walk composite parent). Backed by the three-lens
adversarial workflow recorded in experiment_log.md (build + review +
verify at design commit ffc78ec2), which found ZERO MAJOR and zero
minor findings; the curriculum lens independently re-derived the ledger
rows and I re-derived two by hand (Talu=14, Talu=17) — both correct.

- Recipe: fresh rank-32/alpha-64 adapter via the vendored train_think.py
  (sha e0eca2a2…, byte-identical) with --model-path on the authenticated
  count_walk composite (tree d5fdc55c…, weights ddd7bc4b… — full
  model.safetensors hash checked pre-training), epochs 1, lr 1e-5, batch
  1, grad-accum 8, max-length 4096, w_think 0.2, w_close 0.2, seed 87.
  Corpus sha 66a8d5be… (160 rows, single kind u_state_track); token
  receipt max forward 775 < 4,096 → 0 skips enforced.
- Divergent-skill rationale (recorded pre-event): replay compounding
  BOUNDED at stage 8 by re-saturating its own distribution; a
  non-overlapping skill (state-tracking execution) is a different
  distribution and may add — count_walk's own enumeration dose added
  +0.036 over its parent. Honest prior P(INSTALLED_TRANSFER) ≈ 0.30-0.40.
- Parent authentication fail-closed before training against the in-cell
  sha-pinned merge receipt (sibling original a verification aid only);
  pins fail closed on None; PUBLISHED_ARM_HASHES slot in pre-fill state.

**Verdict:** `PASS_CONTROL_TRAINING`.
