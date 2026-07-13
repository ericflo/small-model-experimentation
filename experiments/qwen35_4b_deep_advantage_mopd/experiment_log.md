# Qwen3.5-4B Deep-Advantage MOPD Experiment Log

## 2026-07-12 — intake and pre-output design

- Created a new result-bearing directory rather than modifying the completed
  two-teacher experiment.
- Selected the deep-only branch because deep independently passed both
  same-prefix audit contrasts in both predecessor blocks; MOPD remained
  untested only because quick was separately mandatory.
- Reused the exact immutable 40/60 soup by SHA-256 instead of constructing a
  numerically new starting checkpoint. New training artifacts have a separate
  external root.
- Froze two new route blocks, the unchanged strict deep-over-quick-and-student
  rule, five-update locality, four 60-deep/20-soup rounds, three primary seeds,
  and unconditional final comparisons against sources, router, controls, and
  sample-more.
- Added two direct mechanism controls: deep targets on one-to-one matched
  non-deep-selected states, and quick targets on the exact selected states.
  The off-policy continuation control and parameter soups remain.
- Copied the parent harness and procedural gym, then adapted quotas, route gate,
  target-cache inventory, locality mixture, controls, and confirmation without
  generating task-model output.
- Passed 50 isolated tests and all 14 family selftests; verified the exact
  quick, deep, and immutable-soup file hashes. The smoke receipt contains no
  task-model generation.
- Committed the complete frozen design at `1ef1f5ad`, pushed it to shared
  `main`, and wrote `runs/preregistration_receipt.json` with byte hashes for all
  frozen files before any Qwen load.

## 2026-07-12 — pinned-model and installation preflight

- Passed all four pinned-runtime semantic probes and a finite Transformers
  training forward pass; vLLM resolved the registered full/piecewise graph
  geometry.
- Revalidated exact quick, deep, and soup checkpoint hashes and merge receipts.
  On the eight fixed canary prompts, every installed checkpoint differed from
  base; quick/deep differed on 8/8, soup/quick on 8/8, and soup/deep on 7/8.
- The installation gate authorizes the fresh two-block route qualification.
  No route evidence or training output exists yet.

## 2026-07-13 — fresh route qualification passes

- Generated two disjoint 192-state soup blocks and 9,216 continuations: three
  policies × four selection + four audit branches. All state, replay, model,
  engine, and branch-hash checks passed.
- Deep routed on 28/26 states. Independent audit deep-minus-soup was
  +0.16499/+0.12205 by block (pooled +0.14209, one-sided lower bound
  +0.12297); deep-minus-quick was +0.20003/+0.14203 (pooled +0.16910, lower
  bound +0.15337). The frozen deep gate passed.
- Quick also passed diagnostically on 29/18 routes in this new replication,
  unlike its predecessor reversal. The locked treatment remains deep-only;
  the result is evidence for, not permission to improvise, the later
  cross-fitted two-teacher design.
- Downstream authorization is exactly `locality_pilot`. No update has run.

## 2026-07-13 — five-update exact-logit locality passes

- The first two online candidate batches supplied 55/60 required deep routes,
  so the frozen runner opened a third batch rather than weakening the quota.
  Three batches yielded 90 deep routes among 576 failed states.
- The final round manifest contains exactly 60 deep capability units, 20 soup
  anchors, and 60 disjoint non-advantage controls. Control matching used 57
  exact cells and three family/kind fallbacks. Its SHA-256 is
  `f4a1eb0848804ddef874ece0afef89a1ea39d84f2717b971575f2ff5f021e0e8`.
- Cached all three policies for 140 samples and 35,147 active positions. The
  cache SHA-256 is
  `20c21a5bb8e8b8058d0b7377929b37fd5e2eca8e55598d61b6ac827503761e76`.
- The pilot completed all five updates with the exact 15-deep/5-soup mixture.
  Training mean corrected top-50 loss was `0.05242`; held-probe loss improved
  `0.04773→0.02947` and overlap improved `0.84840→0.85163`.
- Exact batch-of-one probes measured centered non-target drift `0.02760`,
  relative entropy drop `0.03112`, and target loss `0.01293→0.01170`. Every
  frozen check passed. The probe is one midpoint token per consumed unit, so
  the authorization is literal and deliberately not described as global
  token-position invariance.
- Downstream authorization is exactly `four_round_mopd`. No capability result,
  control comparison, or benchmark event exists yet.

## 2026-07-13 — seed-42 integration round-1 cache recovery

- Full-dose round 0 passed: 20/20 updates, exact 60-deep/20-soup geometry,
  mean corrected loss `0.05669`, probe loss `0.08318→0.05112`, and
  non-decreasing overlap. Its merged receipt is
  `8432e6391ce1f4ce328938163984f490bc424325832c891a58a5dbf35cb06920`.
- Round 1 used three fresh candidate batches. It found 81 deep routes among
  576 failed states and froze the exact 60-deep/20-anchor/60-control quotas
  (56 controls exact-cell, four family/kind).
- Target-cache construction stopped before loading any policy because one
  matched route-control episode tokenized to `3,203 > 3,072`. This was an
  implementation omission, not a registered scientific stop: the fixed
  training length existed, but completion-preserving prompt fitting did not.
- Recovery keeps the frozen `3,072` budget and every completion/target
  position, deterministically left-truncates only oldest prompt tokens, records
  the exact cut in cache/training receipts, and still fails if a completion
  leaves no causal prompt token. Regression coverage reproduces the exact
  `3,203→3,072` case. Existing states, branch ledgers, routes, quotas, and
  model outputs are reused unchanged.

## 2026-07-13 — live integration integrity audit

- Round 1 subsequently passed its frozen gate: 20/20 updates, mean corrected
  loss `0.04901`, probe loss `0.03915→0.02020`, and overlap
  `0.84121→0.84604`. Exactly 60 deep and 20 anchors were consumed with zero
  prompt truncation. The sole 131-token cut remains a cache-only matched
  route-control sample.
- An independent math audit confirmed the teacher-top-50 corrected reverse-KL
  value, gradient, causal indexing, masks, quotas, and no-hint construction.
  It also exposed prospective fail-open edges before they affected a primary
  update.
- Cache creation now fails before model load if a capability or anchor prefix
  would be shortened, and every trainer independently rejects any shortened
  sample selected by its arm. Per-role truncation counts are receipted. The
  known round-1 route control therefore requires a deterministic full-prefix
  replacement from the already-frozen candidate pool before that control may
  run; the primary manifest/cache remain immutable.
- Cache creation, resume, and training now bind stage, frozen config hash,
  top-k, and exact quick/deep/soup paths plus model-config and merge-receipt
  hashes. Existing round-0/1 caches pass the strengthened validator.
- Fixed a pre-control implementation error: off-policy pressure probes had
  taken the first eight lexicographic units (5/3 and 1/7 capability/anchor in
  actual rounds 0/1). They now use and receipt the registered deterministic
  6-capability/2-anchor geometry. The frozen initial-objective-loss scaling
  definition is unchanged.
- Full-round probe entropy contracted `10.28%` in round 0 and `12.33%` in
  round 1. This is not a registered full-round stop—the 10% entropy ceiling is
  specific to locality—so no post-hoc gate was added. It is preserved as a
  collapse-risk warning for confirmation and final interpretation.
- All 58 experiment tests pass after the safeguards. Seed 42 remains in
  progress; no capability or control comparison exists yet.
