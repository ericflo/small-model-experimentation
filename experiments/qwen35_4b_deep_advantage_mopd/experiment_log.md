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
