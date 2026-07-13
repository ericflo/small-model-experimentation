# Counterfactual evidence-acquisition curriculum experiment log

## 2026-07-13 — Direction claimed

- Created a new experiment under `agentic_breadth_installation`, with Active
  Evidence Acquisition and Process Control as supporting programs.
- Direct trigger: the opened trajectory contrast from
  `qwen35_4b_semantic_policy_headroom_tournament` moved the candidate bottleneck
  from post-failure revision to evidence acquisition before the first patch.
- Preserved the predecessor's formal `INSTRUMENT_FAIL`; its trajectories are a
  hypothesis generator, not an inherited positive result.
- Fixed the training parent to transaction replay
  (`1cf5fbca...41ba3`) and the locality/Menagerie anchor to C54 apex
  (`c9331680...08d5`).

## 2026-07-13 — Related-work and collision review

- Ran repository related-work discovery for counterfactual evidence
  acquisition, early hypotheses, tool-mediated specification, and first-patch
  control.
- Reviewed prior active example/trace selection, adaptive evidence budgeting,
  learned trace policy, conditional recovery banking, transaction curriculum,
  and semantic-policy headroom work.
- Audited the newly landed
  `qwen35_4b_early_text_hypothesis_forking` preregistration. It is now named as
  the closest conceptual near-neighbor. Its external early hypothesis bank and
  visible selector do not answer autonomous tool acquisition or weight
  installation; this experiment makes no generic early-proposal novelty claim.
- Confirmed that the concurrent landing added no claim-ledger entry and creates
  no claim-ID collision.

## 2026-07-13 — Design changes from adversarial review

- Made the counterfactual dyad the primary unit: both branches must inspect the
  discriminator, pass their own first-patch executables, and cross-fail.
- Replaced fixed evidence locations with disjoint bank, qualification, and
  transfer path skins across tests, docs, and callsites.
- Added an unseen signature-query transfer skin after reference/symbol
  training and qualification.
- Added deterministic matched-operator nondiscriminating search as a mandatory
  qualification control. Its output must exclude the evidence path and marker;
  correct search evidence must beat it by 30 points in both blocks.
- Added explicit-contract no-search retention to reject an unconditional
  always-inspect policy.
- Strengthened sample-more from equal reserved turns to outcome-blind prefixes
  that overmatch actual sampled and logical model-token costs. The stronger of
  start and incumbent pools is mandatory; full pools are oracle-only.
- Added composed-map receipts from task/query through evidence bytes to first
  patch, following the repository lesson that independently rotating mappings
  can still cancel in composition.
- Added start-to-apex locality feasibility before training. The exact start
  must clear the new 0.10 drift ceiling on the exact new contexts so inherited
  drift cannot be attributed to the acquisition update.
- Added two old-family retention blocks with normal/recovery success guards,
  0.95 rejected/failed transition floors, 0.90 verify/commit floors, interface
  ceilings, and per-family regression bounds.
- Kept entropy and varentropy strictly diagnostic. They do not select rows,
  labels, pressure, checkpoints, or benchmark routes.
- Added a two-phase lifecycle seal: every scientific stop writes an open
  terminal disposition; only a separately pushed documentation commit can
  authorize its closeout receipt; and a post-push verifier requires the closed
  receipt, finished status registry, result brief/chart, generated indexes, and
  `make check` to agree on clean `origin/main`.
- Closed the final adversarial seal gaps before design lock: one exact full-file
  lock schema; canonical model-facing config and checkpoint-role validation;
  pinned model/generation configs and merge receipts; smoke input freshness;
  and reservation/digest-authenticated Menagerie events tied to public Git-tree
  metadata. Expanded tests from 43 to 52 without constructing a model or
  reading benchmark contents.

## 2026-07-13 — Frozen training contract

- Three fixed arms: aligned `evidence_binding`, equal-dose
  `explicit_redundant`, and within-dyad `shuffled_binding`.
- Each arm combines 24 new tasks with 24 prior complete-loop task blocks and is
  padded to 48 rows at each of nine conditional transitions, 432 rows total.
- Each transition receives exactly 16,000 weighted answer tokens per epoch.
  Think loss is zero.
- Training is rank 32, alpha 64, LR 2e-5, dropout 0.05, three epochs, batch 4 ×
  nine-transition accumulation, maximum encoded length 4,096, seed 53.
- The primary arm is fixed before controls; no post-result arm substitution is
  permitted.

## 2026-07-13 — Model-free readiness

- Generated `reports/context_geometry_receipt.json` without loading model
  weights or producing model output. All
  registered synthetic histories fit the 16,384-token window at answer budget
  4,096; worst-case headroom is 10,163 tokens, above the 512-token safety bar.
- Drafted the intake, preregistration, adversarial design review, report, and
  external-artifact plan.
- Deterministic smoke passed: all three arms encoded 432 rows, each of the nine
  conditional transitions has 48 rows and approximately 16,000 weighted action
  tokens, the aligned/shuffled multisets and dyad exchange checks pass, every
  oracle patch cross-fails its counterpart, and exact start/anchor locality
  prompt token IDs match on 48 contexts.
- No model output, adapter, merged candidate, behavioral score, Menagerie event,
  or scientific result exists.
- No file under `benchmarks/` was read or imported.

## Next boundary

1. Finish the post-smoke adversarial implementation review.
2. Commit and push the frozen design directly to `main`.
3. Write, commit, and push the digest-bound design-lock receipt.
4. Run exact start-to-apex locality feasibility.
5. Only a pass may authorize the outcome-free interface ladder.
