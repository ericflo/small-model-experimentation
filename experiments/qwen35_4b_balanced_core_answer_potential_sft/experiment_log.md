# Qwen3.5-4B Balanced-Core Answer-Potential SFT Experiment Log

## 2026-07-12 — Resource-Constrained Fork

The parent long-horizon experiment was paused after 331/1,080 train tasks because observed throughput made
the remaining nine-family schedule incompatible with the user's time budget. No saved shard was lost:
21,184 traces and 97,883,041 sampled thought tokens are present behind atomic SHA-256 receipts.

The user selected the balanced-core funnel. Repository lifecycle rules require a new experiment because
calibration results are already visible. This fork declares those observations, selects the already-leading
three complete family blocks, adds shortest-natural as the strongest calibration control, replaces slow HF
train scoring with a broader parity-gated exact vLLM readout, drops pivot branches, and makes full evaluation
conditional.

At this boundary the GPU is idle. No remaining harvest task, training score, R1 train rollout, SFT update,
or evaluation generation has run under this experiment.

## 2026-07-12 — Immutable Design Anchor

- Prospective README, preregistration, adversarial review, full restartable harness, frozen data, and 40
  passing CPU tests were committed at original `c847615f` before any experiment GPU call.
- The configured guard now points to that commit and its three exact file digests. It fails before model
  load if ancestry or content identity changes.
- After the anchor, the README's first relative link was moved below the generated summary paragraph so the
  repository catalog resolves it from the correct directory. This is a navigation-only repair; the guard
  still verifies that byte-exact prospective design.

## 2026-07-12 — Concurrent-Main Rebase

- Rebasing over three concurrent site commits changed the design anchor from original `c847615f` to
  `cb3d64e3`. All three frozen-file SHA-256 values remained identical.
- The configured ancestry pointer was re-anchored to the rebased commit without changing any design text,
  threshold, split, arm, or code. No experiment GPU call had run.

## 2026-07-12 — Balanced Harvest And Scorer Instrument Stop

- Imported all 331 parent shards at the frozen index digest, then completed exactly 29 Runeward-L3 tasks.
  Final pool: 360 tasks, 23,040 traces, 108,759,239 sampled thought tokens, 22,681 natural closes, four exact
  loops, finite priors on every trace, and no task requiring a top-up.
- The registered task-diverse 32-row joint parity gate then failed closed at 0.692447 > 0.15. No training
  score, R1 train rollout, selection, adapter update, or evaluation had run.
- Inspection of the instrument receipt showed the known long-prefix batch-sensitivity boundary: answer gain
  max 0.147865, joint-likelihood mean-token max 0.054477, empty-answer max 0.156281, and parent-normalized
  joint-gain max 0.692447. No threshold or row was changed.
- Added the pre-outcome amendment to retire vLLM bulk likelihoods and use the single-context Transformers bf16
  reference uniformly. The failed receipt remains evidence; all generation and later evaluation stay on vLLM.

## 2026-07-13 — Post-Score, Pre-Official-Selection Balance Deviation

- Exact scoring completed for all 360 tasks and 22,681 natural traces. Before selection or training, a
  read-only preflight found that the near-best diversity helper could return one row and silently remove the
  entire task from every arm.
- On the frozen scores, unchanged behavior would have retained only 116 tasks, distributed 23 Caravan, 71
  Foundry Ledger, and 22 Runeward. This violates the declared balanced-core estimand.
- Because the complete candidate score bank and the induced imbalance were observed before this repair was
  committed, it violates the preregistration's amendment-timing rule. It is a post-score deviation, not a
  prospective amendment; no later seal can restore that status. Partial incomplete-R1 labels were later
  inspected for cost planning before commit, but did not determine the repair. Official selections, adapters,
  and held-out outcomes remained unseen.
- Repaired the contradiction by keeping near-best diversity when available and otherwise taking the
  deterministic second-ranked trace from the same frozen top-12. Added fail-closed assertions for 360 total
  tasks, 40 per family/level cell, and 720 rows per arm. No selection artifact or adapter existed.
- The same audit found that Trainer seeding occurred after LoRA construction. Added global pre-model seeding,
  an immediate pre-adapter seed reset, receipt provenance, and CPU ordering tests before any training run.
- Hardened training, merge, deployment-probe, and evaluation restart contracts before any adapter existed:
  exact two-epoch exposure/steps, initial LoRA digest, training-lock/final-artifact hashes, complete 128-pair
  merge enforcement, deployed-file fingerprints, and prompt/sampling/runner-bound generation caches.
- Replaced the shuffle control's cyclic heuristic with exact minimum-cost forbidden-edge assignment and made
  Stage-A baseline ties conservative and explicit.
- On the frozen score bank, the balance fallback applies to 5/360 answer tasks and 244/360 joint tasks. Answer
  fallback gaps have median 3.409 and maximum 7.378 nats/answer-token; joint fallback gaps have median 0.893,
  mean 1.410, p90 2.954, and maximum 12.128. The joint arm is consequently a registered hybrid treatment;
  its result will be stratified by mode/gap.
- Shuffle rows now namespace both target and actual-source selection/quality provenance, and their unprefixed
  audit fields describe the trace actually trained. Training receipts are invalidated before artifact
  replacement, exclude themselves from artifact hashes, and are installed atomically for safe restart.

## 2026-07-13 — Training Cost Re-estimate

- In-memory selection over all 360 completed score shards (without writing official selection artifacts)
  gives 32,187,564 two-epoch forward tokens across the five rollout-independent arms. Current-path stress
  receipts imply 9.0--16.9 GPU-hours for those five; allowing the unfinished success arm gives a provisional
  six-arm range of roughly 9.3--20.7 GPU-hours.
- This materially exceeds the user's stated time constraint. Finish and bank the already-running R1 rollout
  and exact selection, but do not start SFT until choosing between the frozen full matrix and a smaller
  prospective follow-up. No licensed shortcut exists inside this frozen experiment before mandatory Stage A.

## 2026-07-13 — Exact Evidence Bank Complete

- Single-context bf16 SDPA scoring finished 360/360 tasks and 22,681 eligible rows in 17,296 seconds.
- R1 finished 360/360 tasks and exactly one rollout for each of the same 22,681 trace IDs in 10,915 seconds.
- Before any seal write, the full read-only validator confirmed all three exact task scopes, every shard hash,
  raw-to-score/R1 source links, per-shard task identity, unique trace IDs, exact score/R1 joins, and the
  natural-close/non-loop eligibility set. The raw pool has 23,040 rows; score and R1 each have 22,681.
- Selection remains absent and blocked pending the committed post-score/partial-rollout deviation seal.

## 2026-07-13 — Pre-Selection Evidence Boundary Sealed

- The seal first repeated the complete read-only raw/score/R1 validation, then added only retrospective
  operation-contract attestations to the three legacy indexes. It records that these contracts were not
  emitted by the original generation/scoring processes.
- Pre-attestation index SHA-256 values are `6aeae76f...24d3`, `c0fed08b...db8`, and `9a9ab75b...71f` for
  raw, exact scores, and R1. Final sealed index SHA-256 values are `f635d060...18fd`, `e2b0a402...5740`, and
  `b116eea5...eea0` respectively; exact full values live in the tracked machine receipts and artifact manifest.
- The amendment receipt binds rebased commit `0a6ccf6d68c79bf80705f48a3de58ad06a0a57ec`, every transitive
  selection/training dependency, all procedural task data, the three final evidence indexes, and the explicit
  post-score/partial-rollout deviation disclosure. It also proves that official selection and adapter
  artifacts were absent at seal time.
- A second seal invocation was byte-idempotent across all three external indexes and all four tracked seal
  receipts. Official selection remains absent until this boundary is committed and pushed.
