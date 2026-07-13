# Qwen3.5-4B Balanced-Core Answer-Potential SFT

**Status:** in-progress · since 2026-07-12 · Reasoning bank built; the six-arm selection training has not been run.

## Status

Prospective resource-constrained follow-up to
`qwen35_4b_long_horizon_answer_potential_sft`.
The original balanced-funnel design was frozen before the remaining 29 harvest tasks, any training-pool
scoring, any SFT, or any held-out evaluation. A selector balance defect was then discovered only after all
candidate scores existed; its repair is transparently classified as a post-score/partial-rollout,
pre-official-selection implementation deviation, not a prospective amendment. Partial R1 success labels were
subsequently inspected for cost planning before the deviation was committed; they did not determine the repair. No official SFT
dataset, adapter, or held-out outcome existed, and no capability result exists yet.

The preserved parent is [linked here](../qwen35_4b_long_horizon_answer_potential_sft/README.md).

This fork preserves the sunk cost of 331 complete, atomic task shards while imposing a hard compute
funnel: finish exactly 360 balanced tasks, score only the independent N=64 pool, skip pivot branching,
train six discriminating arms, and run a small mandatory evaluation before any optional expansion.

## Research Programs

- Primary: `posttraining_and_adaptation`.
- Secondary: `evidence_conditioned_selection` and `test_time_reasoning_budget`.
- Closest near-duplicate: `qwen35_4b_long_horizon_answer_potential_sft`, whose original nine-family,
  95,040-candidate protocol remains frozen and unfinished after calibration and 331/1,080 train tasks.
- Other anchors: C51 (cap-bound answer potential), C28 (own successful thoughts can be
  rationalizations), C50 (the answer-emission seam matters), and C24 (banking gains are driven by
  distinct data rather than repeated exposure).

## Question

On a balanced three-family pool of complete, naturally closed Qwen3.5-4B thoughts, does banking traces
selected by canonical-answer likelihood produce better fresh behavior than banking:

1. length-matched random natural thoughts;
2. R1 answer-success rejection samples;
3. the two shortest eligible thoughts; or
4. the same potential-selected thought multiset reassigned to other tasks?

The second treatment asks whether joint likelihood of the close/answer boundary plus the correct answer
is better than answer-only likelihood.

## Why This Is A Separate Experiment

The parent experiment already exposed calibration results and partial-harvest runtime. Its preregistration
cannot honestly be rewritten. This fork is prospectively frozen after those observations and scopes its
claim accordingly.

Observed parent evidence that informs, but cannot confirm, this design:

- 8,640 calibration traces yielded answer-gain AUROC 0.597 and joint-gain AUROC 0.678;
- top-one answer/joint selections improved R4 answer-rollout success by +6.84/+6.25 points over seeded
  random;
- negative length was stronger (AUROC 0.690 and top-one success 26.56%); and
- the first 331 train tasks required 97,883,041 thought tokens, making the nine-family schedule too slow.

Therefore shortest-natural is a mandatory control, calibration is treated only as design input, and all
claims come from new SFT outcomes on sealed evaluation tasks.

## Model, Firewall, And Inherited Data

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Fresh procedural tasks copied into this experiment from the firewall-clean parent harness.
- No content under `benchmarks/` is read, imported, or used for training.
- Canonical answers are training-time curation instruments and never appear in deployment prompts.
- The 331 inherited task shards remain immutable and are imported by recorded index SHA-256 plus each
  shard's existing SHA-256 receipt. The remaining 29 use the exact same per-task N=64 vLLM protocol.

The inherited source index is
`/workspace/large_artifacts/qwen35_4b_long_horizon_answer_potential_sft/pools/train_independent/index.json`
at SHA-256 `da09176ddf05918712913b4c66ca893f47ed141986f8ed37ef289a63dc37fb63`.
It contains 331 tasks, 21,184 traces, 97,883,041 sampled thought tokens, 20,917 natural closes, and four
exact periodic loops.

## Balanced Core

The core is the first complete family blocks in the parent's pre-existing train order, not families
chosen by score or correctness:

| family | levels | tasks per level | tasks | candidates |
| --- | --- | ---: | ---: | ---: |
| `caravan` | 1--3 | 40 | 120 | 7,680 |
| `foundry_ledger` | 1--3 | 40 | 120 | 7,680 |
| `runeward` | 1--3 | 40 | 120 | 7,680 |
| **total** |  |  | **360** | **23,040** |

The only remaining generation is 29 `runeward` level-3 tasks. The hard stop is 360 tasks: no fourth
family, no pivot branches, and no adaptive enlargement based on outcomes.

## Natural-Thought Protocol

Independent thoughts use temperature 1.0, top-p 0.95, top-k 20, and a 12,288-token natural-close
allowance. A non-loop allowance contact receives one exact-prefix continuation of at most 2,048 tokens.
Only traces that emit their own `</think>`, are not exact periodic loops, and fit the 16,000-token training
record are eligible. No incomplete thought is force-closed into SFT.

If a task has fewer than two eligible traces after N=64, it receives up to four deterministic N=16 top-up
batches. A task still deficient is excluded symmetrically from all trace arms.

## Fast Canonical Scoring

For task prompt `x`, complete thought `z`, boundary `b = </think>\n\nANSWER: `, and canonical answer
`y*`:

```text
answer_gain(z) = log p(y* | x, z, b) - log p(y* | x, empty, b)

joint_gain(z)  = log p(b, y* | x, z) - log p(b, y* | x, empty)
```

The initial candidate instrument used the experiment-local vLLM exact targeted-token readout. It teacher-forces the
observed prefix, reads raw target-token log probabilities, never constrains the sampled token, and bypasses
only the unused vocabulary-rank reduction. Before bulk scoring, 32 fixed inherited calibration traces must
match the Transformers bf16 full-prefix scorer within 0.15 mean nats/token for answer likelihood, joint
likelihood, both empty baselines, and both gains. Any parity failure blocks scoring.

Only the canonical boundary is scored at train scale. The parent's calibration already measured canonical
versus one-newline rank stability (task-macro Kendall tau-b 0.841), so recomputing the second format would
spend substantial prefix work without changing the frozen selector.

The broadened 32-row gate failed before bulk scoring (maximum 0.692447 versus 0.15), exposing known
batch-sensitive long-prefix logits. The threshold was not relaxed. Per the dated preregistration amendment,
vLLM is retired for train likelihoods and every canonical answer/joint score is now computed by the
single-context Transformers bf16 reference uniformly. Generative comparisons remain vLLM-only.

## Selection And SFT Arms

Selection is within task and keeps two full traces per arm:

| arm | trace target | purpose |
| --- | --- | --- |
| `random_natural` | eligible trace nearest each answer-treatment length | long-thought/style control |
| `success_rft` | R1-successful trace nearest each answer-treatment length | ordinary rejection sampling |
| `shortest_natural` | two shortest eligible traces | strongest observed calibration control |
| `answer_potential` | answer-gain quality first, structural diversity second | original treatment |
| `joint_potential` | joint-gain quality first, structural diversity second | close/commit treatment |
| `potential_shuffle` | answer-treatment multiset reassigned within family/level/length | task-specific content control |

Potential selection retains the top 12 by score, takes the best, then the structurally most distant trace
within 0.25 nats per answer token. If that band contains no unused second trace, it deterministically uses
the second-ranked member of the same frozen top 12 so every balanced task still contributes two rows. It
never rewards brevity. The shortest arm is intentionally not token-matched; its lower token dose is part of
the mechanism being tested and is reported.

The pre-selection audit found that this fallback is rare for answer potential (5/360 tasks) but common for
joint potential (244/360 tasks). The joint arm is therefore explicitly interpreted as a best-plus-diverse-or-
second-ranked hybrid, and results must be stratified by selection mode and score gap rather than described as
a uniformly near-best-diverse treatment. This disclosure narrows the selector claim; a commit-and-evidence
seal prevents any further selector change after the completed score bank was inspected.

All arms otherwise use identical QLoRA settings: rank 32, alpha 64, dropout 0.05, two epochs, learning rate
2e-4, batch 1 x gradient accumulation 16, maximum length 16,000, prompt loss 0, thought loss 0.5, and
boundary/answer loss 1.0. Rows and optimizer exposure are matched; `success_rft` is deterministically
oversampled only when it has fewer eligible rows. Adapters and merged checkpoints remain outside git.

## Staged Evaluation

All evaluation is natural-thinking vLLM with a 12,288-token allowance. Merged checkpoints must first
produce a real same-prompt behavioral difference from base.

Mandatory Stage A evaluates base plus all six arms greedily on sealed subsets fixed by task metadata:

| split | construction | tasks |
| --- | --- | ---: |
| core IID | 3 train families x L1--L3 x 20 | 180 |
| core hard | 3 train families x L4 x 20 | 60 |
| held family | brinework/spindle x L1--L3 x first 10 | 60 |

Primary metric: core-IID exact-answer accuracy. Report paired 10,000-resample task bootstraps, parse rate,
natural-close rate, family macro, thought lengths, and actual forward tokens.

For each potential treatment, the strongest trace baseline is the highest-accuracy member of
`random_natural`, `success_rft`, and `shortest_natural`. Stage B triggers only if the treatment:

- beats that baseline by at least 0.03 core-IID accuracy with paired 95% lower bound above zero;
- beats `potential_shuffle` pointwise in the aggregate;
- loses no more than 0.02 parse rate or family macro; and
- has a mathematically reachable registered gate.

If no treatment triggers, the experiment stops after Stage A. If one triggers, Stage B runs all-arm greedy
evaluation on the full inherited IID/hard/held/rendering splits, k=8 only for base, the winning treatment,
its strongest baseline, shortest-natural, and shuffle, and training-seed-43 replication for the treatment
and strongest baseline. A mission-level positive additionally requires the trained method to beat base
sample-more at matched actual forward tokens.

## Verdicts

- `CORE_BANKING_NEGATIVE`: neither potential arm clears the Stage-A trigger.
- `POTENTIAL_BANKING_POSITIVE`: a potential arm clears Stage A and the full Stage-B comparison while
  preserving interface metrics.
- `REPLICATED_BANKING_POSITIVE`: seed 43 has the same sign and the pooled paired interval excludes zero.
- `MISSION_POSITIVE`: replicated positive plus a matched-compute win over base sample-more.
- `SHORTEST_BANKING_LEADS`: shortest-natural is the strongest trace arm; this supports a compression or
  optimization mechanism, not answer-potential selection.

The three-family result cannot support a nine-family claim. A null is a core-scope null, and any broader
follow-up must be a new experiment.

## Run

```bash
.venv-vllm/bin/python experiments/qwen35_4b_balanced_core_answer_potential_sft/scripts/run.py --stage smoke
.venv-vllm/bin/python experiments/qwen35_4b_balanced_core_answer_potential_sft/scripts/run.py --stage full
```

The granular path is `import -> harvest -> parity -> score -> rollouts -> evidence-seal -> select -> train ->
merge -> deployment-probe -> evaluate-stage-a -> analyze-stage-a`, followed only conditionally by Stage B.
`evidence-seal` is a one-time retrospective attestation for the legacy indexes; `select` remains blocked until
that seal and the post-score deviation are committed in the machine amendment receipt.

`full` is resume-only across these commit boundaries: it intentionally stops if the evidence/amendment seal
is not committed, and a fresh selection is not trainable until the byte-identical tracked SFT manifest and
selection summary are committed and pushed. This prevents a one-process run from selecting and immediately
training on an unreviewed dataset. The current execution stops after selection in any case pending the user's
compute choice.

## Results

No capability result yet. The balanced bank is complete: 360 tasks, 23,040 traces, 108,759,239 sampled
thought tokens, 22,681 natural closes, four loops, and zero deficient tasks. The candidate vLLM scoring
instrument failed its strict cross-backend gate before bulk scoring; the reference-scoring amendment above
was frozen before any training score or outcome. Exact single-context reference scoring then completed for
all 22,681 eligible traces in 17,296 seconds, and R1 completed one answer rollout for every scored trace in
10,915 seconds. All 360 raw/score/R1 shards, hashes, task scopes, source links, trace joins, and eligibility
sets passed the read-only pre-seal audit. The retrospective evidence seal is now committed-bound: its
pre-attestation hashes, post-seal index hashes, operation contracts, and post-score deviation disclosure are
recorded in machine-readable receipts. No official selection dataset or adapter exists yet.

## Artifacts

- `idea_intake.md`: routing, novelty, and post-calibration boundary
- `reports/preregistration.md`: original frozen protocol plus dated amendments/deviations
- `reports/design_review.md`: adversarial review and applied fixes
- `configs/default.yaml`: exact counts, seeds, and gates
- `reports/artifact_manifest.yaml`: inherited pool, external scores, adapters, and checkpoints
- `runs/preselection_amendment_receipt.json`: commit-bound code, evidence, and deviation boundary
- `runs/preselection_evidence_seal.json`: exact pre/post index identity and absence checks at seal time
- external root: `/workspace/large_artifacts/qwen35_4b_balanced_core_answer_potential_sft`
