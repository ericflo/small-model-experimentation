# Pre-registration: Answer-Potential Trace SFT

## Freeze Boundary

This document is committed before GPU-scale inference. Later amendments must be additive, dated in
git, and made before observing the affected result. The README gives the practitioner-facing version;
this file controls scientific decisions when wording differs.

## Confirmatory Question

On fresh procedural atom tasks, does SFT on pre-answer Qwen3.5-4B thoughts selected by teacher-forced
canonical-answer likelihood improve fresh greedy accuracy beyond binary successful-answer rejection
SFT, length-matched random-trace SFT, and shuffled-trace SFT at matched data and training steps?

## Fixed Model And Provenance

- Model and trace generator: `Qwen/Qwen3.5-4B` only.
- Pinned revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- No external model, teacher, judge, embedding model, or benchmark training data.
- Procedural atom generators are copied into this standalone experiment from
  `qwen35_4b_gauntlet_breadth_round1`; new seeds create every item.
- Ten families contribute calibration/train/IID/hard data. Two family modules are copied for
  evaluation but are never imported by the calibration, harvest, selection, or training stages.

## Frozen Data Matrix

| split | seed namespace | composition | n |
| --- | ---: | --- | ---: |
| calibration | 61001 | deterministic balanced sample over train families L1-L2 | 64 |
| train | 62001 | 10 families x 2 levels x 30 | 600 |
| IID eval | 63001 | 10 families x 2 levels x 20 | 400 |
| held-family eval | 64001 | 2 families x 2 levels x 25 | 100 |
| hard eval | 65001 | 10 families x L3 x 10 | 100 |

Before inference, assert zero item-ID, exact prompt, canonical prompt digest, and family-generator
seed overlap across splits. Calibration thresholds may use only split `calibration`.

## Inference Protocol

- vLLM runner copied from the repository template, one backend for every scientific inference arm.
- bf16 text-only engine; model limit 16,384; GPU utilization 0.85; concurrency/cudagraph/capacity
  geometry recorded and held fixed across compared arms.
- Thought-only generation stops on token ID 248069 (`</think>`), retains exact raw token IDs, and
  never requests an answer continuation for the potential-treatment pool.
- Calibration `N=32`; full harvest `N=128`; temperature 1.0; top-p 0.95; top-k 20; max 512 sampled
  thought tokens; explicit seed and shard plan.
- The first 16/32/64 candidates in each single N=128 row form nested sample-count diagnostics.

## Scoring Definition

For item `i`, thought tokens `z`, answer-content tokens `y`, and exact rendered prefix `r`:

```text
ll(z)      = sum_t log p_theta(y_t | r(x,z), y_<t)
gain(z)    = ll(z) - ll(empty_thought)
first(z)   = log p_theta(y_1 | r(x,z))
```

Fixed boundary tokens, `ANSWER: `, and EOS are excluded. Sum is primary within task; mean is stored
for diagnostics. Procedurally valid alternative renderings are combined before ranking. Where a
family exposes deterministic decoys, store the canonical-versus-decoy margin and require its sign to
agree with `gain` for a trace to enter the confirmatory quality set.

The vLLM observed-prompt-token logprob extractor must pass: token round-trip, exact span boundary,
manual small-tensor likelihood, finite-value, empty-thought invariance, answer-substitution, and a
small HF parity diagnostic. HF parity validates plumbing only; no HF-generated/scored row enters a
scientific comparison.

## G0 Scorer Gate

For every calibration thought, sample `R=8` fresh short answer continuations at fixed new seeds and
grade them procedurally. Let rollout success be the fraction correct.

Gate passes only if:

1. mean within-task AUROC over mixed tasks is >=0.65;
2. top-one-by-gain task accuracy exceeds seeded random and shortest selection by >=0.10 each;
3. both paired task-bootstrap 95% lower bounds are >0;
4. gain's task-macro AUROC exceeds length and trace mean-logprob AUROC;
5. original traces beat token-shuffled and cyclic foreign-task thoughts after length matching;
6. Kendall tau between rankings under canonical answer formatting and a harmless registered
   whitespace/prefix variant is >=0.80; and
7. >=75% of selected traces have positive gain at a boundary before their first exact answer mention
   or never mention the answer.

Bootstrap: 10,000 deterministic resamples of tasks, seed 61991. AUROC is undefined for single-class
tasks and averaged only over mixed tasks; top-one outcomes include all tasks. Pooled AUROC is reported
but cannot satisfy the gate.

G0 failure terminates the experiment before full harvest/SFT. No threshold may be softened after
seeing G0. Diagnostics may localize the failure but cannot license training.

## Quality-Diversity-Brevity Selector

Freeze from calibration:

- lower trace-prior percentile;
- high-gain quantile or near-best tolerance;
- prefix-compression tolerance;
- normalized trigram duplicate threshold; and
- template/family caps.

Defaults before calibration are: prior floor 5th within-task percentile, high-gain top 10%, near-best
0.10 answer-nats/token, trigram duplicate Jaccard 0.90, two traces/item, family cap 140, and normalized
template cap 12. Calibration may choose among a predeclared grid, recorded before train scoring.

Selection order is fixed: validity -> quality -> farthest-first diversity -> shortest near-best
representative -> stratified global caps. A scalar reward mixing score, distance, and length is
forbidden.

## Binary RFT Baseline

Using the same thought pool, draw one answer continuation per thought at the registered sampling
temperature and retain traces whose sampled answer grades correct. This is deliberately the noisy
one-rollout label that the treatment claims to improve. When several qualify, apply the same
diversity/brevity selector without answer gain. Prompts with no qualifier remain absent from
`success_rft`; the applied comparison reports this coverage difference, while a matched-task
intersection analysis isolates trace quality.

## Pivot/Branch Contingency

Calculate potential-surrogate curves on natural boundaries for the first 64 independent traces.
Branch suffixes immediately before the first registered material drop/plateau until the branch arm
uses the same total forward-token budget as the independent-128 arm. Prefill recomputation counts.

The branch data may enter a trained arm only if top-selected R=8 rollout success beats independent
sampling with a paired lower confidence bound >0, and family/task coverage is noninferior within
0.02. Otherwise record the branch result and train `potential` from independent traces only.

## Matched SFT Matrix

Arms: `empty`, `random_length`, `success_rft`, `potential`, `potential_shuffle`; add
`potential_branch` only after the contingency gate.

- Same base, prompts/answers where defined, final row count, family/level quotas, steps, batch
  geometry, rank 32/alpha 64/dropout 0.05, two epochs, lr 2e-4, seed 42.
- Prompt weight 0; thought 0.2; exact close plus canonical answer 1.0; forced-close context thought 0.
- `random_length` is same-task and nearest in token length to the potential trace.
- `potential_shuffle` is a derangement within family x level x answer-shape x length-bin; no fixed
  points. If a stratum cannot derange, merge the adjacent length bin before dropping rows.
- Hard-trim every arm to the minimum attainable registered count after balancing; report lost task
  coverage. A separate applied-size `potential_all` result may be descriptive, never substituted for
  the matched confirmatory matrix.

Seed-42 screening rule: replicate `potential` and the strongest non-oracle baseline at seed 43 only
if potential's IID greedy delta is >=0.03 and its parse-rate delta is >=-0.01. A positive claim
requires the replicated combined result; a screen failure is a negative/inconclusive terminal result.

## Evaluation And Statistics

All arms x modes `{think@512, no_think}` on the frozen 400 IID tasks. Confirmatory outcome is think@512
greedy exact accuracy. Report family-macro and item-micro values. Paired task bootstrap (10,000,
seed 63991) for arm deltas; training-seed aggregation resamples seed then task when seed 43 runs.

Secondary: 100 held-family, 100 hard, pass@8 with two fixed sample seeds, unique answers, parse rate,
parse-conditional accuracy, natural-close/forced-close, thought tokens, answer-copy, and trace-template
diversity. No held-family or hard result selects thresholds or checkpoints.

Base sample-more uses the same backend and fixed candidate stream. Show accuracy/coverage against
actual cumulative forward tokens, with deployable majority and answer-confidence selection plus an
oracle pass@k ceiling. Training/curation costs are separate and amortized over deployment horizons.

## Terminal Decisions

- `SCORER_NEGATIVE`: G0 fails; no full harvest/SFT.
- `MEASUREMENT_ONLY`: G0 passes, but potential SFT does not beat binary RFT and random-length.
- `LOCAL_POSITIVE`: replicated potential minus success-RFT >=0.05, paired 95% lower bound >0;
  potential also beats random-length and shuffled; parse and family-macro deltas >=-0.02.
- `EFFICIENCY_POSITIVE`: local positive plus median thought length <=70% of success-RFT, or strict
  accuracy/token Pareto dominance.
- `MISSION_POSITIVE`: local positive plus a win over matched-forward-token base sample-more.
- `TRANSFER_POSITIVE`: held-family paired lower confidence bound >0.

Anything not meeting a named rule is scoped descriptively. The report must preserve failed arms,
gates, and amendments.
