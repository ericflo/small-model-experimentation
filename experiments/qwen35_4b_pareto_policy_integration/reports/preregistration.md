# Preregistration

Frozen before task-model training or new evaluation output.

## Primary hypothesis

Correctly routed, same-prefix on-policy MOPD can produce one Qwen3.5-4B policy
whose equal-weight quick/deep joint score is reproducibly above both C54 source
policies, while preserving saturated and never-trained families.

## Policies

- `quick`: QLoRA trained from the pinned base on committed C54
  `sft_blend.jsonl`.
- `deep`: QLoRA trained independently from the same base on committed C54
  `sft_apex.jsonl`.
- `student`: begins from the explicitly merged quick checkpoint.
- No other model may generate, score, judge, label, or teach.

## Strata and splits

- Quick: atoms L1-L2.
- Deep: atoms L3-L6 plus interactive episodes L2/L3/L5.
- Training families: the 12 C54 families present in both committed datasets.
- Transfer: `brinework`, `spindle`, excluded from all training and rollout
  prompt construction.
- Calibration, two qualification blocks, rollout rounds, and two confirmatory
  blocks use the disjoint namespaces frozen in `configs/default.yaml`.
- Benchmark seeds remain unopened until all procedural gates pass.

## Corrected teacher qualification

For each policy's intended stratum, compute paired item-level deltas against
the other source policy. A complementary advantage exists iff:

1. pooled paired mean delta is greater than zero;
2. the one-sided 95% family/level-stratified bootstrap lower bound is greater
   than zero; and
3. both independent seed-block means are greater than zero.

There is no absolute minimum delta. If the incumbent score is at least 0.98 or
fewer than eight effective deficit items remain, the cell is a retention
anchor: equality is acceptable and it cannot veto integration, but regression
beyond 0.02 fails retention.

The quick and deep policies qualify as a Pareto pair only if quick beats deep
on the quick stratum and deep beats quick on the deep stratum under the rule
above. Otherwise there is no complementary pair to integrate.

## Teacher and locality audit

On exact student-visible prompts and prefixes, branch continuations from the
correct teacher must have a positive paired lower bound over reversed routing.
Five-update locality probes (or the smallest implementation-equivalent pilot)
must keep median non-target logit drift at or below 0.10 and entropy loss at or
below the frozen relative ceiling. Failure stops MOPD.

## Integration

Four rollout/update rounds are frozen. Each round samples fresh continuations
from the current student, scores the identical prompt and student prefix with
the routed frozen teacher, consumes each rollout once, and applies the corrected
teacher-top-50 reverse-KL objective. Quick replay occupies 25% of updates.
Round mean KL above 0.10 or non-finite gradients stop training and preserve the
checkpoint.

## Controls

- reversed teacher routing at matched updates and initial KL;
- off-policy distillation on teacher-generated rather than student-generated
  continuations;
- explicit 25/50/75% parameter-delta merges;
- 320-step compute-overmatched union SFT;
- visibly routed two-checkpoint reference;
- incumbent execution-filtered best-of-8 at matched generation cost.

## Primary decision

On two confirmatory blocks, the integrated policy passes only if its
equal-weight quick/deep joint paired delta has a one-sided 95% lower bound above
zero versus each single checkpoint and every one-checkpoint integration
control, with no greater than 0.02 retention regression. The final system must
also beat matched-compute sampling. No fixed positive effect-size bar is used.

Only after this passes may the benchmark CLI be run. Benchmark success requires
positive paired aggregate delta versus each single policy and matched-compute
sampling on the registered quick and medium tiers; report every event without
exclusion.

## Interpretation

- Pass: evidence that on-policy policy-space integration crosses a measured
  single-checkpoint Pareto frontier.
- Qualified teachers but integration failure: evidence for a shared-parameter
  integration/capacity boundary under corrected MOPD.
- No complementary qualification: C54 does not reproduce on the clean
  procedural proxy; not an MOPD result.
- Tiny but statistically credible gains count; large but unstable gains do not.
