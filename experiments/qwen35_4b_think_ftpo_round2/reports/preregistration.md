# Preregistration — entropy-routed think-pivot optimization round 2

Frozen before scoring the full real+shuffled row pools, training any adapter,
or running a round-2 scientific evaluation. A small exploratory census on the
already-regularized real rows established only that entropy/varentropy and
argmax-dominant wrong turns exist; it fixed no outcome threshold and is
reported as design evidence, not a round-2 result.

## Question and arms

Does single-position thought steering become useful when applied only where a
failed continuation is a confident distributional outlier, and is capability
better elicited by **lifting the fruitful alternative** than by demoting the
failed token?

- `base`: pinned Qwen3.5-4B.
- `demote`: published pairwise FTPO on geometry-qualified real pivots.
- `uplift` (primary): positive-only +0.5-logit target on the successful sibling;
  the rejected token is reference-tethered rather than directly suppressed.
- `uplift_shuffled`: identical uplift objective on the parent experiment's
  within-prompt outcome-shuffled rows, matched to the same row count.

`demote` and `uplift` use byte-identical rows. This is the causal comparison of
update direction. `uplift` vs `uplift_shuffled` is the content control.

## Frozen row geometry and dose gate

All distributions are frozen-base next-token distributions at harvest T=0.6.
A row qualifies when: rejected is global argmax; rejected−best-chosen ≥0.5 raw
logits; P(rejected) ≥0.50; entropy ≤1.50 nats; varentropy ≥0.10 nats²; and
P(chosen) ≥0.01·P(top). These absolute thresholds describe a focused but not
fully deterministic "spiky-conflicted" fork. Entropy/varentropy quartiles are
archived for heterogeneity analysis and may not be used to retune the filter.

Real and shuffled pools are independently filtered, then deterministically
downsampled to `min(256, n_real, n_shuffled)`; the same selected real rows feed
both real objectives. Fewer than 128 matched rows stops before training.
Because this dose is necessarily below the published 15–20k regime and below
round 1's 615 rows, a null can only be labeled `LOW_DOSE_NULL`. Directional harm
or control-equivalent harm remains decision-relevant.

## Objectives

Both use LoRA r256/α128, batch-of-1 model forwards with effective batch 16,
bf16, two epochs max, lr 1e-5, and the two-tier raw-logit tether λ=0.4/0.05,
dead zone 0.5. Non-target vocabulary is pinned. `demote` uses round 1's
softplus ε=2 pairwise margin. `uplift` uses the same hinged/tapered form on
`gain = z_chosen − z_ref_chosen`, target +0.5; rejected is non-target. Training
may safety-stop after ≥20% of scheduled updates when objective hit rate ≥0.35.

## Predictions and decision gates

- **P0 geometry:** ≥128 matched rows. Otherwise selector premise fails.
- **P1 targeted mechanism:** each real arm moves its intended target on ≥35%
  of training rows without median absolute non-target drift >0.10 logits.
- **P2 whitebox:** at either think@1024 or 2048, `uplift−base ≥+0.03` success
  and `uplift−uplift_shuffled ≥+0.02`; natural-close drop and answer-limit rise
  must each be ≤2pp. `demote` is adjudicated by the same base bar but is
  secondary. All tasks use fresh procedural seeds.
- **P3 coding agent (north star):** on 72 fresh repository repairs, final
  workspaces are hidden-tested at budget exhaustion (explicit submit rate is a
  separately reported agentic-completion diagnostic). `uplift`
  beats base eight-turn greedy by ≥8pp and the matched-compute union of two
  four-turn base trajectories by ≥5pp; shuffled gain is < half uplift gain.
  Exact task-paired bootstrap intervals are mandatory. Maximum compute is
  eight calls / 6,144 sampled tokens per task in both primary and sample-more.
- **P4 guards:** gym aggregate ≥base−2pp; code greedy and pass@8 no worse than
  10% relative; no-think ≥base−2pp; C49 gate passes for every trained arm.
- **P5 menagerie:** run only if P2 or P3 passes and every P4/termination guard
  passes. Two fresh paired quick seeds; positive if mean uplift−base ≥+0.05
  and neither seed is negative. Otherwise report the measured whitebox result
  without consuming blackbox events.

## Sample-more and oracle boundaries

The repository-agent sample-more baseline is deployable: two independent base
trajectories, success if either repair passes hidden tests, with the same total
turn/token ceilings as one deep trained trajectory. Whitebox best-of-k, if
reported, is labeled non-deployable. Hidden tests and verifier outcomes are
used only for offline row labels/evaluation; they never enter model prompts.

## Outcome labels

- `GEOMETRY_FAIL`: P0 fails; no training.
- `CAPABILITY_CANDIDATE`: P2 or P3 plus all guards; eligible for P5.
- `OBJECTIVE_NEGATIVE`: trained arm harms and shuffled does not explain it.
- `GENERIC_TRAINING_HARM`: real and shuffled arms degrade similarly.
- `LOW_DOSE_NULL`: no meaningful movement and no guard violation.
- `POSITIVE`: P5 passes after the whitebox gates.
