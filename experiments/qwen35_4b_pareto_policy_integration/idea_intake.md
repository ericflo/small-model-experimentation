# Idea Intake

## Program Fit

- Primary program: `agentic_breadth_installation`.
- Supporting programs: `posttraining_and_adaptation`,
  `test_time_reasoning_budget`, and `benchmark_generalization`.
- Closest near-duplicate: `qwen35_4b_specialist_policy_integration`. That run
  stopped before producing a specialist because it treated a fixed absolute
  gain as mandatory even when the baseline was 0.994. This successor does not
  amend or extend its result-bearing directory.
- New decisive anchor: C54 measured a non-convex Pareto pair from the same
  pinned 4B origin: the `blend` policy is strongest in quick/short regimes and
  `apex` is strongest in medium/deep regimes. Data interpolation did not combine
  them.

## Question

Can same-prefix, on-policy policy-space distillation consolidate the measured
C54 quick/deep Pareto pair into one checkpoint, or is the tradeoff genuinely a
shared-parameter capacity boundary?

## Gate Correction

The prior `S0 + 0.10` teacher rule was not scientifically justified. It mixed
up existence, measurement reliability, and practical magnitude. Here a teacher
qualifies when its paired delta is strictly positive, both independent seed
blocks are positive, and a one-sided stratified bootstrap lower bound is above
zero. There is no minimum effect size. More examples, rather than a larger
arbitrary bar, supply resolution.

A cell whose baseline is saturated is a retention anchor. It must not regress,
but it cannot veto a complementary teacher or the integration experiment. The
final integrated system—not every intermediate teacher—must beat
matched-compute sampling.

## Mechanism

Both policies descend from the identical pinned base. The student begins from
the quick policy, generates its own continuations, and receives the correct
teacher distribution at the exact visible prompt and student prefix: quick
teacher on short atoms, deep teacher on long atoms and interactive states.
Corrected teacher-top-50 MOPD pressure is therefore dense without exposing a
solution-conditioned hint. Fresh rollouts each round limit policy lag.

The mechanism is false if correct routing has no continuation advantage over
wrong routing, MOPD does not beat wrong-route/off-policy/parameter-merge and
matched-SFT controls, or one checkpoint cannot improve equal-weight quick/deep
joint utility beyond both single policies and matched sampling.

## Control Plan

- Same-origin single policies: regenerated `blend` and `apex`.
- Inference upper reference: visibly routed two-checkpoint policy; it is not a
  valid one-checkpoint success.
- Integration controls: KL-matched wrong routing, off-policy teacher forcing,
  three explicit parameter merges, and compute-overmatched union SFT.
- Test-time control: execution-filtered `blend` best-of-8 on identical fresh
  procedural items and the final held-out instrument.
- Transfer: `brinework` and `spindle` never occur in either specialist dataset.
- Firewall: only `Qwen/Qwen3.5-4B`; no benchmark content is read or imported;
  programmatic scores never enter prompts; all comparable evaluation arms use
  the pinned vLLM runner and exact backend metadata.

## Decision

Run as a new experiment. Stop only for a scientifically meaningful failure:
uninstalled checkpoints, no reproducible complementary teacher advantage,
failed same-prefix routing/locality, unsafe drift, or integration/control
failure. A small positive effect is not a failure.
