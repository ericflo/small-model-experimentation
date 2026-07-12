# Idea Intake

## Program Fit

- Program: `agentic_breadth_installation`.
- Existing or new program: existing; the question directly follows the
  program's recorded same-prefix routing priority.
- Closest program scorecard reviewed: `knowledge/program_scorecards.md`.
- Related future queue item: `posttraining_method_shared_substrate`; this run
  is narrower and already supported by a measured Pareto pair.

## Prior Evidence

- Anchor 1: `qwen35_4b_pareto_policy_integration` established that coarse
  quick/deep labels do not transport to local teaching advantage and stopped
  before MOPD.
- Anchor 2: `qwen35_4b_gauntlet_frontier` produced same-origin quick/deep
  policies and found a non-convex one-checkpoint frontier whose best joint soup
  is 40% quick / 60% deep.
- Anchor 3: `qwen35_4b_opsd_pressure_locality_audit` and C52 show that plausible
  token supervision can still cause non-local shared-weight drift.
- Closest near-duplicate: `qwen35_4b_pareto_policy_integration`. It tested a
  preassigned stratum teacher after complete trajectories; it did not estimate
  teacher value at student prefixes, abstain, or run any distillation update.

## Novelty Claim

This is the first repository experiment to select between same-origin teachers
using verifier-estimated continuation value at the exact student state, remove
selection bias with disjoint branch outcomes, and require both teachers'
positive advantage over both the current student and each other before MOPD.

## Mechanism

The C54 endpoints contain different local policies, but quick/deep dataset
labels are an unreliable routing proxy. A verifier can measure the quantity the
dense loss otherwise assumes: expected return after the same student prefix.
Selection and audit branches are separated so a noisy maximum cannot certify
itself. Strictly positive routing plus abstention prevents already-correct or
teacher-worse states from receiving ambiguous dense pressure. Same-origin
teachers and the soup initialization should keep token distributions close
enough for corrected top-k MOPD; a frozen-soup anchor and locality gate test
that stability rather than assuming it.

The explanation is false if independent audit continuations do not preserve
the selected teacher's advantage, if only one teacher ever qualifies, if
shuffled/coarse routing matches the trained result, or if exact-logit drift
precedes any capability gain.

## Control Plan

- Baseline: independently regenerated 40/60 parameter soup, plus both source
  teachers.
- Mechanism-falsifying controls: shuffled teacher identities, the old
  quick/deep route, fixed-deep routing, off-policy selected-continuation SFT,
  and explicit parameter soups under matched updates or deployment protocol.
- Shift/robustness: two route-qualification blocks, two final blocks, three
  training seeds, never-trained families, visible two-checkpoint routing, and
  execution-filtered best-of-8.
- Hidden-label boundary: procedural verification may acquire/select training
  states but never enters prompts. Benchmark CLI access is gated; benchmark
  files and item-level results remain unread.

## Evidence Output

- Program evidence update: record whether locally complementary teachers
  exist and, conditionally, whether policy-space integration crosses the soup
  frontier.
- Claim ledger or synthesis update: update synthesis/program surfaces; do not
  add or promote a claim while the repository's claim re-grade remains open.
- Reusable artifact: split-branch route builder/analyzer with deterministic
  state replay, exact branch provenance, abstention, and support receipts.
- Stop/branch condition: route-gate failure stops before locality/training;
  locality failure stops before full MOPD; confirmation failure seals all
  benchmarks and preserves every negative/control.

## Decision

- Run experiment: yes.
- Create program: no.
- Write synthesis only: no; prior work identifies but does not answer this
  state-level estimand.
- Defer: benchmark access until the full procedural decision passes.

