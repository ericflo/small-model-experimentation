# Adversarial Design Review

Date: 2026-07-11. Review performed before any model baseline or training run.

## Verdict

Proceed to implementation and CPU/GPU smokes. Do not proceed from a specialist
to full MOPD unless every capability, same-prefix, and locality gate passes.

## Findings and Required Fixes

### A composition task can be two tasks placed next to each other

Narrative concatenation would let one primitive dominate the score. Required
fix: each compound has one terminal objective whose inputs are produced by the
other primitive; an exact oracle and one-primitive-removal policies are part of
the smoke. Full success for every removal policy must be `<=0.20`.

Resolution: implemented in `compound_core.py`; `cipherkiln` connects inferred
codes to legal protocol actions, `mazeferry` connects navigation to located
typed tools, `patchferry` blocks the chain at the corrupted signature, and
`tripleforge` connects code induction to a typed dependency order.

### Baseline-conditioned item filtering would manufacture headroom

Rejecting each item the incumbent solves would tailor the evaluation to one
checkpoint. Required fix: use a disjoint calibration pool to select structural
levels only, then freeze the confirmatory distribution. Never inspect a model
output when retaining a confirmatory item.

### A stronger endpoint teacher may still be useless on student prefixes

Aggregate teacher accuracy does not prove local guidance where the student
visits. Required fix: continue correct, base, and wrong teachers from identical
frozen prefixes, then force top-token alternatives and measure downstream
reward under the student. Stop if the correct pressure is not actionable.

### Privileged state can leak into the deployed prompt

DAgger experts inspect simulator state. Required fix: keep visible `messages`
separate from expert metadata; assert that every training row ends in a visible
user observation and contains no hidden rule, solution, location table, or
oracle label. MOPD teachers receive the identical observable prompt/prefix and
never simulator state.

### Dense distillation can erase uncertainty and correction branches

Recent OPSD negatives and C52 make ordinary loss curves insufficient.
Required fix: batch-one exact-logit drift, entropy, verification/backtracking
marker rates, and natural closure are stop gates at the five-update pilot and
every saved checkpoint. A failed canonical objective is preserved, not rescued
inside this experiment with a new loss.

### Top-k truncation can optimize the wrong distribution

Naively truncating reverse KL shifts its optimum. Required fix: use MOPD
equation (5)'s published `-p_student + p_teacher` correction on every
teacher-top-k token, and unit-test the `k = |V|` case against full-vocabulary
reverse KL before any GPU update. Do not invent a lumped tail bucket.

### “On-policy” can quietly mean stale replay

Repeated epochs over old student trajectories reintroduce distribution shift.
Required fix: record the checkpoint digest, consume every trajectory once, and
allow at most one optimizer update of lag.

### Wrong-route control can be confounded by larger KL

A random wrong teacher may simply be farther from the student. Required fix:
permute teachers within initial-KL bins and report pressure magnitude, entropy,
and divergence for correct and wrong routing.

### Joint RL can be underfunded by counting only integration compute

Specialists are expensive shared artifacts. Required fix: publish conditional
and end-to-end token ledgers. The headline joint-RL control receives the full
specialist-production-plus-integration budget.

### DAgger or action formatting may contain the entire gain

Required fix: each specialist must beat DAgger-only and compute-overmatched
additional SFT. Action validity and natural close are diagnostics, never reward.
Terminal execution score remains the reward.

### Sample-more can still dominate

Required fix: specialist greedy must beat incumbent execution-filtered
best-of-8 before it is called a teacher. The integrated student must repeat
this comparison on compounds at no greater inference-token cost.

### One high-headroom domain can hide a see-saw regression

Required fix: normalize by each teacher's headroom, report every `I_d`, require
minimum recovery `>=0.50`, and forbid negative domains. Do not clip ratios.

### The composition specialist alone may explain held-out composition

Required fix: the integrated student must exceed the best individual
specialist, explicitly including `T_compose`, by `>=0.10` on held-out compounds.

### Historical exposure weakens the primitive holdout language

`S0` has seen earlier rows from some proxy families. Required fix: call them
“no-new-exposure” rather than never-seen and exclude them from every new replay,
DAgger, reward, and MOPD row. The new compound surface pools are fully held out.

### Qwen3.5 runtime LoRA can silently no-op

Required fix: every adapter is explicitly merged into a full composite, its
applied delta count and hashes are recorded, and a preregistered greedy canary
must differ from `S0`. All result-bearing evaluation uses merged composites.

### Backend equality is not implied by equal seeds

Required fix: every generative comparison uses the same vLLM runner and
batch/decoding configuration. Transformers is an internal measurement backend
only. Any parity failure blocks that arm.

### Training-seed variance can create an integration winner

Required fix: three end-to-end MOPD and joint-RL seeds, frozen paired evaluation
seeds, family/level-stratified episode bootstrap, and explicit effect-size bars.

### Benchmark feedback could silently select the recipe

Required fix: benchmark contents remain unread. One frozen checkpoint becomes
eligible only after all whitebox gates; aggregate CLI output cannot select or
revise training.

## Residual Risks

- Four specialists plus controls are expensive on one GPU; the stop hierarchy
  limits wasted compute but does not make the confirmatory run cheap.
- QLoRA rank may bottleneck integration. Rank is therefore chosen by memory
  fit, not outcomes, and is matched across trainable integration arms.
- The compound families remain synthetic. Primitive no-new-exposure and final
  blackbox transfer are necessary before making a broad claim.
- A null at the specialist stage says nothing about MOPD in settings with
  stronger teachers. It does decisively stop this repository's proposed path.

## Pre-baseline Implementation Note

The first runtime pass exposed two issues before any gym model output. The
copied harness accepted a merged-checkpoint argument while the copied current
runner lacked the corresponding engine field; the runner now loads and hashes
the local composite explicitly. Also, the original 120-step extra-SFT setting
matched optimizer steps but not GRPO's multiple forward passes, so it was
raised to 300 as documented in the preregistration amendment. Both fixes make
the existing controls stricter and were frozen before incumbent calibration.

## Decision Receipt

- CPU compound oracle/necessity gate: required before environment commit.
- Scientific model smoke: required before baseline calibration.
- Specialist gate: required independently for all four domains.
- Same-prefix and locality gate: required independently for all four teachers.
- Integration and compound gates: required before benchmark.
- Negative and stopped stages remain result-bearing artifacts.

## Postmortem Finding (2026-07-12)

### Compound headroom does not imply teacher headroom

The review required a disjoint compound macro below 0.60 and guarded against a
high-headroom domain hiding regressions after training, but it never checked
whether each mandatory specialist's absolute improvement bar was reachable
under its score ceiling. The paired baseline found `ferrier = 0.9940`; the
frozen `+0.10` gate therefore demanded 1.0940 on a `[0,1]` score.

Resolution: a reusable pre-production analyzer now computes every core's
baseline, score ceiling, maximum possible gain, and required absolute target.
It stops before best-of-8 and all training if any core is infeasible. Future
multi-teacher designs must run both gates on disjoint calibration data:

1. deployment-endpoint headroom (for example, held-out composition); and
2. per-teacher theoretical headroom for every mandatory capability producer.

The current experiment is terminal. Swapping the tools family, lowering the
bar, or dropping a required teacher is a new experiment.
