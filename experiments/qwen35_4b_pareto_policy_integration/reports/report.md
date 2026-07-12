# Qwen3.5-4B Pareto Policy Integration Report

## Status

**Stopped negative on 2026-07-12 before teacher audit.** The corrected
specialist gate had no fixed effect-size floor: any replicated, statistically
credible paired gain above zero could qualify. The regenerated `blend` policy
nevertheless lost its intended quick comparison in both independent blocks.
The `apex` policy won deep capability but missed the frozen retention rule.
There was no clean complementary teacher pair, so MOPD was not run.

## Research Program Fit

This is the clean successor to `qwen35_4b_specialist_policy_integration`, whose
fixed `S0 + 0.10` rule became impossible at a saturated baseline. It directly
tests the prerequisite behind the MOPD path suggested by C54: whether C54's
same-origin quick-optimal `blend` and medium/deep-optimal `apex` checkpoints
actually cross over on a contamination-safe state distribution.

## Method

Both rank-32 policies were regenerated independently from the identical pinned
`Qwen/Qwen3.5-4B` revision, explicitly merged, weight-hashed, and behavior-gated
through the same vLLM 0.24 backend. Calibration assigned saturated, explicit
anchor, and never-trained transfer cells to retention; all remaining cells were
fixed as capability cells before qualification.

Qualification used two disjoint procedural seed blocks. Quick capability was
atoms L1-L2; deep capability was atoms L3-L6 plus interactive episodes at
L2/L3/L5. For each intended teacher, qualification required:

1. pooled paired macro delta greater than zero;
2. a one-sided 95% family/level-stratified bootstrap lower bound above zero;
3. both block means above zero; and
4. no retention-cell regression greater than 0.02.

There was no minimum positive effect size. Every arm used greedy decoding,
identical paired items, the exact frozen engine geometry, and the same local
composite provenance checks.

## Reached Evidence

The installation canary passed before any task score was accepted: both
specialists differed from base on 8/8 fixed prompts and from one another on
7/8. Calibration then completed 1,488 items per policy and assigned 56 cells to
capability inference and 52 to retention. All `ferrier`, `brinework`, and
`spindle` cells remained retention-only as frozen.

Qualification completed all four 4,416-item arms—17,664 arm-item evaluations,
or 8,832 paired item identities—with no exclusions. It consumed 18,049,063
sampled tokens and 12,713.08 evaluation seconds.

| Block | Policy | Broad quick | Broad deep | Sampled tokens | Wall seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| 96200 | `blend` | 0.7876 | 0.5297 | 4,748,497 | 3,387.61 |
| 96200 | `apex` | 0.8089 | 0.5813 | 4,244,434 | 2,970.32 |
| 96300 | `blend` | 0.7892 | 0.5191 | 4,737,106 | 3,341.25 |
| 96300 | `apex` | 0.8268 | 0.5819 | 4,319,026 | 3,013.91 |

All nine protocol checks passed: both seed lists, qualification scope, greedy
decode, family identity, calibration authorization and partition, and presence
of capability cells in both strata.

## Terminal Qualification Result

| Intended teacher | Capability pairs | Block deltas | Pooled delta | One-sided 95% LCB | Retention | Decision |
| --- | ---: | --- | ---: | ---: | --- | --- |
| quick `blend` | 768 | −0.00693, −0.03789 | **−0.02241** | **−0.04897** | 12/20 cells pass | fail |
| deep `apex` | 4,032 | +0.04254, +0.04871 | **+0.04563** | **+0.03401** | 26/32 cells pass | fail |

The quick result is decisive for this design: `blend` did not merely miss a
large or practical-gain threshold; its effect had the wrong sign in both
blocks under a rule that would have accepted any credible positive delta.
Broad raw scores agree—`apex` was higher on both quick and deep strata in both
blocks—although the preregistered capability-cell macro, not the broad mean,
made the decision.

The deep result is separately informative. `apex` has a stable intended-stratum
capability advantage, but it is not a clean dominance result. Six retention
cells exceeded the 0.02 regression allowance: `brinework` atom L6;
`glyphgate` episodes L2/L3/L5; and `kilnrite` episodes L2/L5. Its worst
retention delta was −0.09375. Conversely, `blend` failed eight quick retention
cells, with its worst delta −0.21875 on never-trained `spindle` L2.

The machine receipt therefore records `gate.passed=false` and
`downstream_authorization=stop_before_teacher_audit`.

## Interpretation

C54's quick/medium Pareto result is real evidence on its menagerie instrument,
but its labels do not identify a transportable quick/deep teacher routing rule
on this clean procedural proxy. A checkpoint can cross an external aggregate
target while failing to supply a locally better teacher on the state
distribution where distillation must occur. The later C54 model-soup sweep
strengthens the parameter-interpolation negative, but it does not repair this
missing teacher crossover.

This is **not evidence for or against MOPD, OPSD, or the corrected top-k loss**.
Those mechanisms require a better teacher at the student's actual state. Here
the prerequisite failed before teacher scoring or any integration update. It
would be scientifically invalid to run MOPD and call the outcome an integration
test when one coarse route is already worse on both replicated capability
blocks.

The useful next hypothesis is narrower than “try the same teachers harder.”
The cell table contains real heterogeneity: each checkpoint wins some local
states, but the assumed quick/deep labels are too coarse. A new experiment may
pre-register outcome- or state-routed same-prefix distillation: score both
same-origin teachers on disjoint calibration prefixes, route only where a
teacher has positive continuation advantage, and confirm the routing rule on
fresh cells before training. That is the principled version of replacing a
hinted log-probability slogan with an advantage estimator. It must remain a new
experiment with fresh splits and must still beat both checkpoints, a visible
two-checkpoint router, and matched-compute sampling.

For immediate deployment, C54's visible tier router remains the supported
upper reference. It is not a one-checkpoint capability-installation result.

## Unreached by Design

- same-prefix correct-versus-wrong teacher audit;
- exact-logit locality pilot;
- MOPD, wrong-route, and off-policy updates;
- parameter-merge and compute-overmatched union-SFT controls;
- confirmatory evaluation and matched best-of-8;
- every benchmark invocation.

No benchmark content, item, transcript, or result detail was read.

## Artifact Manifest

The terminal receipt is `analysis/specialist_qualification.json`; all four raw
qualification arms are under `runs/policy_eval/`. Large specialist adapters
and explicit merged composites remain external as documented in
`artifact_manifest.yaml`.
