# Verifier-conditioned recovery banking curriculum — report

## Summary

**Stopped negative with a strong mechanism signal.** Conditional transition banking raised trained-family controlled recovery from 48.3% to 91.7%, but the selected reasoning arm failed exact-logit locality (0.303 drift versus 0.15). Transfer and Menagerie remained sealed. Exploratory controls show verifier-conditioned action learning itself is local; nominally 5% plan supervision caused the collateral.

## Research Program Fit

This experiment executed the `agentic_breadth_installation` backlog item created by C54: preserve `failed_patch→changed_patch`, `failed_test→revision`, and `passed_test→commit`, not merely aggregate tool marginals. It resolves that transition-balanced banking is learnable and locally promising, while plan-span pressure remains the bottleneck.

## Method

The frozen C54 apex checkpoint searched 72 fresh procedural repositories across six training families. Successful model patches were minimized and replayed from deterministic public rejected-patch and failed-visible-test states. Each admitted task contributed seven state→action rows; weighted action mass was exactly equal by operator and within-operator transition.

Three r32 QLoRA arms trained for 120 steps from the same warm start:

- `happy_action`: matched operator/transition action mass in successful-path contexts;
- `recovery_action`: actual public failure contexts, zero plan loss;
- `recovery_reason`: byte-identical recovery rows with nominal 5% plan-token mass.

The 60-case calibration block was fresh but used the six training families. A frozen lexicographic selector chose one recovery arm. Locality then compared the selected merged checkpoint against apex on 48 frozen unrelated contexts.

## Results

### Harvest and bank

- 58/72 tasks covered (80.6% versus the 35% bar), with every family at 41.7% or better.
- 143 successful trajectories; 135 preserved pass→commit.
- 57 model repairs survived minimization/collapse.
- 399 rows per arm; 100% replay; every structural, balance, firewall, and length gate passed.

### Calibration

| Arm | Overall | Failed test | Rejected patch | Submit | Invalid/turn | Mean tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.483 | 0.533 | 0.433 | 0.283 | 0.057 | 2,340 |
| happy action | 0.817 | 0.800 | 0.833 | 0.767 | 0.020 | 1,377 |
| recovery action | 0.850 | 0.733 | **0.967** | 0.800 | 0.191 | 1,503 |
| recovery reason | **0.917** | **1.000** | 0.833 | **0.950** | 0.059 | **480** |

The selector chose recovery reason: +0.433 versus base, +0.100 versus happy, +0.067 versus action-only, and +0.189 on the transition composite versus happy. Verification and commit were both 1.0.

### Locality and uncertainty

| Arm | Merge delta norm sum | Non-target drift | Entropy delta | Locality |
| --- | ---: | ---: | ---: | --- |
| happy action | 28.03 | 0.083 | −0.016 | pass |
| recovery action | 29.17 | 0.098 | +0.006 | pass |
| recovery reason | **37.78** | **0.303** | **−0.106** | **fail** |

The nominal 5% plan span was not a 5% realized update. Step-10 loss/gradient were 12.52/1.80 for action-only and 43.61/42.12 for reason; the standard clip bounded magnitude but made early update direction plan-dominated.

Teacher-forced seam audit explains the pressure. Before any new training, the target action-start token was rank 1 with very low entropy for all seven transitions. Failure-specific plan starts were also already rank 1. The imposed ordinary-state plan starts were highly off-policy:

- inspect→patch: target plan token rank ~8,404;
- patch-ok→verify: ~1,163;
- start→inspect: ~135;
- passed-test→commit: 3.

Reason training made every plan and action seam rank 1 with near-zero entropy. Action-only made action seams sharper but left the plan distribution largely natural and passed unrelated-context locality.

## Controls

The matched happy arm shows that most calibration recovery gain comes from balanced action training, not failure conditioning alone. Recovery conditioning adds +3.3pp overall and strongly reallocates success toward rejected-patch states, but also increases invalid actions. Reason supervision restores valid concise execution and adds +6.7pp, at unacceptable shared-weight cost.

The external scaffold and matched-sampling controls were correctly not funded: the selected checkpoint failed a prerequisite locality gate before any held-out family was touched.

## Oracle Versus Deployable Evidence

Host oracles validated fixture truth conditions only. Full training targets came solely from the model's own execution-verified repairs. Hidden executable code/output never entered model context. Calibration hidden tests establish trained-family performance only; because transfer stayed sealed, this is not a breadth or downstream capability claim.

## Interpretation

The experiment rejects its frozen headline recipe but leaves a sharper positive: verifier-conditioned **action** learning is parameter-local at this dose. The failure was placing positive pressure on arbitrary lexical plans whose first tokens were extremely unlikely under natural thinking. Token-mass calibration missed realized CE and gradient scale.

The most efficient next test is locality-first interpolation of the already-trained reason delta. Its 91.7% behavior margin is large, and drift should fall continuously with scale; action-only provides a full-dose locality-pass anchor. This must be a new experiment with the same calibration block for selection and untouched transfer seeds for the first claim-grade test.

## Next Experiments

1. New experiment: merge the reason LoRA at a frozen scale ladder, run locality before behavior, and select only among passing scales plus the action-only anchor.
2. If interpolation retains a calibration advantage, evaluate the frozen winner against base, happy, scaffold, and matched sampling on the untouched four-family transfer blocks.
3. Future retraining should calibrate plan pressure by realized gradient or target surprisal and supervise only genuinely useful, non-rank-1 pivots—not all plan starts.

## Artifact Manifest

Large artifacts live under `large_artifacts/qwen35_4b_verifier_conditioned_recovery_bank`. The committed [result receipt](result_receipt.json) records compact metrics and SHA-256 provenance for harvest, bank, adapters, merges, evaluations, locality, and uncertainty audits. Detailed trajectories and 4B checkpoints remain external as listed in `artifact_manifest.yaml`.
