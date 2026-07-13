# Verifier-conditioned recovery banking curriculum

**Status:** finished

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can execution-selected policy compression install useful coding breadth without deleting rare recovery and completion behavior?
- Prior anchors: C5, C50, C52, and the direct predecessor C54.

## Question

Can Qwen/Qwen3.5-4B learn a transferable coding-agent recovery policy when supervision is balanced at conditional state→action transitions—especially rejected-patch→changed-patch and failed-test→diagnose/revise—rather than only at global INSPECT/PATCH/VERIFY/COMMIT totals?

## Hypothesis

The C54 failure was caused by conditioning collapse: its compact success traces contained no rejected edits or failed tests, so exact operator marginals still taught `failed_test→nothing useful`. Replay-verified failure-state rows should improve controlled recovery on unseen algorithm families. If the effect is real training rather than a promptable rule, it must beat the frozen incumbent, matched happy-path training, an explicit external recovery scaffold, and matched-compute sampling while retaining normal loops and unrelated logits.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e...`, warm-started from the frozen C54 `apex_replay` merged checkpoint.
- Substrate: ten fresh procedural Python repository families with visible and host-only hidden executable tests. Each fixture has an unresolved partial repair used only to construct public failed-test states.
- Train/selection: six families; selection uses new trained-family tasks only.
- Transfer: four wholly different families, followed by a new-seed confirmation block.
- Arms: `happy_action`, `recovery_action`, and `recovery_reason` (the last adds exactly 5% plan loss to byte-identical recovery rows).
- Baselines: frozen start checkpoint, frozen start plus an external recovery reminder, and two independent half-depth trajectories at the same call/token reservation.
- Primary metric: controlled-recovery hidden-test success, paired by task and scenario.
- Retention: normal-start success, verify-after-final-patch, pass→commit, invalid actions, and unrelated-context next-token locality.
- Hidden boundary: the model sees only repository files, issue text, and visible tool/test output. Hidden executable text never leaves the host and no benchmark content is read.

The design is frozen in [reports/preregistration.md](reports/preregistration.md), with the adversarial review in [reports/design_review.md](reports/design_review.md).

## Run

CPU invariants:

```bash
.venv/bin/python experiments/qwen35_4b_verifier_conditioned_recovery_bank/scripts/run.py --smoke
```

GPU integration smoke:

```bash
.venv/bin/python experiments/qwen35_4b_verifier_conditioned_recovery_bank/scripts/run.py --gpu-smoke
```

Staged full run:

```bash
.venv/bin/python experiments/qwen35_4b_verifier_conditioned_recovery_bank/scripts/run.py --full
```

The orchestrator is resumable. It stops before confirmation and Menagerie whenever a prerequisite gate fails.

## Results

**Verdict: stopped at the registered locality gate.** Menagerie and both transfer blocks remained sealed.

The harvest covered 58/72 tasks (80.6%); 57 tasks survived patch minimization and produced 399 replay-clean rows per arm. On the 60-case trained-family calibration block:

| Arm | Recovery success | Failed-test success | Rejected-patch success | Invalid turns | Mean sampled tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen base | 48.3% | 53.3% | 43.3% | 5.7% | 2,340 |
| happy action | 81.7% | 80.0% | 83.3% | 2.0% | 1,377 |
| recovery action | 85.0% | 73.3% | 96.7% | 19.1% | 1,503 |
| recovery reason | **91.7%** | **100%** | 83.3% | 5.9% | **480** |

The frozen selector chose `recovery_reason`: +43.3pp over base, +10.0pp over happy, and +6.7pp over recovery action. It also improved the registered transition composite by +18.9pp over happy. But its centered non-target logit drift was **0.303** versus the 0.15 ceiling, so the run stopped.

Exploratory mechanism controls isolated the damage: happy and recovery-action drift were only 0.083 and 0.098 and both passed locality. Their unrelated entropy changes were −0.016 and +0.006 nats, while recovery-reason reduced it by 0.106 nats. Full result and source checksums are in [reports/result_receipt.json](reports/result_receipt.json).

## Interpretation

Conditional transition banking contains a strong local signal, but this recipe does not establish capability transfer. Action-only failure-state learning is parameter-local and adds +3.3pp over the already-strong happy control, though it produces too many invalid actions. Concise plan supervision repairs those invalids and adds another +6.7pp, but its realized gradient is far larger than its nominal 5% token mass and violates locality.

Entropy/varentropy explains why. Before training, the correct JSON action-start token was already rank 1 at every action seam, and failure-specific plan starts were also rank 1. The imposed plan starts at ordinary pivots were unnatural: rank ~8,404 at inspect→patch, ~1,163 at patch-ok→verify, ~135 at start→inspect, and 3 at pass→commit. Plan SFT forced every one to rank 1 and near-zero entropy. Future plan dose must be calibrated by realized gradient/surprisal, not weighted token count.

## Knowledgebase Update

- Program evidence: updated with the locality-gated negative and action-only positive control.
- Program backlog: queues a new locality-first interpolation experiment; transfer seeds remain untouched.
- Claim ledger/synthesis: deferred until the interpolation follow-up determines whether the signal survives a compliant dose.

## Artifacts

- `src/`: constrained repository environment, recovery scenarios, bank, and pinned vLLM runner.
- `scripts/`: harvest, bank, training, merge, evaluation, diagnostics, analysis, and staged orchestration.
- `configs/default.yaml`: frozen seeds, doses, budgets, and gates.
- `reports/artifact_manifest.yaml`: external artifact locations and regeneration commands.
