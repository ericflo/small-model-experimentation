# Verifier-conditioned recovery banking curriculum

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

Pending the frozen result-bearing run.

## Interpretation

Pending. A gain that does not beat the explicit scaffold is promptable process control, not a weight-level capability unlock. A gain that does not beat matched sampling is not a compute-efficient agentic improvement. A train-family-only gain is memorized protocol, not breadth.

## Knowledgebase Update

- Program evidence: pending.
- Program backlog: pending.
- Claim ledger/synthesis: pending.

## Artifacts

- `src/`: constrained repository environment, recovery scenarios, bank, and pinned vLLM runner.
- `scripts/`: harvest, bank, training, merge, evaluation, diagnostics, analysis, and staged orchestration.
- `configs/default.yaml`: frozen seeds, doses, budgets, and gates.
- `reports/artifact_manifest.yaml`: external artifact locations and regeneration commands.
