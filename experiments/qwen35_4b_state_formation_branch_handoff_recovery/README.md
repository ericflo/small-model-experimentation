# State-Formation Branch Handoff Recovery

**Status:** in-progress · since 2026-07-14 · frozen no-model handoff smoke passes; publish it with the successful G0 before retrying the full-rank positive control

This operational successor lets immutable producer v11 continue after the first branch recovery
successfully replaces the retired failed-G0 receipt at its canonical path. It changes no scientific
arm, model, data, metric, threshold, or branch rule.

## Research Program

- Program: `structured_execution_and_compilers`.
- Direct producer: `qwen35_4b_state_formation_capacity_adjudication`.
- Closest experiment: `qwen35_4b_state_formation_branch_recovery`, whose exact-prefix seam produced
  the valid full-rank seed-7411 G0 but whose frozen retirement guard treats any later occupant of
  that pathname as the retired failure.

## Question

Can downstream producer stages continue without weakening failure preservation once a successful
recovered G0 legitimately occupies the canonical pathname formerly used by the archived failure?

## Hypothesis

The defect is limited to invocation orchestration: the first wrapper asks whether the pathname
exists, not whether it still contains the exact retired failure. A new wrapper can reuse its frozen,
tested path seam while requiring the archived failure and retirement receipts, rejecting a
reappearing mirror or exact failed bytes, and accepting only the exact successful G0 plus its
STARTED/COMPLETE lineage.

## Setup

- Model-bearing stages remain only `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, on producer v11's Transformers backend.
- Smoke loads no model and runs no training or evaluation.
- Successful G0: file SHA-256 `cdc90cd…c68f`, identity `e1f1c906…f89dc`, status
  `MODEL_SMOKE_PASS`, full-rank joint seed 7411.
- Retired failure: SHA-256 `47305826…2c71`; it must remain in the first recovery archive and must
  not reappear at either producer failure path.
- First recovery: source contract `55d0a455…56f3`, smoke SHA-256 `8bf5bb36…6849`.
- Allowed stages and argument shapes remain the first recovery's exact registered set. The producer
  CLI retains canonical path, source snapshot, authorization, setup/control, training, evaluation,
  lock, and receipt gates.
- Hidden-label boundary: smoke opens no contrast or benchmark payload. The wrapper cannot analyze.

## Recovery boundary

The wrapper imports the exact frozen first-recovery source and reuses its safe reads, immutable
writes, producer-context checks, argument-shape policy, stage-specific output validation, and
`src.analysis._canonical_expected_path` context manager. It does not call the first wrapper's
pathname-only `invoke_producer`; it owns new immutable STARTED/COMPLETE receipts and directly calls
the same exact producer CLI under the same path seam.

Every invocation first rehashes the retirement, archived failure, successful G0, first-recovery G0
STARTED/COMPLETE receipts, authorization, first-recovery smoke/source, producer source/model, and the
handoff's own frozen smoke/source. A producer output cannot be adopted unless its handoff STARTED
receipt already exists.

## Run

Freeze and smoke without loading the model:

```bash
.venv/bin/python -B -m unittest discover \
  -s experiments/qwen35_4b_state_formation_branch_handoff_recovery/tests -v
.venv/bin/python -B \
  experiments/qwen35_4b_state_formation_branch_handoff_recovery/scripts/run.py --smoke
```

Commit and push the successful G0, its first-recovery invocation receipts, and this frozen handoff;
require both repository workflows green. Then retry the seed-7411 positive control through this
wrapper. Every later branch-authorized model stage uses the same handoff wrapper; producer analysis
continues through the already-frozen analysis recovery.

## Results

The focused suite passes 8/8. The frozen smoke emitted
`BRANCH_HANDOFF_RECOVERY_SMOKE_PASS` at file SHA-256 `0e0409de…6e5b`, receipt identity
`d6fecf0a…de4c`, and source contract `4d2ffde3…8acd`. All six controls pass: the first wrapper's
pathname-only false rejection is reproduced; retirement, failure archive, successful G0, and its
invocation lineage are exact; and the retired mirror remains absent. Unit controls additionally
reject exact failed bytes at the successful slot, any changed successful-G0 bytes, a reappearing
mirror, and unregistered invocation shapes.

Smoke loaded no model, started no training/evaluation, opened no benchmark or contrast path, and
changed no scientific interpretation. No positive-control or downstream scientific result is
claimed here. Use remains blocked until this complete checkpoint is published and both workflows
are green.

## Interpretation

The valid G0 establishes full-rank feasibility but does not authorize training. This recovery is
only a safe handoff from that G0 to its preregistered positive control. Capacity interpretation
remains unchanged.

## Knowledgebase Update

- Program evidence: no scientific evidence from an invocation repair.
- Program backlog: full-rank seed-7411 positive control remains next after green publication.
- Claim ledger: unchanged.

## Artifacts

- `idea_intake.md`: nearest-neighbor and novelty decision.
- `reports/design_review.md`: adversarial review and stop conditions.
- `configs/default.yaml`: exact producer, recovery, failure, G0, and authorization pins.
- `runs/smoke.json`: frozen no-model handoff smoke.
- `runs/invocations/`: handoff-owned immutable producer invocation provenance.
- `reports/artifact_manifest.yaml`: producer-owned large-artifact boundary.
