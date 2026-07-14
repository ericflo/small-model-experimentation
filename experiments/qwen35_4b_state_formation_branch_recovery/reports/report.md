# State-Formation Branch Authorization Recovery Report

## Status

Operational partial. The first full-rank Stage-B G0 failed before model load while immutable v11
recomputed its LoRA-miss authorization through the already-known nonlexical external-prefix defect.
This recovery archived and retired the exact failure and then produced a valid full-rank seed-7411
G0. Its frozen pathname-only retirement guard cannot hand off to later stages, so an additive
successor is active. This remains mechanics/setup evidence, not a LoRA/full-rank capacity result.

## Method

The recovery imports exact producer v11 `scripts/run.py`, leaves every producer CLI and reviewed
source-snapshot gate active, and temporarily replaces only the exact `src.analysis` path helper used
by downstream branch validation. The seam is limited to the one registered raw prefix and clean
descendants; all other paths retain v11 behavior. A real no-model authorization smoke and
commit-backed failure archival/retirement precede retry.

## Current evidence

- Authoritative producer LoRA-miss receipt: SHA-256 `cb9fee75…818a`, identity `b973bc01…a862`.
- First Stage-B G0 failure pair: SHA-256 `47305826…2c71`, identity `070c23af…aa24`.
- Failure stage: `branch_authorization`; completed checks: none.
- Model load, training/evaluation, benchmark access, sealed contrast access, scientific evidence,
  and downstream authorization: none.
- Focused tests: 14/14.
- Recovery smoke: file SHA-256 `8bf5bb36…6849`, identity `d1135ea2…49b5`, source contract
  `55d0a455…56f3`; all six path/consumer/restoration controls pass.
- Failure archive: file SHA-256 `4fcccea3…45ed`, identity `ff478d40…0ec3`; archive bytes match the
  source pair and occupy a third inode.
- Retirement: file SHA-256 `6e4c8ee3…53ad`, identity `c9abdc59…eae7`; all four archive-commit blobs
  revalidated, only the producer pair removed, and the recovery archive retained.
- Recovered G0: `MODEL_SMOKE_PASS`, file SHA-256 `cdc90cd…c68f`, identity
  `e1f1c906…f89dc`; exact K=1 and round-trip error zero, all 62 full-rank deltas have finite nonzero
  joint gradients and complete finite optimizer state, with about 22.1 GiB free after G0.
- Frozen handoff defect: the next stage was rejected before wrapper STARTED publication because the
  retirement guard tests pathname existence rather than the retired failure's exact bytes/status.

## Interpretation

The original analysis recovery was scientifically correct but operationally incomplete. This
successor proved the consumer seam and produced a valid G0, then revealed that its one-time
retirement guard cannot hand off after the canonical slot is legitimately repopulated. No capacity
conclusion changes; an additive byte/status-aware handoff preserves this frozen result.

## Next action

Publish and validate `qwen35_4b_state_formation_branch_handoff_recovery`, then run the already-
authorized full-rank seed-7411 positive control through that wrapper.
