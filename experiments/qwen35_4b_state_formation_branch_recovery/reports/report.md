# State-Formation Branch Authorization Recovery Report

## Status

Pre-retry. The first full-rank Stage-B G0 failed before model load while immutable v11 recomputed its
LoRA-miss authorization through the already-known nonlexical external-prefix defect. The frozen
downstream recovery smoke now passes, and the exact failure pair is archived but not yet retired.
This is mechanics evidence, not a full-rank or LoRA result.

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

## Interpretation

The original analysis recovery was scientifically correct but operationally incomplete: it could
produce branch receipts that immutable downstream consumers could not reopen. This successor must
prove the same seam at the consumer boundary without broadening it. No capacity conclusion changes.

## Next action

Publish the exact smoke/archive checkpoint, retire the two producer failure paths using that green
commit, publish the retirement, then retry the already-authorized full-rank seed-7411 G0 through the
wrapper.
