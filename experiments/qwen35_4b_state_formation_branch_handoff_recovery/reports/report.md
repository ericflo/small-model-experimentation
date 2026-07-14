# State-Formation Branch Handoff Recovery Report

## Status

The first recovery produced a valid full-rank seed-7411 G0, but its frozen pathname-only retirement
guard stopped the following positive control before producer or model work. The handoff suite and
frozen no-model smoke passed, commit `627254f1…d892` cleared both workflows, and the full-rank
seed-7411 positive control now passes through this wrapper. Producer receipts remain scientifically
authoritative.

## Method

Pin the archived failure, retirement, successful G0, first-recovery G0 invocation lineage, first-
recovery source/smoke, producer source/model, and authorization. Reuse the exact frozen path seam
while replacing only outer invocation orchestration and emitting handoff-owned STARTED/COMPLETE
receipts.

## Results

- Focused tests: 8/8.
- Smoke status: `BRANCH_HANDOFF_RECOVERY_SMOKE_PASS`.
- Smoke file SHA-256 / identity: `0e0409de…6e5b` / `d6fecf0a…de4c`.
- Frozen source contract: `4d2ffde3…8acd`.
- Six smoke controls pass; unit controls reject failed bytes, changed G0 bytes, a reappearing mirror,
  and unregistered invocation shapes.
- Model load, training/evaluation, benchmark/contrast access, and scientific interpretation: zero.
- Full-rank seed-7411 positive control: producer SHA-256 `a0d17e2e…7a16`, identity
  `6708a4d4…9649`; oracle/intact 48/48, adaptation-disabled 0/48, 256 updates, accumulation 16,
  4,096 presentations, and changed full-rank/shared-state parameters.
- Handoff COMPLETE identity: `af0dd15c…2262`; producer result/benchmark/contrast access and
  scientific evidence remain zero.
- Full-rank seed-7412 G0: file SHA-256 `10bf22fc…1d18`, identity `62ecb79e…951dc`; zero K=1/
  round-trip error, complete finite optimizer state for all 62 deltas, finite K=12, 22.1 GiB free.
- Full-rank seed-7412 control: file SHA-256 `1cbbd823…8510`, identity `6575e1d2…4554`; oracle/intact
  48/48, disabled 0/48, 256 updates, accumulation 16, 4,096 presentations, changed full-rank/shared
  state, and zero result/benchmark/contrast/scientific access.
- Full-rank seed-7413 G0: file SHA-256 `021a8444…d635`, identity `4d2316d3…2ff0`; zero K=1/
  round-trip error, complete finite optimizer state for all 62 deltas, finite K=12, 22.1 GiB free.
- Full-rank seed-7413 control: file SHA-256 `8a4af0d6…fde8`, identity `976f28ef…14df`; oracle/intact
  48/48, disabled 0/48, 256 updates, accumulation 16, 4,096 presentations, changed full-rank/shared
  state, and zero result/benchmark/contrast/scientific access.
- Complete matrix: all three independent full-rank G0/control pairs pass; this remains setup-only.

## Interpretation

No capacity conclusion changes. A passing recovery licenses only the preregistered producer stage
that its inputs independently authorize.

## Next action

Publish the complete three-seed setup matrix and require both workflows green. Then run the six
already-mandated Stage-B cells—three LoRA state-only and three full-rank joint—through this wrapper;
no scientific evaluation begins until the producer's exact training barriers authorize it.
