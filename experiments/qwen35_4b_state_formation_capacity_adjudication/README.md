# State-Formation Capacity Adjudication

**Status:** in-progress · since 2026-07-13 · authoritative Stage-A result is `LORA_JOINT_MISS_CONTROLS_REQUIRED`; Stage B LoRA-state-only and full-rank-joint arms are mandatory

## Current status

This is the canonical fresh adjudication of the unresolved LoRA-capacity question from
`qwen35_4b_state_carry_vs_state_bag`. It is not a continuation of either prior checkpoint.
Preregistration, adversarial design review, and the frozen scientific design are complete. The
first Stage-A source-v10 command stopped before model load, train-payload access, optimizer
construction, or training. Its journal reached only `PREPARED`, and its canonical external output
was exactly empty: the attempt JSON helper duplicated a repository-relative destination beneath its
own relative parent. This is a mechanical launch failure, not a LoRA result. The exact journal is
preserved at SHA-256 `209d4c1f…12df`, and failure receipt identity `6b23f95d…b8c2` explicitly records
zero result/sealed/benchmark access and no scientific authority. Source v11 normalizes the
parent/leaf publication boundary, recovers only the exact empty-directory mkdir-before-marker crash
state, canonicalizes validated training and contrast paths before use, and rejects any nonempty
markerless output. Reviewed implementation `7d6cd93f…d278`, full source contract
`5a8ed26d…6666`, and the 363/363 suite pass. Every source-v10 setup artifact is invalid for v11 and
is now preserved in a verified 25-file, 19,566,021-byte archive at receipt identity
`252be000…5d6a3` and files identity `f767eb80…d4b91`. Canonical setup cleanup and an immediate
idempotent replay passed. The archive checkpoint is green at `24733d34`; the canonical empty output
and stale journal were retired only after byte-exact revalidation of their published copies, and the
retirement checkpoint is green at `aa85086f`. Source-v11 non-model setup is now regenerated: CPU
receipt SHA-256 `d46b32bd…0192`, data-manifest SHA-256 `d104a9c0…a22c`, data-contract identity
`43363814…6669`, and empty-ledger identity `01b2724b…41f3f`. All three initialization bundles reopen
canonically, their tracked mirrors are byte-identical and inode-distinct, and their tensor values
exactly reproduce source v10. No model or benchmark was opened and no sealed payload was
decompressed. The non-model checkpoint is green at `bcd57181`. Source-v11 seed-7411 LoRA G0 then
passed at file SHA-256 `9292794f…1264d` and identity `da06c198…94d0f`: exact PEFT parity, zero K=1
and checkpoint-roundtrip error, finite K=12, all 124 LoRA tensors and every recurrent group with
finite nonzero live-joint gradients, no base gradient, and aggregation gradient `6.1676066e-5`. Its
positive control passed at file SHA-256 `ee1de715…c98d8` and identity `c830e65b…0c259`: oracle and
intact fixed-final 48/48, adaptation-disabled 0/48, exactly 256 updates, accumulation 16, and 4,096
presentations. Both remain setup-only evidence with zero benchmark/sealed access. Publish this pair
and require both workflows green before seed 7412. The seed-7411 setup checkpoint is green at
`5536f785`. Source-v11 seed-7412 LoRA G0 then passed at file SHA-256 `c19a6944…6b82d0` and identity
`e2229ad2…da380`: exact PEFT parity, zero K=1 and checkpoint-roundtrip error, finite K=12, all 124
LoRA tensors and every recurrent group with finite nonzero live-joint gradients, no base gradient,
and aggregation gradient `4.4653425e-5`. Its positive control passed at file SHA-256
`1a8d263b…a27a37` and identity `fdb1a16e…cd72a3`: oracle and intact fixed-final 48/48,
adaptation-disabled 0/48, exactly 256 updates, accumulation 16, and 4,096 presentations. Both remain
setup-only evidence with zero benchmark/sealed access. Publish this pair and require both workflows
green before seed 7413. The seed-7412 setup checkpoint is green at `47e160bf`. Source-v11 seed-7413
LoRA G0 then passed at file SHA-256 `87faadfa…aca4d0` and identity `1fa2b054…eec8d5`: exact PEFT
parity, zero K=1 and checkpoint-roundtrip error, finite K=12, all 124 LoRA tensors and every
recurrent group with finite nonzero live-joint gradients, no base gradient, and aggregation gradient
`1.5512113e-3`. Its positive control passed at file SHA-256 `e2e70edf…fc5b57` and identity
`8d74b90d…d689b7c`: oracle and intact fixed-final 48/48, adaptation-disabled 0/48, exactly 256
updates, accumulation 16, and 4,096 presentations. The complete three-seed source-v11 LoRA setup
matrix now passes with zero benchmark/sealed access and no scientific evidence. Publish and require
both workflows green before the three Stage-A LoRA joint cells. The setup checkpoint is green at
`5b7c22fe`. Stage-A seed-7411 LoRA joint training then completed exactly 1,500 fixed updates in
9,217.946 seconds. The canonical classifier returns `COMPLETE` with no errors: run SHA-256
`19cac1e2…7aa60`, run identity `6db22c2f…f81f1`, checkpoint identity `174f3c69…cf9e`, adaptation
state SHA-256 `9af35963…9e71f`, and loop-state SHA-256 `333f8c44…50a5`. All four tracked mirrors are
byte-identical and inode-distinct from their external artifacts; the attempt journal is terminal
`COMPLETE`; all 1,500 optimizer rows are finite. The cell opened only train, no benchmark or sealed
contrast, and authorizes no evaluation until the complete three-cell Stage-A barrier exists. Publish
this training checkpoint and require both workflows green before seed 7412. That checkpoint is green
at `72d526d8`. Stage-A seed-7412 LoRA joint training then completed exactly 1,500 fixed updates in
9,202.149 seconds. The canonical classifier returns `COMPLETE` with no errors: run SHA-256
`0bb1680e…1e05`, run identity `f7d049c6…d0ca`, checkpoint identity `90750967…0324`, adaptation
state SHA-256 `cba9fb80…8c74`, and loop-state SHA-256 `03f4b811…8076`. All four tracked mirrors are
byte-identical and inode-distinct from their external artifacts; the attempt journal is terminal
`COMPLETE`; all 1,500 optimizer rows are finite. The cell opened only train, no benchmark or sealed
contrast, and authorizes no evaluation until seed 7413 completes the Stage-A training barrier.
Publish this training checkpoint and require both workflows green before seed 7413. That checkpoint
is green at `07547f4e`. Stage-A seed-7413 LoRA joint training then completed exactly 1,500 fixed
updates in 9,527.480 seconds. The canonical classifier returns `COMPLETE` with no errors: run
SHA-256 `8bbdb3b3…8b59`, run identity `07f91939…362b`, checkpoint identity `d65d5bb3…e68d`,
adaptation state SHA-256 `2ed8167f…9a65`, and loop-state SHA-256 `97d59a7f…b298`. All four tracked
mirrors are byte-identical and inode-distinct from their external artifacts; the attempt journal is
terminal `COMPLETE`; all 1,500 optimizer rows are finite. The cell opened only train, no benchmark
or sealed contrast. All three cells reopen `COMPLETE`, producing the exact Stage-A training barrier
identity `31d7f6f3…6451`. Publish this checkpoint and require both workflows green before the three
preregistered trigger evaluations. That checkpoint is green at `664d6574`. Seed-7411 trigger
evaluation then completed in 978.507 seconds with exactly 3,072 intact and 3,072
adaptation-disabled rows. Its summary identity is `fd7a516e…9264`; it binds checkpoint
`174f3c69…cf9e`, source v11, and reached-training barrier `6d506285…589e`, and records zero benchmark
or sealed-contrast access. No result values were inspected. An independent production-loader reopen
then stopped before result consumption: `analysis._canonical_expected_path` rejects the experiment's
own registered `../../large_artifacts/...` path after `ROOT / value` preserves `..` lexically. The
same deterministic defect would block `run.py --stage analyze`. It does not invalidate training or
evaluation production, but source v11 cannot emit a terminal analysis receipt. Preserve the v11
artifacts unchanged and freeze a separate analysis-recovery consumer before further evaluation.
The separate recovery experiment passed its exact-prefix smoke at receipt identity
`30353be5…2f8f`, was published at `a6360cc1`, and reached full site/validation coverage at
`e35e071e`; both workflows are green. Seed-7412 trigger evaluation then completed in 1,008.649
seconds with 3,072 intact and 3,072 adaptation-disabled rows. Its summary identity is
`03eaf96b…597d`; intact rows SHA-256 is `dd3f9424…cc4f`, disabled rows SHA-256 is
`743f821a…c085`, and summary file SHA-256 is `a18ba761…f4e2`. The receipt binds checkpoint
`90750967…0324`, exact source v11/config, and records zero benchmark or sealed-contrast access.

An operator validation command accidentally selected the complete `modes` mapping rather than its
three row-lineage fields, printing seed-7412 per-split scientific metrics before seed 7413. This
violated the intended within-stage no-value-inspection procedure and is preserved in
`reports/operator_deviation.md`; no analyzer, classifier, branch, source edit, seed choice, retry,
checkpoint selection, or conditional action occurred. Seed 7413 and every downstream command were
already fixed and remain mandatory regardless of the exposed values. The final result must disclose
that operator blinding was imperfect; it may not use this deviation to stop, repair, or relabel the
precommitted analysis.

Seed-7413 trigger evaluation then completed once, without retry, in 986.884 seconds with exactly
3,072 intact and 3,072 adaptation-disabled rows. Its summary identity is `b35b9e14…da70`; intact
rows SHA-256 is `d62c43a9…b30c`, disabled rows SHA-256 is `683156e9…6cb`, and summary file SHA-256
is `411af149…80a`. The receipt binds checkpoint `d65d5bb3…e68d`, exact source v11/config, the
reached training barrier, and only the three preregistered trigger payloads. It records exact K=1
bypass, zero benchmark or sealed-contrast access, and no branch authorization. Scientific values
were not inspected. The complete three-seed trigger matrix is now present; publish this exact
checkpoint and require both workflows green before the frozen recovery consumer runs
`--phase lora_joint`.

Seed-7413 checkpoint `b326f6cd` passed both workflows. The frozen recovery consumer then executed
the exact immutable v11 analyzer and emitted authoritative producer receipt identity
`b973bc01…a862` (`cb9fee75…818a` on disk), bound by recovery sidecar identity
`d068482a…f40e`. Its status and verdict are `LORA_JOINT_MISS_CONTROLS_REQUIRED`, and its sole next
stage is `run_lora_state_only_and_fullrank_joint`. All 57 required seed×split×depth cells missed the
0.40 joint node+phase+checksum threshold; the best intact required cell was 0.0234375, with per-seed
maxima 0.0234375, 0.015625, and 0.015625. Trained, depth-extrapolation, and joint-shift categories
all miss. The intact-versus-disabled result is `ADAPTATION_CONTRAST_UNCERTAIN`, so Stage A shows that
this LoRA joint recipe did not form the registered state; it does **not** yet show that LoRA capacity
caused the miss. The registered capacity adjudication now requires all three LoRA state-only controls
and all three direct-full-shape joint runs. No contrast split was opened.

The first authorized full-rank seed-7411 G0 then failed at downstream branch-authorization
revalidation before model load: v11's receipt consumer reaches the same nonlexical registered-prefix
defect as its analyzer. Canonical/mirror failure SHA-256 is `47305826…2c71`, identity
`070c23af…aa24`; no check, model load, training/evaluation, benchmark, sealed contrast, or scientific
evidence occurred. The additive `qwen35_4b_state_formation_branch_recovery` successor now passes its
real downstream smoke at identity `d1135ea2…49b5` and has archived the exact failure at archive
identity `ff478d40…0ec3`. Publish that checkpoint, then retire the original failure pair and publish
the retirement before retrying through the frozen wrapper.

Archive commit `bdedabf4…b6b2` passed both workflows. The recovery then verified all committed
archive/source blobs and retired exactly the two producer failure paths at retirement identity
`c9abdc59…eae7`, retaining the exact failure archive. Publish and require both workflows green on
this retirement checkpoint before the recovered full-rank seed-7411 G0 retry.

Retirement commit `e69c1960…c79` passed both workflows. The recovered full-rank seed-7411 G0 then
passed at file SHA-256 `cdc90cd…c68f`, identity `e1f1c906…f89dc`: exact K=1 parity before/after
optimizer and checkpoint-roundtrip error are zero; all 62 direct-full-shape delta tensors have
finite nonzero joint gradients and complete finite optimizer state; K=12 is finite; and about 22.1
GiB remains free. It authorizes only the positive control and records zero training/evaluation,
benchmark, contrast, or scientific evidence. The first recovery's frozen pathname-only retirement
guard then rejected the positive-control handoff before wrapper STARTED publication. The additive
`qwen35_4b_state_formation_branch_handoff_recovery` successor distinguishes exact retired-failure
bytes from the exact successful G0 and passes its frozen no-model smoke at identity
`d6fecf0a…de4c`, source contract `4d2ffde3…8acd`. Publish that checkpoint and require both workflows
green before retrying the positive control through the handoff wrapper.

Handoff commit `627254f1…d892` passed both workflows. The full-rank seed-7411 positive control then
passed at producer file SHA-256 `a0d17e2e…7a16`, identity `6708a4d4…9649`: oracle and intact fixed-
final joint state are 48/48, adaptation-disabled is 0/48, after exactly 256 updates, accumulation 16,
and 4,096 singleton presentations. Both the 892,272,640-parameter full-rank path and shared state
changed; early stopping and checkpoint selection were disabled. Handoff COMPLETE identity is
`af0dd15c…2262`. The pair remains setup-only evidence with zero result-payload, benchmark, contrast,
or scientific-evidence access. Publish and require both workflows green before the registered full-
rank seed-7412 G0/control pair; result training remains blocked on the complete setup matrix.

Seed-7411 setup commit `c507488c…3c06` passed both workflows. Full-rank seed 7412 then independently
passed G0 at file SHA-256 `10bf22fc…1d18`, identity `62ecb79e…951dc`, with zero K=1/round-trip
error, complete finite optimizer state for all 62 deltas, finite/nonzero joint gradients, finite K=12,
and 22.1 GiB free. Its control passed at file SHA-256 `1cbbd823…8510`, identity
`6575e1d2…4554`: oracle/intact 48/48, disabled 0/48, exact 256 updates, accumulation 16, and 4,096
presentations; both full-rank and shared-state parameters changed. Handoff G0/control COMPLETE
identities are `d61d6441…9246` / `6334d72f…c14a`. Both remain setup-only evidence with zero result,
benchmark, contrast, or scientific-evidence access. Publish/green this pair before seed 7413.

Historically, the integrated source-v10 review was `GO` for reviewed implementation
`a5a494b7…6f1c4a` and full source contract `979a9012…f394b7`; the 360/360 suite and exact machine gate
pass. Source-v9 seed-7411 G0 stopped before model load or wrapper construction because the general
no-alias reader rejected Hugging Face's standard snapshot-to-content-blob symlinks. The byte-identical
canonical/mirror failure has file SHA-256 `39ec9625…46ec7` and identity `30af333c…9cfe9`; it opened
only train validation, no benchmark or sealed split, started no training/evaluation, and authorizes
nothing. Source v10 keeps general artifact reads no-symlink while adding a dedicated exact-revision,
exact-basename, content-addressed cache proof. The real nine-file, 9,342,815,919-byte cache passes at
file-set identity `06486f26…d1fe12` without loading the model. Source v10 is published at `3756ce29`
with both workflows green. All source-v9 setup is now preserved in a verified 20-file,
17,655,138-byte archive: file-set identity `7360b00d…1f2650`, receipt file SHA-256
`086d35af…be14e`, and receipt identity `8d5fe94d…33ad5c`. The exact 20-leaf zero quarantine,
canonical cleanup, retained failure mirror, and idempotent replay all pass. The archive checkpoint
was published at `9c1fadde` with both workflows green before source-v10 setup
regeneration began. Source-v10 CPU smoke has SHA-256 `ebfb68fe…17bc5`; the regenerated manifest is
`0b1cca35…7422a`, data contract `5fba6c3c…c252`, and empty-ledger identity
`d0b9eda7…17e04`. All three shared initialization bundles reopen canonically, have byte-identical but
inode-distinct tracked mirrors, and exactly reproduce the archived source-v9 tensor values. Bundle
SHA-256 / sidecar receipt identity by seed: 7411 `e202efb8…3e0d8` / `9f6923d4…147abd`; 7412
`ab0b70c1…9169b` / `cf90d157…806e9`; 7413 `7dca25ea…a7c5c` / `791aa7ec…6201`. No model was loaded,
no benchmark was read, and no sealed split was decompressed. This non-model checkpoint must now be
validated, committed, pushed, and green in both workflows before seed-7411 G0. That checkpoint was
published at `33abfe33` and both workflows passed. Source-v10 seed-7411 LoRA G0 then passed at file
SHA-256 `efde2db9…420f9` and identity `0fe46a0c…7a383`: exact PEFT parity and K=1 behavior, all 124
LoRA tensors and every required common-state group finite/nonzero in the live joint probe, no base
gradient, finite K=12, zero-error checkpoint restoration, and aggregation-scalar gradient
`2.2352195e-5`. Its seed-matched positive control passed at file SHA-256 `04b7f995…a34f6` and identity
`18d02610…ebe89`: oracle accuracy 1.0, fixed-final intact 48/48, disabled 0/48, exactly 256 updates,
accumulation 16, and 4,096 presentations. Both receipts record zero benchmark and sealed-contrast
access and no scientific evidence. Publish and validate this setup-pair checkpoint before seed 7412;
all result stages remain blocked on the complete three-seed setup barrier. Seed-7411 setup commit
`d0642d4a` passed both workflows before seed 7412 began. Source-v10 seed-7412 LoRA G0 passed at file
SHA-256 `e2ef4951…fc18f` and identity `4af55cc3…e30de`, including all 124 LoRA gradients, no base
gradient, exact parity/K=1/roundtrip, finite K=12, and aggregation-scalar gradient `1.0440836e-4`.
Its positive control passed at file SHA-256 `1097d31b…c41e5` and identity `ae4fdb5f…e461b`: oracle
accuracy 1.0, fixed-final intact 48/48, disabled 0/48, exactly 256 updates, accumulation 16, and
4,096 presentations. Both receipts remain setup-only with zero benchmark/sealed access. Publish and
validate this pair before seed 7413; result stages remain blocked on the full setup barrier.
Seed-7412 setup commit `41d587ec` passed both workflows before seed 7413 began. Source-v10 seed-7413
LoRA G0 passed at file SHA-256 `b1a42d2e…5b9256` and identity `1909508f…b47e43`, including all 124
LoRA gradients, no base gradient, exact parity/K=1/roundtrip, finite K=12, and aggregation-scalar
gradient `1.5991002e-3`. Its positive control passed at file SHA-256 `5ea3a552…fa01c7` and identity
`7ed7332a…4f85cc`: oracle accuracy 1.0, fixed-final intact 48/48, disabled 0/48, exactly 256
updates, accumulation 16, and 4,096 presentations. All three source-v10 setup pairs now pass with
zero benchmark/sealed access and no scientific evidence. Publish and validate this final setup
checkpoint before the three Stage-A LoRA joint training cells; no result run has started.
The source-v8 code was published at commit `ee729def` with both workflows green, and the source-d426
archive is now complete: 23 files, 18,927,960 bytes, files identity `1538f2f2…ec3ed0`, receipt file
SHA-256 `9aa04d35…efc1a1`, and receipt identity `e7a71362…818b77`. Independent verification matched every
archive payload and zero-length quarantine leaf, confirmed exact canonical cleanup, and an idempotent
replay made no further change. The combined source-v9/archive checkpoint is published at `9932a560`
with both workflows green. Source-v9 CPU smoke now has SHA-256 `1655354b…45a38`; the regenerated
manifest is `957013ad…a4517`, data contract `3677a0a3…ae761`, and empty-ledger identity
`eb8028bf…9c84dc`. All three shared initialization bundles reopen canonically, have byte-identical
tracked mirrors, and exactly reproduce the archived tensor values. Bundle SHA-256 / receipt identity
by seed: 7411 `5ed9d5c6…0e1b8` / `74ddebb1…31413`; 7412 `15366ea6…dcb2c` /
`01dd7e7c…1ee4b`; 7413 `bda608a2…bf8b4` / `a7ab7d5d…e912f`. No model was loaded, no benchmark was
read, and no sealed split was decompressed during this regeneration. That non-model checkpoint was
published at `ff4a8b9b` with both workflows green before the failed G0. All of it is now invalidated
by source v10 and preserved in the exact source-v9 archive described above.
Under source `3baa7b53…d5c42`, seed 7411 passed LoRA G0, then its 256-update setup
control scored 0/48 exact terminal triples. Review found that the control had presented one singleton
row per optimizer update and omitted the globally frozen accumulation of 16, so each high-entropy row
appeared only five or six times. The scorer, targets, recurrence, gradients, and fixed-final gate were
aligned. The miss is therefore preserved as a setup failure, not evidence about LoRA capacity.

The positive-control source correction keeps the same 48 rows, 256 optimizer updates, seed, state-only objective,
learning rate, dropout, thresholds, initialization, and row order. It now applies the frozen 16-way
accumulation: 4,096 indexed singleton presentations, loss divided by 16, one groupwise clip and one
optimizer step per update. Fixed probes record intact and adaptation-disabled metrics without changing
parameters, mode, or random streams. Any reached failure writes a canonical receipt plus an identical
tracked mirror and still denies result training. The complete source-bound suite passed 171/171, and
independent code and GPU/runtime audits both gave `GO` before live seed-7412 G0 exposed the separate
aggregation-precision defect described below.

Every setup artifact tied to `3baa7b53…d5c42` remains preserved in a verified 20-file archive whose
receipt identity is `1daa86e…e283aa`. The later setup under source `1d1368cf…434b0a` was created,
strictly reopened, and is now preserved in a verified 21-file archive at identity
`13cdcaec…2050b`: CPU smoke SHA-256 `56032f75…7ad43`, data-manifest SHA-256
`85286a95…0cd9`, data contract `891ad784…e9c8`, and empty-ledger identity `b122d490…3c14`.
All three source-bound initialization bundles passed the canonical loader, their tracked receipts were
byte-identical to their external sidecars, and their tensor-value digests exactly reproduced the
archived shared initialization. No sealed contrast row was decompressed, no model was loaded during
regeneration, and the ledger still had zero events. Seed-7411 LoRA G0 passed canonically under
the final source at receipt identity `928e756f…820c`: both PEFT parity regimes have zero observed
error, K=1 is exact, every required recurrent group and all 124 LoRA tensors receive finite nonzero
gradients while the base receives none, the K=12 path is finite, and checkpoint restoration is exact.
The earlier source-3baa G0 pass and 0/48 control miss remain historical mechanics records only. The
corrected source-1d control passed 48/48 after exactly
256 optimizer updates and 4,096 singleton presentations; disabling adaptation at the same fixed final
scores 0/48, confirming that the setup path actually exercised the LoRA update.

Replacement setup under source `d4269bf3…8b36` is now strictly reopened: CPU smoke SHA-256
`1d5a57c9…6fdb`, manifest SHA-256 `e935c31a…1e57`, data contract `8e95991b…d5b`, and empty-ledger
identity `eaf7ba23…b3cb`. All three initialization sidecars are byte-identical to their tracked
mirrors, and every tensor-value digest exactly equals its source-1d predecessor. Regeneration loaded
no model, decompressed no sealed payload during reopening, and left the ledger at zero events.
Seed-7411 LoRA G0 then passed at identity `185835ee…3216`: the repaired aggregation scalar's live
joint gradient is finite and nonzero at `5.3286785e-5`, all other required trainable groups are
finite/nonzero, the frozen base has no gradients, K=1 is exact before and after optimization, K=12
is finite, and checkpoint roundtrip error is zero. The receipt authorizes only its setup control.
That seed-7411 control passed at identity `6a1394d9…cefa`: oracle accuracy 1.0, fixed-final intact
48/48, disabled 0/48, exactly 256 updates, accumulation 16, and 4,096 presentations. It confirms
the repaired setup path still depends causally on the LoRA update.
Seed-7412 G0 then passed at identity `737a8b39…0a89f`. The formerly exact-zero aggregation scalar now
has a present, finite, nonzero live-joint gradient `6.6731358e-5`; every unchanged mechanics gate also
passes. This supports the preregistered BF16 projection/reduction explanation and closes the
precision-repair question without weakening the gate. It is still setup evidence, not a LoRA result.
Its positive control then passed at identity `02a329d9…669a`: oracle accuracy 1.0, fixed-final intact
48/48, adaptation-disabled 0/48, exactly 256 updates, accumulation 16, and 4,096 presentations.
However, the pre-result authorization re-audit found that generic receipt consumption does not yet
re-enforce every model/backend, access, and downstream-authorization claim written by canonical
receipts. Result training remains blocked while that execution boundary is repaired, tested, and all
source-bound setup is archived and regenerated. The control is valid setup evidence for source
`d4269bf3…8b36`, but it cannot authorize a scientific run under a later source contract.

**Historical source-`1d1368cf…434b0a` attempt.** Seed-7412 LoRA G0 stopped at the frozen live-joint
reachability gate. Every one of the 124 LoRA
tensors and every other required recurrent group had a finite nonzero gradient, the base model had
none, and `aggregate_logit.grad` existed and was finite but had norm exactly zero. The pre-repair code
cast that FP32 scalar gate to BF16 before the last-state/mean-state convex mix. Two otherwise matched
seed-7411 G0 executions had aggregate gradients on exact BF16 reduction-grid increments, so an
unchanged retry would risk retry-to-pass rather than adjudicate connectivity. The failure is preserved
at receipt identity `ce3406f8…b634c`; it is setup-mechanics evidence only. No canonical seed-7412 G0
receipt exists, seed 7413 is blocked, and no result training or sealed scoring is authorized. Review
permits only a narrow FP32 convex-mix repair with the same row, masks, objective, schedule, threshold,
and registered nonzero gate. That repair is now implemented under source contract
`d4269bf3…8b36`: the recurrent mean remains BF16, only the scalar convex mix is FP32, and the
completed aggregate is cast back once. G0 failures now persist a nonauthorizing canonical receipt and
an independent byte-identical source-qualified mirror without overwriting existing or symlinked
paths. The complete suite passes 201/201, a CUDA BF16 adversarial probe reproduces legacy gradient
zero versus repaired analytic gradient 0.045, and independent numerical/runtime/archive re-audits
give `GO`. The frozen nonzero gate is unchanged. All source-`1d1368cf…434b0a` setup is archived and
replacement-source seed-7411 and seed-7412 G0/controls passed. A pre-result authorization audit then
found fail-open generic receipt checks; source repair and setup archival/regeneration are required
before seed 7413 or any result-bearing stage.

### Source-v11 operator boundary

Do not promote the preserved source-v10 journal from `PREPARED` to `STARTED`: no model or training
began. Publish source v11 and its tracked failure/journal evidence first. After both workflows pass,
archive every source-v10 setup artifact with
`runs/failures/training_launch_lora_joint_seed7411_source_979a90120e99.json` as the trigger, publish
that archive checkpoint, retire the now-durable canonical empty output and stale PREPARED journal,
then regenerate CPU smoke, all procedural data plus the empty ledger, all three initialization
bundles, and all three LoRA G0/control pairs under source v11. Stage A remains blocked until that
replayed setup matrix is published and green.

The source-v10 instructions below are retained as historical transition evidence.

The frozen GPU runbook is the complete from-zero phase order, not the current resume point. Preserve
`reports/design_receipt.json` and every file it freezes; do not rerun `design-boundary` or rewrite the
preregistration, design review, architecture, runbook, handoff, or default config.
The source-d426 transition is complete at receipt identity `e7a71362…818b77`. Source-v9 non-model
setup was published before its seed-7411 G0 stopped fail-closed at model setup. After source contract
v10 is committed, pushed to `main`, and both repository workflows are green, archive every source-
`5629a3a4…99e236` setup artifact through the registered invalidation helper. That transition is
complete at receipt identity `8d5fe94d…33ad5c`. Publish and validate the archive checkpoint before
regenerating CPU smoke, procedural data/empty ledger, all three initialization bundles, and all three
LoRA G0/positive-control pairs under one final v10 source. Archive commit `9c1fadde` passed both
workflows, and the non-model regeneration is now complete. Publish and validate this non-model
checkpoint before replaying all three LoRA G0/positive-control pairs in seed order.

The exact one-time transition command is:

```bash
EXP=experiments/qwen35_4b_state_formation_capacity_adjudication
.venv/bin/python -B "$EXP/scripts/archive_invalidated_setup.py" \
  --invalidated-source 5629a3a4f12f5720cbd4aea53520d905ed5f03e7e41f3b21df9b4bc92399e236 \
  --trigger-failure "$EXP/runs/failures/g0_lora_seed7411_source_5629a3a4f12f.json"
```

The emitted tracked receipt and external archive are independently verified, published, and green.
The command is retained as the exact historical transition record; do not rerun it against live
source-v10 setup.

A result checkpoint directory is not a completed training cell. Completion requires the exact
external and tracked `TRAINING_COMPLETE` receipts, byte-identical but inode-distinct attempt-marker,
training-metric, and optimizer-step mirrors, the fixed-final checkpoint graph, and the durable
`runs/attempts/training/<slug>.json` journal head in `COMPLETE` state with the exact terminal-run
lineage. The external `run.json` is the last terminal artifact, but the subsequent journal transition
is the completion commit; a crash between them is recoverable but remains incomplete until the exact
published graph finalizes that journal. Evaluation is unavailable until the whole reached training matrix is terminal:
three Stage-A cells, six new Stage-B cells (nine total reached), or three new Stage-C cells (twelve
total reached). A receipt/setup/branch failure before canonical output creation needs no
failed-attempt archive; an existing incomplete canonical output must be archived before a step-zero
retry.

Branch authorization is path- and purpose-specific, not a status string. Stage B accepts only
`analysis/lora_joint_trigger.json` with `LORA_JOINT_MISS_CONTROLS_REQUIRED`; sealed contrast accepts
only `analysis/stage_b_seal.json` with `STAGE_B_CONTRAST_AUTHORIZED`; Stage C accepts
`analysis/stage_b_seal.json` or `analysis/fullrank_joint.json` only when that exact file emits
`FULLRANK_STATE_ONLY_REQUIRED`. `analysis/lora_control.json` is supporting Stage-B evidence, never a
branch authorization. Copies, renamed paths, symlink aliases, mismatched purpose fields, and nested
decoy lineage do not authorize execution.

For an interrupted training cell, pass one existing canonical external or tracked directory to
`scripts/archive_failed_attempt.py`; it automatically captures every existing same-cell companion,
records why the terminal graph is incomplete, and refuses a valid completed pair. Do not move or
delete either side manually. A markerless evaluation retry with multiple historical archives must
also pass the exact 64-character attempt-authority identity via `--attempt-identity`; prefixes and
guessing are rejected. All setup/result producers, analyzers, and both archive helpers share the
ignored `runs/run.lock`, so archive verification and source retirement cannot overlap a cooperating
writer. Cleanup keeps a durable zero-length quarantine skeleton, re-fsyncs it on recovery, and never
uses pathname deletion to dispose of canonical evidence. Commit and push the tracked archive receipt
before retry. At every
verified source, archive, setup, training, evaluation, and analysis checkpoint, run `make check`,
commit the scoped tracked evidence, push `main`, and wait for both `Validate Repository` and
`Publish Research Site` to succeed before the next model-bearing command.

## Research program and prior anchors

- Primary program:
  [`structured_execution_and_compilers`](../../research_programs/structured_execution_and_compilers/charter.md),
  because the scientific endpoint is whether a recurrent latent execution state forms.
- Secondary program:
  [`posttraining_and_adaptation`](../../research_programs/posttraining_and_adaptation/charter.md),
  because the experiment compares a factorized LoRA update with a direct full-shape update and tests
  how joint versus state-only posttraining changes that state. This is secondary program fit, not a
  second estimand or an additional verdict axis.
- Closest anchor: [`qwen35_4b_state_carry_vs_state_bag`](../qwen35_4b_state_carry_vs_state_bag/README.md),
  whose valid rank-32 LoRA pilot failed joint state formation after 300 steps.
- Mandatory capacity anchor:
  [`qwen35_4b_state_carry_vs_state_bag_fullrank_delta`](../qwen35_4b_state_carry_vs_state_bag_fullrank_delta/README.md),
  whose direct-delta pilot also formed almost no state but did not match cross-capacity shared
  initialization or dropout RNG and simultaneously failed non-capacity promotion gates.
- Earlier recurrent negative: [`qwen_fastweight_hook`](../qwen_fastweight_hook/README.md), whose
  256-dimensional answer-supervised hook showed no robust K scaling under larger retests.

The novelty is a fresh, three-seed, fixed-final capacity adjudication that changes only the
registered extra-call adaptation parameterization while explicitly matching shared loop-state
tensors, row order, and adaptation-dropout streams. It separates the original joint objective from a
state-only control and gives state formation its own verdict, independent of Bag, answer-gain,
edge-cut, swap, or sample-more gates.

## Question

Does rank-32 extra-call LoRA prevent the repeated Qwen block from forming the registered deep joint
`(node, phase, checksum)` representation under the original joint training objective? If LoRA misses,
does direct full-rank parameterization rescue formation under the same seeds and controlled
initialization/stochastic streams, or does the failure remain when answer competition is removed?

## Hypothesis

LoRA is plausible: its 62 rank-32 updates act throughout two complete repeated Qwen motifs, while the
carried state remains full width and receives dense state supervision. A valid three-seed LoRA joint
pass would therefore show that low rank does not prevent state formation in this design and would
prohibit the expensive full-rank branch.

If LoRA joint misses, the state-only LoRA control gives a descriptive pattern consistent with answer
loss competition without causally identifying it, while a mandatory full-rank joint arm tests the
practical rank/parameterization concern.
Full-rank relief, with an adaptation-dependent state gain, supports a practical LoRA limitation. If
both joint parameterizations miss and their setup controls are valid, full-rank relief was not
sufficient. If both state-only controls also miss, the registered recipe bottleneck remains
unresolved between supervision/readout architecture and optimization rather than justifying another
rank retry.

## Frozen design

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, through Transformers 5.13.0.
- Runtime: the parent's causal Carry loop over layers 12–19, with eight state-before-query slots,
  untouched non-state memory between extra calls, and exact base K=1 behavior.
- Data: fresh procedural pointer-world rows from the same substrate logic; no benchmark content and no
  reuse of the prior experiments' evaluation rows.
- Training: seeds 7411, 7412, and 7413; exactly 1,500 optimizer steps; fixed final checkpoint only;
  batch one with 16-way accumulation; no pilot, early stopping, seed replacement, or checkpoint
  selection.
- Joint objective: `1.0 * answer + 0.5 * state + 0.05 * fixed_point`.
- State-only control: `0.5 * state + 0.05 * fixed_point`; the answer term is absent from the graph.
- LoRA: rank 32, alpha 64, 0.05 adaptation dropout, active only on extra R applications.
- PEFT compatibility gate: the actual custom hook must match a pinned PEFT `Linear` reference in
  output and A/B gradients under both FP32/dropout-off/autocast-off (`atol=1e-6`, `rtol=1e-5`) and a
  live-like bf16-autocast/dropout-0.05 regime (`atol=2e-3`, `rtol=1e-2`). Both forwards receive
  copied base/A/B/input tensors and the same device RNG reset immediately before each forward, so the
  live-like comparison uses the matched realized mask protocol rather than merely equal nominal
  probabilities.
- Full rank: 62 zero-initialized direct FP32 deltas, 892,272,640 parameters, the same 0.05 adaptation
  dropout and scale 2, active only on the same extra R applications.
- Pairing: a common custom adaptation-hook path, one bit-identical shared loop-state initialization
  bundle per seed, capacity-specific construction RNG, deterministic row order, and a matched
  per-microbatch dropout schedule that excludes capacity and objective from its seed.
- Optimizer isolation: shared loop-state and adaptation parameters are clipped as separate groups, so
  the dense arm's gradient norm cannot rescale the common-module update.
- Evaluation: every checkpoint is scored both intact and with adaptation disabled while retaining its
  trained shared state modules and readout heads.

## Sequential firewall

1. Run the mechanically validated, setup-positive-control-qualified LoRA **joint** objective for all
   three seeds.
2. If LoRA clears every registered state-formation gate, emit
   `LORA_DOES_NOT_PREVENT_STATE_FORMATION` and prohibit all further capacity arms.
3. If a complete, valid LoRA joint result misses any formation gate, its identity-bound analysis
   receipt mandates three LoRA **state-only** controls and three full-rank **joint** runs on the same
   seeds.
4. If full-rank joint misses any trigger cell, run the three full-rank **state-only** controls without
   opening sealed data. If its trigger passes but a sealed absolute cell later misses, open the same
   state-only branch only after applying the all-cell LoRA-pass priority in step 5. These are the only
   paths to the maximum 12 result-bearing runs.
5. If full rank passes its trigger, a dedicated seal opens three fresh contrast splits for both joint
   arms: trained-depth validation, depth extrapolation, and joint shift. The evaluator still runs six
   capacity×seed jobs because each job scores all three splits in both intact and disabled modes.
   A LoRA pass there emits `LORA_TRIGGER_MISS_NOT_REPLICATED_ON_SEALED_CONTRAST` and prohibits a
   rescue claim regardless of full-rank score. Otherwise a full-rank absolute miss still mandates
   state-only. If full rank passes, every LoRA category that failed on trigger must fail again in its
   corresponding sealed domain; otherwise emit
   `LORA_TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED_ON_SEALED_CONTRAST` and prohibit rescue. Extra
   sealed failures cannot substitute for a missing category replication.
6. Missing cells, mechanics failures, initialization/dropout mismatches, or positive-control failures
   authorize repair only. They never become a scientific terminal result and never authorize a seed
   substitution.

## Primary metric and scope

The primary event is exact terminal joint state correctness at K equal to semantic depth: node,
phase, and checksum must all be correct. A parameterization passes only when **every seed×depth cell**
reaches at least `0.40` on trained depths 2–4, unseen depths 5–12, and the depth-5–12 joint held-out
family-plus-surface split. Depth 1 is reported but excluded because K=1 bypasses adaptation. Terminal,
trajectory-step, component, trained-depth, unseen-depth, and joint-shift results remain separate;
none may be pooled to hide a failed depth.

Fresh trigger splits contain 1,024 rows each for validation, depth extrapolation, and joint shift.
After a LoRA miss, a 768-row trained-depth validation contrast at seed 73307 (depths 2–4, 256/depth)
and two 1,024-row deep contrasts at seeds 73305 and 73306 remain sealed until all
three LoRA-joint, three LoRA-state-only, and three direct-full-shape-joint fixed finals and their
intact/disabled trigger evaluations are complete. A dedicated identity-bound
`STAGE_B_CONTRAST_AUTHORIZED` receipt—not the earlier LoRA-miss receipt—must also prove that no
contrast scoring output or prior-open record exists. It is emitted only if direct full shape passes
every trigger cell; a trigger miss mandates state-only immediately and leaves the contrasts unopened.
Only the authorization receipt can open those rows, and only those sealed rows support a
cross-capacity rescue contrast.

An interrupted sealed evaluation may retry the same cell/checkpoint/canonical path only after its
incomplete output is moved to the content-addressed failure archive. Each retry must discover exactly
one newly tracked and content-validated `FAILED_ATTEMPT_ARCHIVED` receipt; that receipt is appended to
the existing access-ledger event and both event and ledger identities are recomputed. The ledger is
atomically and durably replaced while a separate stable lock inode is held, before decompression.
Initial access rejects an archive that predates its event; a completed evaluation cannot be archived
as failed, and an old archive cannot license another retry.

For every checkpoint, the analysis also reports intact minus adaptation-disabled state accuracy.
If intact misses formation while adaptation-disabled passes, it emits
`ADAPTATION_DISABLED_REVERSAL`: removing the trained adaptation improves the readable state enough to
reverse the absolute verdict. Branching still follows the preregistered intact checkpoint, and this
diagnostic prohibits an adaptation-required or direct-recipe-rescue interpretation.
The post-contrast receipt exposes `lora_sealed_contrast_adaptation` separately from the full-rank
sealed adaptation status, so a LoRA sealed reversal cannot be hidden by the cross-capacity terminal
label.
Any `DIRECT_FULLSHAPE_RECIPE_RESCUE` additionally requires preregistered positive
intact-minus-disabled and direct-full-shape-minus-LoRA effects on all three sealed splits, with positive
every-seed effects, no depth reversal, and crossed task-by-seed lower bounds above zero. This prevents
trained shared heads or conditional split reuse from manufacturing a rescue. Model-seed rows are
never treated as independent tasks.

The selection-safe replication rule is category-specific: trigger `trained`, `depth`, and `joint`
failures map to `contrast_validation`, `contrast_depth`, and `contrast_joint`. Every trigger-failed
category must fail again; additional sealed category failures are allowed. The category guard is
applied only after full rank passes its absolute trigger and sealed cells, because a full-rank miss
instead opens the registered state-only control.

This experiment can establish readable state formation and a practical direct-full-shape recipe
signature. It cannot identify mathematical rank alone, causal state use, answer improvement, serial
advantage over Bag, deployment capability, or a win over matched-compute sampling. Those require a
fresh successor.

## Run

The run is deliberately non-monolithic. At the current source-v10 resume point, the source-v9 archive
transition and source-v10 non-model regeneration in **Source-v10 operator boundary** are complete.
The non-model artifacts passed repository validation and both workflows at commit `33abfe33` before
the first model-bearing command; seed-7411 setup passed both workflows at `d0642d4a`, and seed-7412
setup passed both at `41d587ec`. All three G0/control pairs now pass. The final setup pair must be
committed, pushed, and green before Stage A. The completed smoke command was:

```bash
.venv/bin/python -B experiments/qwen35_4b_state_formation_capacity_adjudication/scripts/run.py --stage cpu-smoke
```

Then follow [`docs/gpu_runbook.md`](docs/gpu_runbook.md). Every model-bearing or branch stage must
reopen and verify the exact upstream receipt; a status string copied into another file is not
authorization.

## Expected artifacts

- `idea_intake.md`: duplicate search, novelty, and decision.
- `reports/preregistration.md`: frozen scientific contract and terminal taxonomy.
- `reports/design_review.md`: adversarial pre-run review; required before the design receipt.
- `reports/implementation_review.md`: machine-enforced source-version execution authorization.
- `reports/design_receipt.json`: canonical pre-model identity boundary once frozen.
- `docs/architecture.md`: common loop and adaptation-backend contract.
- `docs/gpu_runbook.md`: phase order, inspections, and recovery rules.
- `docs/research_handoff.md`: rationale and continuity.
- `reports/artifact_manifest.yaml`: tracked/external artifact policy.
- `runs/attempts/training/<slug>.json`: durable per-cell launch and replay history.
- `runs/` and `analysis/`: runtime receipts and results after execution; no result exists yet.
