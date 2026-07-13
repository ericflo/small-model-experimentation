# State-Formation Capacity Adjudication Report

## Status

**In progress; no scientific result exists.** Preregistration, adversarial design review,
implementation review, and the frozen design are complete. The corrected setup-control source passed
171/171 local tests and independent code and GPU/runtime review. Every setup artifact from source
`3baa7b53…d5c42` is durably archived. The later CPU smoke, data manifest, empty sealed-access ledger,
and three shared initialization bundles were regenerated and strictly reopened under source
`1d1368cf…434b0a` before entering their own archive. Seed-7411 LoRA G0 passed canonically at identity
`928e756f…820c`, and its corrected positive control passed 48/48 at identity `8db4595e…2df7`. Seed
7412 then stopped fail-closed during its live-joint G0 probe: every adaptation and recurrent group
except the registered aggregation scalar had finite nonzero gradients, while
`aggregate_logit.grad` was present and finite with norm exactly zero. No canonical seed-7412 G0
receipt was created. The exact failure is preserved at identity `ce3406f8…b634c` as mechanics
evidence only. The narrow aggregation-precision and durable G0-failure repair now passes 201/201
tests and independent numerical/runtime/archive review under source `d4269bf3…8b36`. The complete
source-`1d1368cf…434b0a` setup is now durably archived at receipt identity `13cdcaec…2050b` (21
files, 18,288,790 bytes). Replacement-source CPU smoke, data/empty ledger, and all three shared
initializations are regenerated and strictly reopened under `d4269bf3…8b36`; their manifest is
`e935c31a…1e57`, ledger identity is `eaf7ba23…b3cb`, and tensor values match source-1d exactly. Seed
7411 LoRA G0 now passes at identity `185835ee…3216`, with repaired aggregation-scalar gradient
`5.3286785e-5`, exact K=1 parity, finite K=12 execution, and zero checkpoint-roundtrip error. Seed
7411's positive control also passes at identity `6a1394d9…cefa`: intact 48/48, disabled 0/48 after
exactly 256 updates and 4,096 presentations. Seed 7413 and all result stages remain blocked pending
seed-7412/7413 replay. No training
checkpoint, evaluation row, or terminal analysis from this directory should be cited as scientific
evidence.

Under the invalidated source, seed 7411 passed LoRA G0 and then scored 0/48 on the setup-only control.
The control had omitted the globally frozen accumulation of 16, presenting only one singleton row per
optimizer update. The corrected path retains 256 optimizer updates and every explicit control value,
but uses 16 loss-scaled singleton microbatches per update and records fixed diagnostics. The prior miss
is preserved as mechanics history only. The source-1d setup had an empty contrast ledger and the
same shared tensor values as the earlier archived setup. Source-1d seed-7411 G0 passed every
registered mechanics gate. Its corrected control completed exactly 256 updates and 4,096
presentations, scored 48/48 intact and 0/48 with adaptation disabled, and authorized later-seed setup
under that source. The pre-repair BF16 path cast the FP32 aggregation weight before its convex mix;
matched seed-7411 observations put the scalar gradient on exact BF16 reduction-grid increments.
Review therefore rejects an unchanged retry. The implemented repair forms the same
last-state/mean-state convex mix in FP32 before one cast back to model dtype while retaining the BF16
mean, and persistent G0 failures now have independent canonical/mirror evidence. The frozen
nonzero-gradient gate was not weakened. The source repair invalidated and the archive preserved every
source-1d setup receipt, so replacement replay must begin from seed 7411. Every seed's setup gate must pass before any result-bearing arm
is authorized.

## Why this experiment exists

The parent rank-32 LoRA pilot and its direct-full-rank successor both produced chance-like registered
states. The latter did not close capacity: parameterization-specific construction shifted the shared
state-module initialization and dropout stream, and its state miss occurred alongside unrelated
Carry/Bag and answer-query promotion failures. The raw negatives remain useful, but neither is a
three-seed paired capacity adjudication.

This successor keeps the original joint objective for the first comparison, matches all avoidable
cross-capacity randomness, separates state formation from downstream utilization, and adds a
state-only objective only as a conditional diagnostic.

## Registered method

Three fixed-final 1,500-step rank-32 LoRA joint runs at seeds 7411–7413 run first. Each uses the same
Qwen3.5-4B recurrent Carry loop, fresh procedural rows, state targets, shared initialization bundle,
row order, and capacity/objective-independent adaptation-dropout schedule. Every final checkpoint is
evaluated both intact and with adaptation disabled.

Before training, the actual custom LoRA hook must match pinned PEFT output and A/B gradients on copied
tensors in two regimes: FP32/dropout-off/autocast-off at `atol=1e-6`, `rtol=1e-5`, and live-like
bf16-autocast/dropout-0.05 at `atol=2e-3`, `rtol=1e-2`. The stochastic comparison resets the same
device RNG seed immediately before each implementation's forward and receipts the custom realized
native-dropout mask/call cycle.

An all-seed LoRA formation pass stops the experiment and answers that LoRA does not prevent state
formation under this design. A valid miss mandates three LoRA state-only controls and three
full-rank joint runs. A full-rank trigger miss then mandates three full-rank state-only controls. If
the trigger passes and sealed rows open, the LoRA-all-cell stop has priority; otherwise a full-rank
sealed miss mandates those controls. No Bag, edge cut, swap, answer-gain, or sample-more result can
promote or demote this capacity verdict.

## Registered evidence

The headline event is exact terminal joint node+phase+checksum correctness at K equal to semantic
depth. The absolute threshold is `0.40` in every seed×depth cell: trained depths 2–4, unseen depths
5–12, and depth-5–12 joint family-plus-surface shift. Depth 1 is reported but excluded because its K=1
path has no adaptation. Trajectory-step accuracy, component scores, crossed task-by-seed intervals,
and each split/depth remain separate.

A `DIRECT_FULLSHAPE_RECIPE_RESCUE` must also be adaptation-dependent and selection-safe on a complete
fresh counterpart to every required trigger domain: 768 sealed trained-depth rows at seed 73307 and
two sealed 1,024-row deep/joint-shift splits at seeds 73305 and 73306. Intact full shape must robustly
beat its own adaptation-disabled mode and the paired LoRA joint checkpoints on all three splits. The
earlier LoRA-miss receipt cannot open these rows, and a full-rank trigger miss leaves them unopened
because rescue is already impossible. The evaluator still runs six capacity×seed jobs, with every job
scoring all three splits in both modes. This prevents trained shared heads, incomplete controls, or
reuse of branch-selecting rows from manufacturing a capacity result.

Rescue additionally requires category-matched replication: every LoRA category that failed on
trigger (`trained`, `depth`, and/or `joint`) must fail again on its matching sealed domain. Additional
sealed failures are allowed, but a failure elsewhere cannot substitute. An interrupted sealed job can
reuse its ledger event only after exactly one new content-validated `FAILED_ATTEMPT_ARCHIVED` receipt
is appended; completed evaluations cannot enter that archive path.

## Result table

No result rows have been generated. The eventual report will preserve all reached arms, including
negative controls and branch stops; unreached conditional arms will be labeled prohibited rather than
scored as failures.

## Interpretation contract

- LoRA pass: LoRA does not prevent formation; full rank is prohibited.
- LoRA trigger miss followed by a pass in every sealed LoRA cell across trained depth, extrapolation,
  and joint shift:
  `LORA_TRIGGER_MISS_NOT_REPLICATED_ON_SEALED_CONTRAST`; the fresh miss did not replicate and no
  direct-recipe rescue claim is licensed.
- A valid full-rank trigger miss—or, after excluding the all-cell LoRA sealed stop, a full-rank
  sealed miss—with setup controls valid: full-rank relief is not sufficient. Use the conditional
  state-only controls as descriptive evidence consistent with objective competition, not causal
  identification. If neither state-only arm passes, report
  `FULLRANK_RELIEF_NOT_SUFFICIENT_REGISTERED_RECIPE_BOTTLENECK_UNRESOLVED` and leave
  supervision/readout architecture versus optimization unresolved.
- LoRA still fails somewhere on sealed data, full rank passes every trigger and sealed absolute cell,
  but at least one trigger-failed LoRA category passes its corresponding sealed domain:
  `LORA_TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED_ON_SEALED_CONTRAST`; extra failures elsewhere do
  not license rescue.
- Valid LoRA miss plus direct-full-shape pass and all three robust sealed contrasts:
  `DIRECT_FULLSHAPE_RECIPE_RESCUE`, a practical recipe result rather than mathematical rank
  identification.
- Direct full shape passes every trigger and sealed absolute cell but the contrast is uncertain:
  `DIRECT_FULLSHAPE_RECIPE_PASS_CONTRAST_UNCERTAIN`.
- Any reached arm whose intact checkpoint misses formation while its adaptation-disabled mode passes:
  report `ADAPTATION_DISABLED_REVERSAL`. Keep branching based on intact, prohibit adaptation-required
  or rescue wording for that arm, and preserve the LoRA and full-rank sealed statuses separately so a
  LoRA sealed reversal is not hidden by the terminal capacity label.
- Any mechanics, lineage, shared-initialization, dropout, completeness, or setup-control failure:
  repair and rerun the registered cell; no scientific terminal label.

These outcomes concern readable internal state only. They make no claim about answer use, causal
transport, serial advantage, deployment, capability gain, or beating sample-more.

## Artifacts

The planned tracked and external artifacts are specified in
[`artifact_manifest.yaml`](artifact_manifest.yaml). Runtime checksums are absent until the referenced
artifacts are actually created; they must be filled from generated receipts, never guessed.
