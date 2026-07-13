# Implementation Review

**Source-contract version:** `7`

**Reviewed implementation SHA-256:** `af6d65dfe28dc385e8819e8443cd6b63985101f1bee5be2c248563a4576abcef`

**Status:** `GO`

Source-contract v7 passed integrated authorization, science/protocol, and terminal/recovery review.
This authorization binds exactly the implementation digest above plus the pinned training lock; any
change to a reviewed source or test revokes it automatically. It authorizes only the registered
source-v7 archive/regeneration/runbook sequence and does not make any historical setup artifact valid
under v7.

## Source-v7 integrated authorization

The complete source-bound CPU suite passes **355/355**. The focused terminal boundary passes
**142/142** across stable I/O (31), invalidated-setup archive/recovery (52), failed-attempt
archive/recovery (28), and static execution contracts (31). The independent frozen-science audit
passes **238/238** with no scientific defect: fixed seeds, split matrices, estimand, thresholds,
10,000-draw crossed bootstrap, Stage A/B/C ordering, sealed-contrast logic, and the mandatory
LoRA-negative full-shape/state-only branches remain exactly preregistered. No model or GPU was used
during this review, and no benchmark content was accessed.

The terminal review initially returned `NO_GO` and reproduced defects that ordinary happy-path tests
missed: exceptional context exits skipped canonical rebinding; recursive directory creation could
finish through a renamed-away ancestor; a completed-looking zero quarantine could bypass a failed
leaf fsync; archive payloads and both receipt copies were revalidated only after destructive cleanup;
and markerless evaluation recovery could confuse multiple historical attempts. All are now closed by
source-bound regressions. Stable contexts revalidate roots, ancestors, leaf bindings, and held inodes
on both normal and exceptional exit. Directory creation re-walks every held component from the
trusted root. Both archivers hold and repeatedly hash the exact durable archive while retiring
sources, validate every quarantine leaf before the first truncate, retain exact zero-length
quarantine skeletons instead of unlinking canonical evidence, and re-fsync completed skeletons and
all source parents on recovery. Archive receipts, tracked mirrors, tree membership, hardlink counts,
and full 64-character attempt identities are checked before mutation. All setup/result producers,
analyzers, and archive helpers share the ignored `runs/run.lock`, excluding concurrent cooperating
writers across the validation-to-retirement interval.

Adversarial regressions cover root and ancestor replacement, body-exception races, cross-parent
fsync failure, hardlinks and symlinks, destination collisions, corrupt final quarantine content,
last-leaf truncate-plus-fsync failure, archive-payload and archive-receipt mutation before cleanup,
regenerated canonical names coexisting with completed quarantine markers, partial move/zeroization
resume, multiple historical evaluation archives, malformed/wrong explicit attempt identities, and
exact training-journal recovery. Completion markers dominate without modifying a regenerated
canonical source, but they are never accepted without descriptor revalidation and fresh fsync.

This section supersedes historical prose below that describes cleanup as pathname deletion or
directory removal. Those passages record earlier implementation states; v7 uses no destructive
unlink/rmtree operation on canonical or quarantine evidence. Private transaction-owned staging may
still be discarded only after its exact ownership and destination state are validated.

This source-bound review records the executable go/no-go decision separately from the immutable
scientific design receipt. The runner, initialization and checkpoint lineage, sealed-data firewall,
optimizer receipts, analyzer, and terminal branch taxonomy pass the complete 201-test local
experiment suite plus independent adversarial science, CLI-matrix, and prose audits. The latest
aggregation, G0-failure-persistence, and setup-invalidation coverage passes 55/55 tests, and the
static/CLI/provenance contract passes 27/27.

The first real CPU-smoke attempt exposed an empty-authorization sentinel defect before any model or
result data ran. Execution authorization was retracted while it was repaired. The final runner now
passes `CPU_SMOKE_PASS`; all 23 registered CLI cells have exact exhaustive authorization and
canonical-output coverage, prohibited state-only contrast cells fail before output construction,
irrelevant branch axes cannot silently redirect a stage, and junk or noncanonical receipts/outputs
fail before dispatch. The broader suite also verifies exact seeded order and both optimizer-group
learning rates, adapter-disabled reversal, same-category failure replication, post-contrast reopening
of the three Stage-B arms, crash-safe atomic ledgers, and tamper rejection.

The first live seed-7411 G0 then exposed a Transformers 5.13.0 revision-provenance defect before
wrapper construction or mechanics. The pinned outer Qwen3.5 config resolves the registered commit,
but the causal-LM wrapper retains a derived text config whose `_commit_hash` is `None`. Execution
authorization was retracted. The repaired loader now requires config, every tokenizer asset, the
safetensors index, and every indexed shard to resolve with exact basenames through one canonical
`snapshots/<pinned-commit>` root; derives the common resolved revision from every file; records byte
counts and SHA-256 values; and then forces both loaders to the same exact revision in local-only mode,
with safetensors required for model weights. Runtime `None` is diagnostic only after that proof, while
any non-null mismatch remains fatal. Regression coverage proves malformed/empty indexes, missing or
traversal shards, mixed roots, wrong basenames, wrong commits, and proof-before-loader ordering all
fail closed. The real cache proof covers nine files totaling 9,342,815,919 bytes at the pinned commit.
An independent post-repair audit gives `GO` under the repository's standard trusted-Hugging-Face-cache
threat model.

The corrected seed-7411 retry subsequently reached the final G0 receipt write after passing every
registered in-memory check, but the two-step probe had rebound the canonical `output` path parameter
to a `StateLoopOutput`. The final writer therefore failed before creating a durable receipt. No
sealed contrast, positive control, result training, or evaluation ran, and all setup artifacts bound
to that source contract are preserved in a separate invalidated archive. The repair renames the probe
value, detaches a diagnostic before scalar conversion, and adds an AST regression that prohibits any
store to the `model_smoke` output-path parameter. An independent function-level scan found no
analogous destination shadow in CPU smoke, design freeze, data or initialization preparation,
positive control, training, checkpoint saving, evaluation, or analysis.

That audit also detected that the four hand-authored historical failure/archive receipts had hashed
a trailing newline while the runtime identity contract hashes compact canonical JSON. Their evidence
payloads and archived file manifests were correct; all four claimed identity fields were recomputed
without changing evidence content. A new regression reopens every tracked `runs/failures/*.json`
receipt with the runtime identity function. The full post-repair suite passes 135/135, including the
new destination-shadow and historical-receipt checks. Independent re-review gives `GO`.

That prior `GO` preserved the already frozen scientific design and authorized regeneration of every
invalidated-source setup artifact followed by a fresh attempt at the live gates. It is not scientific evidence
and does not waive G0, positive-control, hardware, parity, or branch-completeness requirements. Any
later mechanical source repair likewise changes the source-contract digest and requires downstream
data, initialization, and setup artifacts to be regenerated.

## Positive-control accumulation correction

The first durable seed-7411 LoRA G0 passed every registered mechanics gate. Its subsequent setup-only
positive control passed the oracle readout gate, completed 256 optimizer updates, and then scored 0/48
on exact terminal `(node, phase, checksum)` correctness. No positive-control pass, result payload,
result training, evaluation, or sealed access occurred. The failure is preserved at identity
`44397a2e278293bf54fe5d172ac4294c565a2a98ab7c8f4faaeb5ee044e8ec7c`.

Three independent failure audits found no target, shape, terminal-index, recurrence, gradient, or
scorer defect. They instead found that `positive_control()` used one singleton presentation per
optimizer update while ignoring the globally frozen `training.gradient_accumulation: 16`. The failed
path therefore made only 256 presentations, exposing each of the 48 high-entropy rows five or six
times. Exact joint chance is 1/256, so 0/48 is ordinary for an unlearned control and is not LoRA
capacity evidence.

An independent frozen-boundary adjudication gives a narrow mechanical `GO`. The decision rests on
evidence fixed before the outcome: accumulation is global alongside the learning rate and clipping
controls already consumed by this function; the frozen architecture defines an optimizer update
after 16 microbatches; the dropout contract is indexed by global microbatch; the smoke overlay changes
both accumulation and positive-control updates; and repository precedent uses the same convention.
This does not license a generic post-failure budget increase.

The corrected control keeps all explicit scientific values unchanged: the same 48 rows and canonical
row hash, seed 73991, exactly 256 optimizer updates, state-only objective, K equal to row depth, LR
`2e-4`, weight decay zero, dropout 0.05, groupwise clip thresholds, oracle threshold 0.99, fixed-final
threshold 0.95, initialization, row order, and no early stopping or checkpoint selection. Each update
now contains exactly 16 sequential singleton microbatches, each loss is divided by 16, both groups are
clipped once, and the optimizer steps once. Global microbatch indices 1--4096 feed the unchanged
dropout preimage. This deterministically gives 16 rows 86 exposures, 32 rows 85, and depth exposure
counts 1368/1368/1360; the implementation fails closed if any count changes.

Fixed observational probes at updates 0, 1, 16, 64, 128, and 256 score all 48 rows in both intact and
adaptation-disabled modes. They record component, joint, terminal, trajectory, depth, loss,
parameter-delta, gradient, clipping, schedule, exposure, and optimizer evidence. A dedicated context
restores train/eval mode and exact CPU/CUDA RNG state, and before/after parameter receipts prohibit a
probe mutation. Only the intact update-256 metric enters the existing pass rule.

Every reached-control exception now atomically writes an identity-bound canonical
`SETUP_CONTROL_FAILED` receipt and an identical source-qualified tracked mirror before re-raising.
Both explicitly deny training, record zero benchmark reads and sealed/result payload access, and
retain all diagnostics completed safely. Training still accepts only `POSITIVE_CONTROL_PASS`.
A new source-bound invalidation helper stages and fsyncs every current setup byte, verifies both
archive receipts, and deletes current setup only after durable verification. It also recognizes
future canonical positive-control pass/failure receipts and requires an identical failure mirror, so
recovery never requires deleting evidence by hand.

The frozen configuration, preregistration, architecture, runbook, handoff, design review, and design
receipt remain byte-unchanged. All setup artifacts bound to source `3baa7b53…d5c42` must be archived,
then CPU smoke, procedural data and empty seal ledger, initialization bundles, and seed-7411 G0 must
be regenerated under the final replacement source before the corrected control runs.

At the archival boundary the source-bound suite contained 154 tests, including 13 positive-control
tests, eight setup-invalidation tests, and 27 static/CLI/provenance tests. Independent code and
CUDA/runtime audits gave `GO`. The then-current setup reopened as exactly 20 files totaling 17,775,495
bytes with file-manifest identity
`115b9dd5e810cb1bcf88e58a4da366370cbb48c4a1fc1adfcdcb37a5283e1a25`, an empty seal ledger, and
collision-free archive targets. The durable receipt now preserves that complete setup at identity
`1daa86e02f7d3f3c612c2f1ec01db89e4967b5b9ce7eb19ba75c752fe1e283aa`; it remains immutable.

The first post-archive repository gate found that successful deletion left the now-empty tracked
`runs/cpu_smoke/` and `runs/setup/` directories on disk. A first cleanup patch removed those
directories, but adversarial review then reproduced two deeper hazards before commit or regeneration:
a failed unlink/fsync could leave a partial setup that the existing archive made non-resumable, and
resolved root/evidence paths could hide symlinks and redirect cleanup outside the canonical tree.

The final helper treats the immutable archive receipt as its transaction journal. Fresh, archive-only,
archive-plus-tracked, and invalid tracked-only states are distinct. Archive-only recovery validates
every archived byte before atomically copying the receipt's original bytes to the tracked mirror.
Resume reconstructs live paths only from an allowlisted, ordered file manifest and accepts each source
only as exact or already absent. It rejects changed sources, unknown residue, nonidentical receipts,
archive extras or special entries, partial temporaries, and symlinks in every canonical root, ancestor,
or evidence leaf before any read or deletion. Cleanup fsyncs each parent batch, removes only the two
required empty tracked roots, preserves the data `.gitignore` and external archive, and verifies all
postconditions plus both immutable receipts before returning. A completed rerun is read-only and
idempotent.

The final source-bound suite passes 171/171: 13 positive-control tests, 25 setup-invalidation/recovery
tests, and 27 static/CLI/provenance tests. Focused correction/recovery coverage passes 38/38, and the
implementation gate passes 7/7. Tests inject archive-only promotion, mid-unlink, `rmdir`, and fsync
failures; arbitrary partial deletion; tampered and unknown live state; tracked/archive mismatch;
special filesystem entries; and symlinked data, runs, archive, trigger, tracked-receipt, and failure-
mirror paths. Independent artifact and runtime/path re-audits both give `GO`. The real source-`3baa`
archive replays as a no-op with all 20 files and 17,775,495 bytes exact, both receipt bytes and mtimes
unchanged, and every cleanup postcondition satisfied. No setup artifact was created under either
intermediate post-archive source. All regenerated setup must bind the source identity produced after
this addendum; the historical archive receipt must not be rewritten.

No result-bearing arm is authorized until its required live setup gates pass.

## Seed-7412 aggregation-precision and G0-persistence repair

The final-source seed-7411 G0 and corrected 48/48 setup control passed, but seed-7412 G0 stopped at
the frozen live-joint all-groups reachability gate. Its gradient tensor for the registered
`aggregate_logit` existed and was finite with exact norm zero. Every one of 124 LoRA tensors, all
four initializer tensors, the step projection, all eight sufficiency tensors, and the damping scalar
had finite nonzero gradients; the frozen base had none. The failure occurred before the joint clip or
optimizer step and is preserved as setup-mechanics evidence at identity
`ce3406f8fa788c08421687d5d6a0843a2eb7035fd254d5086123d895b2bb634c`. It is not evidence that LoRA
failed to form the registered representation.

The scalar is initialized identically at a sigmoid weight of 0.9 and is untouched by the preceding
state-only probes. Under the old implementation it was cast from FP32 to BF16 before broadcasting
through the last-state/mean-state convex mix. Two otherwise matched seed-7411 G0 executions produced
scalar gradients `2.197266076e-5` and `4.394532152e-5`; division by the unchanged sigmoid derivative
maps them exactly to projection increments `2^-12` and `2^-11`. Together with seed 7412's
allocated-but-zero tensor, this makes one-example BF16 projection/reduction quantization a concrete,
falsifiable explanation and argues against simple graph disconnection or magnitude underflow; it is
not a legitimate capacity miss. The factor-two variation under matched code, row, masks,
initialization, environment, and answer loss also makes an unchanged retry a retry-to-pass risk.

The handoff explicitly permits a pre-result dtype/autocast incompatibility repair. The approved
change preserves the existing BF16 recurrence states and computes their mean in BF16 exactly as
before. It keeps the registered sigmoid gate in FP32, converts only the already-computed last and mean
states for the scalar convex mix, executes that mix with autocast disabled, and casts the completed
aggregate back once. K=1 still bypasses initialization, recurrence, and aggregation exactly.
Aggregation remains used only by the optional answer graph, so state-only still gives
`aggregate_logit` no gradient or Adam moments. Parameters, initialization, row, target, objective,
dropout stream, call geometry, steps, thresholds, fixed-final rule, and branch order are unchanged;
the common helper applies identically to LoRA and full rank. No frozen design file or design receipt
changed.

A deterministic adversarial BF16 test constructs a width-256 projection for which the legacy mix
gives scalar gradient exactly zero while the repaired production helper gives the analytic FP32
gradient `0.045`. Separate tests lock mean-before-cast arithmetic, BF16 output, explicit autocast
disablement, one cast back, K=1 bypass, and answer-only aggregate use. The frozen live G0 gate still
requires the scalar gradient to be present, finite, and strictly nonzero; no exemption or tolerance
was added. If the regenerated seed-7412 G0 still produces exact zero, this precision hypothesis is
falsified: the gate must remain failed and any multi-row reachability redesign belongs in a new
preregistered successor.

The same failure revealed that `model_smoke` did not durably emit the preregistered
`SETUP_CONTROL_FAILED` receipt. The public guard now captures initialization immediately after its
canonical loader, then incrementally binds data, authorization, setup, reached checks, dropout, and
gradient diagnostics. Any in-scope exception emits a nonauthorizing canonical failure and a
byte-identical source-qualified tracked mirror before re-raising. One serialization is written to two
independently fsynced staging inodes, then installed without replacement at the mirror first and the
canonical path second. The final files have distinct inodes, so an in-place mutation of one cannot
silently mutate the other. Existing files, broken symlinks, or symlinked ancestors are never
replaced. This establishes `canonical => mirror`, while a crash between installs leaves mirror-only
evidence that blocks replay rather than losing the failure. Pass and failure receipts explicitly
record train-only/zero-sealed/zero-benchmark access and deny result authorization at G0. Receipt
loading hard-rejects `SETUP_CONTROL_FAILED` even if accidentally added to a caller's status allowlist;
positive-control and training entry points also require every pass authorization and access field to
be present with its exact type and value. Successful G0 persistence uses the same fsynced,
no-replacement leaf installation rule, so a file or broken symlink raced into place during the live
attempt is preserved and the pass fails closed.

The invalidation helper now accepts either a G0 pass or canonical G0 failure. A failure must have the
exact phase, source/config/seed binding, exact data and initialization lineage for every reached
check, truthful nulls limited to the corresponding early failure stages, false authorization and
scientific flags, a safe access list, nonempty error fields, and a byte-identical nonsymlink mirror;
its completed checks must be the exact registered prefix allowed at its failure stage, both copies of
initialization lineage must independently match once reached, and it cannot coexist with a positive
control. The archive transaction revalidates every archived failed-G0 and failed-control mirror on
resume and immediately before each cleanup unlink, so a removed or changed non-trigger mirror stops
further deletion even after a partial cleanup.
The canonical failure enters the source archive while its tracked mirror remains in place and should
be used as the archival trigger. The historical seed-7412 receipt remains tracked-only because the
old runner never created a canonical output; it must not be retroactively fabricated or added to the
archive inventory.

Source-contract version 6 adds the focused aggregation and G0-persistence tests. The complete suite
passes 201/201, including 38 setup-invalidation/recovery tests, 14 G0 failure/authorization tests,
three aggregation-precision tests, and 27 static/CLI/provenance tests. Tests inject early and
mid-probe failures, the observed zero-scalar structure, existing and broken-symlink destinations,
symlinked ancestors, a second-link interruption, explicit downstream allowlisting, missing/changed/
symlinked mirrors, unsafe access and authorization, wrong phase/initialization, and failed-G0/control
coexistence. Independent numerical, runtime-persistence, and archive audits give `GO` for this narrow
mechanical repair.

Because source-contract files changed, every setup artifact under source
`1d1368cf064689322d9df7f345e67b026cecccc32d3a7b7514b82f253d434b0a` is invalid for downstream
authorization. Archive it with the preserved seed-7412 failure as trigger, regenerate CPU/data/empty
ledger/all three initialization bundles under the replacement source, and replay G0 and controls
from seed 7411 before any seed-7412 retry. No result-bearing stage is authorized meanwhile.
