# State-Formation Capacity Adjudication

**Status:** in-progress · since 2026-07-13 · frozen design unchanged; source-v8 implementation review `GO`; source-v8 publication/CI and source-d426 archival still required; no result run is authorized

## Current status

This is the canonical fresh adjudication of the unresolved LoRA-capacity question from
`qwen35_4b_state_carry_vs_state_bag`. It is not a continuation of either prior checkpoint.
Preregistration, adversarial design review, and the frozen scientific design are complete. The
integrated source-v8 review is `GO` for reviewed implementation
`f9364c36…b9d873` and full source contract `7991d46a…b1cc88`; the 357/357 suite and exact machine gate
pass. Execution remains ordered: publish this source to `main`, require both workflows green, archive
the source-d426 setup, publish that archive checkpoint, and only then regenerate source-v8 setup.
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

### Source-v8 operator boundary

The frozen GPU runbook is the complete from-zero phase order, not the current resume point. Preserve
`reports/design_receipt.json` and every file it freezes; do not rerun `design-boundary` or rewrite the
preregistration, design review, architecture, runbook, handoff, or default config. After
source-contract v8 is committed, pushed to `main`, and both repository workflows are green, archive
every source-`d4269bf3…8b36` downstream setup artifact through the registered invalidation helper.
Then regenerate CPU smoke, procedural data and the empty contrast ledger, all three initialization
bundles, and all three LoRA G0/positive-control pairs under the one final v8 source before Stage A.

The exact one-time transition command is:

```bash
EXP=experiments/qwen35_4b_state_formation_capacity_adjudication
.venv/bin/python -B "$EXP/scripts/archive_invalidated_setup.py" \
  --invalidated-source d4269bf34f7c80affcc8c1e8a33fee9afddcd912d1bd9dead223e520ee108b36 \
  --trigger-failure "$EXP/runs/failures/pre_result_authorization_audit_failure.json"
```

Inspect the emitted tracked receipt and external archive, run `make check`, commit and push that
archive checkpoint, and wait for both repository workflows before regenerating any v7 setup.

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

The run is deliberately non-monolithic. At the current source-v8 resume point, perform the
source-d426 archive transition in **Source-v8 operator boundary** above first. Only after that archive
checkpoint is committed, pushed, and green should setup regeneration start with the non-model smoke:

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
