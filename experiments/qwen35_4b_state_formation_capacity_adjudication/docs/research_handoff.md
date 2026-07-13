# Research Handoff

## The exact unresolved question

The LoRA parent and direct-delta successor both learned almost none of the registered recurrent
state. It is tempting to say that full rank settled the issue, but the comparison was not paired:
capacity-specific construction shifted the shared state tensors and dropout stream, only one
300-step seed ran, and the full-rank state miss coincided with unrelated Carry/Bag and query-stratum
failures. The correct conclusion is that two unmatched recipes failed, not that rank was eliminated.

This experiment is the clean adjudication. It asks whether the original **joint** objective forms
state with rank-32 LoRA under a sufficiently long, three-seed fixed-final schedule. Full rank is
conditional, not automatic. State-only is a diagnostic control, not a replacement question.

## Why the branch is sequential

Full-rank training is expensive and unnecessary if LoRA already forms state. More importantly, the
user's concern has a directional answer:

- LoRA joint pass: low rank did not prevent formation here; stop.
- LoRA joint miss, direct joint pass with the sealed replication/contrast gates satisfied: the direct
  recipe practically relieved the failure.
- full rank misses a trigger cell—or, after sealed access and absent the LoRA-all-cell stop, a sealed
  cell—with valid setup controls: direct full shape was not sufficient; inspect the conditional
  state-only controls before designing a new representation/supervision successor.

The branch receipt is load-bearing. No full-rank model-bearing stage may run merely because a human
read a low score. It must consume the complete identity-bound LoRA miss analysis. Conversely, a valid
LoRA miss makes the registered follow-up mandatory; it cannot be left as an optional suggestion.

That LoRA-miss receipt licenses Stage-B training only. It cannot open any of the three sealed
contrast splits.
After all LoRA joint, LoRA state-only, and direct-full-shape joint fixed finals and intact/disabled
trigger evaluations are complete, the separate `stage_b_seal` analysis must bind their identities and
prove no prior contrast scoring access. A direct-full-shape trigger miss mandates state-only without
opening the sealed rows. Only a complete trigger pass yields `STAGE_B_CONTRAST_AUTHORIZED` and
licenses the contrast evaluator. `CONTRAST_FIREWALL_NOT_READY` is not a result; actual premature
access burns the rows and requires a fresh successor rather than ledger deletion.

## Why state-only is conditional

Changing from joint to state-only can make the representation easier to learn by removing answer
competition. That is valuable diagnosis, but it would not answer whether rank caused the original
joint-objective miss. Therefore joint always runs first for a capacity, and state-only opens only
after that capacity's joint miss.

The state-only signatures refine, but do not rewrite, the joint comparison:

- state-only pass after joint miss is consistent with multitask/objective competition, but the
  conditional trigger-row comparison does not causally identify it;
- state-only miss despite setup-positive-control pass leaves generalization, transition/readout
  architecture, and optimization unresolved;
- a failed setup overfit/readout control invalidates the arm rather than proving a bottleneck.

## What is actually paired

For a given seed, all reached arms share exact common state-module tensor values, generated row order,
1,500-step schedule, 16-microbatch accumulation, and adaptation-dropout seeds and probe masks. A
common custom hook backend gives LoRA and full rank the same target discovery, call ordering, and
enable/disable seam. Common and adaptation gradient groups are clipped independently.

The parent used PEFT LoRA, so the common hook is not accepted on algebra alone. G0 copies identical
deterministic A/B/base/input tensors into the actual custom hook and a pinned PEFT
`lora.layer.Linear` reference at alpha/r = 2. FP32/dropout-off/autocast-off output and A/B-gradient
parity uses `atol=1e-6`, `rtol=1e-5`; live-like bf16-autocast/dropout-0.05 uses `atol=2e-3`,
`rtol=1e-2`. The stochastic forwards reset the same device RNG seed immediately before each path so
they consume the corresponding native-dropout mask position, and the custom path receipts that
realized mask and its one-call/one-cycle schedule. This is a small reference module, not a second Qwen
model; live target discovery/call checks remain separate.

The factorization still changes parameter count, FLOPs, and optimizer geometry. Say
`DIRECT_FULLSHAPE_RECIPE_RESCUE` or “practical direct-full-shape recipe relief,” not “mathematical
rank proven causal.” Preserve adapter FLOPs and wall time as diagnostics; do not call total compute
matched.

## What the state metric means

The heads predict the complete `(node, phase, checksum)` state before the later query. Joint
correctness requires all three components. Report terminal-state and trajectory-step metrics
separately, by semantic depth and training seed. A pooled trajectory number is dangerous because
shallow and adapter-independent steps can hide deep failure.

The trigger uses fresh validation, depth-extrapolation, and joint-shift splits. If it opens full rank,
three counterpart splits stay sealed until both capacities' fixed-final joint checkpoint identities,
the LoRA state-only controls, and every intact/disabled trigger evaluation are immutable and bound by
the dedicated Stage-B gate: `contrast_validation` has 768 rows at depths 2–4 (256/depth, seed 73307),
while `contrast_depth` and `contrast_joint` each have 1,024 rows at depths 5–12 (128/depth, seeds
73305 and 73306). The evaluator still launches only six capacity×seed jobs because every job scores
all three splits in both modes. If direct full shape already misses a trigger cell, the gate leaves
them sealed and opens the state-only control instead. Otherwise, cross-capacity rescue is judged only
on those post-trigger rows; the trigger rows may describe absolute formation but cannot provide a
selection-safe rescue contrast.

The sealed rows also test whether the LoRA miss itself replicates across every required trigger
domain. If LoRA passes every required sealed seed×depth cell across all three splits, the terminal
result is `LORA_TRIGGER_MISS_NOT_REPLICATED_ON_SEALED_CONTRAST`; no direct-recipe rescue or hard
rank-limit wording is permitted even if full rank scores higher.

If LoRA does not fully pass and full rank misses any absolute trigger or sealed cell, the registered
state-only branch still takes priority. With full rank passing, rescue requires category-matched LoRA
failure replication: trigger `trained`, `depth`, and `joint` failures must fail again in
`contrast_validation`, `contrast_depth`, and `contrast_joint`, respectively. Additional sealed
failures are allowed, but they cannot substitute for a trigger-failed category that passes its
counterpart. That mismatch terminates as
`LORA_TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED_ON_SEALED_CONTRAST`.

Do not reuse a generic “validate all rows” helper before authorization: preauthorization checks of
sealed files stop at manifest identity, compressed SHA-256/size, and gzip header. The authorized
evaluator records its receipt/checkpoint/cell identity in the access ledger before decompression.

If that evaluator is interrupted, preserve the incomplete canonical output with the registered
content-addressed archive helper. A same-cell retry requires exactly one newly tracked and
content-validated `FAILED_ATTEMPT_ARCHIVED` receipt, the identical checkpoint and canonical path, and
a newly empty output directory. The runner appends the new archive lineage to the existing event and
recomputes both event and ledger identities. It atomically replaces the ledger while holding a
separate stable lock inode and fsyncs both temporary file and parent directory before decompression.
Initial access rejects an archive that predates its event. An old receipt cannot license another
retry, multiple new receipts fail closed, and an evaluation with completed `summary.json` cannot be
archived as failed.

Adaptation-disabled evaluation asks whether the trained backend is needed for the readable state.
It leaves the shared initializer, step signal, scalars, and heads intact. It is not the parent's
Carry edge cut and says nothing about whether the coda uses the state.

`ADAPTATION_DISABLED_REVERSAL` is the distinct case where intact misses the required formation matrix
but disabled passes it. Branching remains based on intact, so the reversal cannot become a LoRA pass,
adaptation-required result, or direct-recipe rescue; it is evidence that the trained adaptation
interferes with the readable state at inference. Report it for every reached trigger arm. The sealed
joint analysis reports LoRA and full-rank adaptation statuses separately, so a sealed LoRA reversal
must not disappear behind the cross-capacity terminal label.

## Scope discipline

This experiment deliberately has no Bag arm, answer-gain verdict, query-stratum sign gate, state-edge
cut, donor swap, text comparator, or sample-more stage. Adding one would recreate the mixed verdict
that made the previous capacity conclusion ambiguous. If state forms, a fresh utilization/causal
experiment may ask whether it is used. A readability result alone is not a capability claim.

## Code and artifact map

- `configs/default.yaml`: frozen model, data, objective, pairing, schedule, and gates.
- `src/substrate.py`: fresh procedural worlds and exact state trajectories.
- `src/data_pipeline.py`: split generation, deduplication, and canonical hashes.
- `src/state_loop_model.py`: common Carry wrapper and custom adaptation-hook backend.
- `src/gpu_runner.py`: initialization bundles, mechanics/positive controls, training, fixed-final
  evaluation, checkpoints, and lineage verification.
- `src/analysis.py`: completeness, per-seed/per-depth state gates, paired contrasts, and branch
  authorization.
- `scripts/run.py`: phase-gated CLI.
- `reports/preregistration.md`: authoritative scientific contract.
- `docs/gpu_runbook.md`: operational order.
- `reports/artifact_manifest.yaml`: tracked, external, and omitted artifacts.

## Allowed pre-result fixes

Mechanical repairs may correct an API import, target-name unwrapping, dtype/autocast incompatibility,
path handling, receipt validation, or memory-preserving checkpoint implementation. Every repair needs
a test and a log entry. It must preserve model/revision, loop boundaries, targets, shared tensor
identity, dropout schedule, objectives, seeds, rows, steps, thresholds, fixed-final rule, and branch
order. Regenerate all downstream artifacts whose source/config identity changed.

Changing any scientific item above, relaxing a positive control, selecting a checkpoint, replacing a
seed, reducing full-rank targets, changing rank, switching backend, or adding a utilization gate
requires a new preregistered experiment. OOM or incomplete evidence is not a scientific negative.

## Current state

The directory is in progress and contains no result. Read the adversarial design review and inspect
the canonical `reports/design_receipt.json` before the first model load. Then execute only the next
licensed stage from `docs/gpu_runbook.md`; do not launch a monolithic pipeline or prequeue conditional
dense runs.
