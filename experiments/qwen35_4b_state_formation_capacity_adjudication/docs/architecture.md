# Architecture and Pairing Contract

## Scope

This experiment reuses the parent's continuous Carry computation and changes only the registered
extra-call adaptation recipe. It asks whether a readable recurrent state forms. It does not compare
Carry with Bag and does not infer causal use or answer capability from readability.

Let `P`, `R`, and `C` denote Qwen text layers `[0,12)`, `[12,20)`, and `[20,32)`. The rendered prompt
contains eight state slots before `Query:`. Causal masking prevents those slots and their state heads
from seeing the query kind, choices, or answer.

## Shared recurrent computation

The untouched first pass is:

```text
h_P = P(embed(prompt))
h_1 = R(h_P)                         # adaptation disabled
m   = reset_nonstate_memory(h_1)
s_1 = gather_state_slots(h_1)
```

For `K>1`, a shared state initializer maps `s_1` into the recurrent coordinate system. Every extra
call receives frozen first-pass non-state memory and the previously carried state:

```text
u_t = s_(t-1) + projected_sinusoid(t)
c_t = gather_state_slots(R_adapted(scatter(m, u_t)))
s_t = s_(t-1) + sigmoid(damping) * (c_t - s_(t-1))
```

Only the eight state positions survive between calls. The coda runs once from the registered final
state. K=1 bypasses initialization, aggregation, and every adaptation hook, and must match the pinned
base CausalLM answer-position logits with maximum absolute error at most `1e-5`.

Training uses K=4 on semantic depths 1–4. State supervision is dense: one shared head predicts node,
phase, and checksum at each active state. Evaluation uses K equal to semantic depth. Terminal-state
joint correctness and trajectory-step correctness are stored separately; no component or shallow
step may be substituted for the registered joint gate.

## Common adaptation-hook backend

The implementation discovers the same ordered 62 `torch.nn.Linear` targets in R for both capacities.
A single hook controller owns target discovery, enable/disable context, call ordering, dropout, audit
receipts, and removal. Capacity changes only the learned delta applied after dropout:

```text
LoRA:      y = W_base x + (alpha / rank) * B(A(dropout(x)))
full rank: y = W_base x + 2 * DeltaW(dropout(x))
```

The frozen LoRA values are rank 32, alpha 64, and 16,232,448 adaptation parameters. LoRA A tensors
use an explicit capacity-specific construction seed and B tensors begin exactly zero. Full rank uses
62 FP32 tensors shaped like their targets, 892,272,640 parameters total, and exact zero
initialization. Both use adaptation dropout `0.05`; neither adaptation is active in P, the first R
application, C, or K=1.

Because the parent used PEFT LoRA, formula compatibility is a setup gate rather than an assumption.
Instantiate one pinned `peft.tuners.lora.layer.Linear` reference around a deterministic bias-free
11-to-7 target and the actual custom `AdaptationBank` hook around its copied twin. Copy identical
base/input/A/B tensors into both, use probe rank 5 and alpha 10 (alpha/r = 2), and require output plus
A/B-gradient parity in two regimes:

- FP32, dropout off, autocast off: `atol=1e-6`, `rtol=1e-5`;
- live-like bf16 with CUDA bf16 autocast and dropout `0.05`: `atol=2e-3`, `rtol=1e-2`.

The output dtype and output/gradient shapes must agree, and gradient comparison is performed in
FP32. For the stochastic probe, reset the identical device RNG seed immediately before each forward;
PEFT and the custom path each consume the corresponding first native-dropout mask, while the custom
hook receipts its realized mask and one-call/one-cycle schedule. Equal nominal dropout without this
same-seed position and output/gradient parity is not a pass. This is a small tensor-level reference
test, not a second Qwen model and not a result arm. Exact 62-target discovery and the live recurrent
call schedule are checked separately on the real model.

The two parameterizations intrinsically differ in parameter count, adaptation FLOPs, and optimizer
geometry. A positive is therefore a practical direct-recipe/parameterization result, not a theorem
about matrix rank in isolation.

## Bit-identical shared state initialization

Before model construction, create one external common-state bundle for each seed 7411–7413. It
contains every tensor shared by capacities and objectives:

- state initializer;
- sinusoidal step projection;
- damping and last-versus-mean aggregation logits; and
- node, phase, and checksum heads.

Bundle construction occurs inside an isolated CPU RNG fork. Tensor keys are canonical and its tracked
receipt binds names, shapes, dtypes, per-tensor hashes, a tensor-value digest, file SHA-256, seed,
model/config/source/lock identities, and bundle format version.

Every arm constructs its capacity backend under a separate RNG namespace, then strictly loads the
same seed bundle. It must recompute the digest and prove `torch.equal` for every common tensor.
Construction may deliberately consume unrelated RNG before the second build; equality must still
hold. A mismatch is a setup failure, never evidence.

The hardware pairing is also exact: one NVIDIA RTX 6000 Ada Generation device at compute capability
8.9. Receipts bind its UUID, stable device properties, CUDA/runtime, installed environment lock,
tokenizer, target order, and non-adaptation dropout inventory. Free-memory snapshots are diagnostic
and excluded from cross-arm equality; every stable field must match G0, training, evaluation, and all
reached capacity/objective arms.

## Matched adaptation dropout

All adaptation targets use the same custom dropout primitive. Immediately before each training
microbatch, derive a seed from the canonical tuple:

```text
adaptation-dropout-v1|model-seed|global-microbatch-index|row-id|K
```

Capacity and objective are intentionally absent. The controller resets the adaptation-dropout RNG at
the microbatch boundary and records the ordered target/call/shape schedule. It hashes realized masks
at preregistered first, midpoint, and last probe microbatches; the schedule and probe-mask receipts
must match across every reached capacity/objective arm for a model seed. Model construction,
validation, and coda computation therefore cannot shift the adaptation masks.

G0 inventories every other dropout-like module in the live wrapper and requires it to be absent,
disabled, or configured with probability zero. No stochastic operation may consume the controlled
CUDA stream between the microbatch reset and the registered adaptation calls without appearing in
the receipt. Matching adaptation seeds while an unreceipted dropout remains active is a hard fail.

The live mechanics gate runs the real pinned model through both backends, checks the expected call
count and ordering—186 adaptation calls at K=4 and 682 at K=12—and requires identical probe masks for
corresponding calls. Hook controllers must be removed explicitly before another backend is built;
stale hooks are a hard fail.

After the two-step state-only gradient-onset check, G0 also runs a real K=4 joint backward and
optimizer step through the coda. It requires finite answer/objective loss, exact adaptation calls,
nonzero finite gradients in every trainable group including the aggregation scalar, no base-model
gradient, separate clip receipts, separately read adaptation/common-state learning rates, elapsed
time, and peak memory. Result analysis independently recomputes the registered learning-rate schedule
for each group at every exact step. Thus the primary joint graph and its full-rank feasibility are
tested before a 1,500-step run.

## Objectives

The joint objective preserves the original competition:

```text
joint = 1.0 * answer_loss + 0.5 * state_loss + 0.05 * fixed_point_loss
```

The conditional control is:

```text
state_only = 0.5 * state_loss + 0.05 * fixed_point_loss
```

The state-only graph omits answer loss entirely; it is not `0 * answer_loss`. Both objectives execute
the same recurrent forward and use identical state targets, fixed-point rule, rows, schedule, and
dropout stream. The state-only arm provides a descriptive pattern consistent with objective
competition; because it is conditional and reuses trigger rows, it does not identify that mechanism.
It is not the first or only capacity comparison.

## Optimizer isolation

Both capacities use the frozen AdamW schedule, but adaptation and common state parameters are separate
optimizer/clipping groups. Autocast runs without a GradScaler, so there is no unscale operation.
After the registered 16 microbatches and before each step, the runner computes and records the
adaptation norm and clips that group to its frozen threshold, then independently computes/records and
clips the common-state group, and finally steps once. The ordering is fixed even though the groups are
disjoint.

This prevents the dense adaptation group's much larger norm from changing the clipping multiplier on
the shared state modules. Receipts bind both preclip norms, both applied scales, finite gradients, and
the absence of base-model gradients. The state-only graph has one exact optimizer-state exemption:
the answer-only aggregation scalar receives no gradient or Adam moments. G0's live joint probe must
still prove that scalar is reachable; no adaptation or other common-state tensor may be exempt.

## Adaptation-disabled evaluation

Every fixed-final checkpoint is evaluated twice on identical rows:

- `intact`: its trained adaptation hooks are enabled on extra R calls;
- `adaptation_disabled`: hooks are disabled, while the checkpoint's trained initializer, step
  projection, scalars, and state heads remain loaded.

This is a within-checkpoint causal control for whether readability depends on the adaptation backend.
It is not an edge cut of the carried state and cannot establish downstream causal use. A direct-recipe
rescue label requires the registered intact-minus-disabled effect in addition to absolute state
formation.

The adaptation diagnostic has a distinct `ADAPTATION_DISABLED_REVERSAL` state when intact misses the
required formation matrix but disabled passes it. This records inference-time interference from the
trained adaptation rather than collapsing the case into generic uncertainty. Scientific branching
still follows intact formation, and a reversal cannot satisfy adaptation dependence or rescue. The
LoRA and full-rank trigger analyses report this status; after sealed evaluation, the joint analyzer
reports it separately for both capacities, including a LoRA reversal on the sealed matrix.

## Checkpoint and receipt contract

Only step 1500 is eligible. Each checkpoint metadata file binds capacity, objective, seed, exact
parameter manifest, initialization-bundle path/hash/value digest, data manifest, training-order
digest, dropout schedule/probe digests, groupwise optimizer receipt, model/config/source/lock
identities, branch authorization, final metrics file hash/count, and all payload hashes.

Loaders reopen every upstream receipt and verify its file SHA-256 and canonical receipt identity.
Checking copied status fields is insufficient. Interrupted result runs are non-resumable: preserve the
attempt and restart step zero in a fresh directory with the same registered seed.

The canonical design-boundary receipt binds the idea intake, preregistration, design review, this
architecture, GPU runbook, research handoff, and default config. Those scientific files plus the
separately registered source/tests/implementation review and training lock must be tracked and clean
at the recorded HEAD when the boundary is created. The receipt freezes only scientific design;
every downstream artifact separately binds the current source/test and lock digests. A mechanical
repair therefore preserves the design receipt but invalidates all downstream artifacts made under
the prior implementation digest. The mutable README and result report are intentionally unbound.

Contrast access has an additional one-way firewall. The Stage-B seal analyzer verifies and binds all
nine reached fixed-final checkpoint identities, all intact/disabled trigger-evaluation identities,
the LoRA-state-only analysis, and their common lineage. Without reading contrast payloads, it checks
the access ledger and requires every registered contrast-evaluation/output path to be absent before
adjudicating the full-rank trigger. A trigger miss emits `FULLRANK_STATE_ONLY_REQUIRED` and keeps the
contrasts closed. Only a complete trigger pass emits `STAGE_B_CONTRAST_AUTHORIZED`. Contrast
evaluators and the final full-rank analyzer accept that exact receipt and reject the earlier
LoRA-miss receipt.

The sealed matrix has three data splits: `contrast_validation` (seed 73307, 768 rows, depths 2–4 at
256/depth), `contrast_depth` (seed 73305, 1,024 rows, depths 5–12 at 128/depth), and
`contrast_joint` (seed 73306, 1,024 rows, depths 5–12 at 128/depth). Thus every required trigger
domain has a fresh selection-safe counterpart. Authorization still opens exactly six capacity×seed
evaluation jobs, because each job scores all three splits in both intact and disabled modes.

The post-contrast analyzer tests LoRA intact formation across the complete three-split matrix before
any full-rank terminal label. A LoRA pass emits
`LORA_TRIGGER_MISS_NOT_REPLICATED_ON_SEALED_CONTRAST` and stops regardless of full-rank score. If LoRA
does not fully pass and full rank misses any trigger or sealed absolute cell, state-only remains
mandatory. Only after full rank passes does the analyzer require every LoRA trigger-failed category to
fail again in its corresponding sealed domain: `trained` maps to `contrast_validation`, `depth` to
`contrast_depth`, and `joint` to `contrast_joint`. Additional sealed failures are allowed. A missing
replication emits `LORA_TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED_ON_SEALED_CONTRAST` and stops;
otherwise the analyzer may consider direct-recipe rescue or contrast uncertainty.

Preauthorization data validation is tiered: sealed split files are checked only by manifest identity,
compressed SHA-256/size, and gzip header. No generic validator may decompress or canonical-reopen
them. An authorized evaluator appends its Stage-B receipt, fixed checkpoint/cell identity, and split
to the access ledger before first decompression; later authorized entries must share that receipt and
frozen cell set. Ledger mutation locks a separate stable inode, atomically replaces the ledger after
fsyncing the temporary file, and fsyncs the parent directory. An interrupted same-cell replay also
requires the identical checkpoint and canonical output path plus exactly one newly tracked,
content-validated `FAILED_ATTEMPT_ARCHIVED` receipt. The archived tree must remain byte-identical,
incomplete, and source/design/lock-bound; the evaluator appends its lineage to the existing event,
recomputes event and ledger identities, and durably replaces the ledger before decompression. Initial
access rejects an archive that predates its event. Already-bound or multiple new archives cannot
license a replay, and an evaluation with a completed `summary.json` cannot be archived as failed. A
missing cell, stale identity, unknown prior-open entry, or premature contrast artifact yields
`CONTRAST_FIREWALL_NOT_READY`, never a scientific result. An
actual or unexplained premature-open event cannot be repaired by deleting the ledger or regenerating
the frozen split; it burns those contrast rows and requires a fresh successor for a rescue claim.
