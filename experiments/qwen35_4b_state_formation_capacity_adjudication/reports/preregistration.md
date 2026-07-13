# Preregistration: State-Formation Capacity Adjudication

This contract is frozen before any model-bearing call. CPU-only implementation tests, adversarial
review, and repository validation may precede the design receipt. A live setup or positive-control
result may reveal a mechanical defect, but it may not change a scientific threshold, split, seed,
objective, branch, or terminal rule inside this experiment.

The canonical `reports/design_receipt.json` binds the idea intake, preregistration, adversarial design
review, architecture, GPU runbook, research handoff, and default config: the scientific design only.
At freeze time those files and the separately registered source/tests/implementation review and
training lock must be tracked and clean at HEAD. The receipt records source/lock/HEAD provenance,
while every downstream artifact independently binds the then-current source/test and lock digests.
This permits a setup-only mechanical repair without rewriting scientific design, but invalidates and
requires regeneration of every downstream artifact under the old implementation digest. Mutable
post-result README/report prose is deliberately outside the design boundary.

## 1. Fixed question and estimand

Does rank-32 adaptation on the repeated Qwen block prevent the formation of a jointly sufficient
`(node, phase, checksum)` state under the original joint answer-plus-state objective? If a valid
three-seed LoRA joint result misses, does a direct full-shape recipe rescue state formation under
bit-identical shared initialization, matched row order and matched adaptation dropout?

The primary state event is exact terminal joint correctness at K equal to semantic depth. All three
components must be correct for the task to score one. The capacity estimand is conditional and
sequential: direct full shape is observed only after a complete LoRA joint formation miss.

A positive rescue identifies a **practical adaptation recipe/parameterization**. Full shape changes
parameter count, adaptation FLOPs, and optimizer geometry as consequences of removing factorization.
Even a positive result is not a mathematical identification of rank alone.

## 2. Model, backend, and recurrence

- Only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Transformers 5.13.0, bf16 frozen base, SDPA, cache-free batch-one full-sequence forward.
- One exact `NVIDIA RTX 6000 Ada Generation` device (compute capability 8.9); its UUID, CUDA/runtime,
  installed-lock receipt, and stable setup identity must match across every reached setup, training,
  evaluation, and capacity/objective comparison.
- Exactly 32 text layers with the live repeating `linear, linear, linear, full` motif.
- Prelude layers 0–11, recurrent block R at 12–19, coda at 20–31.
- Eight state slots occur before `Query:`. State heads cannot observe query kind, choices, or answer.
- Only the carried-state graph is used. There is no Bag arm.
- Non-state memory resets to the untouched first-R representation on every extra call; only state
  slots cross calls.
- Training K is 4. Evaluation K equals each row's semantic depth.
- K=1 bypasses state initialization/aggregation and all adaptation. Its answer-position logits must
  match the standard base CausalLM with maximum absolute error at most `1e-5`.
- Every result-bearing arm uses this Transformers path. vLLM is absent because the experiment needs
  training, recurrent internals, and state-head evaluation.

## 3. Adaptation recipes

One custom hook backend discovers and orders the same 62 `nn.Linear` targets in R, owns the explicit
extra-call enable context, and records call schedules. The base weights stay frozen.

### Rank-32 LoRA

- rank 32, alpha 64, scale 2;
- 16,232,448 adaptation parameters over the frozen 62-target manifest;
- A tensors initialized under an explicit capacity-specific RNG namespace;
- B tensors initialized exactly zero; and
- adaptation dropout 0.05 before the low-rank map.

### Direct full shape

- 62 FP32 `DeltaW` tensors shaped exactly like the target weights;
- exactly 892,272,640 parameters;
- scale 2;
- every delta initialized exactly zero; and
- the same adaptation dropout 0.05 before `DeltaW`.

Both recipes are active only on R applications 2..K. The hook/call/dropout path is shared rather than
PEFT for one arm and a separate hook for the other. At initialization, enabled and disabled K=4
logits must be bit-identical because both learned output deltas are zero. K=1 must record zero
adaptation calls.

Before the custom LoRA can answer the parent PEFT-LoRA concern, it must pass a pinned
`peft.tuners.lora.layer.Linear` tensor-reference gate using the actual custom `AdaptationBank` hook.
The small reference is one bias-free 11-to-7 linear target with rank 5, alpha 10, and therefore the
same alpha/r = 2 scale as the result LoRA. Identical deterministic base/input/A/B tensors are copied
into both implementations. Two output-plus-A/B-gradient regimes must pass:

- exact FP32, dropout disabled, autocast off: matching output dtype and output/gradient shapes, with
  outputs and FP32-compared gradients allclose at `atol=1e-6, rtol=1e-5`; and
- live-like bf16, dropout `0.05`, bf16 autocast on CUDA: the same checks at
  `atol=2e-3, rtol=1e-2`.

For the stochastic regime, reset the same device RNG seed immediately before the PEFT forward and
again before the custom forward. Each path has exactly one corresponding dropout consumer, and the
custom hook receipts its realized native-dropout mask/call cycle. Output and gradient parity under
that reset is the matched-mask gate; equal dropout probabilities alone are insufficient. This does
not build or mutate a second Qwen model. The live Qwen G0 separately checks the exact 62 target
names/count and recurrent call schedule.

## 4. Common initialization, row order, dropout, and optimizer controls

For each result seed 7411, 7412, and 7413, construct one serialized common-state bundle before any
capacity model. It contains the state initializer, step projection, damping and aggregation logits,
and node/phase/checksum heads. Construction occurs in an isolated CPU RNG fork. The receipt binds
canonical names/shapes/dtypes, all tensor hashes, a tensor-value digest, file hash, seed, and frozen
model/config/source/environment-lock identities.

Every reached capacity/objective build strictly loads that same per-seed bundle after its
capacity-specific construction. Exact `torch.equal` values and identical tensor digest are required.
G0 deliberately consumes unrelated construction RNG between builds and must still reproduce the
common digest.

Training order is a deterministic function of model seed and global microbatch index. The realized
ordered row digest must match across every reached capacity/objective arm for that seed.

Adaptation dropout uses one pinned custom primitive. Before each microbatch, its RNG seed is derived
from:

```text
adaptation-dropout-v1|model-seed|global-microbatch-index|row-id|K
```

Capacity and objective are excluded. The runner records every ordered target/call/shape schedule and
hashes realized masks at the first, midpoint, and final registered microbatches. Corresponding
schedule and probe-mask hashes must be identical across reached arms. A copied seed value without
realized-mask equality is insufficient.

The live wrapper must also receipt every non-adaptation dropout-like module and require it to be
absent, disabled, or probability zero. An untracked stochastic consumer between the microbatch RNG
reset and adaptation calls invalidates matching.

AdamW uses the same nominal hyperparameters, but common state tensors and adaptation tensors are
separate clipping groups. Autocast runs without a GradScaler; each group is normed, clipped to `1.0`,
and receipted independently before one optimizer step. This prevents the 892M-parameter group's norm
from changing the clipping multiplier applied to common state tensors. An every-step payload binds
both preclip norms, both applied scales, finiteness, separately read adaptation and common-state
learning rates, and the absence of trainable base parameters. Analysis independently recomputes the
registered schedule for each group at every exact step; agreement between the two logged rates cannot
hide a shared scheduling error. The fixed-final receipt also binds complete finite FP32 Adam moments.
State-only has one registered missing-moment exemption for the answer-only aggregation scalar whose
graph is omitted; the live joint G0 must nevertheless prove that scalar has a nonzero finite gradient.

## 5. Fresh procedural data

The generator and transition semantics match the parent substrate, but every row is fresh. No parent
evaluation row and no benchmark content is read or reused.

| Split | Seed | Rows | Depths | Role |
| --- | ---: | ---: | --- | --- |
| `train` | 73301 | 12,000 | 1–4 | result training |
| `validation` | 73302 | 1,024 | 1–4, 256/depth | trained-depth trigger/control |
| `depth_extrapolation` | 73303 | 1,024 | 5–12, 128/depth | LoRA branch trigger, in-family surfaces |
| `joint_holdout` | 73304 | 1,024 | 5–12, 128/depth | LoRA branch trigger, held-out family plus surface |
| `contrast_depth` | 73305 | 1,024 | 5–12, 128/depth | sealed post-trigger capacity contrast |
| `contrast_joint` | 73306 | 1,024 | 5–12, 128/depth | sealed post-trigger joint-shift contrast |
| `contrast_validation` | 73307 | 768 | 2–4, 256/depth | sealed post-trigger trained-depth contrast |

Training uses families `phase_branch` and `checksum_branch`, templates `ledger` and `prose`, and
semantic depths 1–4. Joint-shift splits use held-out `braided_branch` with held-out `compact` surface.
Depth splits retain training families/surfaces. `contrast_validation` independently regenerates the
required trained-depth 2–4 cells with the validation-domain family/surface balance; it omits diagnostic
depth 1 because no adaptation is active there. Cells are balanced before generation rather than
accepted from a rejection-skewed mixture.

Generation rejects repeated complete states and a terminal queried value seen at an earlier step.
Structural fingerprints ignore skin labels and must have zero overlap across all splits. Canonical
row manifests bind counts, order, fields, compressed payload hashes, generator source, and zero
benchmark reads.

The three contrast splits remain unopened to every model/evaluator/analyzer until the LoRA trigger
analysis is sealed. If LoRA passes, they are not evaluated. A LoRA-miss receipt authorizes Stage-B
training, but does not authorize contrast access. After a LoRA miss, a separate Stage-B seal receipt
must bind all three LoRA-joint, three LoRA-state-only, and three full-rank-joint fixed-final
checkpoints plus every intact/disabled trigger evaluation and the LoRA-state-only analysis. It also
verifies common lineage and proves, from the access ledger and absent evaluation/output paths, that
none of the three contrast splits has previously been opened for scoring. The authorization phase may
inspect frozen manifest identities but may not read contrast rows or labels. Before authorization, validation
of a sealed split is limited to its manifest identity,
compressed-file SHA-256/size, and gzip header; no model stage, evaluator, analyzer, or generic manifest
validator may decompress or canonical-reopen its payload. If any full-rank joint trigger cell misses,
the Stage-B seal instead emits `FULLRANK_STATE_ONLY_REQUIRED` and prohibits contrast access because
rescue is already impossible. Only a trigger pass can emit the authorization that licenses either
capacity's contrast evaluator. The authorized evaluator writes an identity-bound access-ledger entry
before first decompression; all later entries must bind the same Stage-B receipt and frozen cell set.
Ledger mutation holds a separate stable lock inode while atomically replacing the ledger, after
fsyncing the temporary file, and then fsyncs the parent directory.
An interrupted same-cell evaluation may replay only at the identical checkpoint and canonical output
path after the incomplete output is content-addressed by the registered archive helper. Every replay
requires exactly one newly discovered, tracked, content-validated `FAILED_ATTEMPT_ARCHIVED` receipt
for that canonical cell; the evaluator appends that archive lineage to the existing ledger event and
recomputes both event and ledger identities before the atomic durable rewrite and decompression.
Initial access rejects any otherwise valid failed-attempt archive that predates its ledger event;
previously bound archives cannot be reused to justify another replay, multiple new archives fail
closed, and an evaluation containing its completed `summary.json` cannot be archived as failed.
This prevents the conditional full-rank contrast from reusing rows that selected or shaped the branch
and avoids spending sealed rows when they cannot change the decision.

## 6. Objectives and fixed-final training

The joint objective is:

```text
1.0 * answer_loss + 0.5 * state_loss + 0.05 * fixed_point_loss
```

The conditional state-only control is:

```text
0.5 * state_loss + 0.05 * fixed_point_loss
```

State loss is the mean node/phase/checksum cross-entropy over the dense active trajectory with shared
heads. State-only omits answer loss from the graph; it does not compute `0 * answer_loss`. Answer
accuracy is reported for joint arms but is never a formation or branch gate.

Every result arm trains exactly 1,500 optimizer steps at batch size one with 16 microbatches per
step, learning rate `2e-4`, weight decay `0.01`, 5% warmup, and no early stopping. Only step 1500 is
saved/eligible. Training losses may be logged, but no validation result selects a checkpoint,
objective, seed, or branch. Interrupted runs are non-resumable and restart at step zero in a new
attempt directory. Seeds cannot be replaced.

## 7. Setup-only validity gates

### G0 mechanics

Before a capacity trains, require the setup gate for each registered model seed. The same
capacity/seed G0 receipt is reused across joint and conditional state-only objectives because it
tests the shared state path and adaptation backend rather than answer-loss weighting. Require:

- exact model/revision/layer/target/count contracts;
- tensor-level custom-hook LoRA output/gradient parity to the pinned PEFT `Linear` reference in both
  exact FP32/dropout-off and live-like bf16-autocast/dropout-0.05 regimes, using the frozen tolerances
  and same-seed dropout-mask protocol in Section 3;
- exact common-bundle equality across deliberately RNG-shifted builds;
- exact zero learned output at initialization and K=1 parity;
- exactly 186 K=4 and 682 K=12 adaptation calls with matched dropout probe masks;
- no active unreceipted non-adaptation dropout or stochastic RNG consumer;
- finite K=12 forward and K=4 backward;
- nonzero finite gradients in adaptation, initializer, step projection, and state heads after the
  registered two-step gradient probe, with no base gradient;
- a live K=4 joint answer-plus-state backward/optimizer probe with finite answer loss, exact call and
  dropout schedule, nonzero finite gradients in every preceding group plus the aggregation scalar,
  no base gradient, separate clipping, elapsed time, and peak memory;
- independently clipped common/adaptation gradient receipts;
- for full rank, finite shape-matched FP32 Adam moments for all 62 deltas, memory peak/headroom, and a
  destructive checkpoint round trip with exact tensor restoration; and
- complete source/config/lock/data/init identities.

The K=4/K=1 row is read only from `train`. K=12 is a separately generated setup-only row from seed
73992, checked for zero structural overlap with every result split; G0 never decompresses trigger or
sealed contrast rows. After G0, at least 4 GiB device memory must remain free.

### Positive-control overfit/readout

Using 48 freshly generated setup rows from control seed 73991, with exactly two rows in every cell of
depths 2/3/4 × query kinds node/checksum × the two training families × the two training templates,
run a fixed 256-update state-only overfit for each reached capacity and model seed, loading that
result seed's registered common-state initialization bundle.
At its fixed final update, intact terminal joint accuracy must be at least 0.95. The analyzer is also
tested against an oracle-perfect synthetic row table and must return at least 0.99 with the expected
task/depth counts. These setup rows are not a seventh result split: their generation receipt is
separate, control checkpoints never initialize result runs, and control rows never enter result
metrics.

This capacity/seed control qualifies the shared state-supervision and readout path for either
objective. It is not rerun with answer loss and cannot be cited as evidence that the joint objective
works.

Failure of G0 or either positive-control component emits `SETUP_CONTROL_FAILED`. Correct mechanics or
receipt code and rerun the same frozen control. It does not authorize a capacity branch and is never a
scientific terminal result.

## 8. Cell metrics and formation statuses

For each task at K=semantic depth, define terminal joint correctness as one iff the final active state
has correct node, phase, and checksum. For each seed×depth×split cell, accuracy is the mean of this
indicator. The frozen absolute gate is `>= 0.40` in **every** required cell; pooling across depths or
seeds cannot pass a failed cell.

Depth 1 is reported but excluded from an adaptation-capacity pass because K=1 bypasses adaptation and
the state initializer. Trained-depth qualification therefore requires validation depths 2, 3, and 4
at or above 0.40 for every seed. Formation then requires every depth 5–12 cell in both
`depth_extrapolation` and `joint_holdout` at or above 0.40 for every seed.

Trajectory-step joint accuracy, node/phase/checksum components, depth 1, answer accuracy, losses,
state-change norms, time, memory, and FLOPs are diagnostics. None can substitute for terminal joint
correctness.

Each capacity/objective arm emits exactly one ordered status:

1. `EVIDENCE_INCOMPLETE`
2. `TRAINED_DEPTH_MISS`
3. `TRAINED_PASS_DEPTH_EXTRAPOLATION_MISS`
4. `TRAINED_AND_DEPTH_PASS_JOINT_SHIFT_MISS`
5. `STATE_FORMATION_PASS`

State-only interpretation is valid only after its trained-depth cells pass. Its deep and joint-shift
status remains explicit; a shallow pass is never called general state formation.

## 9. Adaptation-dependence and capacity contrasts

Every checkpoint is evaluated intact and with adaptation disabled while retaining its trained common
state modules and heads. This contrast never changes the arm's absolute formation status.

For each complete trigger or sealed evaluation bundle, adaptation status is:

- `ADAPTATION_REQUIRED` only when intact passes the absolute formation cells, disabled does not,
  intact-minus-disabled is positive in every seed, no depth has a negative point estimate, and the
  10,000-resample crossed task×seed 95% lower bound is above zero separately on every split in the
  bundle—including `contrast_validation`, `contrast_depth`, and `contrast_joint` for sealed data;
- `ADAPTATION_NOT_REQUIRED_AT_INFERENCE` when disabled also passes every absolute cell;
- `ADAPTATION_DISABLED_REVERSAL` when intact misses at least one required formation cell but disabled
  passes every required cell. Branching and absolute capacity classification still use intact; the
  reversal is reported as adapter interference and cannot support `ADAPTATION_REQUIRED` or a
  direct-recipe rescue; or
- `ADAPTATION_CONTRAST_UNCERTAIN` otherwise.

A direct full-shape joint pass is a robust cross-capacity rescue only when, on all three sealed splits:

- every LoRA formation category that failed on the trigger matrix—`trained`, `depth`, and/or
  `joint`—fails again in its corresponding sealed domain (`contrast_validation`, `contrast_depth`,
  or `contrast_joint`). Additional LoRA sealed failures are allowed; success requires replication of
  the full trigger-failure category set, not merely some failure somewhere on fresh rows;
- full rank passes every absolute seed×depth cell;
- full-rank minus LoRA is positive in every seed;
- no depth has a negative paired point estimate; and
- the crossed task×seed 95% lower bound is above zero separately on each of the three splits.

It must also have `ADAPTATION_REQUIRED`. If LoRA instead passes every sealed absolute cell, emit
`LORA_TRIGGER_MISS_NOT_REPLICATED_ON_SEALED_CONTRAST`, prohibit a full-shape rescue claim, and stop:
fresh evidence shows LoRA can form the registered state even though its trigger result missed.
If LoRA still fails somewhere on sealed data but at least one trigger-failed category passes in its
corresponding sealed domain, then—provided full rank passes every absolute trigger and sealed
cell—emit
`LORA_TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED_ON_SEALED_CONTRAST` and prohibit rescue. The analysis
records trigger-failed categories, sealed-failed categories, and missing replications; additional
sealed failures do not repair a missing corresponding category. A full-rank absolute miss takes
priority over this rescue guard and mandates Stage C.
Otherwise the conditions above support
`DIRECT_FULLSHAPE_RECIPE_RESCUE`, not a mathematical rank claim. Full rank must pass the absolute
cell gates on its trigger and all three sealed splits before contrast uncertainty can be the only
issue.
If those absolute cells all pass but adaptation-dependence or full-rank-minus-LoRA contrast is not
robust, emit `DIRECT_FULLSHAPE_RECIPE_PASS_CONTRAST_UNCERTAIN`. Failure of any full-rank joint
absolute trigger cell, or any sealed cell after the higher-priority all-cell LoRA stop is excluded, is
a joint formation miss and requires the state-only branch.

Bootstrap resampling draws task IDs once and training seeds separately, using 10,000 deterministic
resamples and frozen analysis seed 75301. Reports distinguish unique tasks from model×task rows.

## 10. Sequential branch and terminal taxonomy

### Stage A: LoRA joint

Train and evaluate all three LoRA joint fixed finals on the trigger splits.

- Incomplete or invalid setup: `EVIDENCE_INVALID_REPAIR_REQUIRED`; no later arm is licensed.
- `STATE_FORMATION_PASS`: terminal `LORA_DOES_NOT_PREVENT_STATE_FORMATION`; every later arm is
  `PROHIBITED_BY_LORA_PASS`.
- Any complete miss status: `LORA_JOINT_MISS_CONTROLS_REQUIRED`. Its identity-bound receipt mandates
  LoRA state-only and full-rank joint for seeds 7411–7413.

### Stage B: LoRA state-only and direct full-shape joint

After the LoRA miss receipt, run all six Stage-B cells and complete every intact/disabled trigger
evaluation. The `stage_b_seal` analysis binds all nine reached joint/control checkpoints and trigger
evaluations while confirming the contrast firewall remains unspent. A bare
`LORA_JOINT_MISS_CONTROLS_REQUIRED` receipt can never open a contrast split.

- Missing/invalid cells: `BRANCH_EVIDENCE_INCOMPLETE`; repair only. A failed seal emits
  `CONTRAST_FIREWALL_NOT_READY`. Missing preaccess lineage may be repaired only while the access
  ledger is demonstrably empty. An actual or unexplained premature-open event permanently burns the
  frozen contrast rows; no rescue claim may use them, and replacement rows require a fresh successor.
- Full-rank joint failure in any trigger absolute cell: `FULLRANK_STATE_ONLY_REQUIRED`; prohibit
  contrast access and mandate Stage C directly.
- Full-rank joint trigger pass: `STAGE_B_CONTRAST_AUTHORIZED`; evaluate all three sealed splits for
  both joint capacities, then run the registered full-rank analysis. This is still exactly six
  capacity×seed evaluation jobs—each job emits intact and disabled rows for all three splits.
- Before its terminal checks, that post-contrast analysis reopens the Stage-B-bound full-rank-joint,
  LoRA-joint, and LoRA-state-only trigger evaluations, recomputes all three formation summaries, and
  requires the recomputed LoRA summaries to equal the values cached by Stage B.
- LoRA passes every sealed absolute cell: `LORA_TRIGGER_MISS_NOT_REPLICATED_ON_SEALED_CONTRAST`;
  prohibit a rescue/rank-limit claim and stop, regardless of the full-rank score.
- Otherwise, full-rank joint failure in any sealed absolute cell: `FULLRANK_STATE_ONLY_REQUIRED`,
  which mandates Stage C before applying the rescue guards below.
- With full rank passing every trigger and sealed absolute cell, LoRA fails at least one sealed cell
  but not every category that failed on trigger fails again in its corresponding sealed domain:
  `LORA_TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED_ON_SEALED_CONTRAST`; prohibit a rescue/rank-limit
  claim and stop. Extra failures in other sealed categories do not substitute for the missing
  category replication.
- Full-rank joint formation pass on trigger and sealed absolute cells plus all adaptation and
  cross-capacity contrast gates:
  `DIRECT_FULLSHAPE_RECIPE_RESCUE`.
- Full-rank joint formation pass on every trigger and sealed absolute cell without robust contrast:
  `DIRECT_FULLSHAPE_RECIPE_PASS_CONTRAST_UNCERTAIN`.

If LoRA state-only passes, the report must also say `LORA_CAN_FORM_STATE_STATE_ONLY` and describe the
pattern as consistent with a joint-objective interaction; it may not claim causal identification or
call the Stage B outcome a hard rank limit. These conditional controls reuse trigger rows selected by
the joint miss and are descriptive diagnostics, not selection-corrected cross-objective contrasts.
Every LoRA trigger analysis also reports its adaptation status. The post-contrast analysis separately
reports the LoRA and full-rank sealed adaptation statuses; a LoRA
`ADAPTATION_DISABLED_REVERSAL` on the complete three-split sealed matrix must remain visible even
when the terminal branch is determined by intact formation.

### Stage C: direct full-shape state-only

After an exact `FULLRANK_STATE_ONLY_REQUIRED` receipt from the Stage-B seal (trigger miss) or the
post-contrast full-rank analysis (sealed absolute miss), run all three fixed-final state-only cells.
Before classification, the terminal Stage-C analysis reopens the Stage-B-bound LoRA-state-only and
full-rank-joint trigger predecessors and requires their evaluation lineages to agree exactly with
Stage B. It requires the full-rank state-only runs to match both predecessors on shared
initialization, data, row order, dropout schedule and probes, while reusing the exact full-rank G0 and
positive-control receipts. Formation is recomputed from the current LoRA- and full-rank-state-only
trigger rows. With all setup controls valid, classify the state-only pair:

- both pass: `BOTH_CAPACITIES_FORM_STATE_WITHOUT_ANSWER` — consistent with a joint-objective
  interaction under the registered recipes, not causal identification;
- only full rank passes: `DIRECT_FULLSHAPE_RECIPE_STATE_ONLY_RESCUE` — practical direct-recipe
  support, rank not isolated;
- only LoRA passes: `FULLRANK_CONTROL_REVERSAL`; or
- neither passes: `FULLRANK_RELIEF_NOT_SUFFICIENT_REGISTERED_RECIPE_BOTTLENECK_UNRESOLVED` — the
  experiment cannot distinguish supervision/readout architecture from optimization failure.

Any setup/control/completeness failure supersedes those labels with repair-required status. An
unreached conditional arm is prohibited, not a zero and not a negative result.

## 11. Prohibited inference and repairs

This experiment cannot claim serial advantage, causal state use, answer improvement, capability
gain, deployment value, or a matched-compute sampling win. There is no Bag, edge-cut, swap, textual
state, vLLM, benchmark, or sample-more stage. A readable-state positive licenses a fresh utilization
successor; it does not silently open one here.

Do not change model/revision, loop layers, state slots, target manifest, rank, full-rank shapes,
dropout, objectives, weights, data rows/seeds/counts, train/eval K, optimizer/clipping, steps, result
seeds, positive controls, 0.40 threshold, cellwise rule, bootstrap, fixed-final rule, or branch order
after any model-bearing call. A scientific change requires a new experiment. Preserve negative,
invalid, stopped, and prohibited-branch receipts.
