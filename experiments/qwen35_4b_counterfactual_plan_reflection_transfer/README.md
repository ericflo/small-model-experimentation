# Qwen3.5-4B Counterfactual Plan Reflection Transfer

**Status:** in-progress · since 2026-07-14 · Review 11 blockers are remediated model-free and awaiting exact-SHA Review 12; model/GPU/training/evaluation remain unauthorized

This experiment tests the paper's most actionable claim without relying on its
consciousness framing: can supervision on what the model would say on a later
reflection branch change what it does on an unreflected action branch? The fixed
`READY` seam makes this controlled branch transfer, not a claim about a literal
interrupted internal action state.

## Research Program

- Primary program: `posttraining_and_adaptation`.
- Cross-program fit: `structured_execution_and_compilers` and
  `interpretability_and_diagnostics`.
- Closest near-duplicate: `qwen35_4b_bank_the_thoughts`, which trains the actual
  plan-and-code continuation. Here the treatment receives loss only on a later,
  counterfactual reflection turn and never on the task answer.
- Other anchors: `qwen35_4b_commit_slot_semantic_power_replication` (shared scalar
  J-value negative), `qwen35_4b_jacobian_transport_control_replication` (clean
  supplied-concept J transport positive), and
  `qwen35_4b_tokenizer_eos_residual_mechanics_fresh_replay` (inference-time plan
  materialization did not create correct proposals).

## Question

Can correct, reflection-only SFT on fresh three-step machine-induction contexts
increase held-out answer coverage on the same contexts' unreflected action branch,
beating a within-family shuffled-reflection arm, frozen Qwen3.5-4B, and an end-to-end
matched-compute frozen sampling reservoir? If it does, is the gain specific to reflection framing,
or does an equally sized ordinary auxiliary plan-label branch work just as well?

## Hypothesis

An appended reflection question creates a training branch on which the model must
name the ordered latent plan but not calculate or state the query answer. If the
paper's verbal-disposition mechanism transfers to capability learning, gradients
from that final reflection answer should make the correct plan easier to assemble in
the shared pre-action context. The actual action answer is never a target for the
reflection or auxiliary-label arms. A gain is not task-specific transfer unless
correct reflection beats shuffled reflection under byte-identical contexts and
stepwise token-matched training. It is not reflection-specific unless it also beats
the correct non-reflective auxiliary-label arm.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Task source: experiment-owned procedural list, string, and three-register
  machines. Every task composes exactly three parameter-free primitives, shows seven
  examples, and asks for outputs on three new inputs. The exact ordered plan is the
  only depth-three program consistent with the seven visible examples; globally
  behavior-equivalent spellings and shallower-equivalent programs are excluded.
- Common context: the user supplies the machine and says not to solve; the Assistant
  gives the fixed content-free response `READY`. The next turn branches.
- Reflection branch: asks only for `PLAN: first -> second -> third`; its target never
  contains the exact query-answer string. The target Assistant turn contains a short
  plan statement inside the Qwen thinking channel and the same plan in the final
  answer; prompt and fixed `READY` tokens are fully masked.
- Auxiliary-label branch: replaces only `Pause before solving.` with
  `Provide exact labels.` and uses the identical correct target. Its rendered prompt
  token count must exactly match the reflection branch for every row before training.
- Action branch: asks only for `ANSWER: <JSON outputs>`. Reflection-only training
  never receives loss on this branch or its answer.
- Splits: 216 train, 72 frozen calibration, 144 qualification, and 144 confirmation
  tasks, balanced across three families, plus 48 untouched depth-1/2 retention tasks.
  Programs and behavioral signatures are disjoint.
- Baselines and controls: frozen action; frozen literal-reflection-then-action;
  correct reflection; within-family shuffled reflection; correct non-reflective plan
  labeling; a direct action-branch plan-plus-answer SFT positive control; depth-1/2
  retention; and same-backend sample-more.
- Training parity: QLoRA rank 32/alpha 64/dropout 0.05 on all seven projection
  modules, three epochs, batch 1 × accumulation 18, 36 final-only optimizer steps.
  Every optimizer group contains six rows per family. Correct/shuffled derangement is
  restricted inside that group, so target and forward-token totals must match within
  every step, not merely in aggregate.
- Primary deployable metric: paired exact full-query coverage@16 under identical vLLM
  thinking/answer budgets. Candidate counts 1 and 4 are descriptive. Report every
  family separately.
- End-to-end baseline: frozen Qwen uses the same persistent vLLM engine, prompts,
  thinking/answer caps, and fixed 16-candidate blocks. It stops at the first complete
  preregistered block whose cumulative spend reaches the larger of the two correct
  reflection seeds' full training-plus-confirmation spend in both token-forward
  equivalents and wall time. The stopping process accepts no labels or scores.
- Hidden-label boundary: answers are procedural oracle labels used only for grading
  and direct positive control construction. No `benchmarks/` path may be read,
  imported, or used for training.

## Staged Decision

1. CPU construction must prove exact re-execution, exact-depth feasibility, all
   identity/collision rules, shuffled-target derangement, and answer omission.
2. A tokenizer-only receipt must prove exact rendering, mask boundaries,
   reflection/auxiliary prompt-length equality, and per-step correct/shuffled parity.
   Clean Review 3 authorized this stage only.
3. Frozen calibration must establish a parseable action interface and non-saturated
   headroom before training.
4. Screen seed 47 trains all four arms. The direct positive control must reach 0.50
   coverage@16 and improve over frozen by 0.20. Correct reflection must beat shuffled
   and frozen by at least 0.10 overall and 0.05 in every family, with paired-bootstrap
   lower bounds above zero.
5. Only that pass opens replication seed 53 for the three non-positive-control arms.
   Both seeds must independently pass qualification before the fresh confirmation
   split opens; both must independently pass confirmation. No seed selection or
   ensembling is permitted. Retention must remain within the frozen margins.
6. Final capability promotion additionally requires each seed's correct-reflection
   coverage@16 to strictly beat the compute-stopped frozen reservoir, with a positive
   paired-bootstrap lower bound and no negative family delta. One reservoir is sized
   to the maximum of the two seed costs; a failure to reach both compute units within
   16 blocks is a gate failure, not permission to change the accounting.
7. Reflection-specific interpretation additionally requires correct reflection to
   beat the non-reflective auxiliary arm by 0.05 with a positive paired lower bound.
   Otherwise any capability pass is generic auxiliary-plan transfer.
8. A replicated behavioral pass may open a **new, result-separated experiment** with
   fresh J-fit, J-confirmation, and causal-confirmation data. No J-space fitting or
   ablation may reuse this experiment's behavioral gates.

No generic within-`<think>` correctness scalar is being retried: that exact proposal
was already tested and failed task-held-out controls in
`qwen35_4b_commit_slot_semantic_power_replication`.

## Run

Authorized model-free smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/qwen35_4b_counterfactual_plan_reflection_transfer/scripts/run.py --smoke
```

The full configured CPU construction is also authorized:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/qwen35_4b_counterfactual_plan_reflection_transfer/scripts/run.py --construct
```

Only the pinned tokenizer receipt command is authorized. Model, GPU, training,
evaluation, Jacobian, and benchmark commands remain forbidden.

All tokenizer/training/merge/generation/stage-authorization commands must run from a
separate, clean, detached worktree at the one reviewed authorization SHA. The normal
`main` worktree remains available for commits, rebases, and pushes while execution is
in progress:

```bash
git fetch origin main
git worktree add --detach /workspace/sme-reflection-exec <reviewed-authorization-sha>
cd /workspace/sme-reflection-exec
# Enter every artifact stage through the stage-specific tracked static launcher.
# It supplies the pinned external interpreter, -I -B -S, and a replacement
# environment; direct Python entry is terminal.
TRAINING_LAUNCHER=$PWD/experiments/qwen35_4b_counterfactual_plan_reflection_transfer/scripts/training_launcher
VLLM_LAUNCHER=$PWD/experiments/qwen35_4b_counterfactual_plan_reflection_transfer/scripts/vllm_launcher
# GPU commands pass exactly one physical UUID to the launcher, never an index:
# $VLLM_LAUNCHER --cuda-visible-devices=GPU-<uuid> vllm_runner <arguments>
# Model-free boundary audit (no device selector, tokenizer, model, or writes):
# $TRAINING_LAUNCHER runtime_audit
# $VLLM_LAUNCHER runtime_audit
# Invoke only the stages enabled by that exact committed config, writing outputs
# outside this worktree so it remains clean for the entire staged pipeline.
```

The scripts reject a branch checkout, a dirty tree, a different current directory,
or a later SHA. Do not commit, rebase, or edit inside the execution worktree; remove it
only after all required downstream consumers have authenticated the external
artifacts. Review 7 remediation changed the tokenizer receipt schema and exact
tokenizer/config/script commitments, so the earlier tokenizer receipt
`ddaddd0f7af8a97802ab8f4cfde6c480ef60c94dc74a5c3577dd9db674432079` is historical
and cannot authorize training. A fresh tokenizer-only receipt will be issued from the
next independently reviewed exact SHA; no model or GPU authorization is implied.

After—and only after—a future reviewed config enables confirmation evaluation, the
compute baseline is invoked with both embedded correct-reflection training receipts
and both raw confirmation metadata files. Its CLI intentionally has no label or score
argument:

```bash
$VLLM_LAUNCHER --cuda-visible-devices=GPU-<exact-uuid> run_frozen_reservoir \
  --input <confirmation-prompts> --input-receipt <confirmation-input-receipt> \
  --stage-receipt <confirmation-stage-receipt> \
  --training-receipt <seed-47-training-receipt> \
  --training-receipt <seed-53-training-receipt> \
  --correct-metadata <seed-47-correct-confirmation-metadata> \
  --correct-metadata <seed-53-correct-confirmation-metadata> \
  --output-dir <external-reservoir-directory>
```

`matched_compute_gate.py` then replays both confirmation decisions and the complete
reservoir manifest. `authorize_stage.py --stage final` requires that artifact via
`--matched-compute` in addition to both `--confirmation` decisions. These commands are
documentation, not present authorization.

## Results

The repaired full model-free construction deterministically creates 576 depth-three
tasks plus 48 depth-1/2 retention tasks. It has 576 unique ordered depth-three
programs and behavior signatures, zero cross-split collisions, unique visible exact
plans, complete operation-by-position support in every full split, and zero exact
answer strings in reflection targets. Shuffled supervision preserves
immutable task truth and uses a within-family donor plan that is observably wrong on
the recipient's demonstrations or queries. The construction also emits immutable
four-arm record and optimizer-schedule hashes. A Python audit hook denies file and
directory access beneath the repository benchmark root. This remains model-free and
does not lift the adversarial HOLD.

Review-2 remediation additionally makes all promotion stages receipt-gated, binds
runner metadata to the generated JSONL and exact engine/environment/checkpoint lineage,
requires a real merged-adapter ON/OFF effect, and makes retention and literal-reflection
controls executable. These repairs remain non-authorizing until independently reviewed.

Review-3 remediation now reconstructs prompt and oracle-label bytes from sealed task
code inside every scorer; compares the complete raw and resolved sampling dictionaries;
requires exact task→family/depth mappings and one cross-arm runtime protocol; adds a
dedicated literal-reflection input bundle; validates exact stage schemas and ancestry
before every non-smoke generation; embeds training/stage/tokenizer/PEFT lineage inside
the hashed merged tree; verifies installed packages against the vLLM lock; and performs
a live hybrid-cache token/block preflight before generation. This remains model-free,
non-authorizing work pending clean Review 4.

Review-6 remediation replaces model-level `save_pretrained()` merging with an exact
tensor-level writer. It authenticates and preserves the frozen two-shard index, copies
all 610 unchanged tensors in their source dtype (including 48 F32 tensors), computes
only the 128 registered LoRA updates in FP32, and casts each update back to that
tensor's source dtype. The local composite must carry the byte-exact official config,
may contain only the frozen model files plus retained lineage/receipt, rejects dynamic
or executable checkpoint content, and is loaded with `trust_remote_code=False`.
Physical allocation must cover the complete logical safetensors length. These repairs
remain model-free and non-authorizing pending a fresh Review 7.

Review-7 provenance remediation authenticates all five tokenizer-semantic files and
the complete pinned base snapshot, records the full installed-package/runtime/GPU
identity and charged training compute, carries those commitments through a schema-6
merge receipt, reauthenticates bytes immediately after vLLM engine load, compares
unchanged tensors by raw bytes, and enforces the detached execution-worktree contract
above. It also implements an outcome-blind, fixed-seed frozen reservoir with a
dual-unit compute stop and a transitive two-seed final promotion gate. All work remains
model-free and non-authorizing pending a fresh adversarial verdict.

Review-8 remediation closes the six reproduced false-acceptance classes without a
model or GPU event. Tokenizer loading now uses an exact five-file authenticated local
surface and rejects both extra and missing semantic files. Linux inotify, read leases,
and inode-surface receipts protect tokenizer/config/model engine load windows and
reject a validate→swap→load→restore attack. Raw token arrays reconstruct every
compute-controlling counter; numeric schemas reject booleans and non-finite values;
training, confirmation, and reservoir receipts require one exact GPU identity; and
gradient-checkpointed training is charged at four forward-token equivalents. All
artifact-producing stages also reject ignored worktree state and bind an isolated,
external, hashed interpreter and exact no-extras package inventory. The 86-test
model-free suite and full CPU construction pass. These are implementation results
only. Independent Review 9 passed the tokenizer closure and multiplier-four
consistency but reproduced gaps around the guarded load window, prompt/training token
reconstruction, external environment closure, documented training environment, and
selected-GPU identity. Authorization remains unchanged while those findings are
remediated.

Review-9 remediation now keeps content authentication before and after every
tokenizer/model load inside the active inotify/read-lease guard and binds those exact
content commitments into its receipt. Generation rows persist raw prompt token IDs;
scoring reconstructs prompt spend from those arrays; and training forward tokens must
equal the copied tokenizer-parity total times the fixed three epochs before the
checkpoint multiplier is applied. Artifact stages start under `-I -B -S`, authenticate
the exact stage-specific interpreter, lock, startup-file set, RECORD claims, and full
site-packages file surface before third-party imports, and use the training environment
for Transformers/PEFT/bitsandbytes versus the separate vLLM environment for generation.
GPU receipts now bind exactly one `CUDA_VISIBLE_DEVICES=GPU-...` selector to its
physical UUID row. Receipt schemas were bumped so historical artifacts fail closed.
The resulting 90-test suite and both real environment-authentication passes are green;
no tokenizer, model, GPU, training, evaluation, Jacobian, or benchmark event occurred.
These were implementation results only. Authorization stayed unchanged while
independent Review 10 audited the exact pushed revision.

Independent Review 10 on exact pushed commit
`e0f33860a26ee46d0b64061cf68d70ed7cba05dc` returned HOLD. It passed the
guard-held tokenizer/model transactions, raw prompt and sealed training-token replay,
stage-specific dependency closure, receipt schemas, and structured GPU-identity
propagation. It then reproduced four residual false-acceptance/operational paths: a
swap→import→restore window after environment authentication; an interpreter hash that
is recorded rather than committed plus path-only stdlib/native closure; `-S` dropping
required venv-bin/allowlisted path effects and Mamba re-exec dropping `-I -B -S`; and
selected-device inventory accepted from a PATH-shadowed `nvidia-smi`. Authorization
remained unchanged while those findings were remediated model-free.

Review-10 remediation now holds one immutable import window from pre-import
authentication through the last artifact-relevant import and reauthenticates before
any result write. It pins the resolved interpreter, complete stdlib, executable,
system-library, CUDA-library, and stage-specific site-package surfaces in committed config. System
roots use inotify, inode surfaces, and cryptographic before/after checks; read leases
remain mandatory for the mutable Python environments, while 34 root-owned injected
driver files whose leases are denied by the kernel are explicitly enumerated in the
guard receipt. Loaded native mappings must remain inside those authenticated roots.
Under `-S`, vLLM now derives its bin directory from `sys.executable` and explicitly
activates the authenticated CUTLASS package path. Adaptive Mamba geometry and process
re-exec are removed; the frozen engine is capacity-fitted at 15 sequences with capture
sizes `[1, 2, 4, 8, 15]`. The selected-device query uses the pinned absolute
`/usr/bin/nvidia-smi`, and after CUDA initialization its UUID row must match the sole
active logical device's name and memory. Receipt schemas again invalidate every prior
artifact. Both real detached training/vLLM bootstrap-seal audits and the model-free
suite pass without tokenizer/model/GPU/training/evaluation/Jacobian/benchmark events.
Independent Review 11 audited exact pushed commit
`903842b09209044aa0a48c2f7f7fd59ef3681d2b` and returned HOLD despite 87 tests,
23 subtests, and both exact-SHA CI workflows passing. It found that the scoped
system-file lease fallback is not fail-closed against all mutation mechanisms; the
resolved vLLM interpreter still collapses its bin directory to `/usr/bin` under
`-S`; active CUDA validation compares name and memory but not UUID; later Git, `uv`,
and `nvcc` calls remain PATH-resolved while some bootstrap code runs before the full
tree guard; and replay accepts an empty loaded-native-mapping set. Authorization
remains unchanged while these five counterexamples are converted into fail-closed
regressions and remediated model-free.

Review-11 remediation replaces direct dynamic-Python entry with reproducible static
training and vLLM launchers. The launcher remains as the live parent, supplies a
replacement environment and fixed `-I -B -S` interpreter/dispatcher, and carries its
own open inode across both execs for a three-way parent/proof/path authentication.
All Git, `uv`, `nvcc`, and device-inventory calls now execute pinned bytes through an
authenticated inherited descriptor; PATH is never their trust boundary. The vLLM
tool path preserves the invoked venv symlink rather than resolving it to `/usr/bin`.
Lease fallback is now legal only for an exact read-only file mount whose mount identity
is unchanged; the real host surface has 4,915 leased files and exactly 34 such NVIDIA
mounts. Active CUDA identity compares and records UUID in addition to name and memory,
and loaded-native replay must contain every pinned initial mapping. The model-free
suite passes 92 tests and 23 subtests. Authorization remains unchanged pending a fresh
exact-SHA Review 12.

Both fixed audit stages then passed from a clean detached worktree at exact pushed
commit `da80b2b314b44140f305e3b84bf727583486e882`. Training sealed 33,178 leased files,
34 exact read-only mounts, and 17 loaded native mappings. vLLM sealed 73,330 leased
files, the same 34 mounts and 17 mappings, and discovered the authenticated CUTLASS
path without importing a model. Validate Repository run `29380316080` and Publish
Research Site run `29380316110` both passed. The detached worktree stayed clean and was
removed. These remain model-free implementation facts; Review 12 and all execution
authorization are still pending.

## Interpretation

The paper unlocks a training hypothesis, not an already-demonstrated Qwen capability.
Inference-time semantic materialization already failed in this repository, whereas
counterfactual reflection changes weights using loss on a different branch. The new
experiment exists to distinguish that mechanism from direct plan SFT and from mere
additional sampling. No scientific result exists yet.

## Knowledgebase Update

- Program evidence: unchanged until a model result exists.
- Program backlog: records this active reflection-only mechanism test.
- Claim ledger and shared synthesis: unchanged; no claim ID allocated.

## Artifacts

- `idea_intake.md`
- `configs/default.yaml`
- `src/taskgen.py`
- `src/records.py`
- `src/scoring.py`
- `src/analyze.py`
- `src/vllm_runner.py`
- `src/matched_compute.py`
- `src/runtime_contract.py`
- `configs/pinned_runtime_environments.json`
- `src/load_window_guard.py`
- `src/tokenizer_lineage.py`
- `scripts/run.py`
- `scripts/runtime_launcher.S`
- `scripts/runtime_entry.py`
- `scripts/runtime_audit.py`
- `scripts/training_launcher`
- `scripts/vllm_launcher`
- `scripts/tokenizer_receipt.py`
- `scripts/train.py`
- `scripts/merge_adapter.py`
- `scripts/adapter_behavior_gate.py`
- `scripts/build_eval_inputs.py`
- `scripts/build_literal_reflection_inputs.py`
- `scripts/build_literal_action_inputs.py`
- `scripts/score.py`
- `scripts/score_literal.py`
- `scripts/analyze.py`
- `scripts/run_frozen_reservoir.py`
- `scripts/matched_compute_gate.py`
- `scripts/calibration_gate.py`
- `scripts/retention_gate.py`
- `scripts/authorize_stage.py`
- `tests/test_taskgen.py`
- `tests/test_records.py`
- `tests/test_scoring.py`
- `tests/test_analyze.py`
- `tests/test_vllm_runner.py`
- `tests/test_eval_inputs.py`
- `tests/test_stages.py`
- `tests/test_runtime_contract.py`
- `tests/test_matched_compute.py`
- `reports/artifact_manifest.yaml`
- `reports/power_analysis.md`
