# Adversarial Design Review

## Review 1 — 2026-07-14

**Reviewed commit:** `3eae868d182f4a02848f6415d8eaafdb87465336`

**Verdict:** **HOLD**

**Access:** zero tokenizer/model/GPU/benchmark/hidden/protected events; no review edits

The independent adversary reproduced the committed smoke and then tested the proposed
geometry. It found the original corpus impossible: the string family had only 122
eligible signatures for 192 required rows, and an uncaught state-explosion exception
terminated full construction. It also exhaustively checked the visible examples and
found that 14/30 smoke tasks admitted two to nine depth-three plan spellings.

Additional blockers were:

1. correct-versus-shuffled isolates task-specific auxiliary plan supervision, not a
   reflection-specific effect; a matched non-reflective plan-label arm is required;
2. the fixed `READY` seam is controlled branch transfer, not a literal interrupted
   action state, so claims must be scoped accordingly;
3. direct-control, retention, literal-reflection, and candidate-compute contracts were
   missing or contradictory;
4. tokenizer rendering, exact loss masks, LoRA/optimizer/batching parity, target-token
   accounting, and immutable receipts were not frozen;
5. qualification/confirmation power and seed-independent gates were not executable;
6. the conditional J stage lacked independent fit/confirmation/causal data and fixed
   readout/intervention controls, and therefore must not share behavioral evidence.

### Repair checkpoint status

The following defects are repaired in the working design and covered by executable CPU
tests:

- a finite exact-depth catalog is enumerated before allocation;
- globally behavior-equivalent and shallower-equivalent programs are excluded;
- seven visible demonstrations uniquely identify the exact ordered plan;
- full 216/72/144/144 construction succeeds with zero program or behavior collisions
  and complete per-split operation-position coverage;
- state explosions are rejected, not leaked;
- shuffled donor labels occupy separate supervision fields, preserve task truth, and
  are behaviorally wrong for the recipient;
- validation uses fail-closed exceptions rather than optimization-removable asserts;
- the construction entry point consumes the committed config and installs a Python
  audit-hook benchmark read firewall.

The verdict remains HOLD. No tokenizer/model/GPU/training/Jacobian stage may run until
the remaining design contracts are implemented, committed, and independently reviewed.

### Post-review implementation awaiting Review 2

The working repair adds the previously missing non-reflective correct-label arm,
real retention tasks, exact record and mask code, a frozen QLoRA recipe, within-step
token parity, executable paired gates, explicit literal-reflection accounting, and
two-seed confirmation rules. The J stage has been moved to a required separate future
experiment. These changes are not self-authorizing; Review 2 must inspect their exact
committed revision.

## Review 2 — 2026-07-14

**Reviewed commit:** `1cb3c351b1ca14b518abe7cbff02ac67e6134726`

**Verdict:** **HOLD** for training/evaluation; tokenizer implementation statically
safe, but a separate clean authorization signature is required.

The adversary reproduced full 216/72/144/144 plus 48 retention construction, verified
all 36 optimizer groups and 216 shuffled donors, passed the focused model-free tests,
and found the rendering/mask/trainer logic coherent and fail-closed. It confirmed the
J follow-up is genuinely result-separated.

It then demonstrated that the decision functions could pass a one-task, one-family,
one-depth artifact because they did not require the sealed task-ID sets. It also found
that generation metadata, merged trees, stage ancestry, retention promotion,
literal-reflection execution, adapter ON/OFF behavior, and adapter-tree lineage were
not mechanically enforced. These are blocking provenance defects even though the
statistical design itself is now specified.

During review, one overly broad repository search surfaced unrelated tracked run-log
lines containing hidden-output fields despite the benchmark exclusion. The reviewer
made zero tokenizer/model/GPU/benchmark calls and no edits, but this access mistake
means its tokenizer-safe assessment is not accepted as the required clean independent
authorization.

### Review 2 remediation implemented, pending independent review

- exact sealed task IDs, all three families, and both retention depths are now required;
- scoring binds input/label receipts, vLLM metadata, model/revision, runner, environment
  lock, clean Git state, sampling, seed, engine geometry, merged checkpoint, arm, and
  adapter seed;
- merge and vLLM load bind full artifact trees and exact source arm/seed lineage;
- screen, replication, confirmation, and final promotion use hash-bound stage receipts;
- retention has an executable promotion gate;
- literal reflection now has an exact branch constructor and token-matched base-prefix
  scorer;
- every merged adapter must pass a greedy token/logprob ON/OFF gate before scoring.
- runner metadata now hashes the generated JSONL itself, and every scoring path checks
  that binding before accepting rows.

The repaired model-free suite passes 45 focused tests and reconstructs all 624 sealed
depth-three plus retention tasks with no collision. No authorization changes occur
until these repairs are committed and a fresh clean reviewer returns its verdict.

## Review 3 — 2026-07-14

**Reviewed commit:** `492376af67fd03e8b75210b8bb42ebb297fdbeed`

**Verdict:** **PASS_TOKENIZER_ONLY; HOLD full implementation.**

This was the required clean review: it read only the allowlisted experiment/shared
implementation files, used temporary synthetic fixtures, and made zero tokenizer,
model, GPU, benchmark, protected-output, or web calls. Both exact-commit CI workflows
were also green. It independently passed all 45 focused tests, syntax, and full
construction.

The tokenizer path is authorized because it pins the sole permitted model/revision,
checks EOS 248046, reconstructs sealed records, forbids truncation, validates exact
mask/loss boundaries and optimizer-step parity, and exclusively writes its receipt.
This verdict does not authorize any model, GPU, training, evaluation, J-space, or
benchmark event.

The adversary reproduced eight full-implementation blockers:

1. scoring trusts self-issued prompt/label receipts instead of reconstructing exact
   sealed prompt bytes, label bytes, and per-task family/depth mappings;
2. primary scoring does not compare the complete sampling and resolved-sampling
   dictionaries, so unregistered penalties and custom-prompt flags can pass;
3. family and retention-depth gates check sets rather than the exact per-task mapping
   and balanced sealed counts;
4. the literal branch lacks a dedicated sealed reflection-input constructor;
5. consumers do not fully validate stage-receipt schema/cardinality/ancestry, and
   confirmation generation is not mechanically stage-gated;
6. merge/runner lineage omits parts of the source training receipt, PEFT recipe,
   trainer/Git identity, and prerequisite receipt chain;
7. adapter ON/OFF does not reconstruct exact sealed calibration mappings, and runtime
   parity does not prove installed-lock or exact cross-arm environment identity;
8. the runner lacks the live KV-capacity/preemption preflight required by the pinned
   vLLM operating contract.

Review 3 demonstrated false capability, positive-control, and reflection-specific
passes using substituted constant labels plus internally consistent synthetic
receipts. It also demonstrated qualification with family counts 142/1/1 and retention
with depth counts 1/47 and family counts 46/1/1. These are decisive HOLD findings.
They must be remediated and independently reviewed before any full execution.

### Review 3 remediation implemented, pending Review 4

- `src/eval_inputs.py` is now the single canonical builder for action prompts, exact
  oracle labels, literal-reflection prompts, literal-action branches, receipts, and
  task metadata. Scorers reconstruct these bytes and reject self-consistent forged
  receipts; a regression test reproduces the constant-label attack and fails closed.
- Primary, adapter-gate, and literal paths compare every `SamplingConfig` field plus
  the complete resolved sampling dictionary, including penalties, custom-prompt flags,
  logprob settings, and thinking shuffling.
- Capability, positive-control, calibration, and retention gates require the exact
  per-task family/depth mapping. Every compared row must share one runtime-protocol
  hash, closing the 142/1/1 family and 1/47 depth attacks.
- A dedicated sealed literal-reflection constructor is distinct from the action input.
  Literal action prompts and their receipt are deterministically reconstructed from
  the exact reflection generation.
- Stage receipts have an exact schema, prerequisite claim cardinality, both frozen
  seeds, source hashes, issuer Git/script identity, and clean-worktree requirement.
  The runner requires and validates a stage receipt before every non-smoke model load,
  including confirmation generation.
- Training copies its tokenizer and stage receipts, records trainer Git/code, recipe,
  record, parity, and full adapter-tree identities. Merge validates the complete PEFT
  recipe and embeds the source training/stage/tokenizer/adapter-config files in the
  hashed merged tree; the runner revalidates them at load time.
- Runtime validation checks every locked distribution version and exact cross-arm
  package/GPU/Git/engine/cache identity.
- The runner authenticates pinned Qwen hybrid-cache geometry, then requires both the
  rounded live KV-token inequality and conservative `active_sequences * 11` block
  inequality before generation. The complete terms and margins enter metadata.

The model-free suite now passes 55 focused tests plus syntax and full construction.
All model/GPU/training/evaluation/J/benchmark flags remain false. These changes are
not self-authorizing and require a fresh clean adversarial verdict on their committed
revision.

## Review 4 — 2026-07-14

**Reviewed commit:** `542ba82592d96eafcf56cd5e70bfad948b43b65b`

**Verdict:** **HOLD full implementation.**

Both exact-commit CI workflows were green. The clean adversary passed all 55 focused
model-free tests, syntax, full 216/72/144/144 plus 48 construction, and separate
token- and block-overcommit attacks. It confirmed that Review 3 findings 1–4 and 8
are genuinely closed.

Three high-severity false-acceptance paths remain:

1. stage consumers accept prerequisite claims containing arbitrary or nonexistent
   SHA-256 strings, while the authorizer accepts minimal hand-written decision and
   retention JSON without authentic score ancestry or analyzer identity;
2. the committed merged-checkpoint test accepts `weights.safetensors` containing only
   the bytes `weights` plus self-consistent fabricated lineage, so tree integrity does
   not establish scientific origin or arm/seed identity;
3. installed-package validation parses only `name==version` lock entries and silently
   omits the direct-URL `vllm` requirement, allowing a forged vLLM build while checking
   the other 188 distributions.

No model, GPU, training, evaluation, J-space, or benchmark work is authorized. The
next implementation checkpoint must make the exact attacks fail by resolving and
revalidating prerequisite artifacts, retaining and authenticating the actual source
adapter tensors/tree through merge and consumption, validating a real composite
checkpoint inventory, and enforcing the normalized direct-URL vLLM build pin.

### Review 4 remediation implemented, pending Review 5

- Every task-level score row now carries one exact, hash-bound invocation over the raw
  generation, runner metadata, sealed prompt receipt, sealed labels, and any adapter
  gate. Score artifacts are reconstructed byte-for-byte before they can enter a gate.
- Calibration, qualification/confirmation, and retention artifacts now have exact
  producer/invocation schemas. Consumers replay their score ancestry, analyzer code,
  stage ancestry, and gate calculation. Stage claims contain only absolute artifact
  paths and hashes; copied pass booleans have been removed. Nonexistent, changed, and
  minimal self-consistent prerequisite fixtures are explicit rejection tests.
- Adapter ON/OFF receipts now retain and replay the exact raw base/merged generations,
  metadata, sealed inputs, and labels. Replaying generation provenance also revalidates
  the underlying base or merged model and its stage receipt.
- Merge retains the complete source adapter directory, including actual LoRA tensors,
  start record, training receipt, stage/tokenizer receipts, and PEFT config. The runner
  recomputes its full tree hash, opens the safetensors file, requires exact A/B pairs,
  and binds module count and weight hash to the merge.
- The merged model must contain a real sharded safetensors index, an exact on-disk
  tensor-to-shard mapping, required model/tokenizer assets, at least 100 tensors, and
  at least 5 GB of both indexed tensor bytes and physical shard bytes. The old literal
  `b"weights"` fixture is now a rejection test.
- Direct-URL lock requirements are parsed as wheel pins. Installed vLLM must be exactly
  `0.24.0+cu129`; missing, non-wheel, and forged versions are rejection tests.

The repaired suite passes 61 focused model-free tests, syntax, and full construction.
Authorization remains tokenizer-only. Review 5 must independently attack the exact
committed revision before any model/GPU/training/evaluation stage is opened.

## Review 5b — 2026-07-14

**Reviewed commit:** `d5ed01aceb39bd6164dafee4051ba2d236d576c2`

**Verdict:** **HOLD full implementation.**

The first Review 5 worker was terminated by an automated safety classifier before
returning a verdict. A fresh read-only scientific audit then verified both exact-commit
CI runs, 61 focused tests plus 30 subtests, AST parsing for 35 Python files, and full
216/72/144/144 plus 48 construction with zero collisions or prohibited events.

Stage/gate replay and the direct-URL vLLM fix passed independent negative fixtures.
The reviewer nevertheless reproduced one high-severity checkpoint false acceptance:
`merged_checkpoint_inventory()` accepted `model_type: fabricated`, architecture
`DefinitelyNotQwen`, 100 arbitrary U8 tensor names, and a sparse 5,000,009,064-byte
logical shard that occupied only 12,288 physical bytes. The implementation trusts
logical file length and index `metadata.total_size`; it does not bind the exact pinned
Qwen config, names, shapes, dtypes, or recomputed tensor payload bytes.

More fundamentally, the retained LoRA source and merged tree are authenticated only
independently. Consumption does not prove that each merged module equals its pinned
base tensor plus `B @ A * alpha / rank`. An unrelated loadable composite can therefore
be paired with self-consistent arm/seed lineage.

Authorization remains tokenizer-only. Required remediation is exact base-structure
binding, tensor-derived byte accounting, and deterministic replay of every retained
LoRA delta against pinned base tensors with comparison to the corresponding merged
tensor or an independently verifiable tensor commitment.

### Review 5b remediation implemented, pending Review 6

- `configs/pinned_model_structure.json` freezes public structural metadata for exact
  revision `851bf6e...`: config and weight-index hashes, both official LFS shard hashes
  and sizes, shard-header hashes, exact Qwen model type/architecture, 738-tensor
  name/shape/dtype inventory hash, BF16/F32 counts, and 9,319,737,856 derived bytes.
  Only `config.json`, the index, and HTTP range bytes covering safetensors headers were
  fetched while constructing this contract; no weight payload, tokenizer, model load,
  model request, or GPU event occurred.
- The checkpoint validator parses safetensors headers directly. It recomputes each
  tensor's bytes from shape and dtype, requires contiguous exact offsets and exact file
  length, compares the complete canonical inventory to the pinned commitment, and
  checks allocated filesystem blocks against logical length. Index size metadata is no
  longer trusted, and sparse shards fail closed.
- `src/merge_replay.py` authenticates the locally cached base shards against the exact
  official LFS SHA-256 commitments. It then compares every merged tensor: 610
  unmodified tensors must equal base bit-for-bit, and all 128 frozen target modules
  must exactly equal the BF16 result of `base.float() + (B.float() @ A.float()) *
  alpha/rank`. It records per-adapted-tensor commitments and an aggregate commitment
  for unchanged tensors.
- Merge runs the replay after saving and retaining the source adapter. The vLLM runner
  independently repeats it before accepting the override and requires byte-identical
  replay evidence in the merge receipt. An unrelated same-shape composite or relabeled
  arm therefore cannot reach model initialization.
- Regression fixtures cover sparse logical allocation, non-Qwen structure, derived
  metadata/index mismatch, altered adapted tensors, and altered unmodified tensors.

The model-free suite passes 63 tests. Full execution remains unauthorized pending a
fresh exact-commit Review 6 and green CI.

## Review 6 — 2026-07-14

**Reviewed commit:** `3e144905db852c1c38cef393de7451100a0b86a7`

**Verdict:** **HOLD full implementation.**

The independent read-only review confirmed the exact commit on `origin/main`, green
Validate Repository run `29356780301`, green Publish Research Site run `29356780131`,
all 63 focused model-free tests, AST parsing for all 36 Python files, and full
216/72/144/144 plus 48 construction with zero collisions or prohibited events. It
also independently authenticated the public frozen config, index, both LFS shard
hashes and sizes, both safetensors headers, the 738-tensor mixed-dtype inventory,
the exact 128-module LoRA target set, and the 610 unchanged-tensor set.

The reviewer nevertheless reproduced two independent contradictions that make every
real merge fail before a usable receipt can exist:

- Pinned Transformers 5.13 defaults `save_pretrained()` to a 50 GB shard cap. The
  9,319,737,856-byte model is therefore emitted as one `model.safetensors` with no
  index, while the validator requires `model.safetensors.index.json`.
- Loading with explicit `dtype=torch.bfloat16` casts the official checkpoint's 48 F32
  `A_log`/`dt_bias` tensors to BF16. The exact pinned inventory and unchanged-tensor
  replay correctly reject that dtype/value change even if sharding is repaired.

The audit also reproduced a high-severity configuration false acceptance. The
validator hashes only a structural projection, so exact official tensors plus a config
augmented with dynamic `auto_map`, unrelated `quantization_config`, and executable
remote-code routing retains the accepted projection hash. Because the runner passes
`trust_remote_code=True`, Transformers prefers that dynamic route even for locally
known `qwen3_5`. Exact tensor replay does not authenticate execution semantics.

Finally, the 99% physical-allocation tolerance accepted a synthetic safetensors file
with a punched 4,096-byte payload hole; at model scale it permits roughly 93 MB of
unallocated space. Exact replay remains a semantic backstop, but the static check is
not as fail-closed as documented.

Authorization remains tokenizer-only. Required remediation is a deterministic
tensor-level merge that preserves every untouched tensor's original dtype and exact
value, rewrites only the 128 target tensors, freezes and validates an explicit indexed
shard policy, authenticates the complete base config and forbids executable checkpoint
content, loads local composites with `trust_remote_code=False`, tightens physical
allocation to filesystem-block rounding, and adds model-free regressions for all four
counterexamples. No model, tokenizer, GPU, training, evaluation, benchmark, hidden,
qualification, or confirmation event occurred during Review 6.

### Review 6 remediation implemented, pending Review 7

- The model-level load/save path has been removed. `src/tensor_merge.py` opens the
  authenticated official shards directly, copies all 610 unchanged mmap-backed
  tensors in source dtype, computes only the exact 128 adapter targets using
  `base.float32 + (B.float32 @ A.float32) * alpha/rank`, casts each result back to its
  own source dtype, and emits the same frozen two-shard filenames and byte-exact index.
  No Transformers model object or global dtype coercion is involved.
- `configs/default.yaml` now freezes the tensor writer, exact two-shard placement,
  unchanged/source-dtype rule, update equation, full-allocation rule, and false
  remote-code policy. Merge receipt schema 5 binds that complete contract and a
  tensor-writer receipt; the runner independently reconstructs both.
- Production checkpoint validation now requires the complete official `config.json`
  SHA-256 and byte-exact official weight-index SHA-256, not only a structural
  projection. The runtime root is a minimal allowlist of config, index, two shards,
  receipt, and retained lineage; symlinks, executable suffixes/modes, and every other
  root file or directory are rejected. AutoConfig, AutoTokenizer, and vLLM all use
  `trust_remote_code=False`.
- Physical allocation must now be at least the complete logical file length. The 1%
  tolerance is gone; a synthetic otherwise-valid 10 MB safetensors with a punched
  4,096-byte payload hole fails.
- Regressions exercise two-shard serialization, a mixed BF16/F32 source, exact F32
  preservation, update math/order/source-dtype cast, one-shard policy drift, injected
  `auto_map`, executable-file injection, and a partial sparse hole. The complete
  model-free suite passes 67 tests in the pinned environment, including the real
  safetensors writer (the system-Python run skips only that dependency-gated case).

Authorization remains tokenizer-only. Review 7 must independently attack the exact
committed remediation and its real-runtime feasibility before any model/GPU/training
or evaluation flag changes.

## Review 7 — 2026-07-14

**Reviewed commit:** `2b79995a50c13275a3bacf7fa7cb71ef16525188`

**Verdict:** **HOLD full implementation.**

The fresh read-only audit confirmed the exact clean commit and ancestry, green Validate
Repository run `29358785262`, green Publish Research Site run `29358785257`, all 67
pinned-environment tests (including the real multi-tensor mixed-dtype safetensors
writer), AST parsing for 37 Python files, and full 216/72/144/144 plus 48 construction
with zero collisions or prohibited events. It independently confirmed the official
738-tensor, 9,319,737,856-byte, 690-BF16/48-F32 inventory; 128 adapted/610 unchanged
partition; exact two-shard serialization; partial-hole, executable, and symlink
rejections; representative cross-environment LoRA replay; and host feasibility.

Seven blockers remain:

1. `vllm_runner.py` now records `trust_remote_code: false`, but `provenance.py` still
   requires `true`, so every valid frozen or merged generation fails before scoring.
2. Frozen sample-more still has only the same 16 candidates and caps as each evaluated
   checkpoint. It receives no compute corresponding to 36 optimizer steps, so it is
   not the repository-required end-to-end matched-compute baseline.
3. Loaded bytes are not fully bound to validated bytes. Frozen evaluation and training
   do not authenticate the exact cached source shards before load; training receipts
   omit the exact package/runtime/hardware/base commitments; and merged validation
   releases the path before vLLM opens it, leaving a TOCTOU interval.
4. Tokenizer receipt and training still enable remote code, and evaluation binds only
   selected token semantics rather than the complete tokenizer artifact identity.
5. `torch.equal` is numeric rather than bitwise: an unchanged F32 tensor changed from
   negative zero to positive zero passed the claimed exact replay.
6. Every stage requires its commit to equal current HEAD. The pipeline is feasible only
   if all execution stays in one immutable clean exact-SHA worktree while result commits
   and rebases happen elsewhere; this workflow is neither documented nor enforced.
7. A misindented YAML dash concatenates the rendered-prompt parity gate and exact-mask
   receipt gate into one string, leaving five parsed gates rather than six.

Required remediation is executable provenance parity with false remote-code trust; a
preregistered end-to-end compute unit, amortization horizon, same-backend frozen
sampling reservoir, and decision gate; exact base/tokenizer/environment commitments
at training and evaluation load boundaries plus an immutable handoff; raw-byte replay;
an enforced detached execution-worktree contract; and exact config-schema tests.
Authorization remains tokenizer-only. Review 7 made zero tokenizer/model/GPU/training/
evaluation/Jacobian calls and accessed no benchmark, run, large-artifact, hidden,
qualification, confirmation, cache, or tensor-payload content.

### Review 7 remediation implemented, pending Review 8

- Runtime metadata and provenance now agree on `trust_remote_code=False`. The six YAML
  parity gates have exact schema coverage, and unchanged tensor replay hashes raw
  tensor bytes so signed-zero changes fail.
- The tokenizer receipt authenticates the exact five tokenizer-semantic files at the
  pinned public revision. Training records the complete installed-package lock,
  hardware/runtime, base snapshot, tokenizer snapshot, detached worktree, per-row
  token/mask hashes, and full training compute. Schema-6 merge lineage retains and
  replays those commitments.
- Base, tokenizer, and merged-model commitments are checked before engine creation and
  immediately after vLLM opens the model. A mocked mutation test proves that a changed
  post-load commitment shuts the engine down before generation.
- Tokenizer, training, merge, stage authorization, and generation now require a clean
  detached execution worktree at one exact SHA and root CWD. The normal `main`
  worktree is explicitly reserved for concurrent commits/rebases/pushes.
- Frozen sample-more is now an outcome-blind same-vLLM reservoir. Fixed 16-candidate
  blocks stop only when both token-forward equivalents and wall time match the maximum
  full training-plus-confirmation cost across the two correct-reflection seeds. The
  final stage requires a replayable matched-compute artifact, two independent positive
  paired lower bounds, and nonnegative family deltas.
- The pinned-environment suite passes 80 model-free tests, including real mixed-dtype
  safetensors serialization, exact tokenizer mutation rejection, detached-worktree
  rejection, dual-unit compute stopping, and final-stage matched-gate cardinality.

Authorization remains tokenizer-only. A fresh Review 8 must attack the exact pushed
implementation and its full artifact chain before any model, GPU, training, or
evaluation flag changes.

## Review 8 — 2026-07-14

**Reviewed commit:** `dc95ab8cdea18257ca7630bf59d6594eea70f9e7`

**Verdict:** **HOLD full implementation.**

The independent read-only audit confirmed the exact clean commit at `origin/main`,
green Validate Repository run `29362140847`, green Publish Research Site run
`29362140803`, AST parsing for all 44 tracked experiment Python files, and 54 unique
restricted model-free/synthetic checks. It confirmed false remote-code parity, the six
separate YAML gates, raw-byte tensor replay, two-seed stage cardinality, outcome-blind
reservoir CLI and compute-only stop, maximum-block failure, and transitive final replay.

Six blockers remain:

1. The tokenizer commitment pins five files but does not reject extra semantic files.
   Transformers can resolve `added_tokens.json` and `special_tokens_map.json`; a
   synthetic added-token file beside the five pinned files was accepted. Tokenizer
   users also load by model ID rather than from one closed authenticated local path,
   and downstream code does not enforce the recorded tokenizer class.
2. Post-load integrity rehashes only current filesystem bytes. A
   validate→swap→load→restore race leaves different state resident in vLLM or
   Transformers while both pre/post hashes pass. No opened-inode/load-window or
   resident-state commitment exists.
3. Training records GPU UUID while vLLM omits it, and no validator requires training,
   correct confirmation, frozen block zero, and reservoir blocks to share hardware.
   Synthetic fast-training/slow-evaluation receipts passed the wall-time target.
4. Compute-controlling counters are trusted rather than reconstructed. Synthetic empty
   token arrays with a billion claimed sampled tokens passed scoring, and fabricated
   metadata produced a two-billion-token reservoir charge. Booleans/non-finite numeric
   values are not uniformly rejected.
5. The fixed multiplier of three omits the additional forward recomputation induced by
   the frozen gradient-checkpointed training recipe, so token-forward accounting can
   undercharge training when that unit binds.
6. Ordinary `git status --porcelain` ignores code-bearing ignored files. A malicious,
   timestamp-valid `.pyc` or ignored in-worktree environment can affect execution while
   the detached worktree receipt still reports clean exact-SHA state.

Required remediation is a closed tokenizer file/absence surface loaded only from the
authenticated local snapshot; an inode/event- or resident-state-bound load window with
swap-restore tests; exact GPU/runtime parity; raw token-array reconstruction and strict
finite/non-boolean numeric schemas; conservative checkpoint-aware compute accounting;
and explicit rejection/authentication of ignored executable state and interpreter
provisioning. Authorization remains tokenizer-only. Review 8 read no benchmarks,
protected outputs, caches, qualification/confirmation contents, or tensor payloads and
made zero tokenizer/model/GPU/training/evaluation/Jacobian calls.

### Review 8 remediation implemented, pending Review 9

- Tokenizer provenance is now a closed five-file local view: the source snapshot pins
  hashes and sizes, explicitly requires the two optional semantic files to be absent,
  rejects every extra entry, loads `Qwen2Tokenizer` locally with remote code disabled,
  and enforces the exact tokenizer class in downstream receipts.
- Tokenizer, config, Transformers model, and vLLM engine initialization now run inside
  a Linux load-window guard combining inotify events, read leases, and exact inode/
  namespace surfaces. The regression suite proves that an unchanged read succeeds and
  a validate→rename→malicious-load→restore attack fails despite identical final
  content hashes.
- Training and generation use the same exact GPU identity tuple (name, UUID, driver,
  and total memory). Both correct-confirmation seeds and every frozen reservoir block
  must match the training hardware before wall-time evidence can enter the budget.
- Every generation counter used by scoring or matched-compute replay is reconstructed
  from raw stage, injected, final, and prompt token arrays. Exact schemas reject
  booleans, non-integers, non-finite timing, incorrect forced-close accounting, and
  self-consistent fabricated metadata.
- The frozen gradient-checkpointed recipe is conservatively charged at four forward-
  token equivalents: original forward, two backward-equivalent passes, and the
  recomputed forward. The same multiplier is enforced in training, provenance,
  target-budget construction, reservoir accounting, and tests.
- Artifact-producing commands require an exact detached root CWD, isolated `-I`
  execution, `-B`, an external hashed interpreter, an exact no-extras package
  inventory matching the lock, and a Git status including ignored entries. Extra
  distributions, ignored bytecode, and in-worktree interpreter fixtures now fail closed.

The pinned-environment suite passes 86 model-free/synthetic tests, and the full
authorized CPU construction still produces the frozen 576-task geometry with zero
model calls, GPU events, or benchmark reads. Authorization remains tokenizer-only.
A fresh Review 9 must attack the exact pushed implementation before any additional
execution flag changes.
