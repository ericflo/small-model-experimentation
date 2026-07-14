# Counterfactual Plan Reflection Transfer Experiment Log

## 2026-07-14 — Review-9 remediation, pending Review 10

- Moved both pre-load and post-load content authentication inside every active
  tokenizer/model load-window guard and bound the authenticated commitment into the
  replayable guard receipt. Metadata-preserving pre-guard substitution and
  swap/read/restore attacks now fail closed.
- Persisted raw prompt token IDs and reconstruct prompt spend from them. Training
  compute now replays the exact copied tokenizer-parity forward-token total for the
  arm, multiplies it by the fixed three epochs, and only then charges the preregistered
  checkpoint-aware factor of four. A self-consistent billion-token forgery is rejected
  on both generation and training paths.
- Split artifact execution across the actual pinned training and vLLM interpreters.
  Both start with `-I -B -S`; before any third-party import they authenticate the
  interpreter, stage lock, exact startup-file set, every RECORD claim, and the complete
  importable site-packages file surface. The real training surface has 79
  distributions/28,222 files; vLLM has 189/68,353. One overlapping vLLM RECORD path is
  explicitly counted while the selected final byte surface remains exact and pinned.
- Bound every GPU-stage receipt to a single physical `CUDA_VISIBLE_DEVICES=GPU-...`
  selector and the matching UUID/name/driver/memory row, then requires exact identity
  across training, confirmation, and the frozen reservoir.
- Bumped generation, tokenizer, training, STARTED, merge, runtime, and load-guard
  receipt schemas so all historical evidence fails closed. Syntax and all 90
  model-free/synthetic tests pass; both real environment surfaces authenticate. No
  tokenizer, model, GPU, training, evaluation, Jacobian, or benchmark event occurred.
  Authorization remains unchanged pending independent Review 10 of the pushed SHA.

## 2026-07-14 — Review-9 HOLD

- Independent Review 9 on exact clean commit `73bef40429ccc85ba9b6ddaf7e00629a5fb29c26`
  returned HOLD after confirming both exact-SHA CI workflows and all 86 model-free
  tests.
- It passed the closed tokenizer surface, receipt-schema transitions, and consistent
  checkpoint multiplier of four, but reproduced five remaining gaps: authentication
  outside the guarded load interval, absent raw prompt IDs and unbound training-token
  totals, unauthenticated external startup/import files, an impossible single-vLLM-
  environment training instruction, and host-inventory rather than selected-device
  GPU parity.
- Authorization remains unchanged. No tokenizer, model, GPU, training, evaluation,
  Jacobian, benchmark, or protected-output event occurred during the review.

## 2026-07-14 — Review-8 false-acceptance remediation

- Closed tokenizer provenance to an exact authenticated five-file local surface and
  explicit absent-file set; all tokenizer users load `Qwen2Tokenizer` locally with
  remote code disabled and reject class or file-surface drift.
- Added Linux inotify/read-lease/inode load-window guards around tokenizer, config,
  Transformers model, and vLLM engine initialization. A swap-load-restore regression
  now fails even when the final bytes match the original commitment.
- Reconstruct all generation and compute counters from raw token arrays, reject
  boolean/non-finite numeric values, require exact training/confirmation/reservoir GPU
  identity, and charge checkpointed training at four forward-token equivalents.
- Expanded the detached-worktree contract to include ignored state and an external,
  hashed, isolated `-I -B` interpreter plus an exact no-extras package inventory. All
  artifact-producing stages enforce the same boundary.
- The complete pinned-environment suite passes 86 model-free/synthetic tests, and the
  authorized full CPU construction remains unchanged at 576 unique depth-three tasks
  with zero model, GPU, or benchmark events. Authorization remains tokenizer-only
  pending independent Review 9 of the exact pushed revision.

## 2026-07-14 — Review-7 provenance and matched-compute remediation

- Review 7 held the full implementation on seven reproduced blockers: contradictory
  remote-code provenance, no training-cost sample-more baseline, incomplete
  base/tokenizer/runtime and post-load byte binding, numeric rather than bitwise replay,
  no immutable execution-worktree enforcement, and a malformed parity-gate list.
- Published the first remediation tranche on `main` as `9c8cfed7` after resolving a
  concurrent generated-index rebase conflict. It pins all tokenizer-semantic files,
  carries exact base/tokenizer/runtime/training-compute commitments through schema-6
  merge lineage, reauthenticates after vLLM engine load, hashes raw tensor bytes, and
  enforces one clean detached exact-SHA execution worktree.
- Replaced the equal-16-candidate sample-more placeholder with an outcome-blind frozen
  vLLM reservoir. It accumulates fixed 16-candidate blocks until both token-forward
  equivalents and wall time reach the maximum full training-plus-confirmation cost of
  the two correct-reflection seeds. Labels and scores are not accepted by the stopping
  process.
- Added a replayable final gate: both correct-reflection seeds must strictly beat the
  compute-stopped frozen coverage with positive paired-bootstrap lower bounds and
  nonnegative deltas in all three families. Final stage authorization now requires
  both passing confirmation decisions plus this matched-compute artifact.
- All authorization flags remain unchanged. The historical tokenizer receipt is
  invalid as a training prerequisite under the stronger schema; a fresh exact-SHA
  tokenizer-only receipt awaits Review 8.
- Review 8 on exact `dc95ab8cdea18257ca7630bf59d6594eea70f9e7` returned HOLD.
  It confirmed the structural matched-compute and provenance repairs but reproduced
  six new false-acceptance classes: extra tokenizer-semantic files, swap-load-restore
  TOCTOU, cross-hardware wall matching, fabricated raw/metadata token counters,
  checkpoint-recomputation undercharging, and ignored executable state inside a
  nominally clean worktree. No execution authorization changes occur.

## 2026-07-14 — discovery and scaffold

- Re-read the workspace paper's methods, intervention figures, counterfactual
  reflection section, method ablations, formal sparse-frame definition, and
  multi-token extensions.
- Discovery rejected a generic within-thought correctness-coordinate experiment as a
  duplicate of the terminal J-value line.
- Selected the paper's distinct training claim: loss on a counterfactual reflection
  branch may shape behavior on an untrained action branch.
- Named `qwen35_4b_bank_the_thoughts` as closest near-duplicate and the concurrently
  active on-policy prefix-repair experiment as a non-duplicate neighboring line.
- Created a fresh experiment and a model-free construction smoke. No model,
  tokenizer, GPU, adapter, benchmark, claim allocation, or hidden result was touched.

## 2026-07-14 — adversarial HOLD and construction repair

- Independent review of commit `3eae868d182f4a02848f6415d8eaafdb87465336`
  returned HOLD: proposed string geometry was impossible, state explosion escaped,
  and 14/30 smoke plans were not uniquely identified by visible examples.
- Kept tokenizer/model/GPU/training/Jacobian work sealed.
- Expanded the string and list primitive libraries, removed a list symmetry that had
  insufficient unique slot support, and replaced rejection allocation with exhaustive
  exact-depth catalog enumeration plus deterministic split allocation.
- Required each target to have one global depth-three spelling and exactly one
  depth-three program (with no depth-zero/two alternative) on its seven visible examples.
- Full configured CPU construction now produces 504/504 tasks with zero collisions
  and complete operation-position support. State-exploding candidates are rejected.
- Shuffled arms now preserve task truth in immutable fields and place the donor only
  in explicit supervision fields; every donor is wrong on the recipient's visible or
  query behavior.
- Added a Python audit-hook firewall that denies benchmark-root opens and directory
  enumeration. Remaining review defects concern mechanism isolation, exact rendering,
  training parity, retention, gates, and result-separated Jacobian work.
- Published construction repair commit `83a55cf3887dd681790aeee1e8d1070cea4b8d15`;
  exact Validate Repository run `29345252095` and Publish Research Site run
  `29345252135` both completed successfully.

## 2026-07-14 — design-contract implementation

- Added a 72-task calibration split and 48 real, visible-identifiable exact-depth-1/2
  retention tasks. Full model-free geometry is now 576 depth-three plus 48 retention.
- Added a correct non-reflective auxiliary-label arm. Its target is identical to the
  correct reflection arm; its only semantic change is the first instruction sentence,
  and exact rendered prompt-token equality is a prerequisite gate.
- Restricted correct/shuffled donor permutations within each family and 18-row
  optimizer group. Each group has six tasks per family and identical correct/shuffled
  target multisets, allowing exact per-step token exposure checks.
- Froze the QLoRA recipe, target-only Qwen thinking-channel masks, two training seeds,
  final-only checkpoint selection, vLLM generation geometry, paired-bootstrap effect
  thresholds, per-family breadth gates, positive-control sanity, retention margins,
  and no-seed-selection staging.
- Implemented immutable record construction, token/mask encoding, parity receipts, a
  guarded trainer, and executable decision-gate analysis. None was run with a real
  tokenizer or model because adversarial HOLD remains in force.
- Removed the conditional J stage from this result-bearing experiment. A behavioral
  replication can only license a separate experiment with fresh J fit, confirmation,
  and causal evidence.
- Published design-freeze commit `1cb3c351b1ca14b518abe7cbff02ac67e6134726`;
  exact Validate Repository `29347022799` and Publish Research Site `29347024862`
  completed successfully.
- Review 2 kept training/evaluation on HOLD after reproducing construction and parity:
  incomplete evidence could satisfy decision functions, provenance and stage ancestry
  were not enforced, and literal/ON-OFF/retention controls were not executable end to
  end. Its tokenizer implementation assessment was favorable but not accepted as a
  clean authorization because an overly broad search surfaced unrelated protected-log
  lines.
- Implemented all eight Review 2 remediations model-free: exact evidence sets,
  output/input/model/environment/checkpoint lineage, mechanical stage receipts,
  executable retention and literal-reflection controls, adapter ON/OFF proof, and
  adapter-tree validation. Generated JSONL is now hashed into runner metadata so an
  output cannot be substituted under otherwise valid provenance. Authorization remains
  unchanged pending a fresh independent review of the committed revision.
- Clean Review 3 on exact commit `492376af67fd03e8b75210b8bb42ebb297fdbeed`
  returned `PASS_TOKENIZER_ONLY` and kept all full execution on HOLD. It passed 45
  tests/full construction but reproduced forged-label false passes, unsealed sampling
  fields, imbalanced family/depth false passes, missing literal reflection inputs,
  incomplete stage/adapter/runtime lineage, and absent live KV-capacity preflight.
  Enabled only `authorization.tokenizer`; every model/GPU/training/evaluation/J/benchmark
  flag remains false.
- Published tokenizer-only authorization commit
  `334d11a23b516147cb25007b6db0b6b826fd350e`; Validate Repository `29349549100`
  and Publish Research Site `29349549237` both completed successfully.
- Ran the sole authorized tokenizer receipt. It passed with Qwen2Tokenizer EOS 248046,
  receipt SHA-256 `ddaddd0f7af8a97802ab8f4cfde6c480ef60c94dc74a5c3577dd9db674432079`,
  row hash `5da1f43812d7fd2c3fb50976aba3a557e65f771d6506481499458fe26d305ca8`,
  and zero model/GPU/benchmark events. Correct reflection, shuffled reflection, and
  auxiliary plan-label arms each have exactly 77,020 prompt, 5,164 target, and 82,184
  forward tokens; all 12 optimizer groups match correct versus shuffled.
- Implemented all eight Review 3 full-execution remediations without tokenizer/model/GPU
  events: sealed-byte reconstruction; complete sampling equality; exact task metadata
  and cross-arm runtime identity; dedicated literal-reflection inputs; strict staged
  generation ancestry; embedded training/PEFT/merge lineage; installed-lock and adapter
  ON/OFF parity; and live hybrid KV token/block preflight. The suite passes 55 focused
  tests. Authorization remains tokenizer-only pending a fresh Review 4.
- Clean Review 4 on exact commit `542ba82592d96eafcf56cd5e70bfad948b43b65b`
  returned HOLD. It confirmed the sealed inputs, sampling/task mappings, literal
  branches, and live hybrid-cache checks, but reproduced three false acceptances:
  arbitrary/nonexistent hashes can authorize stages; a dummy byte string can pass as
  a merged checkpoint with self-issued lineage; and the direct-URL vLLM pin is omitted
  from installed-package validation. Authorization remains tokenizer-only while these
  attacks are converted to fail-closed tests and remediated.
- Implemented all three Review 4 remediations model-free. Stage promotion now replays
  exact raw-generation score ancestry and recomputes every gate; merged checkpoints
  retain and authenticate the real source adapter tensors/tree and must pass a static
  4B safetensors/index inventory; adapter ON/OFF evidence is replayed from raw bundles;
  and installed vLLM is checked against the direct-URL `0.24.0+cu129` wheel pin. The
  suite passes 61 focused tests plus full construction with zero model/GPU/benchmark
  events. Authorization remains tokenizer-only pending clean Review 5.
- Review 5b on exact commit `d5ed01aceb39bd6164dafee4051ba2d236d576c2`
  returned HOLD despite passing stage/gate replay and direct-URL vLLM checks. A
  synthetic non-Qwen checkpoint with arbitrary U8 tensors and a 5 GB sparse logical
  shard passed the static inventory while using 12 KB of physical storage. The review
  also confirmed that retained LoRA and merged trees are not yet linked by replaying
  `base + LoRA delta`. No execution authorization changes occur until exact pinned-base
  structure, tensor-derived bytes, and deterministic merge equality are enforced.
- Replaced size-threshold checkpoint validation with a pinned structural contract for
  the exact Qwen3.5-4B revision and a full deterministic merge replay. The validator
  derives bytes from all 738 tensor headers and rejects sparse allocation; merge and
  runner authenticate both official base shards by LFS SHA-256, compare 610 unchanged
  tensors exactly, and replay all 128 LoRA equations. Public config/index/header
  metadata only was fetched; no model weight payload, tokenizer, model call, GPU, or
  benchmark event occurred. Sixty-three model-free tests pass; authorization remains
  tokenizer-only pending Review 6.
- Review 6 on exact commit `3e144905db852c1c38cef393de7451100a0b86a7`
  returned HOLD despite independently authenticating the exact base inventory, replay
  target set, 63 tests, full construction, and both green CI runs. The real merge is
  impossible as written: Transformers 5.13 emits one unindexed shard under its 50 GB
  default, and explicit BF16 loading corrupts the frozen checkpoint's 48 F32 tensors.
  The reviewer also reproduced dynamic-code config injection through `auto_map` because
  only a config projection is checked and local vLLM loading trusts remote code, plus a
  4,096-byte sparse payload hole accepted by the 99% allocation tolerance. Full
  execution remains unauthorized pending tensor-level mixed-dtype-preserving merge,
  explicit indexed sharding, exact config/runtime-code binding, and stricter sparse
  regression fixtures.
- Replaced the failing Transformers model-level merge with a deterministic tensor-level
  writer. It preserves the exact official two-shard index and all 610 unchanged tensors
  in source dtype, including the 48 F32 state-space tensors, while applying the 128
  LoRA deltas in the exact preregistered FP32 operation order and casting each result
  back to its own base dtype. Production validation now requires the byte-exact full
  official config and index, rejects unexpected/executable/symlink checkpoint content,
  enforces full physical allocation, and all local/runtime loads disable remote code.
  Mixed-dtype, one-shard, `auto_map`, executable-injection, and 4,096-byte punched-hole
  regressions pass as part of 67 pinned-environment model-free tests. Authorization
  remains tokenizer-only pending fresh Review 7.
- Review 7 on exact commit `2b79995a50c13275a3bacf7fa7cb71ef16525188`
  returned HOLD. The real tensor writer, mixed-dtype preservation, 67 tests, full
  construction, exact public inventory, and both CI workflows passed, but generation
  provenance still expects remote-code trust while the runner records false; the
  frozen baseline does not include 36-step training compute; base/tokenizer/runtime
  bytes and the merged load handoff are underbound; tokenizer/training still enable
  remote code; numeric equality accepts negative-zero/positive-zero byte changes;
  staged artifacts cannot survive an in-place commit/rebase; and malformed YAML joins
  two parity gates. No execution authorization changes until all seven counterexamples
  have executable fail-closed regressions and a fresh exact-commit review.
