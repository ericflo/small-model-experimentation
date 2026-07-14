# Counterfactual Plan Reflection Transfer Experiment Log

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
