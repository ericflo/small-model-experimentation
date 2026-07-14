# On-Policy Failure-Prefix Universal Curriculum Experiment Log

## 2026-07-14 — Intake

- Opened only after state-table negative commit `003efebb` was pushed to `main` and
  GitHub Validate Repository run `29341508735` and Publish Research Site run
  `29341513334` both completed successfully.
- Ran `make related` for on-policy failure-prefix correction. Selected the existing
  `agentic_breadth_installation` program and named
  `qwen35_4b_universal_state_table_compiler_token_match` as the closest
  near-duplicate.
- Anchored the pivot in C53's failure-forensics/on-policy direction, C56/C59's
  content-bearing serial-compute boundary, C50's deployment-state emission lesson,
  and the operator-capture negatives from interactive/recovery curricula.
- Proposed the same authenticated `close_xi` parent; rejected inheritance from the
  failed scaffold and state-table candidates.
- Reserved fresh construction/parent-rollout/training/local/conditional-aggregate
  seeds 77113/66113/47/88009/78139.
- Authorized intake and CPU design feasibility only. No parent rollout, data
  selection, GPU model generation, training, local capability, merge, or benchmark
  event ran.

## 2026-07-14 — Model-free collection design

- Published intake commit `10ae8923` directly to `main`; Validate Repository run
  `29342538743` and Publish Research Site run `29342538693` both passed.
- Froze 288 fresh truth-audited tasks at construction seed 77,113: 48 each across
  declaration/operation, state transition, bounded induction, probe scoring, repair
  propagation, and commit serialization.
- Separated hidden oracle source (`32589348...1172`) from model-facing rollout input
  (`7a643e96...a5485c`). Fresh local seed 88,009 remains unmaterialized.
- Added exact generated-token prefix masking and model-free failure mining. Fixed ten
  reachable failures per class, a 32-token immediate-commit boundary, and fail-closed
  insufficient-quota behavior.
- Closed the documented vLLM runtime-LoRA silent no-op by requiring an explicitly
  merged `close_xi` composite with exact Qwen3.5-4B architecture fingerprinting.
- Adversarial review verdict is `PASS_PARENT_MERGE`; training, local evaluation, and
  benchmark access remain unauthorized pending observed prefix lengths and a second
  exact-compute review.

Next: commit/rebase/push this design and verify both workflows; then run only the
explicit parent-merge stage and checkpoint its receipt.

## 2026-07-14 — Explicit parent composite merge

- Published design commit `3f75c992` directly to `main`; Validate Repository run
  `29344691083` and Publish Research Site run `29344691096` both passed.
- From that clean checkpoint, ran only `--stage merge-parent`. The explicit composite
  merger loaded `Qwen/Qwen3.5-4B` revision `851bf6e8...d0a` and authenticated the
  `close_xi` adapter as weights/config `16e9dc75...c179` / `de953bd5...7ff`.
- Applied 128 LoRA deltas on CUDA; all 128 were nonzero. Sum/max delta Frobenius norms
  were 159.990169 / 2.824141, with FP32 TF32 disabled and scale 2.0.
- Saved one 8.5-GiB composite shard with SHA-256 `4933f2dd...eb373`. External
  `merge_receipt.json` SHA-256 is `1fbc84b3...5557`; durable log/experiment receipt
  hashes are `fc0b938b...53d2` / `10c3870d...95b`.
- Re-ran the merge authenticator and the exact Qwen3.5-4B architecture-fingerprint
  gate against the saved composite. No generation, training, capability, local, or
  benchmark event ran.

Next: publish and CI-verify this merge receipt, then run only `collect-parent`.

## 2026-07-14 — Authenticated parent rollout collection

- Published parent-merge commit `21e1eb59` directly to `main`; Validate Repository
  run `29345395690` and Publish Research Site run `29345395680` both passed.
- From that clean checkpoint, ran only `--stage collect-parent`: the explicitly
  merged `close_xi` composite generated one greedy natural-thinking completion for
  each of all 288 frozen prompts at seed 66,113 and a 1,024-token cap. The same vLLM
  event used max model length 4,096, max 16 sequences, max 8,192 batched tokens, and
  explicit CUDA-graph sizes 1/2/4/8/16.
- Completed 288/288 rollouts with 170,252 sampled tokens, 61,981 unique/logical input
  prompt tokens, and zero injected or stage-two tokens. Model load plus generation
  took 311.869 seconds; generation throughput was 849.923 sampled tokens/s.
- Preserved rollout/metadata/normalized-log hashes `8010632f...3b17f` /
  `9fe81276...664` / `ed0d4fc4...26b7`; the model runner hash is
  `2099c674...32aaf` and metadata binds generation to commit `21e1eb59`.
- Generation completed atomically, but the original wrapper's postvalidator exited
  only because it demanded runner `git_dirty=false` after the wrapper itself had
  opened an untracked log. Every other frozen contract check passed. The collector
  now captures clean Git state before opening outputs and includes an explicit
  `--recover-completed` path. That path authenticated the completed event, reran no
  generation, and wrote receipt hash `c6b98b79...74fa`.
- Added a repository-wide operational guard and regression test for this self-dirty
  wrapper failure. No rollout outcome was graded, no prefix was selected, and no
  training, capability, local, or benchmark event ran.

Next: publish and CI-verify this rollout checkpoint, then run only the model-free
`mine-prefixes` stage and preserve either the 60-repair inventory or the frozen
insufficient-quota negative.

## 2026-07-14 — Model-free prefix quota satisfied

- Published rebased parent-rollout commit `dbd433e8` directly to `main`; Validate
  Repository run `29346896317` and Publish Research Site run `29346896827` both
  passed.
- From that clean aligned checkpoint, ran only `--stage mine-prefixes`. The miner
  authenticated the committed rollout receipt, metadata, runner, task source, and
  hidden-field boundary, then made zero model calls.
- Graded 288 experiment-owned procedural rows: 230 met at least one frozen failure
  condition, 58 passed all registered conditions, and all 230 failed rows exposed a
  reachable clean thinking-channel prefix.
- Reachable failures for bounded induction, commit serialization, declaration /
  operation, probe scoring, repair propagation, and state transition were
  46/48/35/24/36/41. Every preregistered quota of ten passed without borrowing or
  threshold changes; exactly 60 repair rows were selected.
- Preserved inventory/source hashes `7230af52...dfe7` / `30141538...d84b8`.
  Selected prefixes contain 47,123 masked tokens total (33 minimum, 785.383 mean,
  1,024 maximum). Boundaries were 42 generation caps, ten first tokens beyond the
  commit budget, and eight answer boundaries.
- The severe-prefix mix is a compute-review risk, not a post-hoc reason to change
  selection. No exact-token stream, adapter training, capability measurement, local
  event, merge, or benchmark event ran.

Next: publish and CI-verify this failure-inventory checkpoint. Then materialize
exact-token candidate/control streams and perform the second adversarial compute
review in a separate model-free checkpoint; do not expose training before it passes.

## 2026-07-14 — Model-free exact-compute freeze

- Published rebased prefix-inventory commit `d16beecc` directly to `main`; Validate
  Repository run `29347732698` and Publish Research Site run `29347732815` both
  passed.
- From that clean aligned checkpoint, measured all 60 frozen repairs with the exact
  pinned Qwen tokenizer and the actual training encoder. All 60 fit at length 4,096;
  no selected row was removed or replaced after lengths became visible.
- Deterministically materialized two 320-row streams. Each has exactly 304,313
  unpadded forward tokens, zero skips, 40 optimizer steps, and the same 200
  byte-identical replay rows at aligned positions. Candidate repair/filler blocks
  contain 76,953/28,000 tokens; the disjoint control-variable replay block contains
  104,953.
- Preserved source-token/stream-manifest/control/candidate/final-receipt hashes
  `2ae6aded...654d` / `f836d0a1...93cd3` / `541805df...be6` /
  `9a43f3be...03f1` / `eb08026f...e0cfc`. Final encoded lengths span 329–2,991.
- Audited the non-compute match explicitly: candidate versus control has +33,421
  masked-context tokens, −33,949 think targets, equal close targets, +528 answer
  targets, and −33,421 total target tokens. Nonzero-weight tokens and absolute loss
  mass are 111,983/25,049.4 versus 145,404/31,311.2. Any result must retain this
  target-composition caveat.
- Added a fail-closed training wrapper that authenticates stream receipt, bytes,
  warm start, output path, and hyperparameters; captures clean Git state before
  opening outputs; refuses overwrite; and preserves logs/receipts. Candidate training
  additionally requires the committed control receipt.
- Second adversarial verdict is `PASS_CONTROL_TRAINING`. No model load, adapter
  training, capability measurement, local event, or benchmark event ran.

Next: commit/rebase/push this compute freeze and verify both workflows. Then run only
`train-control` from that published clean checkpoint and immediately preserve its
log and receipt before any candidate event.

## 2026-07-14 — Exact-compute replay control trained

- Published rebased compute-freeze commit `a8529c04` directly to `main` after
  resolving a generated knowledge-index conflict by deterministic rebuild. Validate
  Repository run `29350075815` and Publish Research Site run `29350075883` both
  passed.
- From that clean aligned checkpoint, ran only `--stage train-control`. The wrapper
  reauthenticated design, mining, stream bytes, token receipt, parent adapter, and
  frozen hyperparameters before opening any output.
- The exact trainer encoded 320/320 replay rows with zero skips and performed 40/40
  updates over one epoch, 304,313 forward tokens, batch size one, gradient
  accumulation eight, learning rate `1e-5`, and seed 47. Trainer/wrapper wall times
  were 272.8/292.4 seconds; final training loss was 0.4588.
- Preserved normalized log/receipt hashes `a49076ec...3501` /
  `f78f2069...d6de`. The external adapter config/weights hashes are
  `0dfd9bda...120f` / `bb59d3bd...5154d`; weights are 169,903,320 bytes.
- Structural audit found 256 tensors and 42,467,328 elements, matching the reported
  trainable parameter count. Every tensor was finite and nonzero.
- The preflight Git status was empty at commit `a8529c04`; the recorded post-training
  dirtiness contains only the newly created durable training directory. No candidate,
  capability, local, merge, generation, or benchmark event ran.

Next: publish and CI-verify this control log/receipt. Then run only
`train-candidate`; its direct wrapper must authenticate the committed control receipt,
committed log, and external adapter before loading the model.

## 2026-07-14 — Exact-compute prefix-repair candidate trained

- Published control checkpoint `b690a4b3` directly to `main`; Validate Repository run
  `29351333012` and Publish Research Site run `29351333028` both passed.
- From that clean aligned checkpoint, ran only `--stage train-candidate`. Before model
  load, the direct wrapper reauthenticated the committed control receipt and log,
  external control adapter, token receipt, candidate stream bytes, parent adapter,
  and all frozen hyperparameters.
- The candidate independently restarted from `close_xi`, encoded 320/320 rows with
  zero skips, and performed 40/40 updates over one epoch and 304,313 forward tokens.
  Batch size, gradient accumulation, learning rate, and seed remained 1/8/`1e-5`/47.
  Trainer/wrapper wall times were 282.4/298.2 seconds; final training loss was 1.288.
- Preserved normalized log/receipt hashes `e895c546...ca0` /
  `846d8107...7098`. The external adapter config/weights hashes are
  `91b7db57...37de` / `85811191...0f14`; weights are 169,903,320 bytes.
- Structural audit found 256 tensors and 42,467,328 elements. Every tensor was finite
  and nonzero. The preflight Git status was empty at commit `b690a4b3`; only the new
  candidate log and receipt were created in the tracked tree.
- This completes paired training only. The candidate's lower supervised-token count
  and loss mass remain explicit causal caveats, and no capability measurement, local
  event, merge, generation, or benchmark event ran.

Next: publish and CI-verify this paired-training checkpoint. Then design and freeze
the fresh paired local gate in a separate model-free checkpoint before evaluating
the parent, replay control, or prefix-repair candidate.
