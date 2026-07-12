# Qwen3.5-4B Pareto Policy Integration Experiment Log

## Scaffold

Created as a new experiment scaffold.

## 2026-07-12 — corrected successor accepted

- User rejected the predecessor's fixed `S0 + 0.10` specialist gate as an
  obvious scientific-design error. The correction is not to lower that number;
  it is to remove arbitrary effect-size qualification entirely.
- Teacher existence is now paired `delta > 0` with two positive seed blocks and
  a one-sided stratified-bootstrap lower bound above zero. Saturated cells are
  retention anchors, not vetoes.
- C54 landed between the two experiments and materially changed the best test:
  rather than speculate about four not-yet-trained domain specialists, this run
  attempts to consolidate the already evidenced same-origin quick/deep Pareto
  policies (`blend`, `apex`).
- New experiment directory created rather than rewriting the predecessor.
- No task-model output existed when the config, preregistration, and design
  review were authored.

## 2026-07-12 — design lock

- Pre-output design commit: `6bb8530ac5b1c289fbf9682846317607d46e9673`.
- `runs/preregistration_receipt.json` freezes SHA-256 digests for the config,
  intake, preregistration, and design review. Every non-smoke stage verifies
  both those digests and commit ancestry before loading a model.

## 2026-07-12 — live model preflight

- Pinned base vLLM semantic/runtime smoke passed 4/4 tasks; Transformers prompt
  parity, finite logits, causal-conv and flash-linear-attention fast paths all
  passed under the frozen training lock.
- The scaffold runner had accidentally omitted its local-composite CLI path.
  The first reload therefore stopped at argument parsing after training/merge,
  before producing a local-model score. Reintroduced the proven explicit
  `model_override` path with mutual-exclusion and model-type validation plus a
  regression test.
- Weighted-training smoke completed 8/8 steps on the quick data shape with no
  skips. Explicit merge applied 128/128 nonzero deltas (summed Frobenius norm
  23.00) on CUDA FP32 with TF32 disabled.
- The merged composite then loaded through vLLM, produced the 4/4 semantic
  smoke outputs, and preserved the requested full CUDA-graph decode geometry.

## 2026-07-12 — integration harness locked before policy evaluation

- Corrected teacher-top-50 MOPD now caches full-softmax probabilities at the
  exact student token prefix and consumes 160 distinct trajectories per round;
  no rollout or target span is split or replayed to manufacture update count.
- The five-update locality pilot measures centered non-target logit drift and
  full-vocabulary entropy change before authorizing the four-round run.
- Wrong-route and off-policy controls use the primary arm's deterministic
  rollout selection and rescale backward loss to the primary arm's measured
  initial corrected-top-k pressure in each round. Update count, data pressure,
  and initial objective magnitude are therefore matched explicitly.
- Non-finite loss or gradients and the frozen round-loss ceiling now preserve
  an auditable stopped adapter receipt instead of disappearing as a crashed
  process.
- The two-block final analyzer uses equal quick/deep macro weight, paired
  one-sided bounds against both source policies and every one-checkpoint
  control, separate anchor/transfer retention checks, three training seeds,
  and the execution-filtered best-of-8 hurdle.

## 2026-07-12 — quick specialist regenerated

- The committed C54 blend corpus encoded 2,233/2,240 rows; the seven omitted
  rows exactly match the pre-run encoding audit.
- QLoRA completed 350/350 optimizer steps (2.5 epochs), final logged loss
  0.8077 and whole-run train loss 2.0220 on the NVIDIA L40.
- Explicit composite installation applied 128/128 nonzero LoRA deltas with
  summed Frobenius norm 164.55 (CUDA FP32, TF32 disabled). The merged model
  weight hash is `97bb30362c130fab6525586b39dff9d72ca31be57b72edc879bf03e304ce91cb`.
- No task score was inspected before the independent deep specialist began.
- Before either full specialist was behavior-scored, an eight-prompt
  same-prefix installation canary was fixed (four quick, four deep; maximum
  rendered prompt length 438 tokens). Calibration now requires both merged
  specialists to differ from base and from each other under the same greedy
  vLLM runner; the canary has no task-success threshold.

## 2026-07-12 — pre-evaluation protocol hardening and upstream closure

- Before any calibration or qualification score existed, audit found that the
  imported procedural harness did not forward the config's explicit CUDA-graph
  capture list even though CLI generation did. Commit `ce45c383` forwards the
  frozen geometry and adds a regression test.
- Commit `437a1b87` additionally makes every procedural score fail closed unless
  all generation calls prove the exact local composite, config hash, engine
  dimensions, and resolved full-decode graph geometry. The scientific suite is
  31/31 and repository CI is green.
- C54 concurrently closed its simple adapter-capacity alternative after this
  experiment's design lock: rank-128/alpha-256 APEX scored quick `+0.249` and
  medium `+0.229`, below rank-32 APEX (`+0.308`/`+0.345`). Extra LoRA capacity
  therefore did not dissolve the frontier. This strengthens the motivation for
  the already-frozen policy-space test but does not change an arm, seed, gate,
  or hypothesis here.

## 2026-07-12 — deep specialist regenerated

- The committed C54 APEX corpus encoded 4,662/4,669 rows; the same seven
  pre-audited non-training rows were omitted.
- QLoRA completed the frozen 730/730 optimizer steps (2.5 epochs) with
  whole-run train loss `0.882575`, 7,156.98 seconds wall time, and 16.13 GB
  peak allocated CUDA memory on the NVIDIA L40.
- Explicit composite installation applied 128/128 nonzero LoRA deltas in CUDA
  FP32 with TF32 disabled. Their summed Frobenius norm is `248.5323` (maximum
  `4.1774`), and the merged model weight hash is
  `3bf936150a0a68e80a7a2ef3564334503fdfc3dfb3dcc5f6dfa0b1b12b7cf28b`.
- Both specialist receipts now bind the pinned base revision, exact external
  adapter/composite paths, and model-weight hashes. No specialist task score
  was inspected during either training run.

## 2026-07-12 — specialist installation canary and evaluation-engine preflight pass

- On eight fixed same-prefix greedy prompts, quick and deep each changed all
  8/8 base continuations; the two specialists differed from one another on
  7/8. All prompt, runner, decode, merge-receipt, and nonzero-delta checks
  passed, authorizing independent calibration. This is an installation test,
  not a task-success measurement.
- A separate one-token quick-composite preflight loaded the exact frozen
  evaluation geometry: 16,384-token model/batch limits, 48 sequences, 0.85 GPU
  utilization, and explicit capture sizes `[1,2,4,8,16,24,32,40,48]`.
  vLLM resolved full decode graphs at all nine requested sizes without a Mamba
  clamp or process re-exec, reported 807,029 KV-cache tokens and 49.26x maximum
  full-length concurrency, and bound the output to the quick merge receipt.

## 2026-07-12 — first calibration attempt exposed post-generation ledger bug

- The quick policy completed all 1,344 calibration atoms and 144 interactive
  episodes through turn 17, then failed before writing a score because the
  token ledger expected `turn["policy"]["n_sampled_tokens"]`. The copied
  harness actually stores the slim policy fields directly on each turn.
- This was a bookkeeping-only failure after generation: no score, cell role,
  gate, or analysis artifact existed, and the output directory remained empty.
  The fix reads `turn["n_sampled_tokens"]` through a schema-specific helper;
  a regression test exercises both atom and episode token layouts. The suite is
  now 32/32. The identical frozen calibration seed and protocol will be rerun.

## 2026-07-12 — independent descriptive calibration passes

- The corrected rerun completed 1,488 paired items per specialist. Quick used
  1,642,252 sampled tokens in 1,296.25 seconds; deep used 1,433,797 in 1,163.68
  seconds. Every exact-model, engine, runner, and resolved-graph check passed.
- Calibration assigned 56 cells to capability inference (8 quick, 48 deep)
  and 52 to retention. Every `ferrier`, `brinework`, and `spindle` cell is
  retention-only as frozen; both strata retain capability cells.
- Descriptively, the quick policy scored `0.7939` on the broad quick stratum
  and `0.5256` on deep, while deep scored `0.8224` and `0.5662`. On the
  cell-balanced capability subset, intended-teacher deltas were `-0.0605`
  (quick) and `+0.0332` (deep). These calibration outcomes do not qualify or
  disqualify a teacher and were not used to alter cells, seeds, or thresholds;
  the two frozen qualification blocks remain the decision instrument.

## 2026-07-12 — qualification execution in progress

- Quick policy block 0 (seed 96200) completed all 4,416 items in 3,387.61
  seconds using 4,748,497 sampled tokens. Broad raw means were `0.7876` quick
  and `0.5297` deep; all evaluator provenance checks passed.
- This single-arm output is checkpointed without an advantage interpretation.
  The paired deep arm and second independent block remain frozen and required.
