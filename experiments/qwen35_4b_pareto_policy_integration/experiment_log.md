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
