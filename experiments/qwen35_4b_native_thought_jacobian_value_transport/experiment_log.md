# Qwen3.5-4B Native-Thought Jacobian Value Transport Experiment Log

## Scaffold

Created as a new experiment scaffold.

## 2026-07-12 — intake and adversarial design

- Routed through `make related`; named five close prior lines and the canceled
  native-prefix stages of `qwen35_4b_jacobian_value_transport` as the nearest
  duplicate.
- Froze the replicated lens, band 4--8, natural prefix fractions, full-recompute
  backend, scalar value-coordinate formula, staged gates, and control family.
- CPU enumeration found `negate`-first depth-2 tasks non-identifiable due to
  algebraic reorderings. Excluded it from target support before model calls while
  retaining it as a distractor/second operation.
- Generated 80 unique fresh tasks with zero parent overlap and a one-type visible
  first-operation certificate.
- Completed the 24-threat adversarial review before implementation or any model
  call. Current state is design/CPU smoke only.

## 2026-07-12 — immutable design boundary

- Rebased onto current `origin/main`, then froze design commit
  `b87b67f28586687954c89ba653d22cafe93d6073`.
- Frozen README SHA-256:
  `221992f5ff29f74db16d29996d46a87e3162c49b41e03516812a3959aec692b1`.
- Frozen preregistration SHA-256:
  `62b975cae27bfdf842bdccc06ce395735469c3fbff3f596a5987aee803a56040`.
- No model call has occurred.

## 2026-07-12 — native generation and seam implementation

- Added pinned Transformers batch-one native-thinking generation with explicit
  temperature/top-p/top-k, natural-close-only stopping, full-prefix
  recomputation, and `use_cache=False` on every token.
- Added historical thought-token activation capture, frozen J coordinate reads,
  alias token/rank contracts, and exact causal suffix-invariance smoke checks.
- Implemented the 16-task/48-trace frozen seam gate. It stores natural close,
  parse, correctness, token, seed, and stopping receipts without forcing close.
- CPU suite passes four tests. No model call has occurred; implementation will
  be committed and pushed before model smoke.

## 2026-07-12 — model-smoke attempt 001

- The pinned model loaded, then the tokenizer contract stopped before generation:
  the user instruction literally contained `<think>...</think>`, creating a
  second special-token pair in addition to the chat template's native opener.
- No token was generated and no correctness outcome was observed or written.
- Removed literal delimiter strings from the instruction while retaining native
  thinking through the frozen chat template. No scientific setting changed.

## 2026-07-12 — model-smoke attempt 002

- The repaired one-opener token contract passed. The second smoke stopped before
  generation because an eight-example rendered prompt used 541 tokens, above an
  implementation-only 512-token context safety cap.
- Raised that cap to 768 so the frozen prompt plus 160-thought/8-answer allowance
  fits. Task data, output budget, sampling, checkpoints, gates, and controls are
  unchanged; no correctness outcome was observed or written.

## 2026-07-12 — model-smoke attempt 003

- Token IDs, prompt cardinality, model shape/revision, lens rank, J-coordinate
  finiteness, and generation plumbing passed on two tasks.
- Both traces hit the frozen 160-token thought cap without natural close; no
  final alias was parseable. Correctness was deliberately not computed.
- The same historical thought token differed by 0.0625 across short-prefix and
  full-suffix forwards, above the frozen 1e-3 causal-invariance bar. This is
  consistent with the Qwen hybrid sequence-length numerical sensitivity already
  seen for batch geometry.
- Preserved the failed receipt as `runs/model_smoke/attempt_003_failed.json`.
  Plumbing pass is now reported separately from scientific causal invariance;
  the latter remains fatal before causal confirmation. Neither the 160-token cap
  nor invariance threshold is changed.

## 2026-07-12 — model plumbing pass

- Reran the identical two-task smoke with plumbing and scientific controls
  reported separately. Model/revision, alias/think token IDs, lens ranks, finite
  coordinates, full-recompute generation, and capture pass.
- The receipt remains explicit that causal invariance fails (max 0.0625) and
  both traces were cap-bound. Correctness outcomes remain absent.
- Model smoke now unlocks only the frozen seam calibration; it cannot unlock
  value fitting or causal work by itself.

## 2026-07-12 — terminal natural-seam gate

- Ran the 16 frozen seam tasks once, three traces each, after publishing the
  model-smoke boundary.
- All 48/48 rows contacted the 160-token cap without natural close; natural
  close, parse, and exact success were each 0/48, and mixed tasks were 0/16.
- The cache-free full-prefix run used 7,632 forwards over 389.9 seconds. Every
  stop reason was `think_cap_without_close`.
- Frozen decision: `NO_NATURAL_SEAM`. Prefix-value, control-calibration, and
  causal-confirmation stages are canceled without opening their outcomes.
- Preserve as an interface-budget negative. Next work requires a separate
  preregistered natural-close budget ladder and dynamic per-length patch/control
  geometry; do not rescue this experiment in place.
