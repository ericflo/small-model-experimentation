# Qwen35 4B WHY-Think Scale — Experiment Log

## 2026-07-18 — Dual-channel design freeze (model-free, no GPU stage run)

Built the CORRECTED WHY curriculum (owner directive). The prior WHY curriculum
(`experiments/qwen35_4b_why_scale_ladder`) put its reasoning in inline `#WHY:`
comments and left the `<think>` block minimal. Qwen3.5-4B is a THINKING model
whose coding performance depends on its `<think>` trace (the repo's most-
replicated finding; the coding harness was even mistakenly measuring thinking-OFF
and is being fixed to thinking-on + 8192 budget). Empty-think WHY training risks
DESTROYING the model's native thinking, so this cell puts a GENUINE step-by-step
derivation in the `<think>` channel AND keeps the inline `#WHY:` comments.

Constructed and verified model-free (no seed consumed by a model event; construction
seed 95200, training seed 95201 reserved by design):

- **Generator** `scripts/gen_why_think_curriculum.py` (seed 95200): copied the proven
  59-family scale-capable generator and EXTENDED it with the dual-channel think
  derivation. Each row's `think` is emitted MECHANICALLY from the family AST/shape:
  parse the spec (goal/inputs/output) -> choose an approach FROM THE CODE SHAPE
  (accumulator / builder / running-extreme / branch / search / dict) phrased as a
  decision -> build the solution step by step in construction order -> trace a REAL
  worked example (one of the task's asserts, executed line by line, every value
  byte-true) -> conclude into the answer. The worked-example CORE is a
  deterministic, rng-free string recomputed byte-for-byte at verification.
- **Per-row truth audit** kept AND extended: strip `#WHY:` -> clean code passes all
  asserts; commented code runs identically; marker strippable; every `#WHY:` line-
  specific/non-boilerplate; AND (new) the think has an approach-decision phrase, a
  worked-example trace matching real execution, and is NOT the joined `#WHY:`
  comments; safety/termination; determinism (2 builds identical).
- **Measured (5000-row sample, seed 95200):** 59/59 families across 13 categories;
  100% unique programs; 1196 distinct normalized `#WHY:` templates; **4997/5000
  distinct think skeletons** (the derivation genuinely varies, not one template).
  10000-row: 59 families, 1197 `#WHY:` templates, 9985 distinct think skeletons,
  100% unique.
- **Token budget:** the think lengthens the render, so it is capped. Real pinned
  tokenizer full render (chat + think + `</think>` + answer) over 5000 rows:
  **max 739 tokens, p95 619, median 467, min 330 — 0 over the 4096 cap.**
  Conservative >=3-char/token estimate: max 695.
- **Contamination through 10000 rows:** 663 banned benchmark names after whitelist,
  **0 whole-word hits** over prompt + THINK + answer; **0 distinctive shared code
  7-grams** (78 structural idioms) vs benchmark solutions (HF cache present, aid RAN).
- **Ladder** `data/ladder_manifest.json`: rungs 2000/5000/10000/20000/40000, sha-
  pinned (generator sha + fixture sha + per-rung corpus sha); corpora are large,
  deterministically regenerable, gitignored under `large_artifacts/`.
- **Recipe** (frozen): one fresh r32/a64 QLoRA adapter per rung from the fail-closed
  `base_reserialized` composite, lr 1e-5, batch 1, grad-accum 8, max-length 4096,
  **w_think 0.2** (>0: preserves + shapes thinking), w_close 0.2, **1 epoch every
  rung** (owner directive), seed 95201. Vendored trainer (sha e0eca2a2...) + merger
  (sha cb9af8b4...) byte-identical.
- **Measurement:** the SHARED coding-fitness harness (referenced, not copied), being
  fixed to thinking-on + 8192 budget; base is CO-MEASURED thinking-on per rung, so
  NO hardcoded thinking-off 76.2% anchor is carried.

GPU stages (train/merge/measure per rung) are gated behind staged adversarial
reviews (`reports/compute_review.md`, `merge_review.md`, `measure_review.md`) that
are created later; none has run. This is Phase A of scale-then-RLVR: the peak rung
becomes the SFT foundation for the subsequent RLVR phase.

## Scaffold

Created as a new experiment scaffold, then rebuilt as the dual-channel WHY-think
scale ladder (cloned from `experiments/qwen35_4b_why_scale_ladder`).
