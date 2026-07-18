# Qwen35 4B WHY-Comment Install Experiment Log

## 2026-07-18 — model-free construction frozen

Lifecycle 35, cognitive-core coding bet #4 and the FIRST of the owner-specified
WHY-not-WHAT family. Mission: install real coding capability into base
Qwen/Qwen3.5-4B via a designed, contamination-free curriculum, proven by
transfer. Meta-context: bet #1 (execution-tracing, a PASSIVE skill) was NULL; bet
#2 (self-repair, a LOOP behavior) was a WEAK POSITIVE (HumanEval +3, agentic 8/35
-> 10/35, 3-vs-1 discordant) — loop/behavior curricula beat passive-skill ones.

The bet: teach the 4B WHY a correct answer is correct (its causal/generating
structure), not just WHAT it is — the mechanism most likely to escape
`install != convert`. This INLINE variant puts the rationale as trailing `#WHY:`
code comments bound to the exact line each explains, leaving the think channel
alone. It goes FIRST because it is the CLEANEST test: code comments are INERT to
execution grading, so the WHY hypothesis is tested with ZERO annealing — train on
richly-commented code, eval pass@1 (comments ignored); if the CODE improved, the
WHY-annotation worked, unconfounded. Provenance: NO teacher — the task and its
correct solution are built BY CONSTRUCTION and the `#WHY:` text is emitted
mechanically per line.

Built and verified (no GPU, no commit):

- `scripts/gen_why_comment_curriculum.py` (seed 92450) — 504 `spec -> correct
  solution` rows across 15 parameterized synthetic function families, each
  meaningful line annotated with a trailing `#WHY:` causal comment. Per-row
  verified by real CPython execution: STRIP the `#WHY:` comments and the clean
  code passes ALL asserts; the commented code runs and passes them IDENTICALLY;
  the marker is distinctive and mechanically strippable; every `#WHY:` comment is
  line-specific (references a token on its line) and the comments VARY within the
  row (non-boilerplate). Corpus sha
  `040be350678ea0337b8fe0607f783aba9e9071f789471b0ea00f7ce1ebef2962`; tiers short
  120 / medium 192 / long 192; 3-8 comments per row; 504 unique (commented-code,
  tests) keys; 438 distinct commented sources. Output length: full training
  render max 468 tokenizer tokens (median 329), measured against the pinned
  tokenizer — well under the 4096 cap; ZERO rows truncate.
- `scripts/contamination.py` + `data/contamination/banned_function_names.json`
  (668 benchmark function names, 663 after whitelist) — 0 whole-word hits over
  prompt + think + answer (including the `#WHY:` prose); 0 distinctive shared
  7-grams between the corpus's executable CODE (comments stripped) and benchmark
  solution code (51 shared spans, all pure control-flow idioms). The `#WHY:`
  vocabulary deliberately avoids the benchmark def-name collisions
  (count/find/sort/check/add/compare/maximum/minimum/first/multiply/divisor).
- Vendored `scripts/train_think.py` (sha e0eca2a2…) and `scripts/merge_adapter.py`
  (sha cb9af8b4…), byte-identical to the chain trainer/merger.
- `scripts/train_trial.py` — fail-closed base authentication (in-cell provenance
  copy + tree manifest + full 9 GB weights hash); recipe r32/a64, 1 epoch, lr
  1e-5, seed 92451 (63 optimizer steps).
- `scripts/measure_transfer.py` — invokes the shared fitness harness for both
  arms x both datasets; the grader IGNORES comments (the clean, unconfounded WHY
  test); frozen, TIGHTENED INSTALLED_CODING (>= 3-problem gain) / RETENTION_FAIL /
  NULL consequence, identical to bet #2's rule.
- `scripts/run.py` — checkpointed `--smoke | --stage train | --stage merge |
  --stage measure`; each GPU stage gated behind a staged adversarial review.
- 58 unit tests green (present-only HF-cache aids RUN with the cache; every row
  independently re-executed by a separate assert-based grader); `run.py --smoke`
  green; boundary drills refuse.

Grep-fresh note: construction seed 92450 and training seed 92451 are fresh
repo-wide as SEEDS (the only textual matches are incidental substrings inside
unrelated hashes/floats). No training-seed collision.

No-annealing decision (frozen): the commented model IS the test; a comment-strip
anneal is a documented FUTURE follow-on only if this clears the bar. Pre-committed
next member of the WHY family if NULL: the think-block variant (bet #3).

GPU stages (train/merge/measure) are pending their staged reviews.
