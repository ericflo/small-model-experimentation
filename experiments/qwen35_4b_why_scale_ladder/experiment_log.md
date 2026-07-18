# Qwen35 4B WHY Scale Ladder Experiment Log

## 2026-07-18 — model-free construction frozen (Phase A of scale-then-RLVR)

Phase A of the owner's scale-then-RLVR plan. Bet #4 (`qwen35_4b_why_comment_install`)
gave the program's biggest fast gain — HumanEval 0.7622 -> 0.7927 (+5) on the clean,
comment-inert test — but was underpowered (McNemar p=0.33), flat on the agentic loop,
and did not survive combination. Owner directive: SCALE the WHY curriculum to find
its peak before overfit/collapse, merge the best rung as an SFT foundation, then RLVR.

BLOCKER removed: the sibling generator SATURATES (~75 WHY templates, 438 unique
programs at 504 rows), so naive 20x replay would overfit and read as a false
negative. This cell rebuilds the generator to produce GENUINELY DIVERSE data at
scale and builds the sha-pinned four-rung ladder + train/eval sweep harness.

Built and verified (no GPU, no commit):

- `scripts/gen_why_scale_curriculum.py` (construction seed 94100) — `--rows N`
  produces exactly N verified rows deterministically. 59 parameterized synthetic
  families across 13 categories, each meaningful line annotated with a trailing
  `#WHY:` causal comment drawn from a phrase-pool of TRUE, line-specific rationale
  variants per construct. Per-row verified by real CPython execution: STRIP the
  `#WHY:` comments and the clean code passes ALL asserts; the commented code runs
  and passes them IDENTICALLY; the marker is distinctive and mechanically
  strippable; every `#WHY:` comment is line-specific and the comments VARY within
  the row. Safety: restricted builtins, no imports/I/O, only bounded for-loops
  (never `while`), a per-call step cap that ABORTS and discards. A banned-vocabulary
  self-heal rejects any row that would carry a benchmark name.
  - Diversity (the whole point): 5000-row sample -> 59/59 families, 13 categories,
    1196 distinct normalized WHY templates, 100% unique programs. 10000 -> 1197
    templates. 20000 -> 100% unique programs (raw no-dedup draw ~82%). vs the
    sibling's ~75 templates / 438 programs at 504 rows.
  - Token budget: full training render max 499 tokenizer tokens (median 337, p95
    455) over 5000 rows, measured against the pinned tokenizer — well under the 4096
    cap; 0 rows truncate. Character render max 1567.
- `scripts/contamination.py` + `data/contamination/banned_function_names.json` (668
  benchmark function names, 663 after whitelist; byte-identical fixture to the
  sibling cell, sha `6ea920bc…`) — 0 whole-word hits over every row; 0 distinctive
  shared 7-grams (78 shared structural idioms) at 5000 rows, 0 at 10000 rows. The
  accumulator/list-param pools avoid benchmark code idioms (`total`/`res`/`prod`,
  `arr`/`lst`/`nums`). Prose vocabulary avoids the benchmark def-name words that are
  common English (`power`, `longest`, `answer`, `count`, `sort`, `find`, ...).
- `scripts/build_ladder.py` + `data/ladder_manifest.json` — four rungs
  (2000/5000/10000/20000, seed 94100), each verified + contamination-audited,
  sha-pinned (2000 `608192fa…`, 5000 `2a0fb91a…`, 10000 `d038452c…`, 20000
  `e32584d0…`); the manifest also pins the generator sha and the fixture sha.
  `--verify` regenerates each rung to its pinned sha. The corpora are large and
  deterministically regenerable, so they live gitignored under `large_artifacts/`.
- Vendored `scripts/train_think.py` (sha `e0eca2a2…`) and `scripts/merge_adapter.py`
  (sha `cb9af8b4…`), byte-identical to the sibling.
- `scripts/train_trial.py` — fail-closed base authentication (in-cell provenance
  copy + tree manifest + full 9 GB weights hash); per-rung, parameterized by
  `--rows`, reads the rung corpus + sha from the committed manifest; recipe r32/a64,
  lr 1e-5, batch 1, grad-accum 8, max-length 4096, w_think 0.2, w_close 0.2, seed
  94101. Epoch schedule `epochs = max(1, round(8000/rows))` = 4 / 2 / 1 / 1
  (exposures 8k / 10k / 10k / 20k; optimizer steps 1000 / 1250 / 1250 / 2500) —
  larger corpora need fewer epochs while total exposures stay roughly comparable.
- `scripts/measure_transfer.py` — per-rung SWEEP via the shared fitness harness
  (referenced, not copied): base + each rung composite, HumanEval 164 + MBPP 200,
  greedy pass@1; the grader IGNORES comments (the clean test). Records all four
  numbers + paired McNemar deltas + rung-vs-base problem deltas per rung. NO
  single-shot verdict — the orchestrator assembles the pass@1(rows) curve.
- `scripts/run.py` — `--smoke | --stage gen-ladder | --stage train --rows N |
  --stage merge --rows N | --stage measure --rows N`; each GPU stage gated behind a
  clean pushed main + a committed staged review + the committed ladder manifest.
- 52 unit tests green (diversity at 5000/10000/20000; WHY-truth re-executed by a
  separate assert-based grader; safety/termination; contamination zero at 10000;
  determinism; base auth fail-closed; epoch schedule; ladder-manifest sha pinning).
  `run.py --smoke` green; boundary drills refuse.

Grep-fresh note: construction seed 94100 and training seed 94101 are fresh
repo-wide as SEEDS. No training-seed collision.

GPU stages (train/merge/measure, per rung) are a SWEEP pending their staged
reviews; the orchestrator runs each rung and records the curve to find the peak as
the SFT foundation for the RLVR phase (Phase B).

## 2026-07-18 — Epoch schedule -> 1 epoch everywhere (owner directive)

- Owner flagged the epoch>1 confound: with unlimited unique data, multi-
  epoch on small rungs re-shows examples (memorization) AND varies epochs
  across rungs (confounds the scale variable). Switched epochs_for() to
  return 1 for ALL rungs; scale = pure unique-data volume, every step sees
  fresh data. Extended the ladder to 40000 (generator holds 100% unique
  through 40k, verified). Rungs 2000/5000/10000/20000/40000 @ 1 epoch,
  optimizer steps 250/625/1250/2500/5000. Killed the 4-epoch rung-2000
  before it finished; no measurement taken. Tests updated (52 green).
