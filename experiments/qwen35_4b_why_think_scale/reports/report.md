# Qwen35 4B WHY-Think Scale — Report

**Design-frozen report.** The dual-channel construction (the scale-capable
generator with a genuine `<think>` derivation, the sha-pinned five-rung ladder, the
train/eval harness, the tests) is complete and verified; the per-rung
train/merge/measure GPU stages are gated behind staged adversarial reviews and have
not run. Rung pass@1 numbers (measured thinking-on) and the assembled scaling curve
will be appended to Results as the sweep is read.

## Summary

Phase A of the owner's scale-then-RLVR plan, CORRECTED. The prior WHY curricula
(`qwen35_4b_why_comment_install`, `qwen35_4b_why_scale_ladder`) taught the 4B WHY
each line of a correct solution is correct via inline `#WHY:` comments and left the
`<think>` block minimal. But Qwen3.5-4B is a THINKING model whose coding performance
depends on its `<think>` trace (the repo's most-replicated finding; the shared coding
harness was even mistakenly measuring thinking-OFF, now being fixed to thinking-on +
8192 budget). Training the 4B with an empty/near-empty think target risks DESTROYING
its native thinking. This cell builds the CORRECTED **dual-channel** curriculum: a
GENUINE step-by-step derivation IN the `<think>` block AND the strippable `#WHY:`
comments. Each row teaches the model to think richly (derive the solution and verify
it with a real worked example) in the native channel, then emit clean-but-`#WHY:`-
annotated code. It builds the sha-pinned five-rung ladder (2000/5000/10000/20000/
40000) and the per-rung train/eval sweep, measured THINKING-ON, to find the WHY peak
as the SFT foundation for the RLVR phase — without the empty-think retention hazard.

## Research Program Fit

The program installs real, transferable coding capability into base `Qwen/Qwen3.5-4B`
by designed, contamination-free curricula proven by transfer. Bet #4 (WHY-comment)
was the strongest fast lever but underpowered and comment-only; the scale-ladder
scaled it but with a minimal think block. This cell asks the scaling question the
RIGHT way for a thinking model: with a genuine think derivation trained alongside the
`#WHY:` code, does the WHY signal CLIMB to a peak worth an RLVR foundation, stay flat,
or COLLAPSE — measured thinking-on — while PRESERVING the native thinking the 4B's
coding depends on? HumanEval/MBPP are the fast transfer + retention signal; the
agentic duet-eval is a follow-on confirm on the peak composite.

## Method

- **Dual-channel generator** (`scripts/gen_why_think_curriculum.py`, construction
  seed 95200). `--rows N` produces exactly N verified rows for any N up to ~30000+,
  deterministically. 59 parameterized synthetic families across 13 categories
  (arithmetic accumulation, list reduce, list transform-by-hand, conditional chains,
  parity/modular/digit arithmetic, nested loops, pairwise/adjacent comparisons,
  bounded search, state machines, string manipulation, dict aggregation). Per row:
  - `messages`: a plain `write a function` prompt (spec + signature + public asserts),
    NO instruction to think or comment (both behaviours must be the model's DEFAULT).
  - `think`: a GENUINE forward derivation emitted MECHANICALLY from the family
    AST/shape — parse the spec (goal/inputs/output) -> choose an approach from the
    code SHAPE (accumulator / builder / running-extreme / spread / branch / search /
    dict) phrased as a decision -> build the solution step by step in construction
    order -> trace a REAL worked example (one of the task's asserts, executed line by
    line, values byte-true) -> conclude into the answer. It is NOT the `#WHY:`
    comments joined.
  - `answer`: the CLEAN correct code WITH inline `#WHY:` comments, strippable via the
    distinct marker.
- **Per-row truth audit** (never ship an unverified row), by REAL CPython execution.
  (1) STRIP the `#WHY:` comments -> clean code passes ALL asserts; (2) the COMMENTED
  code runs and passes them IDENTICALLY; (3) the marker is strippable; (4) every
  `#WHY:` is line-specific and varies within the row; (5) the think's worked-example
  trace matches ACTUAL execution — a deterministic rng-free trace CORE
  (`Trace f(args): <var> moves 0 -> 4 -> 12 -> 24, so it returns 24.`) is recomputed
  byte-for-byte at verification and must appear verbatim; (6) the think has an
  approach-decision phrase and is NOT the joined `#WHY:` comments. Safety: restricted
  builtins, no imports/I/O, bounded for-loops only, a step cap. Banned-vocabulary
  self-heal rejects any row (prompt / think / answer) carrying a benchmark name.
- **Contamination firewall** (`scripts/contamination.py`, committed fixture of all
  668 HumanEval + MBPP function names, 663 after the language whitelist). Zero
  whole-word hits over prompt + THINK + answer; a present-only code-only 7-gram aid
  finds zero distinctive shared spans vs the benchmark solutions.
- **Scale ladder** (`scripts/build_ladder.py`). Corpora at 2000/5000/10000/20000/
  40000 rows (fixed seed 95200, different N), each verified + contamination-audited,
  sha-pinned in `data/ladder_manifest.json` (which also pins the generator sha and
  the fixture sha). The corpora are large and deterministically regenerable, so they
  live gitignored under `large_artifacts/`; `--verify` regenerates each rung.
- **Install** (`scripts/train_trial.py` -> vendored `scripts/train_think.py`). One
  fresh r32/a64 adapter per rung, lr 1e-5, batch 1, grad-accum 8, max-length 4096,
  **w_think 0.2** (POSITIVE: preserves + shapes the native thinking — the crux),
  w_close 0.2, seed 95201, from the `base_reserialized` composite (authenticated
  FAIL-CLOSED). Epoch schedule = **1 epoch at every rung** (owner directive):
  unlimited unique data, vary data VOLUME not epochs; optimizer steps = rows / 8 =
  250 / 625 / 1250 / 2500 / 5000.
- **Merge** (vendored `scripts/merge_adapter.py`) with `--base-model` = the base
  composite -> `merged/why_think_<rows>`.
- **Measure** (`scripts/measure_transfer.py` -> SHARED harness, referenced not
  copied). Base and each rung composite, HumanEval 164 + MBPP 200, greedy pass@1,
  identical vLLM path, THINKING-ON (8192 budget). Base CO-MEASURED thinking-on per
  rung; NO hardcoded thinking-off anchor. A SWEEP: all four numbers + paired McNemar
  deltas + rung-vs-base deltas recorded per rung; the orchestrator assembles
  pass@1(rows).

## Results

Pending the sweep. `runs/measure/rung_<rows>.json` will carry
`pass_at_1{base,rung}{humaneval,mbpp}` (base co-measured thinking-on), the pass
counts, the McNemar b/c paired deltas per dataset, and the rung-vs-base problem
deltas; the assembled curve pass@1(rows) locates the peak. Deployable evidence is a
pass@1 gain over the co-measured base (a real code improvement because the grader
ignores comments); the retention guard is the paired dataset staying within tolerance
(and, since thinking is TRAINED not emptied, the native thinking being preserved).

Dual-channel construction facts already established:

- **Diversity.** 5000-row sample: 59/59 families across 13 categories, 1196 distinct
  normalized `#WHY:` templates, **4997/5000 distinct think skeletons** (the
  derivation genuinely varies across families and rows, not one template), 100%
  unique clean programs. 10000-row: 1197 `#WHY:` templates, 9985 distinct think
  skeletons, 100% unique. 20000-row: 100% unique.
- **Contamination**: 663 banned benchmark names after whitelist, 0 whole-word hits
  over every row's prompt + think + answer; 0 distinctive shared 7-grams (78 shared
  structural control-flow idioms) at 10000 rows (HF cache present so the aid RAN).
- **Token budget**: the think lengthens the render, so it is capped. Real pinned-
  tokenizer full render (chat + think + `</think>` + answer) over 5000 rows: **max
  739 tokenizer tokens (median 467, p95 619, min 330) — 0 rows over the 4096 max-
  length cap**, so the trainer's zero-skip contract holds by construction.
  Conservative >=3-char/token estimate: max 695. Character render max 2084.
- **Determinism**: the corpus is a pure function of (seed 95200, N); two rebuilds are
  byte-identical; the ladder `--verify` regenerates every rung to its pinned sha.
- **Unit tests green** (dual-channel: worked-example trace re-executed independently;
  think has approach + trace and is not the joined `#WHY:`; tamper drills refuse a
  corrupted trace value / a think set to the joined comments / a missing approach;
  diversity at 5000/10000/20000; `#WHY:`-truth re-executed by a separate grader;
  safety/termination; contamination zero at 10000; determinism; base auth fail-closed;
  1-epoch schedule; ladder-manifest sha pinning).

## Controls

- Contamination firewall (banned-name audit over prompt + think + answer +
  distinctive code 7-gram overlap), both zero at scale, so any benchmark movement
  cannot be memorization.
- Comments are inert to the execution grader, so any pass@1 gain is a CODE gain — the
  design property that makes this the clean test.
- The think derivation is byte-verified against real execution (the worked-example
  trace's values and final result are recomputed independently), so the taught
  reasoning is TRUE, not a plausible-sounding fabrication.
- The dual channel is itself the control on the empty-think retention hazard: a
  POSITIVE w_think trains a genuine think trace, so a thinking-on retention drop
  (if any) is attributable, not silently baked in by an empty target.
- Base composite authenticated fail-closed (tree + weights) before every rung's
  training and merge.
- Identical measurement path for base and every rung (shared harness), base co-
  measured thinking-on per rung, so deltas are directly comparable.

## Oracle Versus Deployable Evidence

Deployable evidence = a pass@1 gain over the co-measured base on HumanEval/MBPP
(real, held-out `spec -> code` generation, comments ignored by the grader, measured
thinking-on), assembled into the scaling curve. The retention guard (the paired
dataset within tolerance; native thinking preserved) is a control on the forgetting
risk, not a capability claim. The agentic duet-eval is the eventual deployable target
but is a manual follow-on confirm on the peak composite, not gated here. No metric
here uses hidden labels beyond the per-rung pass@1 reads.

## Interpretation

Pending the sweep. A rising-then-peaking curve makes the peak rung the SFT
foundation for the RLVR phase (Phase B) and shows the dual-channel design scales WHY
without the empty-think retention risk; it funds the agentic confirm. A flat curve
reprices the WHY mechanism on this surface and takes RLVR from base. A collapse at the
large rungs bounds the usable dose and makes the small rung the foundation. The
distinctive design goal — teaching WHY WHILE preserving the model's native thinking —
is testable independently as a thinking-on retention read.

## Next Experiments

- Phase B (RLVR) from the peak rung composite as the SFT foundation — the pre-committed
  successor if a rung clears the bar.
- If flat: reprice the WHY family; take RLVR from base.
- If collapse: document the overfit boundary; the small rung is the foundation.
- Follow-on: the agentic duet-eval confirm on the peak composite (thinking-on).

## Artifact Manifest

See `artifact_manifest.yaml` — the rung corpora (large, deterministically
regenerable, sha-pinned in `data/ladder_manifest.json`) and the trained adapters /
merged composites live under `large_artifacts/` (omitted from git); the generator,
contamination fixture, ladder manifest, base provenance copy, and receipts are
in-repo and reproducibility-critical.
