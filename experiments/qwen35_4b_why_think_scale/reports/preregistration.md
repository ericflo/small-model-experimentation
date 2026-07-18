# Preregistration: Dual-Channel WHY-Think Scale Ladder (Phase A of scale-then-RLVR)

Frozen before any model event. This cell builds the CORRECTED **dual-channel**
WHY curriculum (owner directive) and the scale-ladder train/eval harness. It is
Phase A of the owner's scale-then-RLVR plan: scale the WHY curriculum to find its
PEAK, merge the best rung as an SFT foundation, then RLVR. The GPU stages
(train/merge/measure per rung) are gated behind staged adversarial reviews and are
a SWEEP the orchestrator runs rung-by-rung; a flat or collapsing curve is a
preserved finding, never permission to change this contract.

## Why the dual channel: preserve the model's native thinking

Qwen3.5-4B is a THINKING model whose coding performance depends on its `<think>`
trace — the repo's MOST-REPLICATED finding (the shared coding harness was even
mistakenly measuring thinking-OFF and is being fixed to thinking-on + 8192 budget).
The prior WHY curriculum (`experiments/qwen35_4b_why_scale_ladder`) put its
reasoning in inline `#WHY:` comments and left the `<think>` block a single minimal
line. Training the 4B with an EMPTY/near-empty think target risks DESTROYING its
native thinking — a first-order retention hazard for exactly the capability the
program depends on. The correction: put a GENUINE reasoning derivation IN the
`<think>` block AND keep the strippable `#WHY:` comments. Each row teaches: think
richly (derive the solution and verify it with a real worked example) in the native
channel, then emit clean-but-`#WHY:`-annotated code. Both channels carry TRUE
reasoning; the comments remain strippable for a later anneal. **The base coding
baselines are being re-measured thinking-on; the old thinking-off 76.2% is NOT
carried as an anchor** — the base is co-measured thinking-on for every rung.

## The row format (frozen)

- **`messages`**: a plain `write a function` user prompt (spec + signature + public
  asserts), with NO instruction to think or to comment — both behaviours must be the
  model's DEFAULT so they transfer to a plain eval prompt. The prompt is thinking-
  enabled (the trainer renders the think channel).
- **`think`** (the NEW channel): a GENUINE step-by-step derivation, emitted
  MECHANICALLY from the family/AST construction (NOT the `#WHY:` comments joined). It
  reasons FORWARD from spec to code: (1) parse the spec — goal, inputs, output type;
  (2) choose the approach FROM THE CODE SHAPE (accumulator? builder? running extreme?
  branch? search? dict?) phrased as a decision ("I'll keep one running value and
  update it as I go"); (3) build the solution step by step, in construction order;
  (4) trace a REAL worked example — pick one of the task's asserts, step the
  execution mentally line by line, arrive at the value, and confirm it equals the
  expected output — produced by ACTUALLY EXECUTING the program on that input so it is
  true; (5) conclude into the answer. A test enforces the think is NOT the joined
  `#WHY:` comments and requires the approach-decision and worked-example sections.
- **`answer`**: the CLEAN correct code WITH inline `#WHY:` comments (the by-
  construction per-line rationale), strippable via the distinct `#WHY:` marker.

## Every guarantee of the prior cell kept, verified PER ROW

By REAL CPython execution, at generation, on every row (any failure -> rejected and
re-drawn):

1. STRIP the `#WHY:` comments -> the CLEAN code passes ALL asserts (independently re-
   executed by a separate assert-based grader).
2. The COMMENTED code runs and passes them IDENTICALLY (comments inert to Python).
3. Every `#WHY:` is TRUE, line-specific, non-boilerplate, emitted BY CONSTRUCTION
   (never a teacher model); comments vary within a row.
4. The `#WHY:` marker is distinctive and mechanically strippable.
5. **(NEW) The think's worked-example trace matches ACTUAL execution of that input.**
   The trace CORE is a deterministic, rng-free string (`Trace f(args): <var> moves
   0 -> 4 -> 12 -> 24, so it returns 24.`) recomputed byte-for-byte at verification;
   the traced values and the final result are re-derived and must appear verbatim.
6. **(NEW) The think is a real derivation** — it carries an approach-decision phrase
   and a worked-example trace, and it is NOT the joined `#WHY:` comments (normalized
   inequality enforced).
7. Safety/termination: restricted builtins, no imports/I/O, only bounded for-loops
   (never `while`), a per-call step cap that ABORTS and DISCARDS.
8. Determinism: the corpus is a pure function of (seed 95200, N); byte-identical.
9. **Token budget: the think lengthens the render, so it is CAPPED.** The real
   pinned-tokenizer full render (chat + think + `</think>` + answer) over 5000 rows
   is **max 739 tokens, p95 619, median 467 — 0 rows over the 4096 max-length cap**;
   a per-row `>=3-char/token` estimate rejects any oversize render, so the trainer's
   zero-skip contract holds by construction.

## Contamination firewall at scale (unchanged, now audits the think too)

- **Banned-name audit** (`scripts/contamination.py`, committed fixture of all 668
  HumanEval + MBPP function names, 663 after the language whitelist). ZERO whole-word
  hits over every row's prompt + THINK + answer. Generation self-heals: any row that
  would carry a banned token (including in the derivation prose) is rejected.
- **Code-only distinctive 7-gram overlap** (present-only aid). ZERO shared 7-grams
  carrying a distinctive token between the corpus's executable CODE (comments
  stripped) and benchmark solution code, through 10000 rows (78 structural idioms).
- The prompt never asks for comments or thinking, so both behaviours are the model's
  DEFAULT and transfer to a plain HumanEval/MBPP prompt.

## Measured diversity (why this scaling test is valid)

The dual-channel data is genuinely diverse at scale, so a flat curve cannot be
blamed on replayed data and a rising curve is real scaling:

- **59 families across 13 categories**; 59/59 exercised at 5000 rows.
- **100% unique programs** at 5000 and 20000 rows (raw no-dedup draw high).
- **1196 distinct normalized `#WHY:` templates** (5000 rows; 1197 at 10000).
- **4997/5000 distinct think skeletons** (5000 rows; 9985 at 10000) — the think
  derivation genuinely varies across families and rows, NOT one template.

## The frozen scale ladder

- Rungs (fixed construction seed 95200, different N): **2000, 5000, 10000, 20000,
  40000**, each a deterministic verified + contamination-audited corpus, sha-pinned
  in `data/ladder_manifest.json` (the corpora are large, regenerable, gitignored
  under `large_artifacts/`; the manifest pins the generator sha, the fixture sha, and
  each rung's corpus sha256).
- **Treatment per rung**: ONE fresh rank-32/alpha-64 QLoRA adapter (NO warm start)
  from the `base_reserialized` composite via the trainer's `--model-path`. Recipe
  (frozen, identical across rungs): lr 1e-5, rank 32, alpha 64, batch 1, grad-accum
  8, max-length 4096, **w_think 0.2** (a POSITIVE think-loss weight preserves AND
  shapes the native thinking — the crux of this cell), w_close 0.2, training seed
  **95201**.
- **Epoch schedule: ALWAYS 1 EPOCH at every rung** (owner directive). With unlimited
  unique data the scale variable is pure DATA VOLUME — every step sees fresh data, no
  re-showing, no memorization, no epoch confound. Optimizer steps = rows / (batch 1 x
  grad-accum 8) = 250 / 625 / 1250 / 2500 / 5000.
- Base composite (training + merge base): `base_reserialized`, tree `26d8ee48…`,
  weights `b654e033…` (9,078,620,536 bytes), authenticated FAIL-CLOSED pre-training
  against the in-cell provenance copy `data/provenance/base_reserialized.json` (sha
  `25aee794…`), the per-file cheap checks, the full-tree manifest sha, AND the full
  9 GB weights hash.
- Trainer: vendored `scripts/train_think.py` (sha `e0eca2a2…`, byte-identical).
  Merger: vendored `scripts/merge_adapter.py` (sha `cb9af8b4…`).
- Measurement: the SHARED coding-fitness harness
  `experiments/qwen35_4b_coding_fitness_harness/scripts/eval_pass1.py` (referenced,
  NOT copied), being fixed to thinking-on + 8192 budget: base and each rung
  composite, HumanEval 164 + MBPP 200, greedy pass@1, identical vLLM path; the
  grader IGNORES comments (the clean test). The measurement WILL BE thinking-on.

## The consequence: a SWEEP measured thinking-on (no single-shot verdict)

`measure_transfer.py` records, PER RUNG, all four pass@1 numbers (base/rung x
HE/MBPP, counts + fractions), the per-problem paired McNemar b/c deltas, and the
rung-vs-base problem deltas — with the base CO-MEASURED thinking-on. The
orchestrator assembles the curve pass@1(rows). The frozen scale hypothesis, measured
thinking-on:

- **If dual-channel WHY is a REAL scaling signal**, HumanEval/MBPP pass@1 CLIMBS with
  scale to a PEAK, then flattens or collapses. The peak rung is the SFT foundation
  for RLVR. Crucially, because thinking is TRAINED (not emptied), the retention read
  (thinking-on pass@1 not dropping) should hold across rungs.
- **If flat**, the curve is FLAT within paired-test error — dual-channel WHY does not
  scale on this surface, a preserved boundary finding that reprices the WHY family
  and redirects the RLVR phase.
- **If it COLLAPSES**, the synthetic surface overfits and the peak (likely a small
  rung) is the foundation while the collapse bounds the dose.

## Honest priors (computed before any event)

- Precedent FOR: the WHY mechanism gave the biggest fast gain on the cleanest test;
  the dual channel removes the empty-think retention hazard that threatened the
  comment-only version; the surface is maximally divergent from the eval; the
  diversity blocker that would have faked a null is removed.
- Precedent AGAINST: the 504-row gain was not individually significant; synthetic
  curricula in this program tend to reshuffle-without-raising; larger doses of a
  narrow synthetic surface carry overfit/forgetting risk; HumanEval is near a 4B
  ceiling.
- **P(a rung meaningfully beats base on HumanEval, >= +3 problems with retention,
  thinking-on) ~= 0.40**; **P(a clean rising-then-peaking curve) ~= 0.30**; **P(flat
  within error) ~= 0.40**; **P(collapse at the large rungs) ~= 0.25**. A NULL/flat
  curve remains a likely, INFORMATIVE outcome that reprices the WHY family. The
  primary design win — thinking PRESERVED while teaching WHY — is testable
  independently as a retention read.

## Pre-committed next move (frozen)

- If a rung clears the bar: merge the PEAK rung as the SFT foundation and begin the
  RLVR phase from it (Phase B); run the agentic duet-eval confirm on the peak.
- If flat: dual-channel WHY does not scale on this surface; reprice the WHY family
  and take RLVR from base.
- If collapse: the small rung is the foundation and the collapse bounds the dose.

## Standalone and provenance boundary

The PRODUCTION side is carried in-cell: the generator
(`scripts/gen_why_think_curriculum.py`, seed 95200), the contamination module +
committed fixture, the sha-pinned ladder manifest, the in-cell base provenance copy,
the vendored trainer/merger, and the fixed-seed recipe (seed 95201). The rung CORPORA
are large and deterministically regenerable from the in-cell generator + seed + N, so
they live gitignored under `large_artifacts/` and are reproduced by
`scripts/run.py --stage gen-ladder`; the committed manifest pins their shas. The
MEASUREMENT side (the shared fitness harness) is repo-level infrastructure referenced
in place. `benchmarks/` contents are never parsed or read as data; HumanEval/MBPP are
executed through the shared harness, never read as training data.

## Mandatory checkpoint order

1. Model-free construction (this contract, the generator, the contamination fixture,
   the ladder manifest, the tests, the lineage package) — committed, pushed, green.
2. `--stage gen-ladder` regenerates the corpora locally from the committed manifest
   (CPU, model-free, verified against the pinned shas).
3. Adversarial compute review — committed `reports/compute_review.md` carrying
   ``**Verdict:** `PASS_CONTROL_TRAINING`.`` -> `--stage train --rows N` per rung.
4. Training receipt committed; adversarial merge review — committed
   `reports/merge_review.md` carrying ``**Verdict:** `PASS_CONTROL_MERGE`.`` ->
   `--stage merge --rows N`; then the published-rung hashes filled in
   `train_trial.py` (TODO-PIN) and committed.
5. Merge receipt committed; adversarial measure review — committed
   `reports/measure_review.md` carrying ``**Verdict:** `PASS_MEASURE`.`` ->
   `--stage measure --rows N` per rung (thinking-on); assemble the curve; select the
   peak.

## Interpretation limits

The curve prices THESE doses (2000/5000/10000/20000/40000 rows, the frozen recipe,
r32/a64, 1 epoch) of THIS dual-channel WHY curriculum against base at THIS instrument
(greedy pass@1 on HumanEval 164 + MBPP 200, thinking-on). Locating a peak makes that
rung the SFT foundation for the RLVR phase — it does not itself claim a confirmed
agentic gain. The benchmark firewall is unchanged: execution grades, never data reads.
