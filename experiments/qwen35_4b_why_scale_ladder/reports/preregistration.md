# Preregistration: WHY Scale Ladder (Phase A of scale-then-RLVR)

Frozen before any model event. This cell builds the SCALE-CAPABLE, high-diversity,
verified WHY-rationale curriculum generator and the scale-ladder train/eval
harness. It is Phase A of the owner's scale-then-RLVR plan: scale the WHY
curriculum to find its PEAK, merge the best rung as an SFT foundation, then RLVR.
The GPU stages (train/merge/measure per rung) are gated behind staged adversarial
reviews and are a SWEEP the orchestrator runs rung-by-rung; a flat or collapsing
curve is a preserved finding, never permission to change this contract.

## The meta-context: why scale the WHY curriculum

The cognitive-core coding program is installing real, transferable coding
capability into base `Qwen/Qwen3.5-4B` by designed, contamination-free curricula.
Four bets ran; the standout was **bet #4, WHY-comment install**
(`experiments/qwen35_4b_why_comment_install`): a single 504-row curriculum where
each meaningful line of a correct solution carries a trailing `#WHY:` comment
giving the true causal reason that line is correct. Because comments are INERT to
the execution grader, it is the cleanest possible test — a pass@1 gain is an
unconfounded CODE improvement. It produced the program's **biggest fast gain:
HumanEval 0.7622 -> 0.7927 (+5 problems)** — but it was UNDERPOWERED (McNemar
p=0.33, not individually significant), flat on the agentic loop, and did not
survive combination. The owner's directive: **SCALE the WHY curriculum to find
the peak before overfit/collapse, then RLVR from the best rung.**

## The blocker this cell removes: generator saturation

The sibling generator SATURATES — only ~75 distinct WHY reasoning templates and
438 unique programs at 504 rows. Naive 20x replay would just repeat data, so any
"scaling" test would overfit and read as a FALSE NEGATIVE. This cell rebuilds the
generator to produce GENUINELY DIVERSE data at scale so the ladder tests REAL
scaling:

- **>= 50 program families** (up from 15). 59 parameterized synthetic families
  span arithmetic accumulation, list reduce, list transform-by-hand, conditional
  chains, parity/modular/digit arithmetic, nested loops, pairwise/adjacent
  comparisons, bounded search, state machines, string manipulation, and dict
  aggregation. (Measured: 59/59 families exercised at 5000 rows.)
- **Unique-program capacity >= 20000.** A rich name/parameter space (44 clean
  function names x list/int/string params x 12 accumulators x per-family constants)
  gives >= 100% unique clean programs in a 20000-row build (raw no-dedup draw
  ~82% unique — the diversity is genuine, not dedup masking).
- **>= 300 distinct NORMALIZED WHY reasoning patterns** (numbers/vars removed). A
  phrase-pool of TRUE, line-specific rationale variants per code construct yields
  ~1180-1197 normalized templates across each rung — an order of magnitude over
  the bar, and over 15x the sibling's ~75.

## Every guarantee of the sibling cell kept, verified PER ROW

By REAL CPython execution, at generation, on every row (any failure -> rejected
and re-drawn):

1. STRIP the `#WHY:` comments -> the resulting CLEAN code passes ALL the task's
   asserts (independently re-executed by a separate assert-based grader).
2. The COMMENTED code runs and passes ALL the same asserts IDENTICALLY (comments
   are inert to Python — verified, not assumed).
3. Every `#WHY:` is TRUE and line-specific — emitted BY CONSTRUCTION (per line,
   the reason the family chose that construct/bound/op), NEVER fabricated and NEVER
   from a teacher model; each references a token on its line and the comments VARY
   within the row (non-boilerplate).
4. The `#WHY:` marker is distinctive and mechanically strippable (`#` never
   appears except as the marker; stripping reproduces the clean code exactly).
5. Safety/termination: restricted builtins, no imports/I/O, only bounded
   for-loops (never `while`), a per-call step cap that ABORTS and DISCARDS.
6. Determinism: the corpus is a pure function of (seed 94100, N); two rebuilds are
   byte-identical.

## Contamination firewall at scale

- **Banned-name audit** (`scripts/contamination.py`, committed fixture
  `data/contamination/banned_function_names.json`, 668 benchmark function names,
  663 after the language whitelist; identical to the sibling cell). ZERO
  whole-word hits over every row (code + spec prose + `#WHY:` prose). Generation
  self-heals: any row that would carry a banned token is rejected. (`power`,
  `longest`, `answer`, `count`, `sort`, ... are all benchmark def names and are
  kept out of the prose vocabulary.)
- **Code-only distinctive 7-gram overlap** (present-only aid). ZERO shared 7-grams
  carrying a distinctive (non-idiom) token between the corpus's executable CODE
  (comments stripped) and benchmark solution code, at 5000 and 10000 rows. The
  accumulator/list-param pools avoid benchmark code idioms (`total`/`res`/`prod`
  and `arr`/`lst`/`nums` are deliberately excluded).
- Prompt = a plain `write a function` framing (spec + signature + tests) with NO
  instruction to comment, so the WHY-writing behavior is the model's DEFAULT and
  fires on a plain HumanEval/MBPP prompt (real transfer).

## Token budget

Every row's full training render (chat + think + `</think>` + answer) is measured
against the pinned tokenizer at **max 499 tokens (median 337, p95 455)** over 5000
rows — well under the 4096 max-length cap; ZERO rows truncate, so the trainer's
zero-skip contract holds by construction.

## The frozen scale ladder

- Rungs (fixed construction seed 94100, different N): **2000, 5000, 10000, 20000**,
  each a deterministic verified + contamination-audited corpus, sha-pinned in
  `data/ladder_manifest.json` (the corpora themselves are large, regenerable, and
  live gitignored under `large_artifacts/`; the manifest is the standalone
  contract, pinning the generator sha, the fixture sha, and each rung's corpus
  sha256).
- Rung shas: 2000 `608192fa…`, 5000 `2a0fb91a…`, 10000 `d038452c…`, 20000
  `e32584d0…`.
- **Treatment per rung**: ONE fresh rank-32/alpha-64 QLoRA adapter (NO warm start)
  trained from the `base_reserialized` composite via the trainer's `--model-path`.
  Recipe (frozen, identical across rungs except epochs): lr 1e-5, rank 32, alpha
  64, batch 1, grad-accum 8, max-length 4096, w_think 0.2, w_close 0.2, training
  seed **94101**.
- **Epoch schedule** (documented choice): larger corpora need fewer epochs, so
  `epochs = max(1, round(8000 / rows))` — **4 / 2 / 1 / 1** at 2000 / 5000 / 10000
  / 20000. This holds total sample exposures roughly comparable (8k / 10k / 10k /
  20k) while letting the model see each rung enough. Optimizer steps = rows *
  epochs / (batch 1 x grad-accum 8) = 1000 / 1250 / 1250 / 2500.
- Base composite (training + merge base): `base_reserialized`
  (`large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized`),
  tree `26d8ee48…`, weights `b654e033…` (9,078,620,536 bytes), authenticated
  FAIL-CLOSED pre-training against the in-cell provenance copy
  `data/provenance/base_reserialized.json` (sha `25aee794…`), the per-file cheap
  checks, the full-tree manifest sha, AND the full 9 GB weights hash — byte-for-byte
  the sibling cell's gate.
- Trainer: vendored `scripts/train_think.py` (sha `e0eca2a2…`, byte-identical to
  the sibling). Merger: vendored `scripts/merge_adapter.py` (sha `cb9af8b4…`).
- Measurement: the SHARED coding-fitness harness
  `experiments/qwen35_4b_coding_fitness_harness/scripts/eval_pass1.py` (referenced,
  NOT copied): base and each rung composite, HumanEval 164 + MBPP 200, greedy
  pass@1, identical vLLM path; the grader IGNORES comments (the clean test).

## The consequence: a SWEEP that finds the peak (no single-shot verdict)

Unlike the sibling's one-shot two-directional consequence, this cell is a SWEEP.
`measure_transfer.py` records, PER RUNG, all four pass@1 numbers (base/rung x
HE/MBPP, counts + fractions), the per-problem paired McNemar b/c deltas, and the
rung-vs-base problem deltas. The orchestrator assembles the curve pass@1(rows) to
locate the PEAK. The frozen scale hypothesis:

- **If WHY is a REAL but underpowered signal**, HumanEval pass@1 should CLIMB with
  real-diverse scale — the +5 at 504 rows was the low-N corner of a rising curve —
  up to a PEAK, then flatten or collapse (overfit to the synthetic surface /
  forgetting). The peak rung is the SFT foundation for RLVR.
- **If the +5 was noise**, the curve is FLAT within paired-test error across all
  rungs — teaching WHY does not scale, a preserved boundary finding that reprices
  the whole WHY family and redirects the RLVR phase.
- **If scale COLLAPSES** function-writing (a monotone decline / retention loss at
  the large rungs), the finding is that this synthetic WHY surface overfits, and
  the peak (likely a small rung) is the foundation while the collapse bounds the
  dose.

The diversity fix is what makes this test valid: because the data is genuinely
diverse at scale (>= 50 families, >= 300 WHY patterns, ~100% unique programs), a
flat curve cannot be blamed on replayed data, and a rising curve is real scaling.

## Honest priors (computed before any event)

- Precedent FOR: the WHY mechanism gave the biggest fast gain on the cleanest
  possible test; the base carries strong code semantics (76.2% HE) to build on;
  the surface is maximally divergent from the eval; the diversity blocker that
  would have faked a null is removed.
- Precedent AGAINST: the 504-row gain was not individually significant; synthetic
  curricula in this program tend to reshuffle-without-raising; larger doses of a
  narrow synthetic surface carry a real overfit/forgetting risk; HumanEval is near
  a 4B ceiling.
- **P(a rung meaningfully beats base on HumanEval, >= +3 problems with retention)
  ~= 0.45**, above the sibling's ~0.35 because scale + real diversity is the
  owner's best lever and the low-N signal already pointed up; **P(a clean rising-
  then-peaking curve) ~= 0.35**; **P(flat within error) ~= 0.35**; **P(collapse at
  the large rungs) ~= 0.30**. A NULL/flat curve remains a likely and INFORMATIVE
  outcome that reprices the WHY family, not a failure.

## Pre-committed next move (frozen)

- If a rung clears the bar: merge the PEAK rung as the SFT foundation and begin the
  RLVR phase from it (Phase B); run the agentic duet-eval confirm on the peak
  composite.
- If the curve is flat: the WHY-comment mechanism does not scale on this surface;
  reprice the WHY family and take the RLVR phase from base (or the sibling's 504
  composite) instead of a scaled WHY foundation.
- If scale collapses: the small rung is the foundation and the collapse bounds the
  usable WHY dose; document the overfit boundary before RLVR.

## Standalone and provenance boundary (stated plainly)

Per AGENTS.md and `docs/quality_gates.md`, the PRODUCTION side is carried in-cell:
the curriculum generator (`scripts/gen_why_scale_curriculum.py`, seed 94100), the
contamination module + committed fixture, the sha-pinned ladder manifest, the
in-cell base provenance copy, the vendored trainer/merger, and the fixed-seed
recipe (seed 94101). The rung CORPORA are large and DETERMINISTICALLY regenerable
from the in-cell generator + seed + N, so they live gitignored under
`large_artifacts/` and are reproduced by `scripts/run.py --stage gen-ladder`; the
committed manifest pins their shas as the verification contract. The MEASUREMENT
side (the shared fitness harness) is repo-level infrastructure referenced in place.
`benchmarks/` contents are never parsed or read as data; HumanEval/MBPP are
executed through the shared harness, never read as training data.

## Mandatory checkpoint order

1. Model-free construction (this contract, the generator, the contamination
   fixture, the ladder manifest, the tests, the lineage package) — committed,
   pushed, green.
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
   `--stage measure --rows N` per rung; assemble the curve; select the peak.

## Interpretation limits

The curve prices THESE doses (2000/5000/10000/20000 rows, the frozen recipe,
r32/a64) of THIS WHY curriculum against base at THIS instrument (greedy pass@1 on
HumanEval 164 + MBPP 200). Locating a peak makes that rung the SFT foundation for
the RLVR phase — it does not itself claim a confirmed agentic gain. The benchmark
firewall is unchanged: execution grades, never data reads.
