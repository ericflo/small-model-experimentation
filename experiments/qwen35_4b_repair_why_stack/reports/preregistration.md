# Preregistration: Repair + Why Stack (Lifecycle 36)

Frozen before any model event. Lifecycle 36 is the STACK of the cognitive-core
coding program's TWO positive coding bets. One fresh adapter trains from BASE in
one stage on the UNION of the two committed curricula, one merge publishes, and
ONE transfer measurement is read under a frozen, TIGHTENED two-directional
consequence. A RETENTION_FAIL or NULL outcome is a preserved finding, never
permission to change this contract inside this experiment directory.

## The meta-context: two complementary weak positives

The program is installing REAL coding capability into base Qwen/Qwen3.5-4B via
designed, contamination-free curricula, proven by TRANSFER. Four prior bets:

- Bet #1 — execution-tracing (`qwen35_4b_exec_trace_install`), a PASSIVE skill —
  NULL: reshuffled WHICH tasks the 4B solves without raising the count (HumanEval
  +1, MBPP -3, agentic 8/35 flat).
- Bet #2 — self-repair (`qwen35_4b_self_repair_install`), a LOOP behavior — WEAK
  POSITIVE on the agentic target: HumanEval +3 (76.2->78.0%), MBPP retained,
  agentic 8/35 -> 10/35 (asymmetric 3-vs-1 discordant — adding, not the flat 5v5
  reshuffle of bet #1). Underpowered, not individually significant.
- Bet #4 — why-comment (`qwen35_4b_why_comment_install`), WHY causal reasoning —
  WEAK POSITIVE on the FUNCTION target: HumanEval +5 (76.2->79.3%, the program's
  biggest fast gain; 11-vs-6 discordant, McNemar p=0.33), MBPP -2 (retained),
  agentic 8/35 FLAT (symmetric 3v3). Not individually significant. (Recipe note:
  the standard 1-epoch recipe undertrained the high-entropy `#WHY:` target;
  why_comment was retrained at 4 epochs, loss ~0.05, before measurement.)

The cross-bet finding that motivates THIS cell: self_repair and why_comment are
COMPLEMENTARY and target-SPECIFIC — repair moves the agentic loop, why moves
per-function correctness, and NEITHER regresses the other's target. Two believed-in
positive ingredients pointed at disjoint failure modes.

## The bet: stack the two positive ingredients

Train ONE fresh rank-32/alpha-64 LoRA on the UNION of the two curricula and test
whether the combined effect captures BOTH gains — the HumanEval ~+5 (from WHY) AND
the agentic ~10/35 (from repair) — and clears the significance the individual weak
bets could not. This is LEANER than a fresh curriculum: both source corpora are
already built, verified, and committed; this cell COMBINES them, it does not
regenerate. No new generation means no new contamination risk beyond the union.

## The stack design (frozen)

- The union is the two committed corpora COPIED into this cell
  (`data/source_corpora/`, sha-pinned) and combined by `scripts/build_corpus.py`:
  each source sha is verified fail-closed BEFORE combining, the 504 self_repair +
  504 why_comment non-blank JSONL lines are concatenated in the frozen
  COMBINE_ORDER (self_repair, then why_comment) EXACTLY as their bytes appear (no
  re-serialization), then DETERMINISTICALLY shuffled with `random.Random(93570)`
  so the two kinds INTERLEAVE (block-concatenation would show all of one kind then
  the other; interleaving matters for training). Combined sha
  `2462c93ea2a8dcfbd9413e1c6115ed1456ad438e5dabfdc01e924be6148ddbe5`, 1008 rows,
  504 self_repair + 504 why_comment. The final sha is a pure function of the two
  source shas + the shuffle seed and is verified stable across rebuilds.
- 4-EPOCH recipe (inherited, deliberate): the why_comment rows in the union are
  the high-entropy `#WHY:` causal phrasings that undertrained at 1 epoch (why
  bet #4 was retrained at 4 epochs). The stack contains those same rows, so it
  uses the SAME 4 epochs. Frozen recipe: epochs 4, lr 1e-5, rank 32, alpha 64,
  batch 1, grad-accum 8, max-length 4096, w_think 0.2, w_close 0.2. Fixed training
  seed 93571 (union-build seed 93570). 126 optimizer steps per epoch (1008 / 8).

## Frozen identities

- Experiment: `qwen35_4b_repair_why_stack` (lifecycle 36).
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Treatment: ONE fresh rank-32/alpha-64 QLoRA adapter (`repair_why_stack`) trained
  on the 1008-row UNION `data/sft_repair_why_stack.jsonl` (sha `2462c93e…`) from
  the `base_reserialized` composite via the trainer's `--model-path`, with the
  4-epoch recipe above.
- Source corpora (COPIED into `data/source_corpora/`, the standalone lineage):
  - `sft_self_repair.jsonl` — sha
    `920cb228172677f005bdbc4501f593ce60dc7a9c4f22cbf177f05660ffc392cb`, 504 rows,
    from `qwen35_4b_self_repair_install`.
  - `sft_why_comment.jsonl` — sha
    `040be350678ea0337b8fe0607f783aba9e9071f789471b0ea00f7ce1ebef2962`, 504 rows,
    from `qwen35_4b_why_comment_install`.
- Base composite (training + merge base): `base_reserialized`
  (`large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized`),
  tree `26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677`,
  weights `b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db`
  (9,078,620,536 bytes), authenticated FAIL-CLOSED pre-training against the
  IN-CELL sha-pinned provenance copy `data/provenance/base_reserialized.json`
  (sha `25aee794…`), the per-file cheap checks, the full-tree manifest sha, AND
  the full 9 GB weights hash.
- Merge: through the vendored external merger `scripts/merge_adapter.py` (sha
  `cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672`) with
  `--base-model` = the base_reserialized composite, into
  `large_artifacts/qwen35_4b_repair_why_stack/merged/repair_why_stack`.
- Trainer: the vendored `scripts/train_think.py` (sha
  `e0eca2a230dae5d109d418dcb4cc19af05882a770af14350ffd741a8d5e90f01`,
  byte-identical to the sibling cells' trainer), wrapped by the fail-closed
  `scripts/train_trial.py`.
- Measurement: the SHARED coding-fitness harness
  `experiments/qwen35_4b_coding_fitness_harness/scripts/eval_pass1.py`
  (referenced, NOT copied): base and the repair_why_stack composite, both datasets
  (HumanEval 164 + MBPP 200), greedy pass@1, identical vLLM path. The grader
  IGNORES comments. `scripts/measure_transfer.py` invokes it for both arms x both
  datasets.

## Contamination (re-audited on the union)

Both source corpora were individually contamination-audited (0 banned names, 0
distinctive shared 7-grams). Because there is NO new generation, the union carries
no contamination risk beyond that of its two parents, but it is VERIFIED, not
assumed:

- BANNED FUNCTION NAMES (`scripts/contamination.py` + the committed fixture
  `data/contamination/banned_function_names.json`, 668 benchmark function names,
  663 after the language whitelist; name set byte-identical to both parents'
  fixtures). ZERO whole-word hits over all 1008 rows (prompt + think + answer),
  recorded in `data/stack_corpus_receipt.json` and re-checked in `run.py --smoke`
  via `build_corpus.py --verify-corpus`.
- N-GRAM CODE OVERLAP (present-only aid). The union's executable-code 7-grams are
  the UNION of the two parents' code 7-grams; each parent recorded 0 distinctive
  shared 7-grams (docstring/rationale prose excluded, audited by the banned-name
  gate), so the union shares 0 as well — re-verified over the combined corpus when
  the HF cache is present (`tests/test_contamination.py`, 61 shared spans all pure
  control-flow idioms with no distinctive token).

## The frozen, TIGHTENED two-directional consequence (no third state)

Implemented in `measure_transfer.consequence_reading` and unit-tested over its
truth table (including the >=3-problem threshold, the exact 0.02 boundary, and the
retention_fail branch). IDENTICAL rule to bets #2 and #4. Let `he_gain =
treat_HE_passed - base_HE_passed` and `mbpp_gain` in PROBLEMS; retention is a
FRACTION tolerance of 0.02.

- **RETENTION_FAIL** (priority) iff EITHER dataset's pass@1 regresses past 0.02.
  A regression past tolerance can never be an install — the double-dose generation
  shift realized.
- **INSTALLED_CODING** iff (no regression past tolerance) AND the repair_why_stack
  composite beats base by **>= 3 PROBLEMS** on at least one target dataset
  (HumanEval OR MBPP), with the other retained (>= base - 0.02). Frozen claim: the
  two complementary ingredients combined; the stack becomes the program reference
  and funds the agentic duet-eval confirm.
- **NULL** otherwise (no >= 3-problem gain and no regression past tolerance).
  Frozen claim: stacking did NOT capture even the why_comment +5 on HumanEval;
  combined with a flat agentic confirm this prices the two ingredients' individual
  weak signals as likely noise rather than additive real effects.

Also recorded, all DESCRIPTIVE: all four pass@1 numbers (counts + fractions), the
per-problem paired McNemar b/c deltas per dataset, and the per-clause booleans.

## The two-directional reading (frozen)

The stack succeeds only if it captures BOTH targets. The fast HumanEval/MBPP gate
tests the FUNCTION direction (the why_comment gain). The agentic duet-eval tests
the LOOP direction (the self_repair gain) and is the PRIMARY real target, but is a
MANUAL follow-on confirm (base 8/35, self_repair 10/35, why_comment 8/35), NOT
gated in `measure_transfer`. Explicitly: if BOTH the HumanEval gain (~+5 from WHY)
AND the agentic gain (~10/35 from repair) appear, the stack works and the two
individual weak signals are confirmed real. If the stack is flat on both, they
were likely noise. A fast gate that merely reshuffles with the agentic eval flat
is honestly a NULL for the mission — the lesson from bet #1.

## Honest priors (computed before any event)

- Precedent FOR: two believed-in POSITIVE ingredients, each with a real
  target-specific direction, pointed at DISJOINT failure modes and neither
  regressing the other's target — the textbook case for additivity. Both surfaces
  are maximally divergent from the eval; the why signal is inert to grading (a
  positive read is unconfounded); the 4-epoch recipe is the converged one.
- Precedent AGAINST: neither parent was individually significant, so the true
  effects may be small or partly noise; a double-dose (1008 rows, 4 epochs, mixed
  targets) carries a real interference/forgetting risk (two behaviors competing in
  one rank-32 adapter); HumanEval is near a 4B ceiling.
- **P(INSTALLED_CODING, i.e. a >= 3-problem gain with retention) ~= 0.40.** Above
  each parent's prior (~0.30-0.35) because we combine two believed-in positive
  ingredients on disjoint targets, but still below even odds: the parents were
  weak and the double dose could interfere. P(RETENTION_FAIL) ~= 0.10-0.15.
  P(NULL) ~= 0.45-0.50. Stated plainly: NULL remains a likely verdict and it is a
  FINDING that prices the two ingredients (likely noise), not a failure.

## Pre-committed next move (frozen)

- If INSTALLED_CODING: run the manual agentic duet-eval confirm on the
  repair_why_stack composite — the real test of whether BOTH weak signals are
  additive and real. A confirmed agentic gain (>= 10/35 with an asymmetric
  discordant) alongside the HumanEval gain makes the stack the program reference.
- If RETENTION_FAIL: the double-dose generation shift is realized; reconsider dose
  / mix ratio before any confirm.
- If NULL: do NOT re-roll this exact union. A flat stack (fast AND agentic) is
  strong evidence the two individual weak signals were noise; advance to the
  think-block WHY variant (bet #3, still queued) or a different mechanism rather
  than re-stacking.

## Standalone and provenance boundary (stated plainly)

This cell trains from a non-hub composite and combines two prior corpora, so it
carries the FULL reproduction package IN ITS OWN DIRECTORY per AGENTS.md and the
standalone-experiments doctrine. Because it trains from BASE in ONE stage on a
locally-reproducible union, the lineage package is:

- `data/provenance/base_reserialized.json` — the IN-CELL sha-pinned copy of the
  base composite's own merge receipt (the hard fail-closed base gate; sha
  `25aee794…`).
- `data/source_corpora/sft_self_repair.jsonl` + `sft_why_comment.jsonl` — the two
  COPIED, sha-pinned source corpora (the reproduction inputs, per the doctrine:
  lineage = copied ordered SFT datasets, not cross-experiment references).
- `scripts/build_corpus.py` (union-build seed 93570) — reproduces the combined sha
  from the two copies deterministically; `data/sft_repair_why_stack.jsonl` +
  `data/stack_corpus_receipt.json` (the union + its build receipt).
- `data/contamination/banned_function_names.json` — the committed contamination
  fixture (668 benchmark function names). A present-only aid re-derives it from
  the HF cache and asserts equality; absent the cache it is skipped with a note.
- Fixed-seed recipe (training seed 93571) + the vendored `scripts/train_think.py`
  and `scripts/merge_adapter.py` (byte-identical, sha-pinned).

Cross-experiment references (the shared fitness harness, the origin cells, the base
composite's sibling records) are VERIFICATION AIDS ONLY and never the reproduction
path. `benchmarks/` contents are never parsed or read as data; HumanEval/MBPP are
executed through the shared harness, never read as training data.

## Mandatory checkpoint order

1. Model-free construction (this contract, the union + its build script, the
   contamination re-audit, the tests, the lineage package) — committed, pushed,
   green.
2. Adversarial compute review — committed `reports/compute_review.md` carrying
   ``**Verdict:** `PASS_CONTROL_TRAINING`.`` -> `--stage train`.
3. Training receipt committed; adversarial merge review — committed
   `reports/merge_review.md` carrying ``**Verdict:** `PASS_CONTROL_MERGE`.`` ->
   `--stage merge`; then the published-arm hashes filled in `train_trial.py`
   (TODO-PIN) and committed.
4. Merge receipt committed; adversarial measure review — committed
   `reports/measure_review.md` carrying ``**Verdict:** `PASS_MEASURE`.`` ->
   `--stage measure` (the transfer-reading stage), then the manual agentic confirm.

## Interpretation limits

The verdict prices THIS union (1008 rows, 4 epochs, r32/a64) against base at THIS
instrument (greedy pass@1 on HumanEval 164 + MBPP 200). An INSTALLED_CODING verdict
makes the new composite the program reference and funds the agentic confirm — it
does not itself claim a confirmed agentic gain. RETENTION_FAIL funds a dose/mix
reconsideration. NULL prices the two ingredients' individual weak signals. Benchmark
firewall unchanged: execution grades, never data reads.
