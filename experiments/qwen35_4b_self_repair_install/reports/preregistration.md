# Preregistration: Self-Repair Installation (Lifecycle 33)

Frozen before any model event. Lifecycle 33 is the SECOND curriculum bet of the
cognitive-core coding program — cognitive-core coding **bet #2**. One fresh
adapter trains from BASE in one stage, one merge publishes, and ONE transfer
measurement is read under a frozen, TIGHTENED two-directional consequence. A
RETENTION_FAIL or NULL outcome is a preserved finding, never permission to
change this contract inside this experiment directory.

## The meta-context: two prior nulls

- The menagerie aggregate was proven NOT to transfer to real agentic coding
  (McNemar p=1.00 vs base).
- Bet #1 — execution-tracing (`experiments/qwen35_4b_exec_trace_install`) — came
  back NULL: HumanEval 76.2%->76.8% (+1 problem, noise near ceiling), MBPP
  56.5%->55.0% (-3), and the agentic duet-eval EXACTLY FLAT (8/35 -> 8/35, with
  5 base-only and 5 exec_trace-only discordant scenarios). Retention held; the
  capability was RESHUFFLED, not RAISED. LAW recorded: **install != convert**
  extends to coding — installing a passive cognitive primitive reshuffles WHICH
  coding tasks are solved, not HOW MANY.

Two independent curriculum families now reshuffle-without-raising. Bet #2 does
two things differently: (1) it targets the AGENTIC LOOP directly rather than a
passive component, and (2) it tightens the transfer rule so a noise-level bump
cannot read as an install.

## The bet

The base 4B is a strong single-function coder (HumanEval 76.2%) but a weak
multi-step agent (duet-eval 23%). The observed agentic failure mode is a LOOP
failure: the model one-shots a multi-step task, a check fails, and it STOPS
instead of verifying and repairing. This cell installs the CHECK-AND-REPAIR loop
directly via a **self-repair debugging curriculum**.

Each row is one debugging episode: a synthetic function with a docstring spec and
concrete `assert` tests, a bug INJECTED by AST mutation, the concrete failing
test (expected vs got), and the instruction "Diagnose the bug and give the
corrected code"; the target is a short localized diagnosis followed by the
corrected function. Because the bug is INJECTED, the fix is KNOWN — the signal is
SELF-GENERATED and EXECUTION-VERIFIED (no larger teacher), respecting the
provenance constraint. The surface looks like NOTHING in HumanEval / MBPP (those
are `spec -> function`; this is `buggy-function + failing-test ->
corrected-function`), so any benchmark movement is genuine TRANSFER.

## The forgetting risk + mitigation

Trace-only SFT (bet #1) was a known collapse hazard because it biases the model
AWAY from generating code. Self-repair is different by construction: the task
ENDS by emitting CODE (the correction) under a distinct instruction, so it does
not compete with code generation. The dose is still kept moderate (504 rows, 1
epoch, r32/a64) and mixed-difficulty, and a `--mix-retention R` switch (default
OFF) blends in R self-generated `spec -> function` rows if a probe regresses
HumanEval. The frozen RETENTION_FAIL branch is the trigger for that re-run.

## Frozen identities

- Experiment: `qwen35_4b_self_repair_install` (lifecycle 33).
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Treatment: ONE fresh rank-32/alpha-64 QLoRA adapter (`self_repair`) trained on
  the FRESH 504-row self-repair curriculum `data/sft_self_repair.jsonl` (sha
  `920cb228172677f005bdbc4501f593ce60dc7a9c4f22cbf177f05660ffc392cb`) from the
  `base_reserialized` composite via the trainer's `--model-path`, with the frozen
  recipe: epochs 1.0, lr 1e-5, rank 32, alpha 64, batch 1, grad-accum 8,
  max-length 4096, w_think 0.2, w_close 0.2. Fixed training seed 91331
  (construction seed 91330). 63 optimizer steps (504 / 8).
- The designed curriculum (`scripts/gen_self_repair_curriculum.py`, seed 91330):
  13 parameterized synthetic function families (clamp, threshold-count,
  scaled-sum, absolute-gap, largest, even-index-sum, factorial, branch-sum,
  scaled-list, spread, nested-triangle, above-average-count, position-weighted
  sum, running-cap, countdown) spanning a FROZEN mixed difficulty schedule:
  short 120, medium 192, long 192 (504 total). A bug is injected by AST mutation
  from a diverse, seeded set (flipped comparison, wrong arithmetic operator,
  swapped operands, off-by-one loop bound, extra `+ 1` on the return, off-by-one
  constant, shifted index) — all seven kinds present in the corpus. Each row:
  prompt = the buggy function (with docstring) + the `assert` tests + the concrete
  first failure (`returned <got>, expected <expected>`) + "Diagnose the bug and
  give the corrected code."; think = a short localized diagnosis (the failing
  case, the buggy line quoted verbatim, the mechanism, the corrected line);
  answer = the corrected function.
- TRIPLE truth audit (never ship an unverified pair), by REAL CPython execution:
  1. the CORRECT function passes ALL its concrete tests;
  2. the BUGGY function fails AT LEAST ONE test with a WRONG VALUE and RAISES on
     NONE (a crashing or behavior-preserving mutation is rejected);
  3. the correction DIFFERS from the buggy code (exactly one changed line), and
     the shipped failure output matches the actual first failing test.
  Safety/termination: restricted builtins (no imports, no I/O), only bounded
  for-loops (never `while`), a per-call step cap that ABORTS and discards.
- Contamination firewall (`scripts/contamination.py`): (a) a committed banned set
  of every HumanEval + MBPP function name
  (`data/contamination/banned_function_names.json`, 668 names; Python
  keywords/emitted builtins whitelisted) — ZERO whole-word hits over all 504 rows;
  (b) a present-only code n-gram overlap aid — ZERO shared 7-grams carrying a
  distinctive (non-idiom) token between the corpus's executable CODE (docstring
  spec prose excluded — prose is governed by the banned-name gate) and the
  benchmark solution code (80 shared spans, all pure control-flow idioms with no
  distinctive token). Row-level uniqueness (504 unique prompts, task ids, and
  buggy/corrected code pairs).
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
  `large_artifacts/qwen35_4b_self_repair_install/merged/self_repair`.
- Trainer: the vendored `scripts/train_think.py` (sha
  `e0eca2a230dae5d109d418dcb4cc19af05882a770af14350ffd741a8d5e90f01`,
  byte-identical to the chain trainer), wrapped by the fail-closed
  `scripts/train_trial.py`.
- Measurement: the SHARED coding-fitness harness
  `experiments/qwen35_4b_coding_fitness_harness/scripts/eval_pass1.py`
  (referenced, NOT copied): base and the self_repair composite, both datasets
  (HumanEval 164 + MBPP 200), greedy pass@1, identical vLLM path.
  `scripts/measure_transfer.py` invokes it for both arms x both datasets.

## The frozen, TIGHTENED two-directional consequence (no third state)

Implemented in `measure_transfer.consequence_reading` and unit-tested over its
truth table (including the >=3-problem threshold, the exact 0.02 boundary, and
the retention_fail branch). Let `he_gain = treat_HE_passed - base_HE_passed` and
`mbpp_gain` in PROBLEMS; retention is a FRACTION tolerance of 0.02.

- **RETENTION_FAIL** (priority) iff EITHER dataset's pass@1 regresses past 0.02.
  A regression past tolerance can never be an install — the forgetting risk
  realized. Frozen claim: re-run with `--mix-retention` before any agentic
  confirm.
- **INSTALLED_CODING** iff (no regression past tolerance) AND the self_repair
  composite beats base by **>= 3 PROBLEMS** on at least one target dataset
  (HumanEval OR MBPP), with the other retained (>= base - 0.02). Frozen claim:
  the check-and-repair curriculum transfers; self_repair becomes the program
  reference for the agentic confirm.
- **NULL** otherwise (no >= 3-problem gain and no regression past tolerance).
  Frozen claim: a preserved boundary finding that funds the pre-committed RL
  pivot, not a re-roll.

This is the fix for bet #1's letter-of-the-law false positive: a +1- or
+2-problem HumanEval bump near ceiling now reads NULL, not INSTALLED. Also
recorded, all DESCRIPTIVE: all four pass@1 numbers (counts + fractions), the
per-problem paired McNemar b/c deltas per dataset, and the per-clause booleans.

## The agentic eval is the PRIMARY real target (manual follow-on)

HumanEval/MBPP are the fast transfer + retention gate. The AGENTIC duet-eval
(base 8/35, 23%) is the PRIMARY real target and the reason for this curriculum,
but is run MANUALLY as a follow-on confirm on the merged composite (it is not
gated in `measure_transfer`). A verdict here that merely reshuffles the fast
signals, with the agentic eval flat, is honestly a NULL for the mission — the
exact lesson from bet #1.

## Honest priors (computed before any event)

- Precedent FOR: self-repair targets the OBSERVED failure mode (stop-instead-of-
  repair) rather than a passive component; the correction is code, so retention
  risk is structurally lower than trace-only; the base already carries strong
  code semantics to build on; the surface is maximally divergent from the eval.
- Precedent AGAINST: two curriculum families already reshuffle-without-raising
  (menagerie, exec-trace), suggesting narrow static-SFT skill-install may have a
  structural limit for coding CAPACITY; a 504-row 1-epoch dose is deliberately
  moderate; HumanEval is near a 4B ceiling; single-line injected bugs may teach
  a narrow pattern that does not generalize to open-ended repair.
- **P(INSTALLED_CODING, i.e. a >= 3-problem gain with retention) ~= 0.25-0.30.**
  P(RETENTION_FAIL) ~= 0.10-0.15 (lower than bet #1 — the correction is code).
  P(NULL) ~= 0.55-0.65 (the modal outcome; the third reshuffle-not-raise). Stated
  plainly: NULL is the single likeliest verdict, and it is a FINDING that prices
  static-SFT install for coding, not a failure.

## Pre-committed pivot if NULL (frozen)

If this bet is NULL, the program has THREE independent static-SFT curriculum
families that reshuffle-without-raising coding capability. The pre-committed next
move is NOT a fourth static-SFT curriculum re-roll. It is a PIVOT to
REINFORCEMENT LEARNING on the agentic loop itself: reward the plan-act-verify-
repair trajectory with execution-verified outcomes (provenance-clean:
self-generated rollouts graded by real test execution, no teacher), so the model
learns the LOOP POLICY (persist and repair) rather than an imitation of a single
repair step. A NULL here is the evidence that funds that pivot.

## Standalone and provenance boundary (stated plainly)

This cell trains from a non-hub composite, so it carries the model-reproduction
package IN ITS OWN DIRECTORY per AGENTS.md and `docs/quality_gates.md`. Because
it trains from BASE in ONE stage, the lineage package is minimal:

- `data/provenance/base_reserialized.json` — the IN-CELL sha-pinned copy of the
  base composite's own merge receipt (the hard fail-closed base gate; sha
  `25aee794…`).
- `data/sft_self_repair.jsonl` + `data/curriculum_receipt.json` — the self-repair
  curriculum and its build receipt (corpus sha, contamination results, stats).
- `data/contamination/banned_function_names.json` — the committed contamination
  fixture (668 benchmark function names). A present-only aid re-derives it from
  the HF cache and asserts equality; absent the cache it is skipped with a note.
- Fixed-seed recipe (seed 91331) + the vendored `scripts/train_think.py` and
  `scripts/merge_adapter.py` (byte-identical, sha-pinned).

Cross-experiment references (the shared fitness harness, the base composite's
sibling records) are VERIFICATION AIDS ONLY and never the reproduction path.
`benchmarks/` contents are never parsed or read as data; HumanEval/MBPP are
executed through the shared harness, never read as training data.

## Mandatory checkpoint order

1. Model-free construction (this contract, the curriculum, the contamination
   fixture, the tests, the lineage package) — committed, pushed, green.
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

The verdict prices THIS dose (504 rows, 1 epoch, r32/a64) of THIS curriculum
against base at THIS instrument (greedy pass@1 on HumanEval 164 + MBPP 200). An
INSTALLED_CODING verdict makes the new composite the program reference and funds
the agentic duet-eval confirm — it does not itself claim a confirmed agentic
gain. RETENTION_FAIL funds the `--mix-retention` re-run. NULL prices static-SFT
install for coding and funds the pre-committed RL pivot. Benchmark firewall
unchanged: execution grades, never data reads.
