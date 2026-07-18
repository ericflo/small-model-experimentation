# Preregistration: Exec-Trace Installation (Lifecycle 32)

Frozen before any model event. Lifecycle 32 is the FIRST curriculum bet of the
cognitive-core coding program. One fresh adapter trains from BASE in one stage,
one merge publishes, and ONE transfer measurement is read under a frozen
two-directional consequence. A RETENTION_FAIL or NULL outcome is a preserved
finding, never permission to change this contract inside this experiment
directory.

## The bet

The menagerie aggregate was proven NOT to transfer to real agentic coding
(McNemar p=1.00 vs base). The mission pivoted: install REAL coding capability
into BASE `Qwen/Qwen3.5-4B` via designed, contamination-free curricula, proven
by TRANSFER to held-out coding. Base coding baselines (shared fitness harness):
HumanEval 76.2% (a strong single-function coder), MBPP 56.5%, agentic duet-eval
23% (a weak multi-step agent). The gap is agentic/multi-step cognition —
state-tracking across steps.

This cell installs an accurate **mental interpreter** via **execution
tracing**. Random terminating Python programs are generated and RUN with a
tracer to capture ground-truth per-step state; the model trains to reproduce
the trace + final output. This drills state-tracking (the agentic gap), is
SELF-GENERATED and EXECUTION-VERIFIED (respecting the no-larger-teacher
provenance constraint), and looks like NOTHING in HumanEval/MBPP (`code ->
trace`, not `spec -> code`). Any benchmark movement is therefore genuine
TRANSFER.

## The forgetting risk + mitigation (the critical design decision)

Pure trace-only SFT risks catastrophically shifting the model to ALWAYS trace
instead of GENERATE code (cf. answer-only SFT collapsing generation
0.72 -> 0.09). This is the single largest threat to the bet. It is mitigated BY
DESIGN, not by hope:

1. An explicit, distinct instruction ("Trace the following program's
   execution.") that does NOT collide with the `spec -> code` completion prompt
   format the eval uses — the model learns a new, separately-cued behavior
   rather than overwriting code generation.
2. A MODERATE dose: ~400 rows, 1 epoch (50 optimizer steps), r32/a64.
3. A MIXED-difficulty corpus (short/medium/long, biased to medium/long) so the
   task is not monotonous.
4. A prepared retention-mixed switch: `gen_exec_trace_curriculum.py
   --mix-retention R` blends in R self-generated code-COMPLETION rows (from the
   SAME generator, where the model must WRITE code, not trace it). Default OFF —
   the pure-trace probe runs first; if it tanks HumanEval, the mixed variant
   re-runs. The frozen RETENTION_FAIL branch is exactly the trigger for that
   re-run.

## Frozen identities

- Experiment: `qwen35_4b_exec_trace_install` (lifecycle 32).
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Treatment: ONE fresh rank-32/alpha-64 QLoRA adapter (`exec_trace`) trained on
  the FRESH 400-row exec-trace curriculum `data/sft_exec_trace.jsonl` (sha
  `7c5b77ea87438f4fb46b1d6d1b468edb275feee4e915161c9180c109d410e32e`) from the
  `base_reserialized` composite via the trainer's `--model-path`, with the
  frozen recipe: epochs 1.0, lr 1e-5, rank 32, alpha 64, batch 1, grad-accum 8,
  max-length 4096, w_think 0.2, w_close 0.2. Fixed training seed 90211
  (construction seed 90210). 50 optimizer steps (400 / 8).
- The designed curriculum (`scripts/gen_exec_trace_curriculum.py`, seed 90210):
  random terminating Python programs over integer/float/string/list/dict
  assignments, arithmetic, augmented assignment, bounded for/while loops,
  if/elif/else, list append + index-set, dict update, string concat/repeat +
  upper/lower, simple function defs+calls and shallow bounded recursion.
  Difficulty is a FROZEN mixed schedule: short 80 (5-12 steps), medium 160
  (12-25), long 160 (20-47). Each row: prompt = a "Trace the following
  program's execution" framing + the code; think target = the step-by-step
  running-state trace (one line per state change: `<stmt> -> <var>=<val>`);
  answer = the final printed output on one line (`FINAL: <value>`).
- TRIPLE truth audit (never ship an unverified trace):
  1. PRIMARY interpreter — a hand-written diff-emitting tree-walk produces the
     trace, output, and final state.
  2. INDEPENDENT re-execution — the program is rendered to real Python and
     executed by REAL CPython under `sys.settrace` in a restricted namespace
     with a step cap; the trace, output, AND final state are reconstructed from
     actual execution and byte-compared against the primary. A mismatch aborts.
  3. SAFETY/TERMINATION — restricted builtins (no imports, no I/O, no
     filesystem), bounded loops + bounded recursion, a step cap that ABORTS and
     discards the program.
- Contamination firewall (`scripts/contamination.py`): (a) a committed banned
  set of every HumanEval + MBPP function name
  (`data/contamination/banned_function_names.json`, 668 names; Python
  keywords/emitted builtins whitelisted) — ZERO whole-word hits in the corpus;
  (b) a present-only n-gram overlap aid — ZERO shared 7-grams carrying a
  distinctive (non-idiom) token with the benchmark solution code. Row-level
  uniqueness (400 unique prompts, task ids, and program bodies).
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
  `large_artifacts/qwen35_4b_exec_trace_install/merged/exec_trace`.
- Trainer: the vendored `scripts/train_think.py` (sha
  `e0eca2a230dae5d109d418dcb4cc19af05882a770af14350ffd741a8d5e90f01`,
  byte-identical to the chain trainer), wrapped by the fail-closed
  `scripts/train_trial.py`.
- Measurement: the SHARED coding-fitness harness
  `experiments/qwen35_4b_coding_fitness_harness/scripts/eval_pass1.py`
  (referenced, NOT copied): base and the exec_trace composite, both datasets
  (HumanEval 164 + MBPP 200), greedy pass@1, identical vLLM path.
  `scripts/measure_transfer.py` invokes it for both arms x both datasets.

## The frozen two-directional consequence (no third state)

Implemented in `measure_transfer.consequence_reading` and unit-tested over its
truth table (including the retention_fail branch and the exact 0.02 boundary).
Let `he_delta = treat_HE - base_HE`, `mbpp_delta = treat_MBPP - base_MBPP`, tol
= 0.02.

- **RETENTION_FAIL** (priority) iff EITHER dataset regresses past tolerance
  (its drop strictly exceeds 0.02). A regression past tolerance can never be an
  install; this is the forgetting risk realized. Frozen claim: re-run with
  `--mix-retention` before any agentic confirm.
- **INSTALLED_CODING** iff (no regression past tolerance) AND the exec_trace
  composite STRICTLY exceeds base on at least one target dataset (HumanEval OR
  MBPP), with the other retained (>= base - 0.02). Strict improvement uses a
  1e-9 guard; exactly -0.02 on the other dataset still passes retention. Frozen
  claim: the mental-interpreter curriculum transfers; exec_trace becomes the
  program reference for the agentic confirm.
- **NULL** otherwise (no strict improvement and no regression past tolerance).
  Frozen claim: a preserved boundary finding that funds a larger/redesigned
  dose, not a re-roll.

Also recorded, all DESCRIPTIVE, never gating: all four pass@1 numbers, the
per-problem paired deltas (McNemar b/c: only-base-passes / only-treatment-
passes) on each dataset, and the per-clause booleans. The agentic duet-eval is
a follow-on confirm, not gated here.

## Honest priors (computed before any event)

- Precedent FOR: the base ALREADY carries strong latent code semantics
  (HumanEval 76%); state-tracking is a plausibly-latent, drill-able sub-skill;
  the training surface is maximally divergent from the eval, so a gain would be
  real transfer.
- Precedent AGAINST: trace-only SFT is a known collapse hazard (answer-only SFT
  0.72 -> 0.09); a narrow 400-row 1-epoch dose may install the trace behavior
  locally yet not move `spec -> code` generation; HumanEval is already near a
  ceiling for a 4B, so a STRICT gain there is hard, and MBPP is the likelier
  mover.
- P(strict pass@1 gain on >= 1 dataset WITH retention on the other) ≈
  **0.25-0.35**. P(RETENTION_FAIL) ≈ 0.25-0.35 (the forgetting risk is real).
  P(NULL) ≈ 0.35-0.45 (the modal outcome for a first, deliberately-moderate
  dose). Stated plainly: NULL is the single likeliest verdict; it is a FINDING
  that prices the dose, not a failure.

## Standalone and provenance boundary (stated plainly)

This cell trains from a non-hub composite, so it carries the model-reproduction
package IN ITS OWN DIRECTORY per AGENTS.md and `docs/quality_gates.md`. Because
it trains from BASE in ONE stage, the lineage package is minimal:

- `data/provenance/base_reserialized.json` — the IN-CELL sha-pinned copy of the
  base composite's own merge receipt (the hard fail-closed base gate; sha
  `25aee794…`). It pins the method, the model lineage/revision, the tokenizer
  sha, and the weights sha+size.
- `data/sft_exec_trace.jsonl` + `data/curriculum_receipt.json` — the trace
  curriculum and its build receipt (corpus sha, contamination results, stats).
- `data/contamination/banned_function_names.json` — the committed contamination
  fixture (668 benchmark function names), so the banned-name audit is
  standalone. A present-only aid re-derives it from the HF cache and asserts
  equality; absent the cache it is skipped with a recorded note.
- Fixed-seed recipe (seed 90211) + the vendored `scripts/train_think.py` and
  `scripts/merge_adapter.py` (byte-identical, sha-pinned).

Cross-experiment references (the shared fitness harness, the base composite's
sibling records) are VERIFICATION AIDS ONLY and never the reproduction path;
the base composite authenticates against the IN-CELL provenance copy + the
tree/weights hashes. `benchmarks/` contents are never parsed or read as data;
HumanEval/MBPP are executed through the shared harness, never read as training
data.

## Mandatory checkpoint order

1. Model-free construction (this contract, the curriculum, the contamination
   fixture, the tests, the lineage package) — committed, pushed, green.
2. Adversarial compute review — committed `reports/compute_review.md` carrying
   ``**Verdict:** `PASS_CONTROL_TRAINING`.`` → `--stage train`.
3. Training receipt committed; adversarial merge review — committed
   `reports/merge_review.md` carrying ``**Verdict:** `PASS_CONTROL_MERGE`.`` →
   `--stage merge`; then the published-arm hashes filled in `train_trial.py`
   (TODO-PIN) and committed.
4. Merge receipt committed; adversarial measure review — committed
   `reports/measure_review.md` carrying ``**Verdict:** `PASS_MEASURE`.`` →
   `--stage measure` (the transfer-reading stage).

## Interpretation limits

The verdict prices THIS dose (400 rows, 1 epoch, r32/a64) of THIS curriculum
against base at THIS instrument (greedy pass@1 on HumanEval 164 + MBPP 200). An
INSTALLED_CODING verdict makes the new composite the program reference and
funds the agentic duet-eval confirm — it does not itself claim a confirmed
agentic gain. RETENTION_FAIL funds the `--mix-retention` re-run. NULL prices
the dose and funds a larger/redesigned one. Benchmark firewall unchanged:
execution grades, never data reads.
