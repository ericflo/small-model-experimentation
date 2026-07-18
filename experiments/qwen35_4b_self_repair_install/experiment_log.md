# Qwen35 4B Self-Repair Install Experiment Log

## 2026-07-18 — model-free construction frozen

Lifecycle 33, the SECOND curriculum bet of the cognitive-core coding program
(cognitive-core coding bet #2). Mission: install real coding capability into base
Qwen/Qwen3.5-4B via a designed, contamination-free curriculum, proven by
transfer. Meta-context: the menagerie proxy did NOT transfer (McNemar p=1.00),
and bet #1 (execution-tracing) was NULL — it reshuffled which coding tasks the 4B
solves without raising the count (HumanEval +1, MBPP -3, agentic 8/35 flat). Two
curriculum families now reshuffle-without-raising.

The bet: install the CHECK-AND-REPAIR loop directly, targeting the observed
agentic failure mode (one shot, a check fails, the model STOPS instead of
verifying and repairing). Training signal: buggy function (bug injected by AST
mutation, so the fix is KNOWN) + its concrete failing test -> a localized
diagnosis + the corrected code. Self-generated and execution-verified (no larger
teacher), disjoint from HumanEval/MBPP.

Built and verified (no GPU, no commit):

- `scripts/gen_self_repair_curriculum.py` (seed 91330) — 504 debugging episodes
  across 13 parameterized synthetic function families, TRIPLE-verified by real
  CPython execution (correct passes all tests; buggy fails >=1 with a WRONG VALUE
  and crashes on none; correction differs, exactly one changed line; the shown
  failure matches the actual first failing test). Corpus sha
  `920cb228172677f005bdbc4501f593ce60dc7a9c4f22cbf177f05660ffc392cb`; tiers short
  120 / medium 192 / long 192; all seven mutation kinds present (arith_op 123,
  const_offset 107, return_offset 107, compare_op 83, range_bound 65, index_shift
  10, operand_swap 9); 504 unique code pairs. A `--mix-retention R` switch
  (default OFF) is prepared as the forgetting guard (blends self-generated
  `spec -> function` rows).
- `scripts/contamination.py` + `data/contamination/banned_function_names.json`
  (668 benchmark function names) — 0 whole-word hits; 0 distinctive shared
  7-grams between the corpus's executable CODE (docstring spec prose excluded —
  governed by the banned-name gate; excluded WITHOUT creating false cross-line
  grams) and benchmark solution code (80 shared spans, all pure control-flow
  idioms).
- Vendored `scripts/train_think.py` (sha e0eca2a2…) and `scripts/merge_adapter.py`
  (sha cb9af8b4…), byte-identical to the chain trainer/merger.
- `scripts/train_trial.py` — fail-closed base authentication (in-cell provenance
  copy + tree manifest + full weights hash); recipe r32/a64, 1 epoch, lr 1e-5,
  seed 91331 (63 optimizer steps).
- `scripts/measure_transfer.py` — invokes the shared fitness harness for both
  arms x both datasets; frozen, TIGHTENED INSTALLED_CODING (>= 3-problem gain) /
  RETENTION_FAIL / NULL consequence (fixes bet #1's letter-of-the-law +1-problem
  false positive).
- `scripts/run.py` — checkpointed `--smoke | --stage train | --stage merge |
  --stage measure`; each GPU stage gated behind a staged adversarial review.
- 55 unit tests green (present-only HF-cache aids RUN with the cache; every row
  independently re-executed by a separate assert-based grader); `run.py --smoke`
  green; boundary drills refuse.

Grep-fresh note: construction seed 91330 and training seed 91331 are fully fresh
repo-wide as SEEDS (the only textual matches are incidental substrings inside
unrelated hashes/floats). No training-seed collision.

Pre-committed pivot: if this bet is NULL, the next move is NOT a fourth
static-SFT curriculum — it is a PIVOT to reinforcement learning on the agentic
plan-act-verify-repair loop (self-generated rollouts graded by real test
execution, provenance-clean).

GPU stages (train/merge/measure) are pending their staged reviews.
