# Qwen35 4B Self-Repair Install — Report

**Design-frozen report.** The model-free construction is complete and verified;
the train/merge/measure GPU stages are gated behind staged adversarial reviews
and have not run. Results will be appended to the Results section once the sealed
measurement is read.

## Summary

Lifecycle 33 — the SECOND curriculum bet of the cognitive-core coding program
(cognitive-core coding bet #2). Bet #1 (execution-tracing) came back NULL: it
reshuffled which coding tasks the 4B solves without raising the count on
HumanEval, MBPP, or the agentic duet-eval (8/35 -> 8/35). The observed agentic
failure mode is a LOOP failure — the model one-shots a multi-step task, a check
fails, and it STOPS instead of verifying and repairing. This cell installs the
CHECK-AND-REPAIR loop directly: a fresh r32/a64 LoRA trains (1 epoch, seed 91331)
on 504 debugging episodes, each a synthetic function with a bug INJECTED by AST
mutation plus its concrete failing test, targeting a localized diagnosis and the
corrected code; it merges onto base and is measured for TRANSFER + RETENTION on
the shared HumanEval + MBPP fitness harness under a frozen, TIGHTENED
two-directional consequence (INSTALLED_CODING requires a >= 3-problem gain). The
training signal is SELF-GENERATED and EXECUTION-VERIFIED (we injected the bug, so
we know the fix; every pair is confirmed by real execution) and looks like
nothing in the benchmarks (buggy-function + failing-test -> corrected-function,
not spec -> code), so any movement is genuine transfer.

## Research Program Fit

The program's target is the base 4B's agentic/multi-step cognition gap: HumanEval
76.2% (strong function coder) vs duet-eval 23% (weak agent). Bet #1 showed that
installing a PASSIVE cognitive primitive (execution modeling) reshuffles but does
not raise coding capability. Bet #2 targets the ACTIVE failure mode — persistence
and self-correction after a failed check — which is the loop the agentic eval
actually exercises. HumanEval/MBPP serve as the fast transfer + retention signal;
the agentic duet-eval (base 8/35) is the PRIMARY real target, run manually as a
follow-on (not gated by this cell).

## Method

- **Curriculum** (`scripts/gen_self_repair_curriculum.py`, seed 91330). 13
  parameterized synthetic function families over integer/list inputs
  (thresholded counts, scaled sums, factorial/product, largest, spread,
  even-index sum, position-weighted sum, nested-triangle sum, above-average
  count, running-cap, countdown, clamp, absolute-gap). A FROZEN mixed schedule
  biased to medium/long: short 120, medium 192, long 192 (504 total). For each
  row a correct function + concrete `assert` tests are generated and executed; a
  bug is injected by a single AST mutation from a diverse, seeded set (flipped
  comparison, wrong arithmetic operator, swapped operands, off-by-one loop bound,
  extra `+ 1` on the return, off-by-one constant, shifted index — all seven kinds
  present). Each row: prompt = the buggy function (with docstring) + the tests +
  the concrete first failure (`returned <got>, expected <expected>`) + "Diagnose
  the bug and give the corrected code."; think = a short localized diagnosis (the
  failing case, the buggy line quoted verbatim, the mechanism, the corrected
  line); answer = the corrected function.
- **Triple truth audit** (never ship an unverified pair), by REAL CPython
  execution. (1) the correct function passes ALL its tests; (2) the buggy
  function fails AT LEAST ONE test with a WRONG VALUE and RAISES on NONE (a
  crashing or behavior-preserving mutation is rejected); (3) the correction
  differs from the buggy code (exactly one changed line) and the shipped failure
  matches the actual first failing test. Safety/termination: restricted builtins
  (no imports/I/O), only bounded for-loops (never `while`), a per-call step cap
  that aborts runaway code. The committed corpus additionally re-executes
  end-to-end via `--verify-corpus` (used in smoke), and the unit tests
  independently re-grade every row with a separate assert-based grader.
- **Contamination firewall** (`scripts/contamination.py`). A committed banned set
  of all 668 HumanEval + MBPP function names (Python keywords/emitted builtins
  whitelisted) — zero whole-word hits. A present-only code n-gram aid — zero
  shared 7-grams carrying a distinctive (non-idiom) token between the corpus's
  executable code (docstring spec prose excluded — governed by the banned-name
  gate) and benchmark solution code. Row-level uniqueness (504 unique
  prompts/task-ids/code-pairs).
- **Install** (`scripts/train_trial.py` -> vendored `scripts/train_think.py`).
  One fresh r32/a64 adapter, epochs 1, lr 1e-5, batch 1, grad-accum 8, max-length
  4096, w_think 0.2, w_close 0.2, seed 91331 (63 optimizer steps), from the
  `base_reserialized` composite. The base is authenticated FAIL-CLOSED (in-cell
  provenance copy + full tree manifest + full 9 GB weights hash) before training.
- **Merge** (vendored `scripts/merge_adapter.py`) with `--base-model` = the base
  composite -> `merged/self_repair`.
- **Measure** (`scripts/measure_transfer.py` -> SHARED harness, referenced not
  copied). Base and self_repair, HumanEval 164 + MBPP 200, greedy pass@1,
  identical vLLM path; all four numbers (counts + fractions) + per-problem paired
  deltas + the frozen, tightened verdict recorded.

## Results

Pending the sealed measurement. `runs/measure/transfer_summary.json` will carry
`pass_at_1{base,self_repair}{humaneval,mbpp}`, the pass counts, the McNemar b/c
paired deltas per dataset, and the frozen consequence. Deployable transfer
evidence is a >= 3-problem pass@1 gain; the retention guard is the paired dataset
staying within 0.02.

Construction facts already established (model-free):

- Corpus sha `920cb228172677f005bdbc4501f593ce60dc7a9c4f22cbf177f05660ffc392cb`,
  504 rows, all `self_repair`; tiers short 120 / medium 192 / long 192; 3-4 tests
  per row; 504 unique buggy/corrected code pairs.
- Mutation-kind spread (all seven present): arith_op 123, const_offset 107,
  return_offset 107, compare_op 83, range_bound 65, index_shift 10, operand_swap
  9.
- Contamination: 668 banned benchmark names, 0 whole-word hits; 0 distinctive
  shared 7-grams (80 shared spans, all pure control-flow idioms).
- 55 unit tests green (present-only cache aids RUN with the HF cache; every row
  independently re-executed: buggy fails >=1 with a wrong value and crashes on
  none, corrected passes all, they differ).

## Controls

- Contamination firewall (banned-name audit + distinctive code n-gram overlap),
  both zero, so a benchmark movement cannot be memorization.
- Base composite authenticated fail-closed (tree + weights) before training and
  merge; a swapped composite aborts.
- Identical measurement path for both arms (the shared harness), so base and
  treatment pass@1 are directly comparable.
- Tightened consequence rule: a noise-level (<3-problem) bump reads NULL, fixing
  bet #1's letter-of-the-law false positive.

## Oracle Versus Deployable Evidence

Deployable evidence = a >= 3-problem HumanEval/MBPP pass@1 gain (real, held-out
`spec -> code` generation). The retention guard (the other dataset within 0.02)
is a control on the forgetting risk, not a capability claim. The agentic duet-eval
is the eventual deployable target but is a manual follow-on confirm, not gated
here. No metric here uses hidden labels beyond the one-shot frozen verdict read.

## Interpretation

Pending measurement. INSTALLED_CODING: the check-and-repair curriculum transfers;
self_repair becomes the program reference and funds the agentic confirm.
RETENTION_FAIL: the forgetting risk is realized; re-run with `--mix-retention`.
NULL: the third static-SFT curriculum family to reshuffle-without-raising — a
preserved boundary finding that funds the pre-committed RL pivot to the agentic
loop policy, not a fourth static-SFT re-roll.

## Next Experiments

- If INSTALLED_CODING: run the agentic duet-eval confirm on the self_repair
  composite; consider a larger/deeper repair dose.
- If RETENTION_FAIL: re-run the `--mix-retention` variant and re-measure.
- If NULL: execute the pre-committed PIVOT to reinforcement learning on the
  agentic plan-act-verify-repair loop (self-generated rollouts graded by real
  test execution, provenance-clean), rather than another static-SFT curriculum.

## Artifact Manifest

See `artifact_manifest.yaml` — the trained adapter and merged composite live
under `large_artifacts/` (omitted from git); the curriculum, contamination
fixture, base provenance copy, and receipts are in-repo and reproducibility-
critical.
