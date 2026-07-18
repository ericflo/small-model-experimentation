# Qwen35 4B Exec-Trace Install — Report

**Design-frozen report.** The model-free construction is complete and verified;
the train/merge/measure GPU stages are gated behind staged adversarial reviews
and have not run. Results will be appended to the Results section once the
sealed measurement is read.

## Summary

Lifecycle 32 — the FIRST curriculum bet of the cognitive-core coding program.
After the menagerie proxy was shown NOT to transfer to real coding (McNemar
p=1.00 vs base), this cell installs an accurate "mental interpreter" into base
Qwen/Qwen3.5-4B via EXECUTION TRACING: a fresh r32/a64 LoRA trains (1 epoch,
seed 90211) on 400 random terminating Python programs paired with their
execution-verified, step-by-step running-state traces, merges onto base, and is
measured for TRANSFER + RETENTION on the shared HumanEval + MBPP fitness harness
under a frozen two-directional consequence (INSTALLED_CODING / RETENTION_FAIL /
NULL). The training signal is SELF-GENERATED and EXECUTION-VERIFIED (no larger
teacher) and looks like nothing in the benchmarks (`code -> trace`, not `spec ->
code`), so any movement is genuine transfer.

## Research Program Fit

The program's target is the base 4B's agentic/multi-step cognition gap:
HumanEval 76.2% (strong function coder) vs duet-eval 23% (weak agent). The
missing sub-skill is state-tracking across steps. Execution tracing drills that
directly. HumanEval/MBPP serve as the fast transfer + retention signal here; the
agentic duet-eval is the eventual real target (a follow-on confirm, not gated by
this cell).

## Method

- **Curriculum** (`scripts/gen_exec_trace_curriculum.py`, seed 90210). Random
  terminating Python over integer/float/string/list/dict assignments,
  arithmetic, augmented assignment, bounded for/while loops, if/elif/else, list
  append + index-set, dict update, string concat/repeat + upper/lower, simple
  function defs+calls and shallow bounded recursion. Difficulty is a FROZEN
  mixed schedule biased to medium/long: short 80, medium 160, long 160 (400
  total). Each row: prompt = a distinct "Trace the following program's
  execution" instruction + the code; think = the running-state trace, one line
  per state CHANGE (`<stmt> -> <var>=<val>`); answer = `FINAL: <printed output>`.
- **Triple truth audit** (never ship an unverified trace). (1) A hand-written
  diff-emitting primary interpreter produces the trace/output/final-state. (2)
  The program is rendered to real Python and executed by REAL CPython under
  `sys.settrace` in a restricted namespace with a step cap; the
  trace/output/final-state are reconstructed from actual execution and
  byte-compared against the primary — a mismatch aborts and the program is
  discarded. (3) Safety/termination: restricted builtins (no imports/I/O/
  filesystem), bounded loops + recursion, a step cap that aborts runaway
  programs. The committed corpus additionally re-executes end-to-end via
  `--verify-corpus` (used in smoke).
- **Contamination firewall** (`scripts/contamination.py`). A committed banned
  set of all 668 HumanEval + MBPP function names (Python keywords/emitted
  builtins whitelisted) — zero whole-word hits. A present-only n-gram aid — zero
  shared 7-grams carrying a distinctive (non-idiom) token with benchmark
  solution code. Row-level uniqueness (400 unique prompts/task-ids/programs).
- **Install** (`scripts/train_trial.py` → vendored `scripts/train_think.py`).
  One fresh r32/a64 adapter, epochs 1, lr 1e-5, batch 1, grad-accum 8,
  max-length 4096, w_think 0.2, w_close 0.2, seed 90211 (50 optimizer steps),
  from the `base_reserialized` composite. The base is authenticated FAIL-CLOSED
  (in-cell provenance copy + full tree manifest + full 9 GB weights hash) before
  training.
- **Merge** (vendored `scripts/merge_adapter.py`) with `--base-model` = the base
  composite → `merged/exec_trace`.
- **Measure** (`scripts/measure_transfer.py` → SHARED harness, referenced not
  copied). Base and exec_trace, HumanEval 164 + MBPP 200, greedy pass@1,
  identical vLLM path; all four numbers + per-problem paired deltas + the frozen
  verdict recorded.

## Results

Pending the sealed measurement. `runs/measure/transfer_summary.json` will carry
`pass_at_1{base,exec_trace}{humaneval,mbpp}`, the McNemar b/c paired deltas per
dataset, and the frozen consequence. Deployable transfer evidence is the strict
pass@1 gain; the retention guard is the paired dataset staying within tolerance.

Construction facts already established (model-free):

- Corpus sha `7c5b77ea87438f4fb46b1d6d1b468edb275feee4e915161c9180c109d410e32e`,
  400 rows, all `exec_trace`; tiers short 80 / medium 160 / long 160; steps
  5-47, mean ≈ 23; 400 unique programs.
- Contamination: 668 banned benchmark names, 0 whole-word hits; 0 distinctive
  shared 7-grams (34 shared spans, all pure control-flow idioms with no
  distinctive token).
- 51 unit tests green (2 present-only cache aids skip without the HF cache).

## Controls

- Contamination firewall (banned-name audit + distinctive n-gram overlap), both
  zero, so a benchmark movement cannot be memorization.
- Base composite authenticated fail-closed (tree + weights) before training and
  merge; a swapped composite aborts.
- Identical measurement path for both arms (the shared harness), so base and
  treatment pass@1 are directly comparable.

## Oracle Versus Deployable Evidence

Deployable evidence = the strict HumanEval/MBPP pass@1 gain (real, held-out
`spec -> code` generation). The retention guard (the other dataset within 0.02)
is a control on the forgetting risk, not a capability claim. The agentic
duet-eval is the eventual deployable target but is a follow-on confirm, not
gated here. No metric here uses hidden labels beyond the one-shot frozen verdict
read.

## Interpretation

Pending measurement. INSTALLED_CODING: the mental-interpreter curriculum
transfers; exec_trace becomes the program reference and funds the agentic
confirm. RETENTION_FAIL: the forgetting risk is realized; re-run with
`--mix-retention`. NULL: a preserved boundary finding pricing this dose, funding
a larger/redesigned one.

## Next Experiments

- If INSTALLED_CODING: run the agentic duet-eval confirm on the exec_trace
  composite; consider a larger/deeper trace dose.
- If RETENTION_FAIL: re-run the `--mix-retention` variant (retention-mixed
  code-completion blend) and re-measure.
- If NULL: scale the dose (rows/epochs) or redesign the trace surface (e.g.,
  longer programs, more control-flow depth) and re-measure.

## Artifact Manifest

See `artifact_manifest.yaml` — the trained adapter and merged composite live
under `large_artifacts/` (omitted from git); the curriculum, contamination
fixture, base provenance copy, and receipts are in-repo and reproducibility-
critical.
