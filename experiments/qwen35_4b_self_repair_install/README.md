# Qwen35 4B Self-Repair Install

**Status:** in-progress · since 2026-07-18 · cognitive-core coding bet #2 (self-repair loop); design frozen, GPU train/merge/measure stages pending staged reviews

## Research Program

- Program: `agentic_breadth_installation` (cognitive-core coding sub-program)
- Program question: can real, transferable coding capability be INSTALLED into
  base Qwen/Qwen3.5-4B by designed, contamination-free curricula, proven by
  transfer to held-out coding — after the menagerie proxy (McNemar p=1.00) AND
  bet #1 (execution-tracing, NULL) both reshuffled coding tasks without raising
  the count?
- Prior anchors: base coding baselines from
  `experiments/qwen35_4b_coding_fitness_harness` — HumanEval 76.2% (strong
  function coder), MBPP 56.5%, agentic duet-eval 23% (weak agent). Bet #1
  (`experiments/qwen35_4b_exec_trace_install`) was NULL: HumanEval +1, MBPP -3,
  agentic 8/35 flat. The observed agentic failure mode is a LOOP failure — one
  shot, a check fails, and the model STOPS instead of verifying and repairing.

## Question

Does installing the CHECK-AND-REPAIR loop via a SELF-REPAIR debugging curriculum
— buggy functions (bug injected by AST mutation, so the fix is known) + their
concrete failing tests -> a localized diagnosis + the corrected code — TRANSFER
to real coding (HumanEval + MBPP pass@1), and (the primary target) move the
agentic duet-eval, without regressing function-writing?

## Hypothesis

Bet #1 showed that installing a PASSIVE cognitive primitive (execution modeling)
reshuffles but does not raise coding capability. Self-repair targets the ACTIVE
failure mode directly — persist through a failed check, diagnose, and fix. The
task ENDS by emitting code (the correction) under a distinct instruction, so it
does not compete with code generation (lower forgetting risk than trace-only).
Because the training data is `buggy-function + failing-test -> corrected-function`
(nothing like `spec -> code`), any HumanEval/MBPP movement is genuine TRANSFER.
Honest prior on a MEANINGFUL install (>= 3-problem gain with retention): ~25-30%;
NULL is the single likeliest verdict and would fund the pre-committed RL pivot.

## Setup

- Model: only `Qwen/Qwen3.5-4B` (rev `851bf6e8…`); ONE fresh r32/a64 QLoRA
  adapter trained from the `base_reserialized` composite in a single stage.
- Dataset/task source: `data/sft_self_repair.jsonl` — 504 debugging episodes
  built by `scripts/gen_self_repair_curriculum.py` (construction seed 91330).
  Every row is triple-verified by real execution (correct passes all tests, buggy
  fails >=1 with a wrong value and crashes on none, correction differs); the bug
  is injected by AST mutation so the fix is known and provenance-clean.
- Train/eval split: training is `buggy-function + failing-test -> corrected code`;
  evaluation is the held-out `spec -> code` HumanEval (164) + MBPP (200) —
  deliberately disjoint surfaces. The agentic duet-eval is the primary real
  target, run manually as a follow-on.
- Baseline: base Qwen/Qwen3.5-4B on the same shared harness.
- Controls: contamination firewall (whole-word banned-benchmark-name audit, zero
  hits; distinctive code n-gram overlap, zero); the base composite is
  authenticated fail-closed (tree + weights) before training.
- Primary metric: greedy pass@1 on HumanEval + MBPP (shared fitness harness,
  `experiments/qwen35_4b_coding_fitness_harness/scripts/eval_pass1.py`,
  referenced not copied).
- Oracle-only metrics: none gate here; the agentic duet-eval is a follow-on
  confirm, not part of this cell's frozen consequence.
- Hidden-label boundary: the frozen, TIGHTENED two-directional consequence
  (INSTALLED_CODING requires a >= 3-problem gain / RETENTION_FAIL / NULL) is read
  once from the four pass@1 numbers; benchmarks are executed, never read as data.

## Run

Smoke (no GPU, no writes):

```bash
python scripts/run.py --smoke
```

GPU stages (each gated behind a staged adversarial review; see
`reports/preregistration.md` for the exact commands and checkpoint order):

```bash
python scripts/run.py --stage train     # r32/a64, 1 epoch, seed 91331
python scripts/run.py --stage merge     # vendored composite merger
python scripts/run.py --stage measure   # shared HumanEval+MBPP harness, both arms
```

Agentic confirm (manual follow-on on the merged composite — the PRIMARY real
target, base 8/35): run the duet-eval gen4 harness with `--model-override` set to
`large_artifacts/qwen35_4b_self_repair_install/merged/self_repair`, exactly as
bet #1's measure review documented for the exec_trace composite.

## Results

Not yet run. The install/merge/measure stages are gated behind staged reviews.
When run, `runs/measure/transfer_summary.json` records all four pass@1 numbers
(base/treatment x HE/MBPP, counts + fractions), the per-problem paired deltas,
and the frozen, tightened verdict. Separate deployable evidence (transfer) from
the retention guard.

## Interpretation

Pending the sealed measurement. INSTALLED_CODING makes self_repair the program's
reference composite and funds the agentic confirm; RETENTION_FAIL realizes the
forgetting risk and funds the `--mix-retention` re-run; NULL is the third
static-SFT curriculum family to reshuffle-without-raising and funds the
pre-committed PIVOT to reinforcement learning on the agentic loop.

## Knowledgebase Update

- Program evidence updated: pending measurement.
- Program backlog updated: pending measurement.
- Claim ledger updated: pending measurement.

## Artifacts

- `scripts/` — self-repair curriculum generator (AST mutation + safe execution),
  contamination module, vendored trainer/merger, fail-closed `train_trial.py`,
  `measure_transfer.py`, `run.py`.
- `data/sft_self_repair.jsonl` — the 504-row curriculum + `curriculum_receipt.json`;
  `data/provenance/base_reserialized.json`; `data/contamination/`.
- `configs/`, `reports/`, `tests/`
- `reports/artifact_manifest.yaml`
