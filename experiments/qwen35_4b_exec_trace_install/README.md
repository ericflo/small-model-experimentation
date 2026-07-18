# Qwen35 4B Exec-Trace Install

**Status:** in-progress · since 2026-07-17 · model-free construction frozen (curriculum built + triple-verified, pipeline authenticated, tests green); the train/merge/measure GPU stages are gated behind staged adversarial reviews and have not run.

## Research Program

- Program: `agentic_breadth_installation` (cognitive-core coding sub-program)
- Program question: can real, transferable coding capability be INSTALLED into
  base Qwen/Qwen3.5-4B by designed, contamination-free curricula, proven by
  transfer to held-out coding — after the menagerie proxy was shown NOT to
  transfer (McNemar p=1.00 vs base)?
- Prior anchors: base coding baselines from
  `experiments/qwen35_4b_coding_fitness_harness` — HumanEval 76.2% (strong
  function coder), MBPP 56.5%, agentic duet-eval 23% (weak agent). The gap is
  agentic/multi-step cognition (state-tracking across steps).

## Question

Does installing an accurate "mental interpreter" via EXECUTION TRACING — train
the model to reproduce the execution-verified, step-by-step running-state trace
of random terminating Python programs — TRANSFER to real coding (HumanEval +
MBPP pass@1) without catastrophically shifting the model into trace-mode
(forgetting)?

## Hypothesis

The base 4B can write a single function but cannot track state across steps.
Tracing concrete execution drills exactly that latent gap. Because the training
data is `code -> trace` (nothing like `spec -> code`), any HumanEval/MBPP
movement is genuine TRANSFER, not memorization. A moderate dose (400 rows, 1
epoch) with an explicit distinct instruction should install state-tracking
while retaining function-writing. Honest prior on a strict pass@1 gain with
retention: ~25-35% (a real bet, not a sure thing — the forgetting risk is real,
cf. answer-only SFT 0.72->0.09).

## Setup

- Model: only `Qwen/Qwen3.5-4B` (rev `851bf6e8…`); ONE fresh r32/a64 QLoRA
  adapter trained from the `base_reserialized` composite in a single stage.
- Dataset/task source: `data/sft_exec_trace.jsonl` — 400 random terminating
  Python programs + execution-verified traces, built by
  `scripts/gen_exec_trace_curriculum.py` (construction seed 90210). Every trace
  is triple-verified (primary interpreter, real-CPython `sys.settrace`
  re-execution, safety/termination caps).
- Train/eval split: training is `code -> trace`; evaluation is the held-out
  `spec -> code` HumanEval (164) + MBPP (200) — deliberately disjoint surfaces.
- Baseline: base Qwen/Qwen3.5-4B on the same shared harness.
- Controls: contamination firewall (whole-word banned-benchmark-name audit,
  zero hits; distinctive n-gram overlap, zero); the base composite is
  authenticated fail-closed (tree + weights) before training.
- Primary metric: greedy pass@1 on HumanEval + MBPP (shared fitness harness,
  `experiments/qwen35_4b_coding_fitness_harness/scripts/eval_pass1.py`,
  referenced not copied).
- Oracle-only metrics: none gate here; the agentic duet-eval is a follow-on
  confirm, not part of this cell's frozen consequence.
- Hidden-label boundary: the frozen two-directional consequence
  (INSTALLED_CODING / RETENTION_FAIL / NULL) is read once from the four
  pass@1 numbers; benchmarks are executed, never read as data.

## Run

Smoke (no GPU, no writes):

```bash
python scripts/run.py --smoke
```

GPU stages (each gated behind a staged adversarial review; see
`reports/preregistration.md` for the exact commands and checkpoint order):

```bash
python scripts/run.py --stage train     # r32/a64, 1 epoch, seed 90211
python scripts/run.py --stage merge     # vendored composite merger
python scripts/run.py --stage measure   # shared HumanEval+MBPP harness, both arms
```

## Results

Not yet run. The install/merge/measure stages are gated behind staged reviews.
When run, `runs/measure/transfer_summary.json` records all four pass@1 numbers
(base/treatment x HE/MBPP), the per-problem paired deltas, and the frozen
verdict. Separate deployable evidence (transfer) from the retention guard.

## Interpretation

Pending the sealed measurement. INSTALLED_CODING makes exec_trace the program's
reference composite and funds the agentic confirm; RETENTION_FAIL realizes the
forgetting risk and funds the `--mix-retention` re-run; NULL is a preserved
boundary finding that funds a larger/redesigned dose.

## Knowledgebase Update

- Program evidence updated: pending measurement.
- Program backlog updated: pending measurement.
- Claim ledger updated: pending measurement.

## Artifacts

- `scripts/` — curriculum generator + tracer, contamination module, vendored
  trainer/merger, fail-closed `train_trial.py`, `measure_transfer.py`, `run.py`.
- `data/sft_exec_trace.jsonl` — the 400-row curriculum + `curriculum_receipt.json`;
  `data/provenance/base_reserialized.json`; `data/contamination/`.
- `configs/`, `reports/`, `tests/`
- `reports/artifact_manifest.yaml`
