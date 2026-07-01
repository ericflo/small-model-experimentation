# Qwen3.5-4B Neurosymbolic REPL Substrate + Failure Profile

## Research Program

- Program: `structured_execution_and_compilers` (extends claim C1: executable intermediates help).
- Mission: **unearth latent capability in the fixed Qwen3.5-4B weights** — no larger models, no
  distillation, no scaling. The lever is **non-stochastic execution feedback**: an interpreter is a
  calculator, not a bigger model, so letting the 4B draft a program, run it, and refine on the real
  result offloads its *weakness* (mentally simulating execution) while its *strength* (writing/debugging
  code) does the work.

## Why a new substrate

MBPP/HumanEval are saturated for this 4B (coverage ~0.89) and contamination-suspect — on a memorized
benchmark you cannot tell whether feedback unearthed reasoning or jogged a memory. So this is built on a
**procedurally-generated, contamination-free substrate**: each task is a random depth-D composition of
~23 total list-of-int primitives, presented as input/output examples; the model synthesizes a Python
`transform(xs)` matching **held-out** examples (graded functionally — any correct program counts). Novel
by construction; difficulty = composition depth. A reference oracle solves 100% of generated tasks, so
every model failure is real.

## Milestones

1. **Substrate + failure profile** (`scripts/run_baseline.py`): frozen 4B, thinking on — greedy@1
   (deployable) vs pass@k (coverage) vs oracle, by depth. Establishes a hard-but-fair substrate with a
   coverage→deployment gap to target.
2. **Neurosymbolic REPL loop** (`scripts/run_repl.py`): draft → execute on visible → real feedback
   (actual vs expected) → refine, ≤T turns; graded on hidden. Controls: **repl_nofb** (multi-turn without
   execution content) isolates whether feedback *content* matters; **sample_more + visible-select** at
   matched compute is the bar (independent sampling beat every trained arm in the corpus). Central
   question: does execution-grounded self-correction beat independent sampling at equal generation budget?
3. **Bank it** (if M2 positive): QLoRA-SFT the 4B on its *own* successful correction trajectories (no
   teacher); test durable single-shot improvement on **held-out fresh** tasks + check for diversity collapse.

## Reuse

- Sandbox `src/code_env.py` (from `qwen35_4b_retrieval_adapt_verify_scale`: AST safety + `-I` subprocess
  isolation + rlimits; visible/hidden protocol; captures actual outputs for feedback). Runtime
  `src/gen_lib.py` (thinking-budget generation). New: `src/gen_tasks.py` (procedural generator + oracle).

## Run

```bash
../../.venv/bin/python scripts/run_baseline.py --per-depth 15 --depths 1 2 3 4 5 6 --k 6   # M1
../../.venv/bin/python scripts/run_repl.py --per-depth 30 --depths <headroom> --turns 5     # M2
```

## Results

Full write-up in [reports/report.md](reports/report.md).

- **M1** (failure profile): substrate oracle-solvable 100% but hard — frozen 4B (thinking) greedy@1 0.156,
  pass@6 0.244; headroom at depths 1–4 (5–6 dead).
- **M2** (REPL loop): **execution feedback does NOT beat matched-compute sampling.** repl_real 0.287 @ 3.9
  gens vs sample_more 0.338 @ 5; the feedback *content* adds only +0.024 over a paired no-feedback control.
  On a contamination-free substrate, the 4B's ceiling is its own sampling distribution — a clean replication
  of "sample-more is hard to beat" with no memorization confound.
- **M3** (banking): **self-training on the 4B's OWN 189 verified solutions (no teacher) improves held-out
  fresh single-shot** — think-greedy@1 0.224 → 0.319 (+0.095, ~2.2 SE over N=210, +42% relative), pass@5 up
  (no diversity collapse), confirmed on two fresh seeds. Works on this contamination-free substrate where
  the corpus's prior MBPP self-improvement regressed.

**Arc:** cleverer *test-time* readout (execution feedback) does not unearth capability, but *self-training*
on verified self-solutions banks it into the weights — and it needs clean, uncontaminated data to show up.
