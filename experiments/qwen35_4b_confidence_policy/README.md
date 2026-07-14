# Qwen3.5-4B: The Compute-Optimal Confidence Policy (select + abstain + escalate)

**Status:** finished

The confidence arc established three levers in isolation: confidence-select beats
majority vote (C41), the single-token P(True) judge beats sequence mean-logprob
(C46), and answer-token probability predicts correctness for abstention (C40).
None were ever fused into one deployable decision policy, and none were plotted
against sample-more at **equal compute** — the capstone C41/C46 explicitly owed.

This experiment builds that policy on the fixed Qwen3.5-4B and asks: for a
low-confidence (abstained) task, is compute better spent on **more breadth**
(more samples) or **more serial depth** (a higher think budget on the same
model)?

## What it does

**Part 1 — select + abstain frontier (post-hoc, zero new inference).**
`scripts/frontier.py` reuses the cached 244-task MBPP candidate pool from
`qwen35_4b_code_confidence` (9 candidates/task, each with `full_pass` ground
truth, `p_true`, `mean_logprob`, `behavior_signature`). It sweeps the compute
level k and compares selection policies at matched compute, plus the
conf+abstain risk-coverage curve.

**Part 2 — the escalation arm (new think-mode generation).**
`../qwen35_4b_code_confidence/scripts/gen_budget.py` generates think-mode MBPP
candidate pools at two think budgets (256, tight; 2048, generous) over the same
120 tasks, each candidate scored `full_pass` (hidden-assert execution), `p_true`
(no-think C10 judge), with `n_think` recorded for token-matched compute
accounting. `scripts/escalation.py` then compares, at matched token-compute and
with verifier-free confidence-select, whether escalating the abstained tasks to
budget 2048 beats spending the same compute on more breadth at budget 256.

## Result

**The capstone — compute-optimal = confidence-gated adaptive ALLOCATION.** The
deployable answer to "compute-optimal confidence policy" is to spend samples only
where they help: sample greedily once, read its single-token P(True); if high,
**commit** (1 sample); if low, **sample K and conf-select**; abstain below a
floor. This reaches full-pool MBPP accuracy (0.762) at **~4.25 average samples vs
9 for uniform** sampling — a **~2× compute saving**, strictly beating uniform at
7/9 operating points (and 7/9 on HumanEval too). One logit drives selection,
abstention, *and* allocation. (`scripts/adaptive_compute.py`)

**Why it works — the difficulty curve.** Pooling MBPP+HumanEval (312 tasks) and
binning by per-task pass rate, the confidence-select edge over majority-vote
concentrates on hard tasks: **hard** (pass 0.08) +0.045, **medium** (0.54) +0.038,
**easy** (0.97) −0.007. Selection pays exactly where the model is uncertain — the
same tasks abstention flags — so a confidence-gated allocator is self-consistent.
(`scripts/difficulty_curve.py`)

**The verifier-free pieces that hold:**
- **Confidence-select** — argmax single-token P(True) is the best verifier-free
  selector on moderate-difficulty MBPP (k=9: 0.762 > majority 0.742 > logprob
  0.725; per-cand AUROC 0.77), but only *difficulty-dependently*: it **ties**
  majority on easy HumanEval (base pass 0.91, both 0.941).
- **Abstention** — max-P(True) gives a clean risk-coverage curve (68% coverage →
  0.866; solvability AUROC 0.72).

**The null (honest correction).** The tempting *escalate* step — spend the extra
compute on more think budget (depth) for the flagged-hard tasks — was a small-sample
win (n=24: 0.458 vs 0.304) that **did NOT replicate at n=400** (esc−breadth
+0.004…+0.022, every 95% bootstrap CI spanning 0). MBPP nearly saturates the think
budget (median ~86 think tokens at both budget 256 and 2048); the budget-bound tail
is capability-hard, not merely compute-starved. So the compute lever is **breadth
allocation, not depth escalation**.

Codified as **C57**. Part 2 was regenerated on vLLM (~10× faster than the initial
HF pass) at n=400 with bootstrap CIs.

## Layout (all analyses CPU-only, post-hoc)

- `scripts/adaptive_compute.py` — **the capstone**: confidence-gated adaptive
  allocation vs uniform sampling (`--src mbpp|humaneval`).
- `scripts/difficulty_curve.py` — conf-select vs majority by task difficulty.
- `scripts/frontier.py` — selection frontier + abstention risk-coverage
  (`--src mbpp|humaneval`).
- `scripts/escalation.py` — escalation(depth)-vs-breadth with bootstrap CIs.
- `../qwen35_4b_code_confidence/scripts/vllm_gen_budget.py` — **fast** vLLM
  think-mode multi-budget generation (GPU); `gen_budget.py` is the older HF one.
- `runs/*.json` — aggregate results (adaptive_compute, difficulty_curve, frontier,
  escalation) and the two vLLM candidate pools (per-candidate fields only, no
  menagerie contents).
