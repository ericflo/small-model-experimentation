# Qwen3.5-4B: The Compute-Optimal Confidence Policy (select + abstain + escalate)

**Status:** in-progress · since 2026-07-13 · powering up the escalation subset n and a menagerie-arbitrated deployment check (C57 next-tests)

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

**Survives (robust):**
- **Confidence-select is the best verifier-free selector at every k** (k=9:
  conf 0.762 > majority 0.742 > logprob 0.725 > greedy 0.701; per-candidate AUROC
  0.77 on the vLLM pools; execution-line 0.840 when a visible test exists; oracle
  0.848).
- **Abstention gives a clean risk-coverage curve** (100% coverage 0.757 → 68%
  0.866 → 43% 0.902; solvability AUROC 0.72).
- **Depth modestly beats breadth on the overall frontier** — pure-2048 0.593 >
  pure-256 0.581 at matched high-k.

**Reversed on power-up (the honest correction):**
- The first-pass n=24–60 result — that *selectively escalating* the abstained
  tail to a higher think budget beats matched-compute breadth (hardest-20%: 0.458
  vs 0.304) — **did NOT replicate at n=400**: esc−breadth is +0.004…+0.022 with
  every 95% bootstrap CI spanning 0. It was a small-sample artifact, caught by
  this claim's own pre-registered power-up. Likely mechanism: MBPP nearly
  saturates the budget (the 4B self-limits to ~108–172 think tokens even at
  budget 2048), so the serial-compute lever is too weak to differentiate here.

The robust deployable core is **SELECT + ABSTAIN**, both verifier-free from one
logit: sample k, pick argmax single-token P(True), abstain below a max-P(True)
threshold. Extra compute is modestly better spent on depth than breadth overall,
but there is no measured benefit to selectively escalating the hard tail on
budget-saturated MBPP — see C57's next-tests for the budget-bound-task follow-up.

Codified as **C57** (corrected). Part 2 was regenerated on vLLM (~10× faster than
the initial HF pass) at n=400 with bootstrap CIs.

## Layout

- `scripts/frontier.py` — Part 1 post-hoc frontier + risk-coverage (CPU).
- `scripts/escalation.py` — Part 2 escalation-vs-breadth analysis (CPU).
- `../qwen35_4b_code_confidence/scripts/gen_budget.py` — think-mode multi-budget
  generation (HF backend, GPU).
- `runs/frontier.json`, `runs/escalation.json` — aggregate results.
- `runs/pool_think_b256.json`, `runs/pool_think_b2048.json` — the two candidate
  pools (aggregate per-candidate fields; no menagerie contents).
