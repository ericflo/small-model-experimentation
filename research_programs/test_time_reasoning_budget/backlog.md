# Backlog

## Done

- `qwen35_4b_thinking_budget_scaling`: MBPP scaling curve; oracle-vs-deployable decomposition;
  shuffled-thinking control. (+15pp greedy, overthinking optimum ~1024, much of the gain is compute/scaffold.)
- `qwen35_4b_thinking_budget_controller`: fixed-rule visible-test escalation controller — an
  efficiency win (Pareto-dominates fixed budgets except the peak), bounded by C2 false-passes.
- `qwen35_4b_thinking_separability_probe`: per-layer probes on answer-token activations. Correctness
  is moderately decodable (AUC 0.64–0.76); thinking raises decodability; shuffled ≈ real in
  decodability (representational side is noisy across experiments). Weak probe signal on C2 false-passes.
- `qwen35_4b_thinking_content_vs_compute`: foreign-task-thinking ladder. The model uses thinking as
  CONTENT (foreign collapses to 0.04 — solves the wrong problem); at the efficient budget the gain is
  coherent reasoning (real 0.86 > shuffle ≈ no_think 0.74). **Corrected** the earlier "mostly
  compute/scaffold" claim (greedy-metric artifact; held mainly at high budgets / representationally).

## Next Experiments

- **Filler/pause-token arm**: B contentless tokens (no semantics) to isolate PURE compute — foreign
  adds *misleading* content (which the model follows), not contentless compute, so it can't separate
  compute from content. Completes the {compute, presence, relevance, order} attribution.
- **Foreign/shuffle/real ladder at a high budget** (1024/2048) to confirm the coherence advantage
  shrinks (overthinking) — i.e. that the "compute/scaffold" reading is the high-budget regime.
- **Learned thinking-budget controller with richer visible signals** (token entropy/logprob,
  self-consistency): can it close the deployable→oracle gap (0.89→0.93)? (Queue: `thinking_budget_controller`.)
- Thinking-budget sweep on the **silent_executor substrate** (modular-program execution):
  its CoT collapsed to 0% at len-24 at a fixed 768-token budget — does a larger thinking
  budget rescue length generalization? Direct pressure-test of the silent-compute thesis.
- **Distill long-thinking into short/no-think**: SFT on the model's own correct long-think
  traces, measure retained accuracy per token (compression of reasoning).
- **Thinking as verifier vs generator** under matched token budget: spend the budget on a
  thinking *generator* vs a thinking *verifier/selector* over no-think candidates — which
  buys more deployable accuracy? (Couples to evidence_conditioned_selection / C2.)
- Replicate the sweep + controller on a **harder substrate** (full MBPP / LiveCodeBench / math):
  does the optimum move, does the C2 false-pass rate grow?

## Required Controls

- Baseline: `no_think` (enable_thinking=False) — the corpus's universal setting.
- Mechanism-falsifying control: **shuffled-thinking** (scramble the model's own thinking
  tokens, then force the answer) and **truncated-thinking** — isolates thinking *content*
  from the `<think>` scaffold + extra compute.
- Matched-compute control: compare conditions at equal total forward tokens (the corpus's
  matched-compute methodology), so gains are not just "more tokens".
- Shift check: difficulty / length slices (does the optimal budget move with task length?).

## Stop Conditions

- Retire/demote if, across substrates and at matched compute, thinking never raises the
  deployable line beyond shuffled-thinking — i.e., only the scaffold + extra compute help.
- Branch a new program if a thinking-budget *controller* becomes a rich line of its own.
