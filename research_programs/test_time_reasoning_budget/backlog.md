# Backlog

## Done

- `qwen35_4b_thinking_budget_scaling`: MBPP scaling curve; oracle-vs-deployable decomposition;
  shuffled-thinking control. (+15pp greedy, overthinking optimum ~1024, much of the gain is compute/scaffold.)
- `qwen35_4b_thinking_budget_controller`: fixed-rule visible-test escalation controller — an
  efficiency win (Pareto-dominates fixed budgets except the peak), bounded by C2 false-passes.

## Next Experiments

- **Learned thinking-budget controller with richer visible signals** (token entropy/logprob,
  self-consistency across 2 cheap samples): can it close the deployable→oracle gap (0.89→0.93) the
  single visible test leaves? (Queue: `thinking_budget_controller`.)
- **Stronger content control** (`thinking_content_vs_compute_control`): substitute a *different*
  task's thinking (remove token-presence, not just order) to settle reasoning vs compute+scaffold.
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
