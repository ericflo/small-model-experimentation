# Evidence

## Seed Experiments

- [`qwen_python_shaped_silent_executor`](../../experiments/qwen_python_shaped_silent_executor/reports/qwen_python_shaped_silent_executor_report.md):
  the only corpus experiment that ever enabled native thinking. Its CoT baseline reached
  62.5% on length-4 programs (746 emitted tokens) but collapsed to 0% on length-24/32 at a
  **fixed ~768-token thinking budget that was never swept**. It was framed as a foil for
  "silent latent compute" (which was itself a controlled negative). This is the contrast the
  program exists to revisit: was the collapse a reasoning limit or a *budget* limit?

## Corpus-Wide Fact (verified)

- Across all 155 experiments, native thinking is disabled (`enable_thinking=False` ×48,
  `True` ×0 besides the one seed); `<think>` blocks are stripped as boilerplate. "Budget"
  always means evidence/probe/tool/program/sample budget, never reasoning tokens. So the
  reasoning-budget axis is genuine, verified white space.

## Anchor Experiments

- [`qwen35_4b_thinking_budget_scaling`](../../experiments/qwen35_4b_thinking_budget_scaling/reports/report.md)
  (n=100 MBPP test, k=8; numbers independently recomputed from raw data and audited) — the sweep.
- [`qwen35_4b_thinking_budget_controller`](../../experiments/qwen35_4b_thinking_budget_controller/reports/report.md)
  (offline, reuses the sweep's greedy answers) — the deployable controller.

## Confirmed Claims

- **Native thinking is a deployable win the corpus disabled.** Greedy pass@1 0.76 → 0.91 (+15pp);
  the deployable line moves *more* than the oracle ceiling (pass@8 0.91 → 0.96) and the
  oracle−deployable gap *narrows* — the opposite of the C2-based prior. Paired-robust (17 fail→pass
  vs 2 pass→fail at think_1024 vs no_think; McNemar p≈0.001). So C2 (coverage ≫ deployable
  selection) does **not** hold for the thinking axis on MBPP.
- **More thinking is not monotonically better.** Broad optimum ~512–1024 tokens then decline;
  `unbudgeted` (greedy 0.84) is worse than a cap (0.91). Shape corroborated by greedy and pass@1.
- **A visible-test budget controller is an efficiency win, not an accuracy win.** A draft→escalate
  rule Pareto-dominates every fixed budget except the peak (matches ~0.88 at 113 mean thinking
  tokens vs fixed think_256/512 at 246–404), but cannot beat the best fixed budget (think_1024,
  0.91). Its deployable gap to the oracle ceiling (0.93) is bounded by visible-test false-passes
  (~8–11%) — the C2 effect made concrete on the thinking axis.

## Negative / Cautionary Findings

- **Much of the "thinking" benefit is not coherent reasoning.** A shuffled-thinking control
  (scramble the model's own thinking tokens, keep count/scaffold) reproduces most of the gain;
  at 2048 shuffled = real. Evidence that coherent reasoning *order* adds beyond compute + scaffold
  + token-presence is weak and budget-dependent. Needs a stronger control (substitute a different
  task's thinking) before claiming the gain is "reasoning."
- The exact optimum (1024) and the never-solved-bucket effect (3/9 tasks) rest on small n /
  single-seed; treat as suggestive, not pinned.

## Current Read

Turning thinking on is a real, cheap deployable lever the corpus left unused — but it is a
*budget* to be controlled, and the controller experiment shows the budget knob is mostly an
**efficiency** lever (near-iso-accuracy at much lower cost), not a new accuracy frontier: a
near-optimal fixed budget already sits close to the oracle, and the deployable controller is
capped by C2 false-passes. Combined with the sweep's shuffle-control caveat (much of the gain is
compute/scaffold, not reasoning), the honest read is: *use thinking, cap it, and a cheap controller
buys back most of the cost — but do not over-credit it as reasoning or as an accuracy unlock.*
Priority follow-ups: a learned controller with richer visible signals (entropy, self-consistency)
to chase the oracle gap; the stronger content control; and replication on harder substrates where
the optimum may move and the C2 false-pass rate may grow.
