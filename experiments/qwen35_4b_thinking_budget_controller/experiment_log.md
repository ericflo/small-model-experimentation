# Qwen3.5-4B Thinking-Budget Controller Experiment Log

## Scaffold

New experiment under `test_time_reasoning_budget`, following `qwen35_4b_thinking_budget_scaling`.

## Design

Offline experiment — no new generation. The sibling sweep already produced, per task and per
thinking budget, the greedy answer and its full-test pass. We copy those greedy records + the MBPP
test metadata into `data/` (self-contained), re-verify the **visible** test (first assert) for each
stored answer, and simulate visible-test escalation controllers vs fixed budgets vs an oracle ceiling
on a deployable accuracy-vs-mean-thinking-token Pareto.

## Result

The escalation controller Pareto-dominates every fixed budget except the peak: the 2-tier
`no_think → 1024` rule reaches 0.88 deployable full-test accuracy at 113 mean thinking tokens
(vs fixed think_256 0.87@246, think_512 0.87@404). It does not beat the best fixed budget
(think_1024 0.91@507) — an efficiency win, not an accuracy win. The deployable gap to the oracle
ceiling (0.93@132) is bounded by visible-test false-passes (`false-visible-commit` ~8–11%), the C2
effect. See `reports/report.md`.

## Notes

- Mean-thinking-token figures here use the greedy generation's `n_think` and differ slightly from
  the sweep report's sampled means (e.g. think_1024 507 vs 530).
- Visible-test re-verification reuses the sandbox approach from the sibling experiment (fork
  subprocess, rlimits, 10s timeout + retry); cached to `data/greedy_with_visible.jsonl`.
