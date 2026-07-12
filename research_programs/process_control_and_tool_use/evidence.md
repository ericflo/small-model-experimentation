# Evidence

## Seed Experiments

- [qwen35_4b_adaptive_tool_controller](../../experiments/qwen35_4b_adaptive_tool_controller/reports/report.md)
- [qwen35_4b_tool_state_policy_lora](../../experiments/qwen35_4b_tool_state_policy_lora/reports/report.md)
- [qwen35_4b_live_tool_dagger](../../experiments/qwen35_4b_live_tool_dagger/reports/report.md)
- [qwen35_4b_adaptive_evidence_budget_policy](../../experiments/qwen35_4b_adaptive_evidence_budget_policy/reports/qwen35_4b_adaptive_evidence_budget_policy_report.md)

## Current Read

Tool-state policies are a promising way to make small models useful inside iterative systems, but they need visible-only evaluation discipline.

- [qwen35_4b_repo_search_compress_bank](../../experiments/qwen35_4b_repo_search_compress_bank/reports/report.md):
  exact marginal balance over inspect/patch/verify/commit installed a perfect
  four-step trained-family path but regressed family-disjoint repository
  success 49/72→25/72. After failed tests the control revised with another
  patch 24/26 times; compact revised 0/48 times, while commit after pass stayed
  intact. Process-control curricula must preserve verifier-conditioned
  transitions and changed recovery actions, not merely terminal success traces
  or operator totals.

- [qwen35_4b_specialist_policy_integration](../../experiments/qwen35_4b_specialist_policy_integration/reports/report.md):
  no new tool policy was trained. The only registered tools family scored
  0.994 under the installed incumbent, leaving 0.006 maximum headroom against a
  frozen `+0.10` qualification rule. This is a substrate-selection negative:
  tool-policy experiments need an unsaturated core before adaptation quality is
  testable.
