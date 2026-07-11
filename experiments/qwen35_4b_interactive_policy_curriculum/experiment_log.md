# Interactive policy curriculum: oracle DAgger to execution-reward RL Experiment Log

## 2026-07-11 — intake, preregistration, and CPU smoke

- Routed to `agentic_breadth_installation`; closest duplicate is C53's static
  gauntlet frontier, and the novelty is live visited-state supervision plus
  complete-trajectory reward.
- Copied the C53 firewall-clean gym into a self-contained follow-up experiment.
- Froze five incremental training families, three incremental transfer
  families, disjoint seed namespaces, terminal-only reward, DAgger/RL gates,
  matched controls, and conditional Menagerie rules.
- Adversarial design review caught five material hazards before GPU spend:
  indexed imported oracles were not state-aware; validity shaping would repeat
  C50; raw entropy would repeat C52; transfer replay could invalidate holdouts;
  and injected close tokens were not policy actions. All are fixed in code or
  preregistration.
- CPU smoke passed: 6 curriculum tests, 13 vLLM wrapper tests, and all 14 gym
  family selftests. No result-bearing GPU stage or Menagerie event has run.
