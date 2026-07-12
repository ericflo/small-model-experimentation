# Qwen3.5-4B Commit-Slot Semantic Power Replication Log

## 2026-07-12 — Intake, power correction, and design

- Created as a distinct fixed-cap replication after the parent's terminal
  five-versus-six mixed-task near miss.
- Rejected decoder calibration and a larger cap because three parent post-hoc
  residual policies underperformed and fixed-1,024 semantic evidence is not yet
  task-level stable.
- Initial 64-task/stage draft had only ~59% approximate power at the observed
  parent effect. Increased both seam stages to the calculated N=113 for 80%.
- CPU smoke passes 322 unique exact-depth tasks, zero overlap with five parents,
  balanced support, exact lens hash, and reachable gates.
- Completed 60-point adversarial review before any model call. Outcomes unopened.
