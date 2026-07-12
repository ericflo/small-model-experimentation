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

## 2026-07-12 — Outcome-blind smoke and implementation audit

- Passed pinned model, architecture, lens rank, tokenizer/slot, finite-logit,
  cache, data-hash, and power-hash contracts at 8,514,319,872 peak bytes.
- Stored no task correctness, chosen alias, trace text, or comparison.
- Audited task-bootstrap units, strict lower-bound gate, alias diversity, exact
  row counts, shuffled multiset, and confirmation hash locks before selection.
