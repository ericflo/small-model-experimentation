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

## 2026-07-12 — Powered seam qualification

- Completed 339/339 fixed-cap paths and all 1,130 slot/control rows in
  11,669.621 seconds; every native path contacted cap 1,024.
- Real ordered thought scored 92/339 versus 46/339 exact shuffle and 11/113
  no-thought: +13.57pp and +17.40pp.
- One-sided task-bootstrap lower bound for real-minus-shuffle was +8.85pp;
  32 tasks mixed outcomes; correct/chosen breadth reached 11/12 aliases; both
  unmasked interface gates passed.
- Automatic `POWERED_COMMIT_SLOT_SEAM_QUALIFIED`; only the hash-locked untouched
  confirmation is authorized. J stages remain unopened.

## 2026-07-12 — Independent powered confirmation

- Completed the one authorized untouched stage: 339/339 fixed-cap paths and all
  1,130 slot/control rows in 11,690.539 seconds. Every path contacted cap 1,024.
- Ordered thought scored 98/339 versus 47/339 exact-token shuffle and 8/113
  no-thought: +15.04pp and +21.83pp independently of qualification.
- The registered one-sided task-bootstrap lower bound over shuffle was +9.44pp;
  31 tasks mixed outcomes; correct/chosen support reached 10/12 aliases; all
  interface and finite-row gates passed.
- Automatic terminal seam decision `POWERED_COMMIT_SLOT_SEAM_REPLICATED`. No
  selection row was pooled to make the decision.

## 2026-07-12 — Post-confirmation adversarial audit

- Added a deterministic 20,000-resample stagewise audit. Two-sided task
  intervals were [7.96pp, 19.17pp] and [8.26pp, 21.83pp]; the independent stage
  effects did not differ detectably.
- Paired ordered-only versus shuffled-only wins were 60:14 and 64:13. Correct
  alias mention did not explain success in either stage.
- Preserved the key nuisance: target identity remains heterogeneous. One
  confirmation target had zero real successes and shuffle beat real for two
  targets. Any J/value successor must beat alias identity, correct-alias
  activity, and ordinary slot margin on task-held-out units.
- Confirmation licenses a new code/audit boundary only. J/value/control/causal
  commands remain fatal-unavailable until that boundary is committed.
