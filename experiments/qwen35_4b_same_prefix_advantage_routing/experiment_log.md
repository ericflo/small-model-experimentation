# Qwen3.5-4B Same-Prefix Advantage Routing Experiment Log

## 2026-07-12 — intake and design

- Created a new experiment rather than extending the terminal Pareto
  qualification directory.
- Re-read the repository scorecards, C50-C54 evidence, model/vLLM playbooks,
  lifecycle rules, and the predecessor's harness, raw qualification receipt,
  preregistration, design review, and primary-paper map.
- Rechecked the 2026 primary literature. The design adopts MOPD's same-origin
  corrected top-k loss, SRPO's “do not densely distill already-correct
  samples” lesson, and the OPD failure literature's distribution/locality
  diagnostics.
- Froze the scientific correction: teacher advantage means verified
  continuation return on the identical student state, not a teacher's
  aggregate rank or hinted token log-probability. Four samples route and four
  disjoint samples audit; no arbitrary positive margin exists.
- Chose the independently regenerated 40% quick / 60% deep soup as the student
  so a pass must extend the strongest existing joint checkpoint, not merely
  recover from one weaker endpoint.
- No task-model output existed while the intake, config, preregistration,
  design review, literature review, and implementation plan were written.

## 2026-07-12 — upstream power correction incorporated before lock

- A 29-commit shared-main advance landed before this experiment created its
  immutable receipt. C54 pooled nine apex medium events and corrected the
  earlier favorable n=3 `+0.345` draw to `+0.321 ± 0.017 SE`; the tier router
  reaches the medium ceiling but does not decisively clear the old bar.
- The teacher checkpoints and local scientific question are unchanged: no
  quick/deep rank is assumed and both must win on independent same-prefix
  audit branches. The visible tier router is now an even more direct baseline.
- The frozen downstream power rule now requires at least eight medium
  benchmark events. No task-model output had run, so this is a legitimate
  pre-lock correction rather than an outcome-dependent amendment.
