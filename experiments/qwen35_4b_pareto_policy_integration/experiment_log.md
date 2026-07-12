# Qwen3.5-4B Pareto Policy Integration Experiment Log

## Scaffold

Created as a new experiment scaffold.

## 2026-07-12 — corrected successor accepted

- User rejected the predecessor's fixed `S0 + 0.10` specialist gate as an
  obvious scientific-design error. The correction is not to lower that number;
  it is to remove arbitrary effect-size qualification entirely.
- Teacher existence is now paired `delta > 0` with two positive seed blocks and
  a one-sided stratified-bootstrap lower bound above zero. Saturated cells are
  retention anchors, not vetoes.
- C54 landed between the two experiments and materially changed the best test:
  rather than speculate about four not-yet-trained domain specialists, this run
  attempts to consolidate the already evidenced same-origin quick/deep Pareto
  policies (`blend`, `apex`).
- New experiment directory created rather than rewriting the predecessor.
- No task-model output existed when the config, preregistration, and design
  review were authored.
