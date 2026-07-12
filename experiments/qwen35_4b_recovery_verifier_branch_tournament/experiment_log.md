# Public-verifier recovery branch tournament Experiment Log

## 2026-07-12 — intake, mechanism audit, and preregistration

- Direct predecessor stopped because λ=.18 tied action-only at 55/80 on
  confirmation; their hidden union was 63/80 on both transfer blocks.
- Froze the public rule: select action only when action final-visible passes and
  candidate does not; otherwise candidate.
- Retrospective qualification reproduced 60/80 on both source blocks, 95.2%
  union capture, and +6.25pp over exact random policy choice. These are design
  facts, not prospective evidence.
- Added four new procedural families with initial/partial/oracle executable
  gates and disjoint dev/confirm seeds.
- Matched the two-policy 12,288-token reservation with two full six-call
  trajectories from each source policy. Pass-if-either hidden coverage is the
  primary sample-more control.
- Adversarial review fixed exact-random rather than single-draw gating,
  control-first union feasibility, and the separation between a capability
  producer and later curriculum compression.
- No prospective Qwen output or Menagerie seed has been exposed.

## 2026-07-12 — prospective development and feasibility stop

- Base 49/80; λ=.18 greedy 59/80; action greedy 59/80.
- Equal-reservation pass-if-either controls: λ=.18 59/80, action 60/80.
- Deterministic mixed-policy union: 60/80, only one exclusive win per source.
- All three frozen union-feasibility checks failed; recorded
  `PROSPECTIVE_DEV_INFEASIBLE` before public selector scoring.
- Confirmation, winner-bank production, and Menagerie remained sealed.
- Forensics: all 20 shared failures were `atomic_reservations`, despite 100%
  two-turn changed-patch behavior. Traces oscillated between atomic validation
  and input-copy invariants; action sample-more solved one of 20 cases.
- Strategic pivot: source selection is closed for this line. Queue diverse
  transactional scaffold-distillation with existing recovery replay to shift
  proposal coverage while preserving conditional transitions.
