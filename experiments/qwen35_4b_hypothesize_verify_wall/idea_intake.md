# Idea intake: hypothesize-and-verify meets the structure wall

- Source: forest-review workflow 2026-07-08 (cross-arc connector lens, arc-map pair #1) after C47 landed; selected by user from four proposals.
- Novelty check: no experiment prompts or trains an enumerate-and-verify strategy on the DSL substrates (grep across experiments/*/scripts confirmed; nearest priors are C44's shift-specific hint prompt at 0.00, depth_wall_anatomy's plan-GIVEN discrimination arms, and C45's affine-only install). Not in the future-experiment queue; the 2026-07-08 decision record's training slot targets grammar induction on the meta-induction substrate, not the DSL wall.
- Why now: C36 and C45 are both terminal claims whose own logic demands this cell; every outcome updates a law. C47 just sharpened the audit standard (frozen paired evals, pre-registered gates) this design inherits.
- Stop-branch conditions: trap gate (oracle-skelfill < 0.85 -> substrate misconfigured, stop); install gate (dsl_sft fails to reproduce trace format on held-out d1 -> report install failure, skip transfer claims); harvest/train infra reused from C36/C45 with the torch-2.12 OOM patch ported.
- Cost envelope: ~9-11 GPU-h, one day, two QLoRA trains (C45 regen + DSL SFT).
