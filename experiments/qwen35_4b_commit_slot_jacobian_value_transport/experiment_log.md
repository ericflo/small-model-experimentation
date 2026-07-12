# Qwen3.5-4B Commit-Slot Jacobian Value Transport Experiment Log

## 2026-07-12 — Intake and design

- Created as a distinct answer-interface change after terminal close-only
  `FORCED_COMMIT_SEAM_FAIL`.
- Fixed action: append close plus `First:`; restrict only the next choice to the
  12 public aliases; never supply answer identity.
- Retained no-thought slot, unmasked logits, and matched close-only free-form
  controls; adversarial review added an exact-length token-multiset shuffle as
  the load-bearing coherent-content control before design freeze.
- Tightened seam headroom to 20%--80% and required real-over-shuffled gains in
  both selection and untouched confirmation.
- First CPU generation attempt caught an exact seed collision with the prior
  causal split. Replaced the entire new seed block before model calls.
- Final CPU smoke: 96/96 unique fresh exact-depth tasks, zero overlap with four
  parents, exact lens hash, reachable gates.
- Completed adversarial review before any model call; outcomes unopened.

## 2026-07-12 — Outcome-blind model smoke

- Anchored README, preregistration, 58-point adversarial review, semantic config,
  lens, and all four procedural splits to design commit `73deac8a`.
- Passed exact revision/architecture, tokenizer, rank-24 lens, one-token alias,
  fixed-slot, finite-logit, and cached forward contracts.
- Peak allocated GPU memory: 8,515,461,632 bytes. Correctness, trace text, and
  the chosen alias were not recorded; no scientific result opened.
- A second outcome-blind implementation audit added runtime exact-multiset
  shuffle hashes, full-vocabulary alias mass, row cardinality, control-finiteness,
  and observed-baseline feasibility receipts before selection.
