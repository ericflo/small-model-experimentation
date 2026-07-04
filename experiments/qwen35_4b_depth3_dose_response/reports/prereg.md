# Pre-registration: depth-3 dose-response — data-limited or representational cap?

Logged 2026-07-04, before eval data (design hardened by an adversarial workflow review; `reports/design_review.md`).
C22 found tool-seeded banking crosses the depth-3 wall but WEAKLY (think coverage@16 0.00→0.125; deployable
no-think ~0). Question: is the weak depth-3 install **DATA-LIMITED** (strengthens with more depth-3 pairs) or a
**REPRESENTATIONAL CAP** (plateaus)?

## Method
- Explorer: CPU interpreter brute-search over the substrate's own 16-op DSL (no external model); solved 640/640
  depth-3 tasks. Render to code.
- Doses (NESTED, log-spaced): C21's exact depth-1+2 pairs + N tool-found depth-3 pairs, N ∈ {40, 160, 640}.
  QLoRA r32/α64, epochs=3 held constant. → banked_{40,160,640}.
- Eval on ONE frozen paired held-out set, dedup vs the 640-superset training rules by **function-signature AND
  op-composition** (0 leakage by construction; leakage reported). Primary: **think coverage@16 at depth-3**,
  n=80. Deployable: **no-think coverage@16 at depth-3** (dense; greedy@1 confirmatory only). Guardrail:
  depth-2 think coverage (scaffold drift).

## Predictions (locked)
- **P1:** think coverage@16 at depth-3 is monotone non-decreasing across N ∈ {base=0, 40, 160, 640}.
- **P2 (the decision):** DATA-LIMITED if think coverage@16 rises materially from N=40 to N=640 (Δ ≥ +0.10,
  top-dose CI above the low-dose CI); REPRESENTATIONAL-CAP if it plateaus (N=640 within noise of N=160, CIs
  overlap) — stated as CI-overlap equivalence, never a bare non-significant p.
- **P3 (deployable):** does the depth-3 install reach deployable no-think coverage@16 > 0.05 at the top dose,
  or stay ~0 (test-time-only even at 640)?
- **P4 (memorization control):** 0 eval-task solutions (function-sig or op-composition) present in training, so
  any rise reflects generalization to NOVEL depth-3 rules, not enumerating the finite DSL.

## Honest limits (deferred from the review, scope)
Single nested realization, one training seed (no seed error bars); fixed epochs=3 (so 'more depth-3 data' is
confounded with 'more depth-3 gradient exposure' — the same event physically; the total-compute vs
data-diversity split via fixed-step + upsampled-40 control is NOT done here); harvested depth-3 are the
search-easy tail (a plateau could be an easy-bias ceiling, flagged). These bound the strength of a "cap" claim.
