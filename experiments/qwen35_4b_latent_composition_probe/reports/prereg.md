# Pre-registration: is the generation wall representational or expressive?

Logged 2026-07-03, before any probe data. The C13–C18 arc mapped the fixed 4B's compositional generation wall
entirely from the OUTSIDE (behaviorally, input→output). This looks INSIDE for the first time. When the model
fails to identify a depth-3 composition (bare identification ≈ 0, C17), is the correct composition
**linearly present in its residual stream** (present-but-unexpressed → latent capability, the mission's core
thesis) or **absent** (a genuine information gap)?

## Method

Fresh verified-depth, collapse-rejected `list` tasks, depths 1/2/3, ~500 each (disjoint), 8 visible examples.
For each task, render the canonical no-menu identification prompt (I/O examples → infer `transform`,
`enable_thinking=False`) and capture the residual-stream vector at the **last prompt token** at every layer
via `gen_lib.activations` → `[N, 33, 2560]` (32 layers + embedding). This is the "has read all examples,
about to answer" state.

**Probe targets** (decoded from activations, per layer, held-out 30% test, stratified):
- **first-op name** (multiclass over the ~16 primitives) — the hardest to read off surface I/O at depth 3
  (buried under two later ops), so decoding it implies the model internally inverted the composition.
- **per-primitive presence** (16 binary probes; report macro-F1) — softer signal, more surface-confoundable.

Probe = per-layer standardize → PCA(128) → L2 logistic regression. Report best-layer held-out accuracy.

**Baselines / controls:**
- **chance** (empirical base rates of first-op / each primitive).
- **shuffled-label probe** (labels permuted) — must collapse to chance (leakage guard).
- **layer-0 (embedding) probe** — surface-token baseline; deeper-than-layer-0 decoding = the model's added
  representation, not raw I/O statistics.
- **behavioral first-op naming** — prompt the model to NAME the first op (given the op menu); accuracy is the
  model's own OUTPUT-level access to the same fact.
- **behavioral identification pass@1** (think) on the same tasks — the generation rate (context: ≈0 at d3).
- **depth-1 positive control** — the model can generate single ops, so the probe MUST decode first-op near
  the ceiling at depth 1; validates the methodology. Depth 3 is the test.

## Predictions (locked)

- **P1 (methodology valid):** depth-1 best-layer first-op probe ≥ 0.80 (the model plainly represents ops it
  can generate); shuffled-label probe ≤ chance + 0.05 at every depth.
- **P2 (the test):** at depth 3, best-layer first-op probe ≥ 3× chance AND ≥ layer-0 + 0.15. (Chance ≈ 1/16
  ≈ 0.06, so ≥ ~0.18.) Refuted if depth-3 probe ≤ chance + 0.05 (≈ absent).
- **P3 (representation ≫ expression = latent):** at depth 3, best-layer first-op probe ≥ behavioral first-op
  naming + 0.15, while identification pass@1 ≈ 0 — the info is linearly present but the model neither names
  nor generates it. STRONG form: the layer profile peaks in mid/late layers (computed, not surface).
- **P4 (partial-structure gradient):** first-op probe accuracy decreases monotonically with depth
  (1 > 2 > 3) — the deeper the composition, the less the model has inverted it — but stays above chance at 3.

## Decision mapping

- **LATENT** (P2 ∧ P3): the composition is represented but not expressed ⇒ the wall is an *expression/readout*
  failure, not a knowledge gap. Direct next step: activation steering — add the probe direction to the
  residual stream at generation time and test if identification rises. The mission's dream case.
- **ABSENT** (P2 refuted): depth-3 probe ≈ chance ≈ layer-0 ⇒ the model does not compute the composition;
  the wall is a genuine information/support gap. Retires "clever readout / steering" hopes for this wall and
  reinforces that only proposal-shifting (banking, tools; C18/C12) can cross it. A clean, important negative.
- **PARTIAL** (P2 holds, P3 refuted): the model represents AND expresses the same partial info (probe ≈
  behavioral) ⇒ no hidden latent capability beyond what behavior already shows.

## Honesty notes

- "Linearly decodable" ≠ "usable for generation": a positive result shows the info is *present*, not that the
  model can *route* it into a correct program. The steering follow-up is what would test usability.
- Low-n / high-dim (2560-dim, ~350 train/depth) → PCA(128) + strong L2 + held-out test + shuffled-label
  control guard against overfitting; the shuffled control is the decisive leakage check.
- Surface confound on presence probes is controlled by the layer-0 baseline; first-op is the confound-robust
  primary target.
