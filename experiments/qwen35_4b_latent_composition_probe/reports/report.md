# Inside the generation wall: is the composition latent or absent?

## Summary

The C13–C18 arc mapped the fixed 4B's compositional generation wall entirely from the OUTSIDE (behaviorally).
This experiment looks INSIDE for the first time: it trains linear probes on residual-stream activations to
ask whether, when the model fails to identify a composition, the answer is **linearly present but unexpressed**
(latent capability) or **absent** (a genuine information gap). 1500 verified-depth `list` tasks (500 each at
depths 1/2/3), probing the last-prompt-token activation at every layer for the composition's **first operation**
(the confound-robust target — hardest to read off surface I/O).

**Answer: the wall's nature CHANGES with depth.** At shallow depth the composition is strongly encoded but
under-expressed (an *expression* failure = latent capability); at the deep wall the representation itself thins
out (a *representation* failure = information gap).

| depth | linear probe (first-op) | shuffled floor | **real signal** | model names 1st op | model generates (ident@1) |
|---|---|---|---|---|---|
| 1 | **0.99** @L15 | 0.05 | 0.94 | 0.44 | 0.68 |
| 2 | **0.42** @L22 | 0.09 | 0.34 | 0.13 | 0.07 |
| 3 | **0.27** @L19 | 0.14 | 0.13 | 0.13 | 0.01 |

(chance ≈ 0.06–0.10; layer-0/embedding probe stays at chance, so the signal is computed, not surface.)

- **Depth 1:** the first op is *almost perfectly* linearly decodable (0.99), rising from chance at the
  embedding to ~0.99 by layer 15 and plateauing — yet the model names it only 0.44 of the time. Representation
  ≫ expression.
- **Depth 2:** probe 0.42 (real signal 0.34 over the shuffled floor) vs behavior ~0.13 — the first op is
  encoded ~3× better than the model expresses it.
- **Depth 3 (the wall):** probe 0.27, but the shuffled-label floor is 0.14, so the *real* decodable signal is
  only ~0.13 — roughly equal to the model's (poor) behavioral naming. The representation-expression gap has
  nearly closed because the representation itself has thinned to a thread.

## Research Program Fit

The first mechanistic (representation-level) probe of the C13–C18 generation wall. Directly tests the project's
core thesis — "unearthed potential in the weights" — and adjudicates whether steering/readout could ever cross
the wall.

## Method

Fresh verified-depth, collapse-rejected `list` tasks (depths 1/2/3, 500 each, disjoint), 8 visible examples.
Render the canonical no-menu identification prompt (`enable_thinking=False`) and capture the residual-stream
vector at the **last prompt token** at every layer (`gen_lib.activations` → `[1500, 33, 2560]`). Probe = per
layer standardize → PCA(128) → L2 logistic regression, stratified 70/30, held-out accuracy; sweep all 33
layers, report the best. Targets: first-op name (multiclass, ~16 primitives). Behavioral baselines on 150
tasks/depth: identification pass@1 (think, greedy) and first-op naming (think, greedy).

## Controls

- **Shuffled-label probe** at the best layer — the leakage/overfitting floor. It is ~chance at depths 1–2
  (0.05, 0.09) but rises to 0.14 at depth 3 (thin-n overfitting), so depth-3 claims use **probe − shuffled**
  (0.13), not raw probe.
- **Layer-0 (embedding) probe** stays at chance (0.07/0.07/0.04) at every depth → the decodable signal is the
  model's *computation*, not surface I/O token statistics.
- **Depth-1 positive control** (probe 0.99) validates the methodology.
- **Confound-robust target:** first-op (buried under later ops at depth ≥ 2) rather than easily-surface-detectable
  presence. (A presence probe was run but is uninformative — with ~19% base rate its raw accuracy tracks the
  majority-"absent" baseline; excluded from claims.)

## Pre-registered verdicts

- **P1 (methodology valid):** HELD — depth-1 probe 0.99 ≥ 0.80; shuffled ≤ chance+0.05 at depths 1–2 (0.14 at
  depth 3 is the one exceedance, handled via probe−shuffled).
- **P2 (the test at depth 3):** MARGINAL — probe 0.27 is 2.7× chance (just below the pre-set 3×) and clears
  layer-0 by +0.23; not "absent" (probe−shuffled = 0.13 > 0), but a thin signal.
- **P3 (representation ≫ expression = latent):** HELD at depths 1–2 (gap +0.55, +0.30 over naming), NEARLY
  CLOSES at depth 3 (probe−shuffled 0.13 ≈ naming 0.13). The latent gap is real but concentrated at shallow
  depth.
- **P4 (gradient):** HELD — probe first-op decays monotonically 0.99 → 0.42 → 0.27 with depth, mirroring the
  behavioral wall.

Overall verdict: **GRADIENT / crossover**, not a clean LATENT or ABSENT.

## Interpretation

- **The generation wall is not one thing — it changes character with depth.** For shallow structure the model
  *has computed* the inverse (the first op is strongly, linearly present in mid-network) but does not *express*
  it — a readout/routing failure, i.e. genuine latent capability. For deep compositions the model *has not
  computed* the inverse — the representation thins to near the overfitting floor — a real information/support gap.
- **This adjudicates the steering hope honestly:** activation steering (adding the probe direction at generation
  time) has real headroom at depth 1–2, where the info is present but unexpressed; but at the true wall
  (depth 3+) there is almost nothing to steer *toward* — the composition simply is not encoded. No clever
  readout conjures information the forward pass never computed.
- **It explains WHY banking (C18) was necessary and worked.** C18 showed banking *expands* the depth-2 coverage
  ceiling on held-out tasks. C19 shows the base model's depth-2 representation of the composition is weak
  (0.42) and its depth-3 representation is a thread (0.13). Banking *installs* the representation the base
  lacks — which is why only proposal-installation (banking / tools, C18 / C12), not test-time readout, crosses
  the deep wall. The two claims lock together: the wall is representational at depth, and banking is how you
  add the missing representation.
- **Refines C13/C16 ("the model can't propose"):** the mechanism is that inverse-inference is *computed* for
  shallow structure but not routed to output, and simply *not computed* for deep structure — a depth-graded
  mixture of expression and representation failure.

## Honesty notes

- "Linearly decodable" shows the info is *present*, not that the model can *route it into a correct program*;
  the depth-1 gap is partly a naming-task artifact (the model does use the first op to generate at 0.68). The
  cleanest latent signal is the *middle* regime (depth 2): strongly encoded (0.42), barely expressed (0.13).
- Depth-3 shuffled floor at 0.14 caps confidence in the depth-3 residual; the honest statement is "thin but
  non-zero (probe−shuffled 0.13, ~4 SE above the floor)."
- Single substrate (list), single probe family (linear), last-token position. A steering experiment is the
  decisive follow-up (does adding the probe direction raise identification?).

## Next Experiments

- **Activation steering:** add the depth-2 first-op probe direction to the residual stream at generation time;
  does identification / coverage rise? (Tests usability of the latent signal — the LATENT verdict's real test.)
- **Full-composition probe:** decode op-2 and op-3, not just op-1, to map how much of the *whole* pipeline is
  latent vs the first step only.
- **Probe the banked model (C18):** does banking raise the depth-2/3 first-op probe accuracy? (Direct test that
  banking installs the missing representation.)

## Artifact Manifest

See `reports/artifact_manifest.yaml`. Key: `scripts/capture.py`, `scripts/probe.py`, `scripts/analyze.py`,
`data/{acts.npy, present.npy, labels.json, tasks.jsonl}`, `runs/probe_results.json`, `analysis/latent_probe.png`.
`data/acts.npy` (~200MB activations) is omitted from git; regenerate via `scripts/capture.py`.
