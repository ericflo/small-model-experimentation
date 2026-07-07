# When does the model's structure beat brute-force search? Never on this substrate — the scissors widens with depth

## Motivation
C34 established that at depth-3, brute-force structure-search + value-fill + execution-select deploys at 0.975,
dominating the banked model (0.46), and left open: *"brute-force wins because the depth-3 structure space (4096) is
enumerable; the model's structure-pruning would only become a deployable lever when the space is too large to
brute-force."* This tests that hypothesis at **depth-4** (structure space 16^4 = 65536, 16× larger), with a
depth-4-banked model (banked_d4) and held-out depth-4 tasks.

## Result (held-out, min-depth-verified)
| depth | structure space | model structure-cov (banked, forward pass) | brute-full deploy (tool) | model − brute gap |
|---|---|---|---|---|
| 3 | 4,096 | 0.512 (banked_1280) | 0.975 | −0.46 |
| 4 | 65,536 | **0.100** (banked_d4) | **0.967** | **−0.87** |

**The hypothesized regime does NOT appear — the scissors widens, it never crosses:**
1. **Banking's structure-installation collapses with depth:** banked_d4's structure-coverage at depth-4 is only
   **0.10**, vs 0.51 for banked_1280 at depth-3. The deeper-banked model barely proposes the deeper structure.
2. **Brute-force stays near-perfect:** deploy 0.967 at depth-4 (vs 0.975 at depth-3). The 8-visible filter is
   **depth-invariant** — still only ~6 skeletons survive it at depth-4 (vs ~2 at depth-3), so structure-search +
   execution-consensus near-solves while the space is enumerable.
3. So the model-vs-brute deploy gap **grows** from −0.46 (depth-3) to −0.87 (depth-4).

## Why the model never wins
- **Banking's structure degrades faster than brute's cost grows intractable.** Brute's cost is exponential
  (16^depth: 4096 → 65536 → 1M), so its enumeration ceiling is ~depth-5. But banking's structure has already
  collapsed to 0.10 at depth-4 (≈0 at depth-5). There is a regime (depth-5+) where **neither** works cheaply —
  brute intractable, model collapsed — but the model never *wins*.
- **The model's structure can't be cheaply injected into a search anyway.** Recovering it from behavior costs a
  full 16^depth enumeration (the model-guided run took 45+ min at depth-4 before it was killed and the brute
  number taken directly), and direct op-sequence generation is broken (0.00 even at depth-1, C32).

## Implication
Closes the C32→C33→C34→C35 arc. The compositional wall is **structure** (C32); banking installs structure into
the **forward pass** (C33) but that installation **collapses with depth** (0.51 → 0.10); and **with an interpreter
available, brute-force structure-search + value-fill + execution-select dominates the weights outright** (C34,
C35) up to its exponential-cost ceiling. The model — even banked — is never the better deployable
structure-proposer when the tool is available. Banking's value is forward-pass-only (no-interpreter), and it
degrades with depth.

## Honest scope
- List DSL, 16 op-types; depth-3 (from C33/C34) vs depth-4 (n=60). banked_d4 is one depth-4-banked model.
- The "model can't be cheaply injected into search" point is about the *behavioral-inference* and
  *op-seq-generation* routes; a step-wise decompose-search guided by the model's op-ranking (C25) is a different
  (linear-cost) route — but C25 found the model's step-1 ranking is ~chance (planner wall), so it is bottlenecked
  at seeding; and here banked_d4's structure-cov (0.10) shows the forward-pass structure has collapsed anyway.

## Artifact Manifest
See `reports/artifact_manifest.yaml`. Uses external banked_d4 adapter (scratchpad).
