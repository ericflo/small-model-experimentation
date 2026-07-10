# Up to depth 4, the tested banked structure does not beat brute-force search

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

**The hypothesized regime does not appear in the measured depth-3/4 cells:**
1. **The tested depth-4 banked model has much lower structure coverage:** banked_d4's structure-coverage at
   depth-4 is only **0.10**, vs 0.51 for a different model, banked_1280, at depth-3. The comparison is not
   dose- or curriculum-matched, so it is a two-model trend rather than a controlled causal depth effect.
2. **Brute-force stays near-perfect:** deploy 0.967 at depth-4 (vs 0.975 at depth-3). The 8-visible filter is
   **depth-invariant** — still only ~6 skeletons survive it at depth-4 (vs ~2 at depth-3), so structure-search +
   execution-consensus near-solves while the space is enumerable.
3. So the model-vs-brute deploy gap **grows** from −0.46 (depth-3) to −0.87 (depth-4).

## Why there is no measured crossover through depth 4
- **The tested banked coverage drops while brute remains effective.** Brute's cost is exponential
  (16^depth: 4096 → 65536 → 1M), so depth 5 is the first projected stress regime. No depth-5 guided-search or
  banked-model result was run here; whether a model-guided method wins there is explicitly open.
- **The model's structure can't be cheaply injected into a search anyway.** Recovering it from behavior costs a
  full 16^depth enumeration (the model-guided run took 45+ min at depth-4 before it was killed and the brute
  number taken directly), and direct op-sequence generation is broken (0.00 even at depth-1, C32).

## Implication
Through depth 4 on this list DSL, **with an interpreter available, brute-force structure-search + value-fill +
execution-select dominates the tested banked models**. The 0.51→0.10 banked comparison is suggestive but
crosses two differently trained models, so it does not by itself establish a causal depth-collapse law.
Banking's demonstrated value here is forward-pass-only; the unmeasured depth-5 regime remains the relevant
test for cheap model-guided pruning.

## Honest scope
- List DSL, 16 op-types; depth-3 (from C33/C34) vs depth-4 (n=60). banked_d4 is one depth-4-banked model with
  an undocumented/dose-unmatched recipe relative to banked_1280.
- The "model can't be cheaply injected into search" point is about the *behavioral-inference* and
  *op-seq-generation* routes; a step-wise decompose-search guided by the model's op-ranking (C25) is a different
  (linear-cost) route — but C25 found the model's step-1 ranking is ~chance (planner wall), so it is bottlenecked
  at seeding; and here banked_d4's structure-cov (0.10) shows the forward-pass structure has collapsed anyway.

## Artifact Manifest
See `reports/artifact_manifest.yaml`. Uses external banked_d4 adapter (scratchpad).
