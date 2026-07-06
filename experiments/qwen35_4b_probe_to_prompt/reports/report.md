# Externalize the latent readout: knowing the FULL first op elicits depth-2 (where steering failed), but the op-TYPE C19 decodes only narrows sampling — the PARAMETER is the deployable bottleneck

## Motivation
The arc's throughline: deployable capability moves only via weight edits (banking) or externalization
(tool-search); every pure test-time readout has failed. Specifically — C17: test-time SELECTION is free but
adds no coverage. C19: the composition's first-op is linearly DECODABLE from the base residual far above
behavior at depth-1/2 (0.99/0.42) but thins to a thread at depth-3. C20: that decodable direction is NOT
steerable (ActAdd inert). The untried seam: **externalize** the latent readout — decode the first-op with
C19's probe and put it in the CONTEXT (shift the proposal, the only lever C17 says works), rather than steering
the residual (C20) or selecting post-hoc (C17).

## Method
Refit C19's linear first-op probe (standardize+PCA128+L2-logistic) on 1500 training tasks — replicates C19
(depth-1 0.99@L15, depth-2 0.45@L21, depth-3 0.23@L19). On FRESH fsig-disjoint eval tasks (n=100/depth), decode
the first-op from the base model's own activation and generate depth-2/3 code under six arms: **no-hint**;
**neutral** (placebo line, format control); **oracle-type** (TRUE first-op type = the probe's ceiling);
**oracle-full** (TRUE op WITH parameter = ceiling if param-binding is the bottleneck); **probe** (decoded type);
**wrong** (RANDOM wrong type, content-causality control). Metrics: greedy@1 + coverage@6, no-think.

## Results (no-think, fsig-disjoint eval, n=100/depth)
| arm | d2 greedy@1 | d2 cov@6 | d3 greedy@1 | d3 cov@6 |
|---|---|---|---|---|
| no-hint | 0.030 | 0.050 | 0.000 | 0.010 |
| neutral (placebo) | 0.020 | 0.050 | 0.000 | 0.000 |
| oracle-TYPE | 0.020 | **0.190** | 0.010 | 0.030 |
| **oracle-FULL (+param)** | **0.190** | **0.310** | 0.010 | 0.030 |
| probe (C19 readout) | 0.010 | 0.060 | 0.000 | 0.000 |
| wrong (content ctrl) | 0.000 | 0.000 | 0.000 | 0.000 |

Probe EVAL accuracy (fsig-disjoint): depth-2 **0.32**, depth-3 0.18 (vs majority 0.19/0.11). Layer-0 (embedding)
probe: **0.05 / 0.02 = chance** (leak control passes).

1. **Externalization ELICITS deployable depth-2 capability where steering (C20) was inert.** oracle-full lifts
   depth-2 greedy@1 **6×** (0.030→0.190) and coverage **6×** (0.050→0.310). Telling the model the concrete first
   op via the PROMPT works — the **first test-time intervention in the whole arc to move deployable capability**
   (decode→prompt succeeds where decode→steer, C20, failed).
2. **The deployable bottleneck is the PARAMETER, not the op-TYPE C19 decodes.** oracle-TYPE lifts *coverage*
   (0.050→0.190 — knowing the type narrows the sampling search) but NOT *greedy* (0.020); only oracle-FULL (with
   the parameter) makes it single-shot deployable. So the quantity C19 found latent (op type) is
   coverage-relevant, not greedy-deployable.
3. **The C19 type-only probe can't cash out.** probe-hint ≈ no-hint (greedy 0.010, cov 0.060). The effect is
   *genuine self-elicitation* — on the 32% probe-correct tasks, probe-hint coverage 0.156 vs no-hint 0.094
   (+0.062); on probe-wrong tasks it slightly hurts (−0.015) — but at 0.32 accuracy it washes out.
4. **Graded by depth, exactly as C19 predicts.** Everything is real at depth-2 (latent headroom 0.42) and ≈0 at
   depth-3 (thread — even oracle-full only 0.010): no readout conjures information the forward pass never
   computed.
5. **Controls clean.** neutral placebo ≈ no-hint (not a format effect); wrong-hint HURTS (content-causal, like
   C28's T_corrupt); layer-0 probe at chance (the readout is the model's COMPUTATION, not surface-readable I/O).

## Implication
The latent readout (C19) IS usable at test time — by **externalizing** it (decode→prompt), not by steering
(C20). This adds the first test-time lever that moves deployable capability, bounded by the representation
(works depth-2, fades depth-3). But it also finds a NEW wall: the *decodable* quantity (op TYPE) is not the
*deployable* one (the PARAMETER). Knowing which op-type narrows sampling (coverage) without fixing the greedy
mode; the concrete parameter is the missing piece.

## Next
- **Decode the (op, PARAMETER)** — is the parameter latently decodable from the residual too? If a full-op probe
  reaches useful accuracy, a probe-hint would deliver the oracle-full lift (training-free elicitation). If the
  param is NOT decodable, that pinpoints exactly what the forward pass fails to compute.
- Compliance instrumentation (parse the generated first-op) and ≥2 seeds; depth-1 sanity (no disjoint tasks in
  the tiny op-space).

## Artifact Manifest
See `reports/artifact_manifest.yaml`. Probe pickle + activations (~large) moved out of repo.
