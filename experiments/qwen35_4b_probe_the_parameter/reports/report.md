# Is the parameter latent? The op-TYPE is model-computed (elicitable); the PARAMETER is just read off surface I/O

## Motivation
C30 found the DEPLOYABLE bottleneck for depth-2 elicitation is the concrete first op's **parameter** (oracle-full
lifts greedy@1 6×, oracle-type does not; the C19 type-only probe nets to zero). This asks the sharp question:
is the parameter **model-latent** (elicitable, like the op-type C19 found) or merely **surface-readable** off
the I/O examples? If latent → a full-op probe-hint delivers the lift training-free; if not → that pinpoints
exactly what the forward pass computes (the type) vs reads off surface (the param).

## Method
Fit two probes per depth on 600/depth training activations (last-prompt-token residual): a 16-way op-TYPE probe
(C19) and a 32-way CONCRETE-op probe (op+param). **The review caught a critical flaw:** the layer-0 probe is a
degenerate surface control — activations are at the fixed template tail, and RoPE makes the embedding-layer
vector identical across tasks, so "layer-0 ≈ chance" is meaningless. The real control is an **external classifier
on raw I/O features** (list lengths, sums, min/max, elementwise diffs, sortedness) with **no 4B forward pass**.
Decodability on a large fsig-disjoint eval (600/depth, activation-only); deployability arms (n=130/depth) split
by param vs non-param first ops.

## (A) Decodability: model probe vs external-I/O surface baseline (depth-2, fsig-disjoint)
| target | model probe (residual) | external I/O (no 4B) | chance |
|---|---|---|---|
| op-TYPE | **0.413** | 0.272 | ~0.06 |
| CONCRETE (op+param) | 0.258 | 0.163 | ~0.03 |
| PARAM \| type | 0.493 | **0.529** | 0.303 |

- **The op-TYPE is genuinely MODEL-LATENT** — the residual probe (0.413) clearly beats the surface classifier
  (0.272). The model computes the first-op type (C19, now with a *proper* baseline).
- **The PARAMETER given the type is SURFACE-READABLE** — a trivial I/O classifier (0.529) decodes it as well as
  / better than the model residual (0.493). No privileged model knowledge: the param is just read off I/O
  magnitudes (e.g. `add_k(3)`: sum(out)−sum(in)=3·len).
- Depth-3: op-type probe 0.193 ≈ surface 0.198 — at the wall the type is no longer model-latent either (the
  "thread", C19).

## (B) Deployability on PARAM-first-op tasks (no-think, n=148)
| arm | greedy@1 | cov@6 |
|---|---|---|
| no-hint | 0.000 | 0.007 |
| oracle-type (type only) | 0.007 | 0.034 |
| **oracle-full (op+param)** | **0.095** | **0.169** |
| probe-full (32-way probe) | 0.014 | 0.034 |
| surface-full (external I/O) | 0.027 | 0.054 |
| wrong-param (true type, wrong param) | 0.000 | 0.007 |

- **The parameter IS the deployable bottleneck** (isolated to param tasks): oracle-full 0.095 ≫ oracle-type
  0.007. Confirms C30.
- **The model probe barely delivers, and the cheap surface pipeline delivers MORE:** probe-full 0.014 <
  surface-full 0.027 — you do not need the 4B for the parameter. wrong-param 0.000 (content-causal).
- **Two-term check (textbook clean):** probe-full deploys *exactly* like the oracle on tasks it decodes
  correctly (0.091 = 0.091) and like no-hint on those it gets wrong (0.0 = 0.0) — a faithful readout that simply
  can't decode enough (26% concrete accuracy).
- (Non-param tasks: surface-full HURTS (0.009 < no-hint 0.045) — the surface classifier is bad at the structural
  op-types the model computes, consistent: surface is good at magnitudes/params, bad at model-latent types.)

## Implication
Sharp localization of the wall and of C30: the forward pass genuinely **computes the op-type** (a real latent
capability, elicitable training-free via C30's externalization) but has **no privileged representation of the
parameter** — it just reads it off surface I/O, which any trivial classifier does equally. So the training-free
latent-elicitation ceiling is the op-type; the parameter is not a model-latent thing to unearth. This closes the
C30 loop: the "deployable bottleneck" (param) is real but is not a locked-in-the-weights capability — it's
feature-engineering the model isn't special at.

## Next
- Compose type-probe (model) + param-from-I/O (surface) into one hint pipeline — does the hybrid deliver
  oracle-full's lift? (both pieces are cheap/training-free.)
- Depth-3: nothing to elicit (type also surface-parity at the wall).

## Artifact Manifest
See `reports/artifact_manifest.yaml`. Probe pickle + activations moved out of repo.
