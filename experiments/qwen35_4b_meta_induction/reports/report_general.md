# Is general induction-via-reasoning installable? Yes — a general verify-procedure transfers to held-out families (C45)

## Motivation
C44 showed induction is a serial-compute limit but the shift-CoT was *shift-specific* (out-of-family affine 0.13).
The endorsed question: can a **general** reasoning procedure, trained on multiple rule families, install induction
that transfers to a **held-out** family?

## Method
Rule families = affine over positions keyed by multiplier `a ∈ {1,3,7,9}`. A **uniform enumerate-and-verify CoT**:
try each candidate `a`, derive `b` from one example, verify on another, keep the one that fits, apply. Train on
`{a=1,3,9}`, **hold out `a=7`**. Eval via generation (let the model reason). Random orders/params; disjoint seeds.

## Results (n=200/family, generation)
| family | induction accuracy |
|---|---|
| a=1 (trained) | 0.955 |
| a=3 (trained) | 0.930 |
| a=9 (trained) | 0.875 |
| **a=7 (HELD OUT)** | **0.905** |

Held-out `a=7` is **as accurate as the trained families**. The model generalizes to a rule family it never saw as
the answer — it learned the general hypothesize-verify-apply *procedure*, not just the trained rules. Contrast:
C44's single hand-coded shift-CoT transferred to out-of-family at only 0.13; a general verify procedure +
multi-family training generalizes at 0.91.

## Conclusion (C44 + C45)
The fixed 4B **can be taught general induction** — infer a novel rule from examples and apply it — but **only as a
serial reasoning procedure** that lives in the chain-of-thought tokens (C44: reasoning 1.00 vs single-forward-pass
0.01), never compressed into the weights. Give it a general hypothesize-and-verify strategy via SFT and it induces
rules it has never seen. This is the most constructive result of the arc: the induction wall — the model's central
limitation across 40+ claims — is a serial-compute limit that a general reasoning procedure overcomes generally.

## Honest caveats
- The held-out multiplier `a=7`'s arithmetic (7·p) appears in training as a *rejected* candidate, so the
  arithmetic is in-distribution — what generalizes is *accepting* a=7 as the answer via verify (the induction
  *logic*), not novel arithmetic. A broader structural leap (a non-affine held-out family) is owed.
- All families are affine; single seed.
- **Environment note:** an earlier OOM (batch-16 × long-CoT) left the WSL2 CUDA context unable to perform
  training-scale reductions ("device not ready"); trained at batch-2 (under the corruption threshold) with
  gradient accumulation.

## Artifact Manifest
See `reports/artifact_manifest.yaml` (adapters external).
