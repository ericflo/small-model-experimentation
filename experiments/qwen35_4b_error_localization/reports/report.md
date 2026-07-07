# Can the model localize its own errors in multi-step reasoning? Yes — the confidence dip marks the origin

## Motivation
C40 showed single-step implicit metacognition (answer-token probability predicts correctness). This asks whether
that signal is **step-resolved**: on a multi-step chain, does per-step confidence drop at the step where the model
first goes wrong — enabling deployable error localization and targeted repair?

## Method (review-hardened)
The model advances k steps in a cyclic order over depth-4–7 chains, via **scaffolded decoding**: force the
`Step i: <digit>` format and read the digit distribution at each step (genuine live per-step commitment
confidence; no prose, no truncation, exact position alignment). Ground truth = **local correctness** (m_i ==
successor of the model's *own* previous output m_{i-1}), so a local error is a genuine slip and the first local
error is the origin. **Familiar (natural) order** — the model applies +k but slips (~31%/step, genuine arithmetic
errors). Novel/reversal orders were dropped: forced-scaffold makes the model apply a systematic *wrong* rule (no
single origin — the review's failure case).

**The make-or-break control — de-trending:** confidence *rises* with step position (0.66→0.96), so a naive
"lowest-confidence = error" could just track position. We subtract the per-position mean and require localization
to survive on the **residual**, plus baselines: uniform 1/D, always-last-step, position-prior.

## Results (600 chains)
- **Per-step error prediction survives de-trending:** AUROC 0.75 (de-trended) vs 0.73 (raw) — not a position
  artifact.
- **The dip marks the origin:** mean de-trended confidence by offset-from-first-error is minimized *exactly* at
  offset 0 (−0.15), high just before (+0.23 at −2), recovering after.
- **Localization (single-slip chains, n=137, well-posed):**

| method | localization accuracy |
|---|---|
| **de-trended residual (position-controlled)** | **0.56** |
| raw confidence | 0.64 |
| position-prior baseline | 0.36 |
| always-last | 0.01 |
| uniform 1/D | 0.19 |

- **Targeted repair (oracle-downstream, redo from located step):** fixes 0.56 of single-slip chains at avg 3.8
  steps vs 5.6 for redo-all — cheaper.

## Conclusion
The model's per-step confidence is **step-resolved**: it carries *where* it slipped, not just *that* it did. C40's
implicit metacognition composes over multi-step reasoning and enables deployable targeted repair.

## Honest caveats
- **Multi-slip chains (n=224):** the argmin finds *an* error 0.76 of the time but the *first* only 0.27 (several
  low-confidence steps compete) — localization is strongest when the model slips once (38% of error-chains).
- Execute-mode arithmetic slips (familiar order); forced-scaffold competence is lower than free-form (the scaffold
  strips reasoning) — this is deliberate, to produce genuine per-step slips. Single seed.

## Artifact Manifest
See `reports/artifact_manifest.yaml`.
