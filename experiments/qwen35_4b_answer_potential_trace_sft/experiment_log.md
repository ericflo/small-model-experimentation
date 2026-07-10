# Qwen3.5-4B Answer-Potential Trace SFT Experiment Log

## Design Freeze

The experiment was scaffolded after repository-wide related-work review. The agreed plan was written
into `README.md` and `reports/preregistration.md`, hardened in `reports/design_review.md`, and must be
committed before any GPU-scale generation. Closest duplicate: C28 / `qwen35_4b_bank_the_thoughts`.

No benchmark contents were read. No GPU-scale work had run at this boundary.

## 2026-07-10 — Implementation And GPU Smoke

- Preserved and committed the complete design at `3441dd23` before GPU-scale work.
- Copied the procedural atom generators into separate training and evaluation-only held-family
  registries; generated all frozen splits and passed ID/prompt/digest/seed disjointness checks.
- Added verifier-equivalent answer sets. Excluded stallwright from confirmatory potential scoring
  because its arbitrary order/name aliases make the finite string event combinatorial.
- First four-trace smoke exposed vLLM 0.24's unused full-vocabulary prompt-rank compile. No scientific
  result was observed. Added the dated preregistration amendment and documented the footgun.
- The corrected targeted-likelihood smoke passed: 4/4 traces scored with finite values, exact CUDA
  graph geometry resolved, live KV capacity fit, and fresh continuation sampling completed. The
  deliberately tiny 64-token thoughts all required force-close, so their rollout accuracy is not a
  scientific diagnostic.

## 2026-07-10 — G0 Recovery Note

The complete G0 GPU artifacts were written before the final CPU reduction hit a configuration-key
typo (`premention` was read as `premember`). No model output was lost or regenerated. The orchestrator
now has an explicit `--stage analyze-g0` recovery path that verifies and reduces the saved artifacts;
the typo and recovery are part of the provenance rather than hidden by rerunning inference.

## 2026-07-10 — Terminal G0 Result

- Reduced 2,048 sampled thoughts and 14,848 disjoint-seed answer rollouts from the saved artifacts.
- G0 failed with 3/8 stored criteria passing: answer gain AUROC 0.6167 < 0.65; top-one uplift was
  +0.0733 over seeded random and +0.0582 over shortest, both below the required +0.10; the
  pre-answer-mention fraction was 0.5690 < 0.75.
- Mechanism controls were positive: real thoughts beat token-shuffled (+0.5554 nats) and foreign
  thoughts (+4.7906 nats), and answer-format rank stability was Kendall tau 0.8301.
- Thought prior log-probability had not been requested during generation. Its comparator was therefore
  unavailable, serialized as JSON `null`, and failed closed. Other independent failures already made
  the verdict negative.
- Diagnosed a deployment-seam failure: 99.37% of thoughts contacted the 512-token cap and answer
  rollouts parsed only 13.21%, while parsed answers were 86.90% correct.
- Exercised `--stage full`; the guard wrote `runs/full_refusal.json` and refused the N=128 harvest/SFT
  exactly as preregistered. No adapter or external model artifact was created.
- Final decision: `SCORER_NEGATIVE`. Preserve the modest signal and controls, but do not retune or
  scale this experiment after seeing the gate.
