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
