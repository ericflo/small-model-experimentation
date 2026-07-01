# Evidence

## Seed Experiments

- [qwen35_4b_retrieval_adapt_verify_scale](../../experiments/qwen35_4b_retrieval_adapt_verify_scale/reports/final_report.md)
- [qwen35_4b_foofah_selective_program_fallback](../../experiments/qwen35_4b_foofah_selective_program_fallback/reports/report.md)
- [qwen35_4b_foofah_program_ensemble_consensus](../../experiments/qwen35_4b_foofah_program_ensemble_consensus/reports/report.md)
- [qwen35_4b_independent_retrieval_consensus](../../experiments/qwen35_4b_independent_retrieval_consensus/reports/final_report.md)

## Key Result

- [qwen35_4b_generator_verifier_gap](../../experiments/qwen35_4b_generator_verifier_gap/reports/report.md)
  (claim C10): the C2 wall is **plumbing, not capability**. A frozen 4B's black-box self-verification is
  weak/yes-biased with no-think (AUROC 0.77) but strong with thinking (AUROC 0.93); its own zero-training,
  deployable thinking-verifier selects best-of-8 to close **75%** of the pass@1(0.771)→oracle(0.890) gap
  (no-think 24%). Foreign-solution reject rate 1.00. So the model CAN tell its own good solutions from bad —
  once it thinks — and selection has real headroom.

## Current Read

The biggest strategic gap is selection under deployable evidence — and C10 says that gap is *fixable*: the
model's own thinking-verifier is a strong, deployable, zero-training selection signal. Future selection work
should (a) benchmark against the thinking-verifier before building trained selectors, (b) treat native
thinking as a verification lever (not only generation), and (c) still report oracle coverage only as a
diagnostic with the deployable decision rule as the main object. Top follow-up: wire the thinking-verifier
into a controller (vs/with the visible test) and measure the deployable accuracy-vs-token Pareto.
