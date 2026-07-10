# Evidence

## Seed Experiments

- [qwen35_4b_retrieval_adapt_verify_scale](../../experiments/qwen35_4b_retrieval_adapt_verify_scale/reports/final_report.md)
- [qwen35_4b_foofah_selective_program_fallback](../../experiments/qwen35_4b_foofah_selective_program_fallback/reports/report.md)
- [qwen35_4b_foofah_program_ensemble_consensus](../../experiments/qwen35_4b_foofah_program_ensemble_consensus/reports/report.md)
- [qwen35_4b_independent_retrieval_consensus](../../experiments/qwen35_4b_independent_retrieval_consensus/reports/final_report.md)

## Key Result

- [qwen35_4b_partial_structure_search](../../experiments/qwen35_4b_partial_structure_search/reports/report.md)
  adds a hard negative boundary to confidence-based control. On 7,200 unfinished type-prefix children,
  thinking P(viable) had pooled AUROC 0.557 but task-macro AUROC 0.506 and recall@4 0.251; task-shuffled visible
  evidence was no worse. No-think was modestly stronger (0.556 AUROC, 0.303 recall) but still below the frozen
  actionability threshold. Thus C10/C46-style confidence selection does not transfer automatically from
  completed candidates to existential partial reachability. Search controllers must be evaluated within task
  and at the actual retained beam, and should expose residual evidence rather than only symbolic prefix names.

- [qwen35_4b_generator_verifier_gap](../../experiments/qwen35_4b_generator_verifier_gap/reports/report.md)
  (claim C10): the C2 wall is **plumbing, not capability**. A frozen 4B's black-box self-verification is
  weak/yes-biased with no-think (AUROC 0.77) but strong with thinking (AUROC 0.93); its own zero-training,
  deployable thinking-verifier selects best-of-8 to close **75%** of the pass@1(0.771)→oracle(0.890) gap
  (no-think 24%). Foreign-solution reject rate 1.00. So the model CAN tell its own good solutions from bad —
  once it thinks — and selection has real headroom.

- [qwen35_4b_verifier_selector_showdown](../../experiments/qwen35_4b_verifier_selector_showdown/reports/report.md)
  (matched-cost follow-up): on one k=8 pool, the thinking-verifier is **Pareto-dominated** when a cheap visible
  test exists — standalone 0.860 ≈ visible-only 0.850 at ~5× cost; the deployable sweet spot is **visible +
  free no-think verifier (0.870)**, 83% of the pass@1→oracle gap. Expensive thinking-verification only earns
  its cost in verifier-only settings. So the C2 wall is fixable with *cheap* plumbing.

- [qwen35_4b_code_confidence](../../experiments/qwen35_4b_code_confidence/reports/report.md)
  (claim C46, MBPP leg): in the verifier-free code regime, single-token P(True) is the selector to beat. MBPP
  P(True) selection 0.762 beats public-output majority 0.721 and random 0.696. When a visible test exists,
  execute it first (MBPP visible-test execution 0.816), then use confidence for abstain/route or no-test
  settings.

- [qwen35_4b_humaneval_code_confidence](../../experiments/qwen35_4b_humaneval_code_confidence/reports/report.md)
  (claim C46, HumanEval replication): in the strict no-public-probe setting, P(True) selection 0.835 beats
  mean-logprob 0.787 and random 0.766, with oracle pass@8 0.872. This is the clean no-verifier selection
  replication.

## Current Read

The biggest strategic gap is selection under deployable evidence — and C10/C46 say that gap is *fixable* with
cheap plumbing: execute visible evidence when it exists, then use a free no-think P(True) readout for
verifier-free selection, abstention, or routing. Thinking-verification is reserved for verifier-only settings
where its added cost beats the no-think readout. Future selection work
should (a) benchmark against the thinking-verifier before building trained selectors, (b) treat native
thinking as a verification lever (not only generation), and (c) still report oracle coverage only as a
diagnostic with the deployable decision rule as the main object. Top follow-up: wire the thinking-verifier
into a controller (vs/with the visible test) and measure the deployable accuracy-vs-token Pareto.

The partial-structure result narrows that optimism: confidence is useful when correctness is readable from a
completed candidate, but an existential unfinished-state judgment can collapse to task difficulty. Any
controller follow-up must report task-macro discrimination, sibling recall at the deployed beam, a task-
shuffled evidence canary, and prefill-inclusive compute—not just pooled AUROC.
