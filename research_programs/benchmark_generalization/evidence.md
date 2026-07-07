# Evidence

## Seed Experiments

- [factor_recombination_ladder](../../experiments/factor_recombination_ladder/reports/factor_recombination_ladder_report.md)
- [feature_factorized_rule_diversity](../../experiments/feature_factorized_rule_diversity/reports/feature_factorized_rule_diversity_report.md)
- [targeted_bridge_allocation](../../experiments/targeted_bridge_allocation/reports/targeted_bridge_allocation_report.md)
- [qwen35_4b_sketch_coverage_shift_probe](../../experiments/qwen35_4b_sketch_coverage_shift_probe/reports/qwen35_4b_sketch_coverage_shift_probe_report.md)

## Current Read

The imported tracks include strong shift probes. Future work should make shift evaluation a cross-program norm rather than a special case.

- [qwen35_4b_language_reasoning_wall](../../experiments/qwen35_4b_language_reasoning_wall/reports/report.md) (claim C37): the compositional wall does NOT exist in language. The model chains depth-3+ multi-step SIMULATION in natural language near-perfectly (no-think), unlike the depth-3 formal-composition wall -- the wall is formal-modality-specific, not a general multi-step limit. Made-up-relation control confirms it is MODALITY not a semantic prior. Formal-dict triggers code-mode. Tests SIMULATION (C13), not the C32/C36 proposal wall.

- [qwen35_4b_language_proposal_wall](../../experiments/qwen35_4b_language_proposal_wall/reports/report.md) (claim C38): the structure-PROPOSAL wall persists in language. Depth-1 dissociation -- the model EXECUTES a given rule (0.86) but cannot INDUCE one from examples (0.00 no-think; 0.50 think). The model is an executor, not an inducer, in language as in formal domains. Complement to C37 (simulation intact): the wall's two components dissociate by modality -- execution formal-specific, induction modality-general.

- [qwen35_4b_icl_retrieval_vs_induction](../../experiments/qwen35_4b_icl_retrieval_vs_induction/reports/report.md) (claim C39): in-context learning is RETRIEVAL of familiar structure, not INDUCTION of novel structure. The model EXECUTES a novel rule perfectly (0.97) but cannot INDUCE it from examples (0.12=chance), while it induces a familiar rule far better (0.45); more examples don't rescue novel induction. Unifies the arc: executor/retriever of pretrained structure, not inducer of novel structure.

- [qwen35_4b_metacognitive_boundary](../../experiments/qwen35_4b_metacognitive_boundary/reports/report.md) (claim C40): the model knows when it will fail IMPLICITLY (answer-token probability predicts per-item correctness at AUROC 0.95 within a surface-matched cell, >> surface baseline 0.61) but NOT EXPLICITLY (self-verification P(True) 0.46 = chance; verbalized confidence a constant 100). Deployable: read logits for a confidence/abstain signal, never the self-report.

- [qwen35_4b_confidence_guided_compute](../../experiments/qwen35_4b_confidence_guided_compute/reports/report.md) (claim C41): beat sample-more with the model's own uncertainty. Confidence-select (argmax P(answer), verification-free) 0.62 beats flat self-consistency 0.48 at every budget; max P(answer) predicts solvability (AUROC 0.83) for abstention. Turns C40's calibrated confidence into a deployable compute tool; selection+abstention (not allocation) is the win.
