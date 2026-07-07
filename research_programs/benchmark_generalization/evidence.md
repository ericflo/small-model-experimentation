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
