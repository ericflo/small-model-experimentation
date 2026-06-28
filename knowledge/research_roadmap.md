# Research Roadmap

## Priority 1: Evidence-Conditioned Selection

Goal: convert candidate-pool coverage into deployable accuracy.

Suggested experiments:

- counterexample generation for visible-pass hidden-wrong candidates,
- independent implementation consensus across retrieved algorithms,
- verifier training on public-test-augmented evidence,
- abstaining selectors that optimize precision before recall,
- selector calibration under held-out families and longer horizons.

Best prior anchors:

- [qwen35_4b_retrieval_adapt_verify_scale](../experiments/qwen35_4b_retrieval_adapt_verify_scale/reports/final_report.md)
- [qwen35_4b_foofah_selective_program_fallback](../experiments/qwen35_4b_foofah_selective_program_fallback/reports/report.md)
- [qwen35_4b_operator_inventory_search_pilot](../experiments/qwen35_4b_operator_inventory_search_pilot/reports/qwen35_4b_operator_inventory_search_pilot_report.md)

## Priority 2: Structured Compiler Scaling

Goal: scale typed-slot, latent-register, and executable-ABI compilers while preserving generalization and paraphrase consistency.

Suggested experiments:

- replicate high-performing structural compiler runs across seeds,
- add harder length and compositional splits,
- compare direct program text, latent slots, and typed bytecode under identical data,
- measure failure modes by step, operator, and state prefix.

Best prior anchors:

- [qwen_structural_latent_compiler_expansion](../experiments/qwen_structural_latent_compiler_expansion/reports/structural_latent_compiler_expansion_report.md)
- [qwen_compiler_multiseed_reattribution](../experiments/qwen_compiler_multiseed_reattribution/reports/qwen_compiler_multiseed_reattribution_report.md)
- [qwen_typed_bytecode_expert_iteration](../experiments/qwen_typed_bytecode_expert_iteration/reports/qwen_typed_bytecode_expert_iteration_paper.md)

## Priority 3: Operator And Skill Inventories

Goal: make library growth useful without letting retrieval noise dominate.

Suggested experiments:

- train shortlisters over larger operator banks,
- active operator disambiguation with generated inputs,
- retrieval plus verified consensus rather than single retrieved skill cards,
- inventory coverage stress tests with held-out primitives.

Best prior anchors:

- [qwen35_4b_operator_inventory_search_pilot](../experiments/qwen35_4b_operator_inventory_search_pilot/reports/qwen35_4b_operator_inventory_search_pilot_report.md)
- [qwen_verified_skill_memory_rag](../experiments/qwen_verified_skill_memory_rag/reports/qwen_verified_skill_memory_rag_report.md)
- [qwen35_4b_inventory_shortlister_training](../experiments/qwen35_4b_inventory_shortlister_training/README.md)

## Priority 4: Active Evidence Acquisition

Goal: ask for or synthesize the few examples/tests that collapse the right uncertainty.

Suggested experiments:

- active example selection with learned uncertainty rather than simple disagreement,
- generated tests with expected-output-free agreement checks,
- budgeted evidence policies trained on visible-only features,
- family-specific acquisition policies for date/time and numeric transforms.

Best prior anchors:

- [qwen_active_example_acquisition](../experiments/qwen_active_example_acquisition/reports/qwen_active_example_acquisition_report.md)
- [qwen35_4b_adaptive_evidence_budget_policy](../experiments/qwen35_4b_adaptive_evidence_budget_policy/reports/qwen35_4b_adaptive_evidence_budget_policy_report.md)
- [qwen35_4b_active_counterexample_trace_selection](../experiments/qwen35_4b_active_counterexample_trace_selection/reports/qwen35_4b_active_counterexample_trace_selection_report.md)

