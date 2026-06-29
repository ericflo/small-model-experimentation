# Claim Index

Generated from `knowledge/claims/claim_ledger.json`. Edit the ledger, not this file.

- Claims: 8

## Status Counts

| Status | Claims |
| --- | ---: |
| Confirmed | 5 |
| Negative | 1 |
| Open | 1 |
| Promising | 1 |

## Program Counts

| Program | Claims |
| --- | ---: |
| `active_evidence_acquisition` | 1 |
| `algorithmic_memory_and_retrieval` | 1 |
| `benchmark_generalization` | 1 |
| `collective_experimentation_infrastructure` | 2 |
| `evidence_conditioned_selection` | 2 |
| `interpretability_and_diagnostics` | 1 |
| `operator_and_skill_inventories` | 1 |
| `posttraining_and_adaptation` | 1 |
| `process_control_and_tool_use` | 2 |
| `reliability_and_safety` | 4 |
| `structured_execution_and_compilers` | 1 |

## C1: Structured intermediates improve small-model reliability

- Status: `Confirmed`
- Programs: `structured_execution_and_compilers`, `reliability_and_safety`
- Summary: Executable, typed, latent, or otherwise structured intermediates repeatedly create checkable surfaces that direct final-answer prompting lacks.
- Implication: Future direct-answer tasks should include a structured-output or executable-evidence baseline before making broad claims.

### Evidence

- [`qwen_structural_latent_compiler_expansion`](../../experiments/qwen_structural_latent_compiler_expansion/reports/structural_latent_compiler_expansion_report.md)
- [`qwen35_4b_foofah_selective_program_fallback`](../../experiments/qwen35_4b_foofah_selective_program_fallback/reports/report.md)
- [`qwen35_4b_operator_inventory_search_pilot`](../../experiments/qwen35_4b_operator_inventory_search_pilot/reports/qwen35_4b_operator_inventory_search_pilot_report.md)

### Next Tests

- Compare direct text, typed bytecode, latent slots, and stateful executors on one shared suite.
- Ablate supervision source separately from representation.

### Avoid

- Another isolated structured-output win without direct-output and corrupted-intermediate controls.

## C2: Candidate coverage does not imply deployable accuracy

- Status: `Confirmed`
- Programs: `evidence_conditioned_selection`, `reliability_and_safety`
- Summary: Candidate pools can contain hidden-correct answers while visible evidence still selects or commits incorrectly.
- Implication: Candidate-pool work should report oracle coverage, deployable selection, false-pass rate, precision, recall, and abstention separately.

### Evidence

- [`qwen35_4b_retrieval_adapt_verify_scale`](../../experiments/qwen35_4b_retrieval_adapt_verify_scale/reports/final_report.md)
- [`qwen35_4b_foofah_selective_program_fallback`](../../experiments/qwen35_4b_foofah_selective_program_fallback/reports/report.md)
- [`qwen35_4b_independent_retrieval_consensus`](../../experiments/qwen35_4b_independent_retrieval_consensus/reports/final_report.md)

### Next Tests

- Build visible-only selector benchmarks with family-held-out candidate pools.
- Convert oracle-ceiling reports into deployable-gap scorecards.

### Avoid

- Claiming selection progress from hidden-oracle coverage alone.

## C3: Plain prompt memory is not enough

- Status: `Negative`
- Programs: `algorithmic_memory_and_retrieval`, `operator_and_skill_inventories`
- Summary: Naive retrieved skill cards are a weak memory mechanism unless retrieval contributes candidates, tests, constraints, or verifier inputs.
- Implication: Memory experiments should include random, mismatched, and constraint/test/candidate variants rather than only prompt-context variants.

### Evidence

- [`qwen_verified_skill_memory_rag`](../../experiments/qwen_verified_skill_memory_rag/reports/qwen_verified_skill_memory_rag_report.md)
- [`qwen35_4b_verified_algorithm_retrieval_adaptation`](../../experiments/qwen35_4b_verified_algorithm_retrieval_adaptation/reports/final_report.md)

### Next Tests

- Compare retrieved examples, retrieved algorithms, retrieved tests, and retrieved failure cases on one task family.

### Avoid

- Treating retrieval as useful because it looks semantically relevant but is never verified.

## C4: Active evidence must be coupled to downstream decisions

- Status: `Promising`
- Programs: `active_evidence_acquisition`, `process_control_and_tool_use`, `evidence_conditioned_selection`
- Summary: Actively selected examples and probes can help, but the acquisition objective must optimize the final selector, verifier, or controller decision.
- Implication: Evidence-acquisition work should report downstream decision lift per evidence budget, not only probe informativeness.

### Evidence

- [`qwen_active_example_acquisition`](../../experiments/qwen_active_example_acquisition/reports/qwen_active_example_acquisition_report.md)
- [`qwen35_4b_active_counterexample_trace_selection`](../../experiments/qwen35_4b_active_counterexample_trace_selection/reports/qwen35_4b_active_counterexample_trace_selection_report.md)
- [`qwen35_4b_learned_active_trace_policy`](../../experiments/qwen35_4b_learned_active_trace_policy/reports/qwen35_4b_learned_active_trace_policy_report.md)

### Next Tests

- Compare active probe policies by selector lift under identical budgets.
- Separate expected-output-free probes from oracle-only diagnostics.

### Avoid

- Optimizing examples or probes without showing commit, repair, or selection improvement.

## C5: Adaptation must beat strong frozen alternatives

- Status: `Open`
- Programs: `posttraining_and_adaptation`, `process_control_and_tool_use`
- Summary: LoRA, DPO, distillation, GRPO, and DAgger are promising, but their value is unclear unless compared against frozen sampling, verification, retrieval, and tool-control alternatives.
- Implication: Post-training runs should include frozen inference-time alternatives and explicit deployability boundaries.

### Evidence

- [`qwen35_4b_constrained_coverage_dpo`](../../experiments/qwen35_4b_constrained_coverage_dpo/reports/final_report.md)
- [`qwen35_4b_live_tool_dagger`](../../experiments/qwen35_4b_live_tool_dagger/reports/report.md)
- [`qwen35_4b_oracle_process_grpo`](../../experiments/qwen35_4b_oracle_process_grpo/reports/qwen35_4b_oracle_process_grpo_report.md)

### Next Tests

- Run one update method against frozen inference-time alternatives on the same selector or compiler benchmark.

### Avoid

- Adapter wins whose labels are hidden-only or whose artifacts are not auditable.

## C6: Controls are the difference between a result and a story

- Status: `Confirmed`
- Programs: `benchmark_generalization`, `interpretability_and_diagnostics`, `reliability_and_safety`
- Summary: The strongest imported lessons come from experiments that separate mechanism from confounders through corrupted, shuffled, random, frozen, length-held-out, or family-held-out controls.
- Implication: Future claims should name the mechanism-falsifying control before running expensive work.

### Evidence

- [`factor_recombination_ladder`](../../experiments/factor_recombination_ladder/reports/factor_recombination_ladder_report.md)
- [`feature_factorized_rule_diversity`](../../experiments/feature_factorized_rule_diversity/reports/feature_factorized_rule_diversity_report.md)
- [`qwen_structural_compiler_attribution_ablation`](../../experiments/qwen_structural_compiler_attribution_ablation/reports/structural_compiler_attribution_ablation_report.md)

### Next Tests

- Build a shared generalization and control suite used across compiler, selector, memory, and adaptation work.

### Avoid

- Treating train or IID lift as evidence for generalization.

## C7: Reliability requires explicit hidden-label and artifact boundaries

- Status: `Confirmed`
- Programs: `reliability_and_safety`, `collective_experimentation_infrastructure`
- Summary: The repository should preserve small reproducible artifacts while excluding trained adapters and labeling oracle-only evidence separately from deployable evidence.
- Implication: Reports and scaffolds must separate deployable evidence from hidden-oracle evaluation, and trained adapter directories must remain out of git.

### Evidence

- [`qwen35_4b_reliability_exec_opsd_audit`](../../experiments/qwen35_4b_reliability_exec_opsd_audit/reports/final_report.md)
- [`qwen35_4b_real_sample_verify_commit`](../../experiments/qwen35_4b_real_sample_verify_commit/reports/qwen35_4b_real_sample_verify_commit_report.md)
- [`.gitignore`](../../.gitignore)
- [`scripts/validate_repository.py`](../../scripts/validate_repository.py)

### Next Tests

- Add stronger hidden-label boundary and artifact-manifest checks as new experiments are created.

### Avoid

- Checking in trained adapters or letting oracle-only metrics become headline deployment claims.

## C8: The repository itself is a research instrument

- Status: `Confirmed`
- Programs: `collective_experimentation_infrastructure`
- Summary: Program charters, generated indexes, claim ledgers, scorecards, intake notes, templates, validation, and CI are part of the experiment system because they determine whether future work compounds.
- Implication: New experiments should update program evidence and shared claims when they change what future work should believe.

### Evidence

- [`research_programs/registry.yaml`](../../research_programs/registry.yaml)
- [`knowledge/program_scorecards.md`](../../knowledge/program_scorecards.md)
- [`docs/quality_gates.md`](../../docs/quality_gates.md)
- [`.github/workflows/validate.yml`](../../.github/workflows/validate.yml)

### Next Tests

- Measure whether scorecards and intake notes reduce duplicate proposals and improve citation of prior evidence.

### Avoid

- Treating process as complete when it no longer improves research memory.
