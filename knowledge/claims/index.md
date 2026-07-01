# Claim Index

Generated from `knowledge/claims/claim_ledger.json`. Edit the ledger, not this file.

- Claims: 10

## Status Counts

| Status | Claims |
| --- | ---: |
| Confirmed | 5 |
| Negative | 1 |
| Open | 1 |
| Promising | 3 |

## Program Counts

| Program | Claims |
| --- | ---: |
| `active_evidence_acquisition` | 1 |
| `algorithmic_memory_and_retrieval` | 1 |
| `benchmark_generalization` | 1 |
| `collective_experimentation_infrastructure` | 2 |
| `evidence_conditioned_selection` | 3 |
| `interpretability_and_diagnostics` | 1 |
| `operator_and_skill_inventories` | 1 |
| `posttraining_and_adaptation` | 1 |
| `process_control_and_tool_use` | 3 |
| `reliability_and_safety` | 4 |
| `structured_execution_and_compilers` | 1 |
| `test_time_reasoning_budget` | 2 |

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

## C9: Native thinking's accuracy gain is coherent reasoning content at every budget (unused lever)

- Status: `Promising`
- Programs: `test_time_reasoning_budget`, `process_control_and_tool_use`
- Summary: Enabling Qwen3.5-4B's native thinking (disabled corpus-wide) raises deployable MBPP greedy pass@1 +15pp (0.76->0.91); greedy accuracy is non-monotonic (optimum ~1024, then decline), though sampled full-pass is roughly flat across budgets. A visible-test budget controller is an efficiency (not accuracy) win, bounded by C2 false-passes. NATURE OF THE GAIN (decomposed via filler/shuffle/foreign controls across budgets 512/1024/2048): the model USES thinking as CONTENT -- splicing a different task's thinking (foreign) collapses accuracy to ~3-4% because the model follows it to the wrong problem; pure forward compute (contentless '.' filler) ~= no_think at every budget; relevant-but-scrambled thinking (shuffle) ~= no_think; and coherent reasoning (real) is the ENTIRE gain (real-shuffle = +0.105/+0.108/+0.150 at 512/1024/2048 -- it GROWS with budget). So the thinking accuracy gain is coherent reasoning CONTENT at every budget, not compute, scaffold, or token-presence. The earlier 'mostly compute/scaffold, not reasoning' read was wrong -- it only appeared through a greedy-metric lens and a shuffle-protocol artifact in the scaling run (its '2048 shuffle~=real'). Separately, at the REPRESENTATIONAL level a linear probe on the answer-token activation finds correctness moderately decodable and thinking raises decodability, but shuffle>=real there (noisy, overlapping CIs) -- i.e. coherence improves ACCURACY without clearly improving internal correctness-decodability.
- Implication: Re-baseline CoT-substitute results against a fair, budgeted native-thinking baseline. Native thinking's accuracy benefit is genuine reasoning content the model uses (credit it as such) -- but judge 'is it reasoning?' with content controls (filler + foreign), not a single greedy number, which misled here.

### Evidence

- [`qwen35_4b_thinking_budget_scaling`](../../experiments/qwen35_4b_thinking_budget_scaling/reports/report.md)
- [`qwen35_4b_thinking_budget_controller`](../../experiments/qwen35_4b_thinking_budget_controller/reports/report.md)
- [`qwen35_4b_thinking_separability_probe`](../../experiments/qwen35_4b_thinking_separability_probe/reports/report.md)
- [`qwen35_4b_thinking_content_vs_compute`](../../experiments/qwen35_4b_thinking_content_vs_compute/reports/report.md)
- [`qwen35_4b_overthinking_content_ladder`](../../experiments/qwen35_4b_overthinking_content_ladder/reports/report.md)
- [`qwen_python_shaped_silent_executor`](../../experiments/qwen_python_shaped_silent_executor/reports/qwen_python_shaped_silent_executor_report.md)

### Next Tests

- High-budget (1024/2048) ladder to confirm the coherence advantage shrinks under overthinking.
- Contamination-controlled / harder substrate where the no-think baseline is weaker (more headroom).
- A learned controller using internal/uncertainty signals vs the visible-test C2 wall.

### Avoid

- Claiming thinking gains are 'reasoning' without a content control that removes token-presence.
- Pinning an exact optimal budget from single-seed n=100 gaps.

## C10: The C2 selection wall is plumbing not capability: a thinking-verifier closes most of it

- Status: `Promising`
- Programs: `evidence_conditioned_selection`, `test_time_reasoning_budget`
- Summary: For a frozen Qwen3.5-4B on MBPP, intrinsic black-box self-verification (judge a candidate correct/incorrect from the A/B logit, no execution/hidden tests) is WEAK with no-think (balanced-acc 0.627, AUROC 0.773, heavy yes-bias: says 'correct' 0.91 vs base pass 0.771) but STRONG with thinking (balanced-acc 0.827, AUROC 0.926). The model's own thinking-verifier, zero-training and fully deployable (no execution), selects best-of-8 at 0.860, closing 75% of the pass@1(0.771)->oracle(0.890) gap (no-think closes only 24%); foreign-solution reject rate is 1.00. So the C2 coverage-vs-selection wall is a plumbing/evidence problem, not a verification-CAPABILITY limit -- the selection program has real headroom and the lever is thinking-augmented self-verification (cheaper/stronger than the trained selectors the corpus favored). This inverts C9: thinking helps VERIFICATION (+0.20 balanced-acc) at least as much as GENERATION -- its deepest value may be helping the model recognize correct answers, not just produce them.
- Implication: Selection work should try the model's own thinking-verifier before elaborate trained selectors; and native thinking should be evaluated as a verification/selection lever, not only a generation lever. Wire the thinking-verifier as a controller signal (vs/with the visible test).

### Evidence

- [`qwen35_4b_generator_verifier_gap`](../../experiments/qwen35_4b_generator_verifier_gap/reports/report.md)

### Next Tests

- Wire the thinking-verifier as the controller signal vs/with the visible test; deployable accuracy-vs-token Pareto.
- Verify a think-generated (harder-negative) candidate pool, and on a contamination-controlled substrate.
- Iterated generate->self-verify->revise self-correction loop driven by the strong thinking-verifier.

### Avoid

- Reporting no-think self-verification as strong -- it is yes-biased (0.91) and weak (AUROC 0.77).
- Treating verifier-selected accuracy as the oracle -- it is deployable but below the pass@k ceiling.
