# Program Scorecards

Use these scorecards before opening a new experiment. They are intentionally short: the goal is to route ideas, avoid duplicate variants, and choose the next result that would most change the repository's shared beliefs.

For evidence-linked durable claims, use [claims/index.md](claims/index.md).

## Structured Execution And Compilers

- Program: [charter](../research_programs/structured_execution_and_compilers/charter.md)
- Current read: structured intermediates remain strong, but type-only partial viability is oracle-useful and model-unreadable; exact depth-5 list-DSL search is also cheaper than projected.
- Best next experiment: measure the exact depth-6 resource crossover, then test residualized partial states only if learned pruning is actually needed.
- Strong anchors: `qwen35_4b_partial_structure_search`, `qwen35_4b_crosssubstrate_structure`, `qwen35_4b_structure_search_scaling`.
- Avoid repeating: another type-only P(viable) judge, pooled-AUROC launch, or model-guided search without measured brute wall time.
- Evidence that advances the program: causal ablation showing which intermediate structure transfers across family, length, or paraphrase shifts.

## Evidence-Conditioned Selection

- Program: [charter](../research_programs/evidence_conditioned_selection/charter.md)
- Current read: confidence can rank completed candidates, but existential partial reachability is not automatically readable; C51 further shows that a trace score may contain real task-relevant signal while conditioning on a post-thought state the model rarely reaches.
- Best next experiment: close the exact-pool visible-selector gap; for thought selection, test joint autonomous-close-plus-answer potential only after termination/parse gates, not a larger C51 pool.
- Strong anchors: `qwen35_4b_partial_structure_search`, `qwen35_4b_generator_verifier_gap`, `qwen35_4b_code_confidence`, `qwen35_4b_answer_potential_trace_sft`.
- Avoid repeating: pooled-AUROC confidence claims, type-only partial judges, answer-only potential over cap-bound traces, or accuracy gains that hide abstention/commit-rate changes.
- Evidence that advances the program: deployable selection gains under family-held-out candidate pools and adversarial visible examples.

## Active Evidence Acquisition

- Program: [charter](../research_programs/active_evidence_acquisition/charter.md)
- Current read: active examples and probes are promising, but the acquisition objective must be tied to downstream decision quality.
- Best next experiment: compare active probe policies by downstream selector lift under a fixed evidence budget.
- Strong anchors: `qwen_active_example_acquisition`, `qwen35_4b_active_counterexample_trace_selection`, `qwen35_4b_learned_active_trace_policy`.
- Avoid repeating: optimizing probe informativeness without showing that decisions improve.
- Evidence that advances the program: budget-normalized gains from probes that do not rely on expected answers at deployment time.

## Algorithmic Memory And Retrieval

- Program: [charter](../research_programs/algorithmic_memory_and_retrieval/charter.md)
- Current read: memory helps when it supplies verifiable candidates, constraints, or tests; naive context stuffing is weak.
- Best next experiment: compare retrieved examples, retrieved algorithms, retrieved tests, and retrieved failure cases on one task family.
- Strong anchors: `qwen_verified_skill_memory_rag`, `qwen35_4b_verified_algorithm_retrieval_adaptation`, `learned_sparse_slot_executor`.
- Avoid repeating: retrieval demos without negative retrieval controls or verification of the retrieved artifact.
- Evidence that advances the program: memory improves transfer while random or mismatched memory fails under the same budget.

## Operator And Skill Inventories

- Program: [charter](../research_programs/operator_and_skill_inventories/charter.md)
- Current read: inventories can scale coverage, but only if search and shortlisting remain reliable as the bank grows.
- Best next experiment: stress a skill/operator shortlister as distractor inventory size increases.
- Strong anchors: `qwen35_4b_operator_inventory_search_pilot`, `qwen35_4b_operator_inventory_scaling_stress`, `qwen35_4b_inventory_shortlister_training`.
- Avoid repeating: small-bank wins that do not test distractors, type collisions, or compositional reuse.
- Evidence that advances the program: graceful degradation curves and recovery strategies for large noisy inventories.

## Posttraining And Adaptation

- Program: [charter](../research_programs/posttraining_and_adaptation/charter.md)
- Current read: adaptation can reshape behavior, but C50-C53 now expose three boundaries: train on deployable commit states with an outcome-valid label; prove that the parameter update preserves neighboring behavior; and expect the emission-policy recipe to make one large step before reaching a second wall. A true label can beat shuffled labels yet still lose to base when shared-weight collateral dominates.
- Best next experiment: beyond-recipe mechanisms against C53 (tool-found scaffold banking or carefully guarded on-policy optimization); separately, locality-first positive thought steering (+0.25 or context-gated) stopped at an exact-logit preflight before any larger harvest.
- Strong anchors: `qwen35_4b_gauntlet_breadth_round1`, `qwen35_4b_gauntlet_frontier`, `qwen35_4b_bank_the_thoughts`, `qwen35_4b_answer_potential_trace_sft`, `qwen35_4b_think_ftpo_round2`.
- Avoid repeating: adapter or preference runs whose artifacts cannot be audited, hidden-label wins without frozen alternatives, SFT launched from a scorer that missed its outcome gate, or scaling a sparse steering recipe whose non-target drift already failed locality.
- Evidence that advances the program: a trained behavior beats strong frozen/tool baselines without hidden-label leakage.

## Process Control And Tool Use

- Program: [charter](../research_programs/process_control_and_tool_use/charter.md)
- Current read: small models need explicit control policies for tools, budgets, and commit/repair decisions.
- Best next experiment: compare STOP/MORE, commit/repair, and tool-choice policies on one benchmark with strict cost accounting.
- Strong anchors: `qwen35_4b_adaptive_tool_controller`, `qwen35_4b_tool_state_policy_lora`, `qwen35_4b_adaptive_evidence_budget_policy`.
- Avoid repeating: tool-use demos that omit the no-tool, fixed-tool, and random-tool baselines.
- Evidence that advances the program: policy lift survives latency, token, and tool-call ceilings.

## Benchmark Generalization

- Program: [charter](../research_programs/benchmark_generalization/charter.md)
- Current read: many mechanisms look good in-family; the repository needs standard transfer stress before strategic claims harden. C46 shows the confidence toolkit survives MBPP->HumanEval only after the signal is re-expressed as a single-token P(True) readout.
- Best next experiment: compositional-grammar induction as the C45 stress test, plus a small cross-program generalization suite used by compiler, selector, memory, and adaptation work.
- Strong anchors: `factor_recombination_ladder`, `feature_factorized_rule_diversity`, `targeted_bridge_allocation`.
- Avoid repeating: reporting only IID or narrow held-out splits for a mechanism meant to generalize.
- Evidence that advances the program: transfer across substrate, family, length, and real-task variants with a clear failure taxonomy.

## Interpretability And Diagnostics

- Program: [charter](../research_programs/interpretability_and_diagnostics/charter.md)
- Current read: diagnostics are useful when they explain why a mechanism transfers or fails, not when they are post-hoc decoration.
- Best next experiment: pressure-test a winning structured method with attribution, corrupted intermediates, and locality audits.
- Strong anchors: `qwen_structural_compiler_attribution_ablation`, `qwen35_4b_opsd_pressure_locality_audit`, `qwen_full_table_consistency_reranker`.
- Avoid repeating: probes that do not change the next experiment or expose a falsifiable mechanism.
- Evidence that advances the program: a diagnostic predicts which variants will fail before the final metric is observed.

## Reliability And Safety

- Program: [charter](../research_programs/reliability_and_safety/charter.md)
- Current read: reliability depends on precision, abstention, artifact hygiene, and explicit hidden-label boundaries.
- Best next experiment: create a reliability scorecard applied to selectors, verifiers, and tool controllers.
- Strong anchors: `qwen35_4b_reliability_exec_opsd_audit`, `qwen35_4b_real_sample_verify_commit`, `qwen_readable_candidate_verifier`.
- Avoid repeating: impressive accuracy reports without commit-rate, abstention, artifact, and leakage checks.
- Evidence that advances the program: reliability metrics expose regressions that ordinary accuracy hides.

## Collective Experimentation Infrastructure

- Program: [charter](../research_programs/collective_experimentation_infrastructure/charter.md)
- Current read: the repository now has program structure, validation, CI, and collaboration templates; the next lift is faster research navigation.
- Best next experiment: measure whether a new agent can find prior evidence and propose a non-duplicate experiment faster using scorecards and intake records.
- Strong anchors: `knowledge/research_program_index.md`, `knowledge/claims/initial_claims.md`, `docs/quality_gates.md`.
- Avoid repeating: adding process that slows pilots without improving memory or decision quality.
- Evidence that advances the program: navigation artifacts reduce duplicate proposals and improve citation of prior evidence.

## Test-Time Reasoning Budget

- Program: [charter](../research_programs/test_time_reasoning_budget/charter.md)
- Current read: native thinking is a real coherent-content lever, but its budget and termination are workload-specific. C51 shows that cap-bound traces can make a score counterfactual; C52 shows that entropy/varentropy can route useful forks without making a shared-weight edit context-local.
- Best next experiment: the registered 16k+ loop-control line; for deployed-budget thought steering, require an exact-logit locality preflight before any fresh outcome harvest.
- Strong anchors: `qwen35_4b_thinking_content_vs_compute`, `qwen35_4b_overthinking_content_ladder`, `qwen35_4b_answer_potential_trace_sft`, `qwen35_4b_think_ftpo_round2`.
- Avoid repeating: thinking-budget wins without content controls, calibration on a different workload class, cap-bound score interpretation, larger-N harvesting before termination/locality works, or treating high varentropy as a monotone “push harder” signal.
- Evidence that advances the program: a controller or distillation that Pareto-beats fixed budgets, and a content control that isolates genuine reasoning from compute + scaffold + token-presence.

## Agentic Breadth Installation

- Program: [charter](../research_programs/agentic_breadth_installation/charter.md)
- Current read: breadth-first expert iteration remains the first blackbox-arbitrated install (+0.223/+0.294 menagerie quick; C49/C50), but C53 closes same-recipe scaling at a robust second wall. A different attempt to install breadth by editing outcome-conditioned thought forks also failed: confident-outlier pull-up preserved a real-label advantage over shuffled training, but missed exact-logit locality and held-out repository coding (39/72 vs base 43/72). Signal placement at the answer-emission seam transfers once; sparse thought-token loss does not imply a sparse model change.
- Best next experiment: beyond-recipe mechanisms against C53 — tool-found scaffold banking, C29-guarded on-policy optimization, and residual-axis failure forensics. Keep any further think steering behind a locality-only preflight rather than another full capability run.
- Strong anchors: `qwen35_4b_gauntlet_breadth_round1`, `qwen35_4b_gauntlet_frontier` (C53), `qwen35_4b_think_ftpo_round2` (C52).
- Avoid repeating: evaluating adapters through vLLM runtime LoRA (C49 silent no-op — on-vs-off gate mandatory); full-weight near-self-distillation; filtering away deployment-critical force-closed states; comparing HF and vLLM menagerie scores; scaling the exhausted emission-policy recipe; or scaling confident-wrong-turn LoRA while non-target drift exceeds 0.10.
- Evidence that advances the program: a beyond-recipe method that exceeds the C53 blend ceiling on fresh blackbox events, or a locality-passing thought intervention that beats both deep base and matched-compute sampling on held-out agentic tasks.
