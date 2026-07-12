# Program Scorecards

Use these scorecards before opening a new experiment. They are intentionally short: the goal is to route ideas, avoid duplicate variants, and choose the next result that would most change the repository's shared beliefs.

For evidence-linked durable claims, use [claims/index.md](claims/index.md).

## Structured Execution And Compilers

- Program: [charter](../research_programs/structured_execution_and_compilers/charter.md)
- Current read: structured intermediates remain strong, but type-only partial viability is oracle-useful and model-unreadable; exact depth-5 list-DSL search is also cheaper than projected. A fixed `First:` slot repaired answer mode (41/48 unmasked alias tops) without yet making semantic choice reliable: the 1,024 hint was task/alias concentrated and missed its mixed-task gate.
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
- Current read: adaptation can reshape behavior, but the beyond-C53 line now exposes five launch boundaries: train on deployable commit states with an outcome-valid label; prove the update preserves neighboring behavior; prove every absolute gate is feasible; prove a proposed teacher is actually better on the student's same-prefix state distribution; and preserve verifier-conditioned recovery rather than merely matching operator marginals. The corrected Pareto run found `blend` negative on quick in both blocks, while success-only repository compression improved trained families but regressed family-disjoint transfer 49/72→25/72.
- Best next experiment: a fresh same-prefix continuation-advantage routing pilot frozen on disjoint calibration states, or a verifier-conditioned recovery bank/external scaffold controller with transition-level and locality preflights. Stop either line before training unless its necessary mechanism gate replicates.
- Strong anchors: `qwen35_4b_gauntlet_breadth_round1`, `qwen35_4b_gauntlet_frontier`, `qwen35_4b_bank_the_thoughts`, `qwen35_4b_answer_potential_trace_sft`, `qwen35_4b_think_ftpo_round2`, `qwen35_4b_specialist_policy_integration`, `qwen35_4b_pareto_policy_integration`, `qwen35_4b_repo_search_compress_bank`.
- Avoid repeating: adapter or preference runs whose artifacts cannot be audited, hidden-label wins without frozen alternatives, SFT launched from a scorer that missed its outcome gate, starting an infeasible mandatory arm, distilling from coarse external policy labels without replicated same-prefix advantage, scaling a sparse steering recipe whose non-target drift already failed locality, or assuming a success-only canonical trace teaches verifier-conditioned recovery.
- Evidence that advances the program: a trained behavior beats strong frozen/tool baselines without hidden-label leakage.

## Process Control And Tool Use

- Program: [charter](../research_programs/process_control_and_tool_use/charter.md)
- Current read: small models need explicit control policies for tools, budgets, and commit/repair decisions. Exact operator marginals are not enough: a compact repository bank preserved commit after pass but learned zero patch recovery after 48 failed tests, regressing held-out success 49/72→25/72.
- Best next experiment: compare STOP/MORE, commit/repair, and tool-choice policies with strict cost accounting and explicit failed-patch/failed-test transition gates; a recovery curriculum must contain changed second actions, not only successful terminal traces.
- Strong anchors: `qwen35_4b_adaptive_tool_controller`, `qwen35_4b_tool_state_policy_lora`, `qwen35_4b_adaptive_evidence_budget_policy`, `qwen35_4b_repo_search_compress_bank`.
- Avoid repeating: tool-use demos that omit the no-tool, fixed-tool, and random-tool baselines, or banks that balance action counts while deleting verifier-rejection contingencies.
- Evidence that advances the program: policy lift survives latency, token, and tool-call ceilings.

## Benchmark Generalization

- Program: [charter](../research_programs/benchmark_generalization/charter.md)
- Current read: many mechanisms look good in-family; the repository needs standard transfer stress before strategic claims harden. C46 shows the confidence toolkit survives MBPP->HumanEval only after the signal is re-expressed as a single-token P(True) readout. The Pareto qualification negative adds the reverse warning: a held-out instrument ranking can be real on that instrument yet fail to define a useful teacher ordering on the clean training proxy. Native-thinking termination is also non-portable: a 512--1024 scale that often closed on MBPP produced 0/48 natural closes at 1,024 on fresh list induction.
- Best next experiment: compositional-grammar induction as the C45 stress test, plus a small cross-program generalization suite used by compiler, selector, memory, and adaptation work.
- Strong anchors: `factor_recombination_ladder`, `feature_factorized_rule_diversity`, `targeted_bridge_allocation`.
- Avoid repeating: reporting only IID or narrow held-out splits for a mechanism meant to generalize.
- Evidence that advances the program: transfer across substrate, family, length, and real-task variants with a clear failure taxonomy.

## Interpretability And Diagnostics

- Program: [charter](../research_programs/interpretability_and_diagnostics/charter.md)
- Current read: diagnostics are useful when they explain why a mechanism transfers or fails, not when they are post-hoc decoration. Late answer-position J coordinates are writable but non-transporting; the early context-local clamp replicates 48/48 consequences with 1,440/1,440 exact control rows. Fixed `First:` syntax finally repaired answer mode after three interface stops, but its best semantic cell (15/48 at 1,024 versus 11/48 shuffled) had only five mixed tasks, missed the frozen six-task gate, and had task-level intervals crossing zero. Native-thought J value remains untested.
- Best next experiment: a powered fresh fixed-1,024 seam replication with task-level real-minus-no-thought and real-minus-shuffled uncertainty gates; only an independently replicated pass may resume J value with identity and dynamic per-length controls.
- Strong anchors: `qwen_structural_compiler_attribution_ablation`, `qwen35_4b_probe_to_prompt`, `qwen35_4b_jacobian_value_transport`, `qwen35_4b_context_local_jacobian_clamp`, `qwen35_4b_jacobian_transport_control_replication`, `qwen35_4b_native_thought_jacobian_value_transport`, `qwen35_4b_native_thought_seam_budget_ladder`, `qwen35_4b_forced_commit_jacobian_value_transport`, `qwen35_4b_commit_slot_jacobian_value_transport`.
- Avoid repeating: probes that do not change the next experiment, next-token writing tests presented as reasoning transport, promoting a perfect point estimate past a failed control gate, or presenting an oracle donor as a deployable gain.
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
- Current read: native thinking is a real coherent-content lever, but its budget, termination, and emission interface are workload-specific. After 0/48 natural closes and 12.5%--18.8% close-only parse, fixed `First:` syntax repaired answer mode; at 1,024 real thought reached 31.2% versus 25.0% no-thought and 22.9% shuffled. That hint was not stable enough: five mixed tasks missed six required and task-bootstrap intervals crossed zero. C52 separately shows that uncertainty can route forks without making shared-weight edits local.
- Best next experiment: a powered fixed-1,024 answer-slot replication with more fresh task units and task-level uncertainty gates; separately retain the 16k+ loop-control line.
- Strong anchors: `qwen35_4b_thinking_content_vs_compute`, `qwen35_4b_overthinking_content_ladder`, `qwen35_4b_answer_potential_trace_sft`, `qwen35_4b_native_thought_seam_budget_ladder`, `qwen35_4b_forced_commit_jacobian_value_transport`, `qwen35_4b_commit_slot_jacobian_value_transport`, `qwen35_4b_think_ftpo_round2`.
- Avoid repeating: thinking-budget wins without content controls, calibration on a different workload class, cap-bound score interpretation, larger-N harvesting before termination/locality works, or treating high varentropy as a monotone “push harder” signal.
- Evidence that advances the program: a controller or distillation that Pareto-beats fixed budgets, and a content control that isolates genuine reasoning from compute + scaffold + token-presence.

## Agentic Breadth Installation

- Program: [charter](../research_programs/agentic_breadth_installation/charter.md)
- Current read: breadth-first expert iteration remains the first blackbox-arbitrated install (+0.223/+0.294 menagerie quick; C49/C50), but C53 closes same-recipe scaling at a robust second wall. Beyond-wall work has now failed at four distinct prerequisites: non-local thought edits, semantic-operator capture under live-state DAgger, teacher construction, and success-only policy compression. The compact repository arm preserved operator marginals yet deleted failure-conditioned revision, improving trained families +16.7pp while dropping family-disjoint transfer −33.3pp. The two specialist/MOPD lines stopped before the mechanism—first on an impossible bar, then on absent clean teacher crossover.
- Best next experiment: replace coarse tier routing with a fresh disjoint same-prefix continuation-advantage router, or test verifier-conditioned recovery/external scaffolding with changed-revision, held-out transfer, and locality gates before training escalation.
- Strong anchors: `qwen35_4b_gauntlet_breadth_round1`, `qwen35_4b_gauntlet_frontier` (C53/C54), `qwen35_4b_specialist_policy_integration` (feasibility stop), `qwen35_4b_pareto_policy_integration` (teacher-transport stop), `qwen35_4b_think_ftpo_round2` (C52), `qwen35_4b_interactive_policy_curriculum`, `qwen35_4b_repo_search_compress_bank`.
- Avoid repeating: evaluating adapters through vLLM runtime LoRA (C49 silent no-op — on-vs-off gate mandatory); full-weight near-self-distillation; filtering away deployment-critical force-closed states; comparing HF and vLLM menagerie scores; scaling the exhausted emission-policy recipe; starting infeasible specialists; treating an external tier winner as a local teacher without same-prefix evidence; scaling non-local thought LoRA; broad unbalanced DAgger that can replace scarce verify/commit pivots; or success-only canonical banks whose operator totals are balanced but recovery transitions are absent.
- Evidence that advances the program: a beyond-recipe method that exceeds the C53 blend ceiling on fresh blackbox events, or a locality-passing thought intervention that beats both deep base and matched-compute sampling on held-out agentic tasks.
