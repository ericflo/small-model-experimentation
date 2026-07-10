# Program Scorecards

Use these scorecards before opening a new experiment. They are intentionally short: the goal is to route ideas, avoid duplicate variants, and choose the next result that would most change the repository's shared beliefs.

For evidence-linked durable claims, use [claims/index.md](claims/index.md).

## Structured Execution And Compilers

- Program: [charter](../research_programs/structured_execution_and_compilers/charter.md)
- Current read: structured intermediates are one of the strongest seed signals, but representation and supervision effects are still entangled.
- Best next experiment: compare direct text, typed bytecode, latent slots, and stateful executors on one shared task suite with identical splits.
- Strong anchors: `qwen_structural_latent_compiler_expansion`, `qwen_typed_bytecode_expert_iteration`, `latent_executor`.
- Avoid repeating: another isolated positive compiler run without a direct-output baseline and harder held-out split.
- Evidence that advances the program: causal ablation showing which intermediate structure transfers across family, length, or paraphrase shifts.

## Evidence-Conditioned Selection

- Program: [charter](../research_programs/evidence_conditioned_selection/charter.md)
- Current read: candidate pools often contain correct outputs, but visible evidence is not yet strong enough to commit reliably.
- Best next experiment: build an abstaining visible-only selector benchmark with precision, recall, coverage, and hidden-oracle gap reported separately.
- Strong anchors: `qwen35_4b_retrieval_adapt_verify_scale`, `qwen35_4b_foofah_program_ensemble_consensus`, `qwen35_4b_independent_retrieval_consensus`.
- Avoid repeating: claiming a selector improved accuracy when it only lowered commit rate or used hidden labels.
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
- Current read: adaptation can reshape behavior, but oracle-supervised gains and deployable gains must be separated carefully; induction gains live in serial reasoning tokens, not a forced forward pass.
- Best next experiment: compositional-grammar reasoning-SFT after C45 -- held-out combinations, held-out productions, and held-out composition-depth with execute-ceiling, token-budget, and value-fill gates.
- Strong anchors: `qwen35_4b_constrained_coverage_dpo`, `qwen35_4b_live_tool_dagger`, `qwen35_4b_oracle_process_grpo`.
- Avoid repeating: adapter or preference runs whose artifacts cannot be audited or whose labels are not deployable.
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
- Current read: the corpus disabled native thinking everywhere; turning it on is a real deployable lever (MBPP greedy +15pp, claim C9), but a budget to control (overthinking hurts) and partly a compute/scaffold effect rather than reasoning.
- Best next experiment: a deployable STOP/MORE controller over the thinking budget vs the fixed ~1024 optimum at matched mean tokens; and a stronger content control (substitute a different task's thinking).
- Strong anchors: `qwen35_4b_thinking_budget_scaling`, `qwen_python_shaped_silent_executor`, `qwen35_4b_adaptive_evidence_budget_policy`.
- Avoid repeating: reporting a thinking-budget win without a shuffled/foreign-thinking control or without the overthinking decline; claiming an exact optimal budget from single-seed gaps.
- Evidence that advances the program: a controller or distillation that Pareto-beats fixed budgets, and a content control that isolates genuine reasoning from compute + scaffold + token-presence.

## Agentic Breadth Installation

- Program: [charter](../research_programs/agentic_breadth_installation/charter.md)
- Current read: the first blackbox-arbitrated install in the corpus — breadth-first expert iteration on a firewall-clean 12-family gym moved menagerie quick +0.223/+0.294 on two fresh paired seeds (HF backend, deterministic; claims C49/C50), with gym-internal transfer to never-trained families (+0.54/+0.61); the C43/C45/C48 locality laws do not extend to this regime, and the causal lever was gradient placement at the answer-emission seam (recovery arm + weighted loss), not dose.
- Best next experiment: round-3 re-harvest with the round-2 model (queued: `gauntlet_round3_expert_iteration`) plus the recovery-arm-only and breadth-vs-matched-dose ablations that split emission-seam repair from axis competence.
- Strong anchors: `qwen35_4b_gauntlet_breadth_round1`.
- Avoid repeating: evaluating adapters through vLLM runtime LoRA (C49 silent no-op — on-vs-off gate mandatory); training full-weight on the model's own verbatim naturally-closed chains (near-self-distillation); filtering training data to naturally-closed chains only (excludes the deployment-critical force-closed state); comparing HF-backend and vLLM-backend menagerie scores.
- Evidence that advances the program: compounding menagerie deltas across iteration rounds on fresh paired seeds; ablations attributing the delta between protocol-emission repair and axis competence; medium/slow-tier confirmations once the host fla-kernel fault is resolved.

## Agentic Breadth Installation

- Program: [charter](../research_programs/agentic_breadth_installation/charter.md)
- Current read: the first blackbox-arbitrated install in the corpus — breadth-first expert iteration on a firewall-clean 12-family gym moved menagerie quick +0.223/+0.294 on two fresh paired seeds (HF backend, deterministic; claims C49/C50), with gym-internal transfer to never-trained families (+0.54/+0.61); the C43/C45/C48 locality laws do not extend to this regime, and the causal lever was gradient placement at the answer-emission seam (recovery arm + weighted loss), not dose.
- Best next experiment: round-3 re-harvest with the round-2 model (queued: `gauntlet_round3_expert_iteration`) plus the recovery-arm-only and breadth-vs-matched-dose ablations that split emission-seam repair from axis competence.
- Strong anchors: `qwen35_4b_gauntlet_breadth_round1`.
- Avoid repeating: evaluating adapters through vLLM runtime LoRA (C49 silent no-op — on-vs-off gate mandatory); training full-weight on the model's own verbatim naturally-closed chains (near-self-distillation); filtering training data to naturally-closed chains only (excludes the deployment-critical force-closed state); comparing HF-backend and vLLM-backend menagerie scores.
- Evidence that advances the program: compounding menagerie deltas across iteration rounds on fresh paired seeds; ablations attributing the delta between protocol-emission repair and axis competence; medium/slow-tier confirmations once the host fla-kernel fault is resolved.
