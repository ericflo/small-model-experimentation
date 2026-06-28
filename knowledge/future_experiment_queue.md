# Future Experiment Queue

Generated from `knowledge/future_experiment_queue.json`. Edit the JSON source, not this file.

This queue is intentionally broader than the imported prototype corpus. It is a launchpad for future experiments, candidate programs, infrastructure work, and falsifiable probes.

- Proposals: 35
- Existing research programs covered: 11 / 11
- Candidate program lines: 5

## Status Counts

| Status | Proposals |
| --- | ---: |
| `infrastructure` | 6 |
| `program-seed` | 15 |
| `ready-for-intake` | 14 |

## Priority Counts

| Priority | Proposals |
| --- | ---: |
| `P0` | 12 |
| `P1` | 16 |
| `P2` | 7 |

## Candidate Program Lines

### Multimodal And Embodied Small Models

- Candidate id: `multimodal_and_embodied_small_models`
- Focus: Test whether structured execution and evidence-selection patterns survive when inputs include images, UI state, audio, or physical observations.

### Data Generation And Synthetic Curriculum Design

- Candidate id: `data_generation_and_synthetic_curriculum_design`
- Focus: Learn which synthetic curricula create transfer rather than benchmark-specific competence.

### Small-Model Collaboration

- Candidate id: `small_model_collaboration`
- Focus: Measure whether multiple specialized small models outperform one single-policy small model under fixed budget.

### On-Device And Latency-Constrained Agents

- Candidate id: `on_device_and_latency_constrained_agents`
- Focus: Find mechanisms that preserve gains under strict inference, memory, latency, and tool-call ceilings.

### Human-Agent Research Interfaces

- Candidate id: `human_agent_research_interfaces`
- Focus: Build interfaces that help people and agents inspect, branch, and critique experiment evidence quickly.

## By Program

### Structured Execution And Compilers

- Proposals: 4

- `typed_vs_latent_vs_text_compiler_suite` (`P0`, `ready-for-intake`): Which structured representation causes generalization: typed bytecode, latent slots, text programs, or execution feedback?
- `compiler_shift_replication_grid` (`P0`, `ready-for-intake`): Which compiler-style gains survive seed, length, operator, and paraphrase shifts?
- `supervision_causality_ablation` (`P1`, `ready-for-intake`): Is the lift from state-prefix supervision, final-answer supervision, program-token supervision, or data filtering?
- `multimodal_visual_table_transform_probe` (`P1`, `program-seed`): Do executable intermediates help small models transform visual table inputs?

### Evidence-Conditioned Selection

- Proposals: 6

- `visible_abstention_selector_benchmark` (`P0`, `ready-for-intake`): Can a selector improve precision without hiding failures behind an unreported commit-rate drop?
- `deployable_gap_scorecards` (`P0`, `ready-for-intake`): Which oracle-ceiling results remain promising after visible-only constraints are made explicit?
- `stop_more_controller_visible_labels` (`P0`, `ready-for-intake`): Can a controller decide when to gather more evidence using only visible training labels?
- `independent_consensus_vs_selector_pool` (`P1`, `ready-for-intake`): When does independent implementation consensus beat a trained selector on the same candidate pool?
- `operator_bank_shortlister_scaling` (`P1`, `ready-for-intake`): Can a shortlister preserve selected accuracy as operator banks grow?
- `multi_model_role_ensemble_pool` (`P1`, `program-seed`): Can generator, verifier, critic, and evidence-acquirer roles outperform a single small-model policy under fixed total budget?

### Active Evidence Acquisition

- Proposals: 5

- `expected_output_free_counterexample_tests` (`P0`, `ready-for-intake`): Can acquired tests distinguish candidates without requiring hidden expected outputs?
- `acquisition_policy_comparison_pool` (`P0`, `ready-for-intake`): Which acquisition rule wins when uncertainty, disagreement, information gain, and learned policies share one pool?
- `constraint_memory_counterexample_generator` (`P1`, `program-seed`): Can retrieved failures become counterexample generators instead of answer hints?
- `family_aware_evidence_policy` (`P1`, `program-seed`): Can acquisition policies adapt to date/time, numeric, table, and code task families without overfitting?
- `active_disambiguation_type_collision` (`P2`, `program-seed`): Can active probes separate operators that share type signatures but differ semantically?

### Algorithmic Memory And Retrieval

- Proposals: 4

- `memory_mode_comparison_suite` (`P0`, `ready-for-intake`): Which memory use is actually useful: prompt context, constraints, tests, candidates, or failure cases?
- `independent_consensus_vs_selector_pool` (`P1`, `ready-for-intake`): When does independent implementation consensus beat a trained selector on the same candidate pool?
- `constraint_memory_counterexample_generator` (`P1`, `program-seed`): Can retrieved failures become counterexample generators instead of answer hints?
- `compressed_memory_budget_probe` (`P2`, `program-seed`): Can memory remain useful when context and storage are severely compressed?

### Operator And Skill Inventories

- Proposals: 3

- `operator_card_schema_probe` (`P1`, `ready-for-intake`): What operator-card fields make inventories reusable across experiments?
- `operator_bank_shortlister_scaling` (`P1`, `ready-for-intake`): Can a shortlister preserve selected accuracy as operator banks grow?
- `active_disambiguation_type_collision` (`P2`, `program-seed`): Can active probes separate operators that share type signatures but differ semantically?

### Posttraining And Adaptation

- Proposals: 3

- `posttraining_method_shared_substrate` (`P0`, `ready-for-intake`): How do SFT, DPO, process distillation, and DAgger compare on one candidate/evidence substrate?
- `adapter_free_repro_manifest_audit` (`P0`, `infrastructure`): Can every trained-run experiment be audited without committing adapters or checkpoints?
- `hard_negative_training_transfer` (`P1`, `program-seed`): Does training on hard negatives improve real selection, or does it overfit visible failure artifacts?

### Process Control And Tool Use

- Proposals: 7

- `acquisition_policy_comparison_pool` (`P0`, `ready-for-intake`): Which acquisition rule wins when uncertainty, disagreement, information gain, and learned policies share one pool?
- `stop_more_controller_visible_labels` (`P0`, `ready-for-intake`): Can a controller decide when to gather more evidence using only visible training labels?
- `noisy_tool_controller_stress` (`P1`, `program-seed`): Do process policies still help when tools are flaky, delayed, or misleading?
- `latency_ceiling_selector_rerun` (`P1`, `program-seed`): Which selector or tool-control gains survive strict latency, token, and call ceilings?
- `pressure_diagnostic_preflight` (`P2`, `program-seed`): Can cheap token-pressure and execution-pressure probes predict when an expensive run is worth doing?
- `ui_state_tool_execution_probe` (`P2`, `program-seed`): Can process-control policies transfer from text/table tools to UI-state observations?
- `committee_verifier_critic_loop` (`P2`, `program-seed`): Can a committee decide when critique or verification is worth another step?

### Benchmark Generalization

- Proposals: 6

- `typed_vs_latent_vs_text_compiler_suite` (`P0`, `ready-for-intake`): Which structured representation causes generalization: typed bytecode, latent slots, text programs, or execution feedback?
- `compiler_shift_replication_grid` (`P0`, `ready-for-intake`): Which compiler-style gains survive seed, length, operator, and paraphrase shifts?
- `memory_mode_comparison_suite` (`P0`, `ready-for-intake`): Which memory use is actually useful: prompt context, constraints, tests, candidates, or failure cases?
- `posttraining_method_shared_substrate` (`P0`, `ready-for-intake`): How do SFT, DPO, process distillation, and DAgger compare on one candidate/evidence substrate?
- `family_aware_evidence_policy` (`P1`, `program-seed`): Can acquisition policies adapt to date/time, numeric, table, and code task families without overfitting?
- `synthetic_curriculum_transfer_bakeoff` (`P1`, `program-seed`): Which synthetic curriculum source transfers best: human-designed, model-generated, or failure-mined?

### Interpretability And Diagnostics

- Proposals: 3

- `supervision_causality_ablation` (`P1`, `ready-for-intake`): Is the lift from state-prefix supervision, final-answer supervision, program-token supervision, or data filtering?
- `diagnostics_failure_slicing_template` (`P1`, `ready-for-intake`): Which failure slices should every major experiment report before influencing strategy?
- `pressure_diagnostic_preflight` (`P2`, `program-seed`): Can cheap token-pressure and execution-pressure probes predict when an expensive run is worth doing?

### Reliability And Safety

- Proposals: 10

- `visible_abstention_selector_benchmark` (`P0`, `ready-for-intake`): Can a selector improve precision without hiding failures behind an unreported commit-rate drop?
- `expected_output_free_counterexample_tests` (`P0`, `ready-for-intake`): Can acquired tests distinguish candidates without requiring hidden expected outputs?
- `deployable_gap_scorecards` (`P0`, `ready-for-intake`): Which oracle-ceiling results remain promising after visible-only constraints are made explicit?
- `adapter_free_repro_manifest_audit` (`P0`, `infrastructure`): Can every trained-run experiment be audited without committing adapters or checkpoints?
- `hidden_label_boundary_linter` (`P0`, `infrastructure`): Can repository checks force every new result to label oracle-only and deployable evidence boundaries?
- `noisy_tool_controller_stress` (`P1`, `program-seed`): Do process policies still help when tools are flaky, delayed, or misleading?
- `hard_negative_training_transfer` (`P1`, `program-seed`): Does training on hard negatives improve real selection, or does it overfit visible failure artifacts?
- `reproducibility_scorecards_high_impact` (`P1`, `infrastructure`): Which high-impact experiments are reproducible enough to guide future programs?
- `proposal_deduplication_audit` (`P1`, `infrastructure`): Can the repo detect repeated experiment ideas before they become new runs?
- `failure_mined_curriculum_generator` (`P2`, `program-seed`): Can failure slices automatically generate new training cases that reduce repeated errors?

### Collective Experimentation Infrastructure

- Proposals: 8

- `hidden_label_boundary_linter` (`P0`, `infrastructure`): Can repository checks force every new result to label oracle-only and deployable evidence boundaries?
- `agent_evidence_navigation_test` (`P0`, `infrastructure`): Can a new agent find relevant prior evidence faster using repository indexes than by raw search alone?
- `operator_card_schema_probe` (`P1`, `ready-for-intake`): What operator-card fields make inventories reusable across experiments?
- `diagnostics_failure_slicing_template` (`P1`, `ready-for-intake`): Which failure slices should every major experiment report before influencing strategy?
- `reproducibility_scorecards_high_impact` (`P1`, `infrastructure`): Which high-impact experiments are reproducible enough to guide future programs?
- `proposal_deduplication_audit` (`P1`, `infrastructure`): Can the repo detect repeated experiment ideas before they become new runs?
- `static_research_navigation_dashboard` (`P1`, `infrastructure`): Can a static dashboard help humans and agents choose better next experiments than Markdown indexes alone?
- `evidence_branching_ui_probe` (`P2`, `program-seed`): What interface lets a researcher branch from one result into controls, replications, or new programs fastest?

### Data Generation And Synthetic Curriculum Design

- Proposals: 2

- `synthetic_curriculum_transfer_bakeoff` (`P1`, `program-seed`): Which synthetic curriculum source transfers best: human-designed, model-generated, or failure-mined?
- `failure_mined_curriculum_generator` (`P2`, `program-seed`): Can failure slices automatically generate new training cases that reduce repeated errors?

### Human-Agent Research Interfaces

- Proposals: 2

- `static_research_navigation_dashboard` (`P1`, `infrastructure`): Can a static dashboard help humans and agents choose better next experiments than Markdown indexes alone?
- `evidence_branching_ui_probe` (`P2`, `program-seed`): What interface lets a researcher branch from one result into controls, replications, or new programs fastest?

### Multimodal And Embodied Small Models

- Proposals: 2

- `multimodal_visual_table_transform_probe` (`P1`, `program-seed`): Do executable intermediates help small models transform visual table inputs?
- `ui_state_tool_execution_probe` (`P2`, `program-seed`): Can process-control policies transfer from text/table tools to UI-state observations?

### On-Device And Latency-Constrained Agents

- Proposals: 2

- `latency_ceiling_selector_rerun` (`P1`, `program-seed`): Which selector or tool-control gains survive strict latency, token, and call ceilings?
- `compressed_memory_budget_probe` (`P2`, `program-seed`): Can memory remain useful when context and storage are severely compressed?

### Small-Model Collaboration

- Proposals: 2

- `multi_model_role_ensemble_pool` (`P1`, `program-seed`): Can generator, verifier, critic, and evidence-acquirer roles outperform a single small-model policy under fixed total budget?
- `committee_verifier_critic_loop` (`P2`, `program-seed`): Can a committee decide when critique or verification is worth another step?

## Queue

| Priority | Status | Effort | Proposal | Programs | Question | Next step | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | ready-for-intake | medium | `visible_abstention_selector_benchmark` | [Evidence-Conditioned Selection](../research_programs/evidence_conditioned_selection/charter.md), [Reliability And Safety](../research_programs/reliability_and_safety/charter.md) | Can a selector improve precision without hiding failures behind an unreported commit-rate drop? | Run idea intake against selector and oracle-ceiling anchor experiments. | [source](../research_programs/evidence_conditioned_selection/backlog.md) |
| P0 | ready-for-intake | medium | `expected_output_free_counterexample_tests` | [Active Evidence Acquisition](../research_programs/active_evidence_acquisition/charter.md), [Reliability And Safety](../research_programs/reliability_and_safety/charter.md) | Can acquired tests distinguish candidates without requiring hidden expected outputs? | Start from active counterexample and model-in-loop counterexample experiments. | [source](../research_programs/active_evidence_acquisition/backlog.md) |
| P1 | ready-for-intake | medium | `independent_consensus_vs_selector_pool` | [Evidence-Conditioned Selection](../research_programs/evidence_conditioned_selection/charter.md), [Algorithmic Memory And Retrieval](../research_programs/algorithmic_memory_and_retrieval/charter.md) | When does independent implementation consensus beat a trained selector on the same candidate pool? | Use existing independent retrieval and counterexample selection anchors as prior evidence. | [source](../knowledge/research_roadmap.md) |
| P0 | ready-for-intake | small | `deployable_gap_scorecards` | [Evidence-Conditioned Selection](../research_programs/evidence_conditioned_selection/charter.md), [Reliability And Safety](../research_programs/reliability_and_safety/charter.md) | Which oracle-ceiling results remain promising after visible-only constraints are made explicit? | Seed from selector, code ABI, and coverage-gate reports. | [source](../research_programs/evidence_conditioned_selection/backlog.md) |
| P0 | ready-for-intake | large | `typed_vs_latent_vs_text_compiler_suite` | [Structured Execution And Compilers](../research_programs/structured_execution_and_compilers/charter.md), [Benchmark Generalization](../research_programs/benchmark_generalization/charter.md) | Which structured representation causes generalization: typed bytecode, latent slots, text programs, or execution feedback? | Use compiler, latent executor, and structured slot anchors to define the suite. | [source](../research_programs/structured_execution_and_compilers/backlog.md) |
| P0 | ready-for-intake | medium | `compiler_shift_replication_grid` | [Structured Execution And Compilers](../research_programs/structured_execution_and_compilers/charter.md), [Benchmark Generalization](../research_programs/benchmark_generalization/charter.md) | Which compiler-style gains survive seed, length, operator, and paraphrase shifts? | Pick three anchor experiments from the compiler program scorecard. | [source](../research_programs/benchmark_generalization/backlog.md) |
| P1 | ready-for-intake | medium | `supervision_causality_ablation` | [Structured Execution And Compilers](../research_programs/structured_execution_and_compilers/charter.md), [Interpretability And Diagnostics](../research_programs/interpretability_and_diagnostics/charter.md) | Is the lift from state-prefix supervision, final-answer supervision, program-token supervision, or data filtering? | Start from structured slot and dense supervision ladder anchors. | [source](../research_programs/structured_execution_and_compilers/backlog.md) |
| P0 | ready-for-intake | medium | `memory_mode_comparison_suite` | [Algorithmic Memory And Retrieval](../research_programs/algorithmic_memory_and_retrieval/charter.md), [Benchmark Generalization](../research_programs/benchmark_generalization/charter.md) | Which memory use is actually useful: prompt context, constraints, tests, candidates, or failure cases? | Use independent retrieval and feature-factorized rule experiments as anchors. | [source](../research_programs/algorithmic_memory_and_retrieval/backlog.md) |
| P1 | program-seed | medium | `constraint_memory_counterexample_generator` | [Algorithmic Memory And Retrieval](../research_programs/algorithmic_memory_and_retrieval/charter.md), [Active Evidence Acquisition](../research_programs/active_evidence_acquisition/charter.md) | Can retrieved failures become counterexample generators instead of answer hints? | Run related-work discovery for counterexample memory and active acquisition. | [source](../research_programs/algorithmic_memory_and_retrieval/backlog.md) |
| P1 | ready-for-intake | small | `operator_card_schema_probe` | [Operator And Skill Inventories](../research_programs/operator_and_skill_inventories/charter.md), [Collective Experimentation Infrastructure](../research_programs/collective_experimentation_infrastructure/charter.md) | What operator-card fields make inventories reusable across experiments? | Sample operator inventory experiments and extract common fields. | [source](../research_programs/operator_and_skill_inventories/backlog.md) |
| P1 | ready-for-intake | medium | `operator_bank_shortlister_scaling` | [Operator And Skill Inventories](../research_programs/operator_and_skill_inventories/charter.md), [Evidence-Conditioned Selection](../research_programs/evidence_conditioned_selection/charter.md) | Can a shortlister preserve selected accuracy as operator banks grow? | Use inventory shortlister and scaling-stress experiments as anchors. | [source](../research_programs/operator_and_skill_inventories/backlog.md) |
| P2 | program-seed | medium | `active_disambiguation_type_collision` | [Operator And Skill Inventories](../research_programs/operator_and_skill_inventories/charter.md), [Active Evidence Acquisition](../research_programs/active_evidence_acquisition/charter.md) | Can active probes separate operators that share type signatures but differ semantically? | Identify collisions from current operator inventory runs. | [source](../research_programs/operator_and_skill_inventories/backlog.md) |
| P0 | ready-for-intake | medium | `acquisition_policy_comparison_pool` | [Active Evidence Acquisition](../research_programs/active_evidence_acquisition/charter.md), [Process Control And Tool Use](../research_programs/process_control_and_tool_use/charter.md) | Which acquisition rule wins when uncertainty, disagreement, information gain, and learned policies share one pool? | Derive pool features from active trace policy and adaptive evidence budget experiments. | [source](../research_programs/active_evidence_acquisition/backlog.md) |
| P1 | program-seed | medium | `family_aware_evidence_policy` | [Active Evidence Acquisition](../research_programs/active_evidence_acquisition/charter.md), [Benchmark Generalization](../research_programs/benchmark_generalization/charter.md) | Can acquisition policies adapt to date/time, numeric, table, and code task families without overfitting? | Use current active evidence experiments to define feature families. | [source](../research_programs/active_evidence_acquisition/backlog.md) |
| P0 | ready-for-intake | medium | `stop_more_controller_visible_labels` | [Process Control And Tool Use](../research_programs/process_control_and_tool_use/charter.md), [Evidence-Conditioned Selection](../research_programs/evidence_conditioned_selection/charter.md) | Can a controller decide when to gather more evidence using only visible training labels? | Anchor on adaptive tool controller and adaptive evidence budget policy experiments. | [source](../research_programs/process_control_and_tool_use/backlog.md) |
| P1 | program-seed | medium | `noisy_tool_controller_stress` | [Process Control And Tool Use](../research_programs/process_control_and_tool_use/charter.md), [Reliability And Safety](../research_programs/reliability_and_safety/charter.md) | Do process policies still help when tools are flaky, delayed, or misleading? | Use process-control experiments with explicit tool states as anchors. | [source](../research_programs/process_control_and_tool_use/backlog.md) |
| P0 | ready-for-intake | large | `posttraining_method_shared_substrate` | [Posttraining And Adaptation](../research_programs/posttraining_and_adaptation/charter.md), [Benchmark Generalization](../research_programs/benchmark_generalization/charter.md) | How do SFT, DPO, process distillation, and DAgger compare on one candidate/evidence substrate? | Use DPO, DAgger, and distillation experiments to define shared data. | [source](../research_programs/posttraining_and_adaptation/backlog.md) |
| P1 | program-seed | medium | `hard_negative_training_transfer` | [Posttraining And Adaptation](../research_programs/posttraining_and_adaptation/charter.md), [Reliability And Safety](../research_programs/reliability_and_safety/charter.md) | Does training on hard negatives improve real selection, or does it overfit visible failure artifacts? | Start from offline hard-negative and constrained coverage DPO anchors. | [source](../research_programs/posttraining_and_adaptation/backlog.md) |
| P0 | infrastructure | small | `adapter_free_repro_manifest_audit` | [Posttraining And Adaptation](../research_programs/posttraining_and_adaptation/charter.md), [Reliability And Safety](../research_programs/reliability_and_safety/charter.md) | Can every trained-run experiment be audited without committing adapters or checkpoints? | Use artifact manifest index and readiness matrix to prioritize missing manifests. | [source](../research_programs/posttraining_and_adaptation/backlog.md) |
| P1 | ready-for-intake | small | `diagnostics_failure_slicing_template` | [Interpretability And Diagnostics](../research_programs/interpretability_and_diagnostics/charter.md), [Collective Experimentation Infrastructure](../research_programs/collective_experimentation_infrastructure/charter.md) | Which failure slices should every major experiment report before influencing strategy? | Apply to one compiler, one selector, and one posttraining anchor. | [source](../research_programs/interpretability_and_diagnostics/backlog.md) |
| P2 | program-seed | medium | `pressure_diagnostic_preflight` | [Interpretability And Diagnostics](../research_programs/interpretability_and_diagnostics/charter.md), [Process Control And Tool Use](../research_programs/process_control_and_tool_use/charter.md) | Can cheap token-pressure and execution-pressure probes predict when an expensive run is worth doing? | Use token-pressure and tool-control experiments as anchors. | [source](../research_programs/interpretability_and_diagnostics/backlog.md) |
| P0 | infrastructure | small | `hidden_label_boundary_linter` | [Reliability And Safety](../research_programs/reliability_and_safety/charter.md), [Collective Experimentation Infrastructure](../research_programs/collective_experimentation_infrastructure/charter.md) | Can repository checks force every new result to label oracle-only and deployable evidence boundaries? | Prototype against high-impact selector and oracle-ceiling reports. | [source](../research_programs/reliability_and_safety/backlog.md) |
| P1 | infrastructure | medium | `reproducibility_scorecards_high_impact` | [Reliability And Safety](../research_programs/reliability_and_safety/charter.md), [Collective Experimentation Infrastructure](../research_programs/collective_experimentation_infrastructure/charter.md) | Which high-impact experiments are reproducible enough to guide future programs? | Use experiment readiness and claim evidence counts to select anchors. | [source](../research_programs/reliability_and_safety/backlog.md) |
| P0 | infrastructure | small | `agent_evidence_navigation_test` | [Collective Experimentation Infrastructure](../research_programs/collective_experimentation_infrastructure/charter.md) | Can a new agent find relevant prior evidence faster using repository indexes than by raw search alone? | Use existing discovery workflow and related-work CLI. | [source](../research_programs/collective_experimentation_infrastructure/backlog.md) |
| P1 | infrastructure | small | `proposal_deduplication_audit` | [Collective Experimentation Infrastructure](../research_programs/collective_experimentation_infrastructure/charter.md), [Reliability And Safety](../research_programs/reliability_and_safety/charter.md) | Can the repo detect repeated experiment ideas before they become new runs? | Generate cases from future queue items and recent scaffolded ideas. | [source](../research_programs/collective_experimentation_infrastructure/backlog.md) |
| P1 | program-seed | medium | `multimodal_visual_table_transform_probe` | `multimodal_and_embodied_small_models`, [Structured Execution And Compilers](../research_programs/structured_execution_and_compilers/charter.md) | Do executable intermediates help small models transform visual table inputs? | Run intake using Foofah and table-transform anchors. | [source](../knowledge/future_program_seeds.md) |
| P2 | program-seed | medium | `ui_state_tool_execution_probe` | `multimodal_and_embodied_small_models`, [Process Control And Tool Use](../research_programs/process_control_and_tool_use/charter.md) | Can process-control policies transfer from text/table tools to UI-state observations? | Prototype with static UI states before live browser control. | [source](../knowledge/future_program_seeds.md) |
| P1 | program-seed | medium | `synthetic_curriculum_transfer_bakeoff` | `data_generation_and_synthetic_curriculum_design`, [Benchmark Generalization](../research_programs/benchmark_generalization/charter.md) | Which synthetic curriculum source transfers best: human-designed, model-generated, or failure-mined? | Reuse bridge, recombination, and held-out primitive split patterns. | [source](../knowledge/future_program_seeds.md) |
| P2 | program-seed | medium | `failure_mined_curriculum_generator` | `data_generation_and_synthetic_curriculum_design`, [Reliability And Safety](../research_programs/reliability_and_safety/charter.md) | Can failure slices automatically generate new training cases that reduce repeated errors? | Start with diagnostic failure-slicing template once available. | [source](../knowledge/future_program_seeds.md) |
| P1 | program-seed | medium | `multi_model_role_ensemble_pool` | `small_model_collaboration`, [Evidence-Conditioned Selection](../research_programs/evidence_conditioned_selection/charter.md) | Can generator, verifier, critic, and evidence-acquirer roles outperform a single small-model policy under fixed total budget? | Use consensus and selector anchors to define the shared pool. | [source](../knowledge/future_program_seeds.md) |
| P2 | program-seed | medium | `committee_verifier_critic_loop` | `small_model_collaboration`, [Process Control And Tool Use](../research_programs/process_control_and_tool_use/charter.md) | Can a committee decide when critique or verification is worth another step? | Start from STOP/MORE controller and consensus experiments. | [source](../knowledge/future_program_seeds.md) |
| P1 | program-seed | small | `latency_ceiling_selector_rerun` | `on_device_and_latency_constrained_agents`, [Process Control And Tool Use](../research_programs/process_control_and_tool_use/charter.md) | Which selector or tool-control gains survive strict latency, token, and call ceilings? | Pick an existing adaptive controller with a clear smoke path. | [source](../knowledge/future_program_seeds.md) |
| P2 | program-seed | medium | `compressed_memory_budget_probe` | `on_device_and_latency_constrained_agents`, [Algorithmic Memory And Retrieval](../research_programs/algorithmic_memory_and_retrieval/charter.md) | Can memory remain useful when context and storage are severely compressed? | Use memory mode comparison suite once its schema exists. | [source](../knowledge/future_program_seeds.md) |
| P1 | infrastructure | medium | `static_research_navigation_dashboard` | `human_agent_research_interfaces`, [Collective Experimentation Infrastructure](../research_programs/collective_experimentation_infrastructure/charter.md) | Can a static dashboard help humans and agents choose better next experiments than Markdown indexes alone? | Start with generated data files already produced by make catalog. | [source](../knowledge/future_program_seeds.md) |
| P2 | program-seed | medium | `evidence_branching_ui_probe` | `human_agent_research_interfaces`, [Collective Experimentation Infrastructure](../research_programs/collective_experimentation_infrastructure/charter.md) | What interface lets a researcher branch from one result into controls, replications, or new programs fastest? | Use high-impact claim evidence as the first anchor set. | [source](../knowledge/future_program_seeds.md) |
