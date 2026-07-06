export const meta = {
  name: 'extract-chart-specs',
  description: 'Per-experiment: extract verified chart specs from result data for native site charts',
  phases: [
    { title: 'Extract', detail: 'one agent per experiment reads data files and drafts chart specs' },
    { title: 'Verify', detail: 'adversarial number check against the cited sources' },
  ],
}

const EXP_IDS = ["adaptive_cognitive_kernel", "belief_filter_executor", "bridge_dose_recombination_curriculum", "counterexample_rule_repair", "cyclic_transition_ladder", "dense_latent_query_executor", "dense_supervision_ladder", "dense_teacher_distillation", "end_to_end_structured_slot_executor", "episodic_echo_ttt", "execution_conditioned_repair", "factor_recombination_ladder", "feature_factorized_rule_diversity", "joint_register_executor", "latent_executor", "learned_sparse_slot_executor", "query_filter_executor", "qwen35_4b_active_counterexample_trace_selection", "qwen35_4b_adaptive_evidence_budget_policy", "qwen35_4b_adaptive_tool_controller", "qwen35_4b_balanced_discriminative_bridge", "qwen35_4b_bucket_belief_probe_ranker", "qwen35_4b_code_abi_compiler_heldout_primitive_pilot", "qwen35_4b_code_abi_oracle_coverage_ladder", "qwen35_4b_constrained_coverage_dpo", "qwen35_4b_counterexample_directed_dsl", "qwen35_4b_decompose_compose_frontier", "qwen35_4b_deployable_information_ceiling_sweep", "qwen35_4b_depth_wall_anatomy", "qwen35_4b_diversity_keyed_coverage_gate", "qwen35_4b_executable_program_posttraining", "qwen35_4b_foofah_adaptive_program_budget_router", "qwen35_4b_foofah_direct_vs_abi", "qwen35_4b_foofah_ephemeral_program_induction", "qwen35_4b_foofah_external_transform_gate", "qwen35_4b_foofah_program_ensemble_consensus", "qwen35_4b_foofah_program_repair_agent", "qwen35_4b_foofah_program_strategy_portfolio", "qwen35_4b_foofah_selective_program_fallback", "qwen35_4b_foofah_strategy_discovery_live", "qwen35_4b_generator_verifier_gap", "qwen35_4b_graphir_self_repair", "qwen35_4b_humaneval_adaptive_budget", "qwen35_4b_independent_code_abi_coverage_gate", "qwen35_4b_independent_retrieval_consensus", "qwen35_4b_inventory_shortlister_training", "qwen35_4b_joint_shortlister_ladder", "qwen35_4b_learned_active_trace_policy", "qwen35_4b_live_tool_dagger", "qwen35_4b_model_in_loop_counterexamples", "qwen35_4b_neurosymbolic_repl_substrate", "qwen35_4b_offline_hard_negative_coverage_dpo", "qwen35_4b_operator_inventory_scaling_stress", "qwen35_4b_operator_inventory_search_pilot", "qwen35_4b_opsd_pressure_locality_audit", "qwen35_4b_oracle_distilled_semantic_verifier", "qwen35_4b_oracle_probe_synthesis_mdp", "qwen35_4b_oracle_process_grpo", "qwen35_4b_overthinking_content_ladder", "qwen35_4b_passk_coverage_rl", "qwen35_4b_prefix_value_guided_search", "qwen35_4b_real_sample_verify_commit", "qwen35_4b_reliability_exec_opsd_audit", "qwen35_4b_retrieval_adapt_verify_scale", "qwen35_4b_sampler_portfolio_scheduler", "qwen35_4b_sketch_coverage_shift_probe", "qwen35_4b_static_bridge_ceiling_breaker", "qwen35_4b_strategy_token_diversity_lora", "qwen35_4b_substrate_coverage_ladder", "qwen35_4b_thinking_budget_controller", "qwen35_4b_thinking_budget_scaling", "qwen35_4b_thinking_content_vs_compute", "qwen35_4b_thinking_separability_probe", "qwen35_4b_tool_state_policy_lora", "qwen35_4b_trained_vs_frozen_repair_mdp", "qwen35_4b_transform_abi_compiler_pilot", "qwen35_4b_typed_sketch_synthesis", "qwen35_4b_unsaturated_frontier_active_bridge", "qwen35_4b_verified_algorithm_retrieval_adaptation", "qwen35_4b_verified_edit_closure", "qwen35_4b_verifier_guided_self_improvement", "qwen35_4b_verifier_selector_showdown", "qwen_action_conditioned_vm_echo_policy_iteration", "qwen_active_crystallizer_public_gate", "qwen_active_example_acquisition", "qwen_batched_transduction_consistency", "qwen_budgeted_action_value_compiler", "qwen_candidate_conditioned_trace_verifier", "qwen_candidate_trace_verifier", "qwen_checkpoint_scheduled_state_compiler", "qwen_compiler_multiseed_reattribution", "qwen_complete_program_trace_reranker", "qwen_compositional_curriculum_abi", "qwen_constrained_abi_parser", "qwen_context_trace_verifier", "qwen_counterexample_guided_ephemeral_program", "qwen_counterexample_guided_projection", "qwen_counterfactual_episodic_icl", "qwen_counterfactual_icl_public_multiseed", "qwen_counterfactual_trace_preference_distillation", "qwen_crystallized_trace_abi_tournament", "qwen_dense_state_dagger_vm_agent", "qwen_disagreement_probe_program_induction", "qwen_episodic_soft_prompt_task_vectors", "qwen_extrapolation_bound_abi", "qwen_fastweight_hook", "qwen_full_table_consistency_reranker", "qwen_fuyu_vm_grpo_echo", "qwen_hidden_vm_curriculum_repair", "qwen_hidden_vm_mixed_domains", "qwen_hidden_vm_onpolicy_canonical_repair", "qwen_inpolicy_vm_echo_distillation", "qwen_iterative_repair_policy", "qwen_large_abi_nested_compiler", "qwen_latent_beam_program_compiler", "qwen_learned_active_interrogation", "qwen_learned_repair_verifier", "qwen_lora_parser_compiler", "qwen_lora_typed_bytecode_trace_compiler", "qwen_mixed_domain_trace_verifier", "qwen_noisy_row_program_crystallizer", "qwen_numeric_copy_compiler", "qwen_onpolicy_repair_compiler", "qwen_oracle_distilled_acquisition_policy", "qwen_pairwise_table_judge", "qwen_prefix_state_process_verifier", "qwen_program_only_executable_abi", "qwen_progressive_repair_compiler", "qwen_public_prose_abi_gate", "qwen_python_shaped_silent_executor", "qwen_readable_candidate_verifier", "qwen_real_task_abi_coverage_gate", "qwen_recurrent_vm_repair_policy", "qwen_recursive_ephemeral_program_induction", "qwen_recursive_task_decomposition", "qwen_register_structured_runtime", "qwen_register_token_latent_compiler", "qwen_register_trace_refiner", "qwen_search_augmented_rollout_distillation", "qwen_semantic_prefix_value_model", "qwen_shared_parser_compiler", "qwen_slot_repair_distillation", "qwen_slot_stability_compiler", "qwen_span_free_compiler", "qwen_state_ladder_compiler", "qwen_structural_compiler_attribution_ablation", "qwen_structural_latent_compiler_expansion", "qwen_structured_bridge", "qwen_support_contrastive_meta_icl", "qwen_tail_repair_stability_critic", "qwen_teacher_distilled_slot_compiler", "qwen_trace_bootstrap_retention", "qwen_trace_procedure_depth_stress", "qwen_typed_bytecode_expert_iteration", "qwen_verified_skill_memory_rag", "qwen_verifier_guided_slot_repair", "qwen_vm_agent_echo_qlora", "qwen_vm_echo_trace_distillation", "real_transform_abi_gate_with_counterexamples", "rule_family_diversity_scaling", "sampled_query_filter_executor", "sparse_support_memory_executor", "structured_slot_initializer_ladder", "targeted_bridge_allocation", "trace_keyed_symbol_repair"];

const CHART_SCHEMA = {
  type: 'object', required: ['charts'],
  properties: {
    charts: {
      type: 'array', maxItems: 4,
      items: {
        type: 'object',
        required: ['title', 'kind', 'y_label', 'y_format', 'series', 'note', 'source', 'headline'],
        properties: {
          title: { type: 'string', description: 'plain-language chart title, no slugs' },
          kind: { enum: ['bar', 'line'] },
          x_label: { type: 'string' },
          y_label: { type: 'string' },
          y_format: { enum: ['number', 'percent01', 'percent100'] },
          categories: { type: 'array', items: { type: 'string' }, maxItems: 10, description: 'bar charts only: x-axis category labels' },
          series: {
            type: 'array', minItems: 1, maxItems: 5,
            items: {
              type: 'object', required: ['label'],
              properties: {
                label: { type: 'string' },
                values: { type: 'array', items: { type: 'number' }, maxItems: 10, description: 'bar: one value per category' },
                points: { type: 'array', items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }, maxItems: 40, description: 'line: [x,y] pairs, x numeric' },
              },
            },
          },
          note: { type: 'string', description: 'one plain sentence: the takeaway a reader should get (<=140 chars)' },
          source: { type: 'string', description: "where every number comes from: repo-relative file path(s) inside the experiment dir, or 'README table' / 'report table'" },
          headline: { type: 'boolean', description: 'exactly one chart per experiment is the headline' },
        },
      },
    },
  },
}

function extractPrompt(id) {
  return `Repo: /home/ericflo/Development/small-model-experimentation. Experiment: experiments/${id}/

You are extracting CHART SPECS so the research site can render this experiment's results as clean native charts (replacing ugly matplotlib PNGs). Work in this order:
1. Read the README and the primary report in the experiment dir to understand what the experiment found.
2. Find the numbers behind that finding in the experiment's own artifacts: runs/**/*.json, reports/**/*.csv|json, analysis/**, checkpoint manifests — or, when machine files lack them, markdown tables in the README/report.
3. Produce 1-4 chart specs that TEACH the result to a reader. The first/most important one gets headline: true (exactly one headline when any chart exists). Prefer: the headline comparison the finding cites; then supporting curves (budget/depth/round/step sweeps), ablations, or per-condition breakdowns.

HARD RULES:
- Every plotted number must be copied exactly from a cited source (file path or 'README table'/'report table'). NEVER read numbers off PNG figures. NEVER estimate or invent.
- kind 'bar' for categorical comparisons (conditions, arms, ablations): give 'categories' (<=10) and each series 'values' aligned to them (<=5 series; total bars <= 20).
- kind 'line' for sweeps over a numeric x (thinking budget, composition depth, training round, dose, steps): each series has 'points' [[x,y],...] sorted by x (<=40 points).
- y_format: 'percent01' when values are rates in 0-1; 'percent100' when already 0-100; else 'number'. All series in one chart share the y scale — do not mix metrics with different scales in one chart.
- Labels are plain language ('frozen', 'self-trained', 'greedy@1'), not file keys. Titles state what is compared, notes state what it means.
- Small is beautiful: 2-6 bars beats 20; pick the conditions the report itself highlights.
- If the experiment genuinely has no meaningful numeric result series (pure infra/source-only), return {"charts": []}.

Your structured output is consumed by a build script — return only the object.`
}

function verifyPrompt(id, charts) {
  return `Repo: /home/ericflo/Development/small-model-experimentation. Experiment: experiments/${id}/

You are the NUMBER VERIFIER for chart specs extracted for the research site. For EVERY series value/point in the specs below, open the cited source (file in the experiment dir, or the README/report markdown table) and confirm the number appears there exactly or as a trivial transform (0.224 <-> 22.4%; 43.2 <-> '43.2%'). Also check: series lengths match categories; y_format matches the value scale; x points sorted; labels/titles/notes are faithful to what the source measures (no overclaiming); exactly one headline chart.

- Fix any wrong value, label, y_format, or note USING ONLY the sources.
- DROP any chart whose numbers you cannot locate, and any chart that misrepresents the experiment.
- Keep charts you verified. Return the FINAL corrected charts array (possibly empty). Default to dropping when uncertain.

SPECS TO VERIFY:
${JSON.stringify(charts, null, 1)}`
}

phase('Extract')
const results = await pipeline(
  EXP_IDS,
  id => agent(extractPrompt(id), { label: `extract:${id}`, phase: 'Extract', schema: CHART_SCHEMA, effort: 'medium' }),
  (draft, id) => {
    const charts = draft && draft.charts ? draft.charts : []
    if (!charts.length) return { id, charts: [] }
    return agent(verifyPrompt(id, charts), { label: `verify:${id}`, phase: 'Verify', schema: CHART_SCHEMA, effort: 'medium' })
      .then(v => ({ id, charts: v && v.charts ? v.charts : [] }))
  }
)

const byId = {}
let total = 0
for (const entry of results.filter(Boolean)) {
  byId[entry.id] = { charts: entry.charts }
  total += entry.charts.length
}
log(`${Object.keys(byId).length} experiments processed, ${total} verified charts`)
return { experiments: byId }