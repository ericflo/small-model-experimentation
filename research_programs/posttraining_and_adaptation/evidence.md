# Evidence

## Seed Experiments

- [qwen35_4b_constrained_coverage_dpo](../../experiments/qwen35_4b_constrained_coverage_dpo/reports/final_report.md)
- [qwen35_4b_offline_hard_negative_coverage_dpo](../../experiments/qwen35_4b_offline_hard_negative_coverage_dpo/reports/final_report.md)
- [qwen35_4b_live_tool_dagger](../../experiments/qwen35_4b_live_tool_dagger/reports/report.md)
- [qwen35_4b_oracle_process_grpo](../../experiments/qwen35_4b_oracle_process_grpo/reports/qwen35_4b_oracle_process_grpo_report.md)

## Key Result

- [qwen35_4b_learn_from_failures](../../experiments/qwen35_4b_learn_from_failures/reports/report.md) (claim C29): preference training on the model's OWN failures does NOT close the coverage->deployable gap -- DPO collapses generation (greedy@1 & coverage crash; pre-DPO 2AFC=0.81 verifier but preference-optimizing it destroys the model). The gap closes with MORE SFT: SFT_2x triples greedy@1 (0.037->0.113). Extends prior MBPP DPO work to the controlled depth-3 substrate. Limits: DPO not heavily constrained, single seed.

- [qwen35_4b_bank_the_thoughts](../../experiments/qwen35_4b_bank_the_thoughts/reports/report.md) (claim C28): banking correct decomposition PLANS beats banking ANSWERS on deployable depth-3 -- three fresh QLoRA on matched data (A=prompt->code, T=prompt->plan->code, T_corrupt=mismatched plan); T coverage@16 0.325 vs A 0.200; content-causal (T_corrupt collapses to 0.113, below A); test-time channel (T no-think 0.013). Resolves C26/C27 (thinking helps once the reasoning is banked). Limits: synthetic plans (Phase 2 = model's own thoughts), step-1-think eval incomplete, single seed.

- [qwen35_4b_decompose_compose_frontier](../../experiments/qwen35_4b_decompose_compose_frontier/reports/report.md)
  (claim C12): banking search+interpreter-harvested solutions (QLoRA-SFT, no teacher) that the model could NOT
  monolithically sample EXTENDS its frontier into the weights — monolithic held-out pass@5 0.125→0.237
  (+0.112, ~2.6 SE), depth-3 pass@5 4×. This breaks M4's coverage bound: self-training on data from OUTSIDE
  the sampling support (harvested by tool-augmented search) pulls it into the distribution. Frontier
  extension without a teacher.
- [qwen35_4b_neurosymbolic_repl_substrate](../../experiments/qwen35_4b_neurosymbolic_repl_substrate/reports/report.md)
  (claim C11): the corpus's **first self-training WIN**. QLoRA-SFT on the 4B's OWN 189 execution-verified
  solutions (no teacher) improved held-out fresh single-shot on a contamination-free substrate: think-greedy@1
  0.224→0.319 (+0.095, ~2.2 SE, N=210), pass@5 up (NO diversity collapse), two seeds. This **works where the
  corpus's MBPP self-improvement regressed** (`qwen35_4b_verifier_guided_self_improvement`) — implicating
  contamination/substrate, not the method. (Test-time execution-feedback self-correction, by contrast, did
  NOT beat matched-compute sampling — same experiment, M2.)

## Current Read

Adaptation is useful only when the target behavior is well specified and controls expose whether training
changed the intended mechanism — AND on a substrate where a gain is even measurable. C11 shows honest
self-training (own verified solutions, no teacher) banks capability into single-shot on a CONTAMINATION-FREE
substrate, reversing the corpus's earlier "self-training loses to sample-more" reads that were likely
confounded by contaminated/saturated benchmarks. Priority: scale the self-training loop (expert iteration),
test cross-substrate transfer, and re-run the failed MBPP self-improvement with contamination controls.
