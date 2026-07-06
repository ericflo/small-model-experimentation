# Evidence

## Seed Experiments

- [qwen_structural_compiler_attribution_ablation](../../experiments/qwen_structural_compiler_attribution_ablation/reports/structural_compiler_attribution_ablation_report.md)
- [qwen35_4b_opsd_pressure_locality_audit](../../experiments/qwen35_4b_opsd_pressure_locality_audit/reports/final_report.md)
- [qwen35_4b_reliability_exec_opsd_audit](../../experiments/qwen35_4b_reliability_exec_opsd_audit/reports/final_report.md)
- [qwen_full_table_consistency_reranker](../../experiments/qwen_full_table_consistency_reranker/reports/qwen_full_table_consistency_reranker_report.md)

## Current Read

Diagnostics should become standard infrastructure. They are how future agents avoid retesting the same mistaken explanations.

- [qwen35_4b_probe_to_prompt](../../experiments/qwen35_4b_probe_to_prompt/reports/report.md) (claim C30): EXTERNALIZING the latent readout (decode C19's first-op probe -> inject as a PROMPT hint) elicits deployable depth-2 (oracle_full 6x) where steering (C20) was inert -- the first test-time lever to move the wall. But the decodable op-TYPE only narrows sampling; the PARAMETER is the deployable bottleneck, so the type-only probe nets to zero. Graded by depth (fades at depth-3 thread). Layer-0 leak control at chance.

- [qwen35_4b_probe_the_parameter](../../experiments/qwen35_4b_probe_the_parameter/reports/report.md) (claim C31): sharp localization of C30 -- the op-TYPE is MODEL-LATENT (residual probe 0.41 > external-I/O baseline 0.27) but the PARAMETER is SURFACE-READABLE (external I/O 0.53 >= probe 0.49; and surface-hint deploys 0.027 > probe-hint 0.014). The forward pass computes the type (elicitable) but only reads the param off surface I/O. Real surface control = external classifier on raw I/O features (the last-token layer-0 probe is degenerate under RoPE).
