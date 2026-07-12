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

- [qwen35_4b_jacobian_value_transport](../../experiments/qwen35_4b_jacobian_value_transport/reports/report.md)
  (unclaimed while the ledger re-grade is open): an averaged token-Jacobian
  coordinate is strongly writable at one late layer but does not transport.
  On an untouched 24-item confirmation split, the layer-24 intervention changed
  the direct concept report on 18/24 items (75%; random 0/24, logit-lens 5/24),
  while an arbitrary prompt-local consequence changed on 0/24 at every tested
  layer. Its direct margin crossed zero with intervention strength while the
  consequence margin stayed flat. G0 correctly cancelled thought-prefix value
  and task patching. This separates three properties that prior probe work often
  conflated: decodability, local writability, and downstream transport. Scope:
  the random control was not exact realized-delta-norm matched, so direct J
  specificity remains provisional; the transport failure and adjacency failure
  are unaffected.

- [qwen35_4b_context_local_jacobian_clamp](../../experiments/qwen35_4b_context_local_jacobian_clamp/reports/report.md)
  (unclaimed while the ledger re-grade is open; terminal `INVALID_CONTROL`): the
  corrected early selected-token clamp produced the full semantic-transport
  signature on 48 untouched mappings. All-24 J changed direct key and mapped
  digit on 48/48; pair J changed the digit on 47/48; wrong-donor J changed 48/48
  to its own digit; concept logit lens and random changed 0/48. Full donors also
  exposed a sharp causal window: four bands through 16–20 were 24/24, bands
  20–24 and 24–28 were inert. Yet one of 96 random rows missed the frozen
  realized-norm tolerance (1.155e-5 > 1e-5), and bf16 rounding left up to 5.7%
  realized J-span projection despite pre-cast orthogonality. The result cannot be
  promoted. It sharply prioritizes a fresh quantization-aware control
  replication and keeps native-thinking continuation gated.

- [qwen35_4b_jacobian_transport_control_replication](../../experiments/qwen35_4b_jacobian_transport_control_replication/reports/report.md)
  (unclaimed while the ledger re-grade is open; terminal
  `REPLICATED_J_TRANSPORT`): fresh exact-control replication resolves the parent
  invalidity. All-24 J changed 48/48 direct keys and 48/48 separately computed
  mapped digits. Two independent random arms and the concept logit lens changed
  0/48; wrong-donor J produced its own key/digit 48/48 and the registered target
  0/48; pair J reached 46/48 consequences. All 480 calibration and 960
  confirmation post-bf16 control-layer rows met relative norm <=1e-5 and
  realized J-span projection <=0.01. Both paired bootstrap intervals were
  [1,1]. This establishes an oracle, context-local, causally consumed concept
  state on the procedural lookup substrate and unlocks native-thought work. It
  does not install capability: target donor identity remains supplied.

- [qwen35_4b_native_thought_jacobian_value_transport](../../experiments/qwen35_4b_native_thought_jacobian_value_transport/reports/report.md)
  (unclaimed; terminal `NO_NATURAL_SEAM`): the first licensed native-thought
  successor stopped before value fitting. All 48/48 frozen 160-token traces on
  16 fresh identifiable tasks hit the thought cap without natural close; parse,
  success, and mixed-task counts were zero. Model smoke also found up to 0.0625
  historical-token activation drift across different suffix lengths. This is not
  evidence against J-space value: the natural answer seam was unreachable. It
  requires a fresh cap-selection/confirmation experiment and dynamic per-length
  control geometry before causal patching.

- [qwen35_4b_native_thought_seam_budget_ladder](../../experiments/qwen35_4b_native_thought_seam_budget_ladder/reports/report.md)
  (unclaimed; terminal `NO_BUDGET_SELECTED`): the separate frozen
  256/512/1024 selector also found no natural seam. All 48/48 selection traces
  used all 1,024 thought tokens, so natural close, parse, usable-prefix, and
  mixed-task counts were zero at every paired rung; the 24-task confirmation
  split remained unopened. All rows passed the cached-forward audit. A
  post-decision token diagnostic found 0/48 exact short-period 256-token tail
  loops, so loop breaking is not licensed. This still precedes J-space value.
  It closes the natural-cap branch on this workload and redirects the next test
  to an explicitly deployed forced-commit policy with C51 parse/headroom gates
  and per-length post-bf16 controls.

- [qwen35_4b_forced_commit_jacobian_value_transport](../../experiments/qwen35_4b_forced_commit_jacobian_value_transport/reports/report.md)
  (unclaimed; terminal `FORCED_COMMIT_SEAM_FAIL`): explicitly deploying the
  close token did not create a usable emission seam. Across 48 paired traces,
  forced parse was 6/48, 8/48, and 9/48 at caps 256/512/1024; exact success was
  1/48 at every cap; only one task mixed outcomes; and 41--46/48 answers hit the
  16-token answer cap. A post-decision EOS-tolerant parser raised parse only to
  7/11/10 and correctness to 1/2/2, leaving every gate failed. Confirmation,
  value, and causal splits stayed sealed. This localizes the next interface
  change: close is a mode delimiter, not an answer slot. A fresh syntax-only
  `First:` slot may test semantic alias choice while close-only remains control.
