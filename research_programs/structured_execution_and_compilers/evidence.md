# Evidence

## Seed Experiments

- [qwen_structural_latent_compiler_expansion](../../experiments/qwen_structural_latent_compiler_expansion/reports/structural_latent_compiler_expansion_report.md)
- [qwen_compiler_multiseed_reattribution](../../experiments/qwen_compiler_multiseed_reattribution/reports/qwen_compiler_multiseed_reattribution_report.md)
- [qwen_typed_bytecode_expert_iteration](../../experiments/qwen_typed_bytecode_expert_iteration/reports/qwen_typed_bytecode_expert_iteration_paper.md)
- [latent_executor](../../experiments/latent_executor/README.md)
- [structured_slot_initializer_ladder](../../experiments/structured_slot_initializer_ladder/README.md)

## Key Result

- [qwen35_4b_neurosymbolic_repl_substrate](../../experiments/qwen35_4b_neurosymbolic_repl_substrate/reports/report.md)
  (claim C11): a **fresh, contamination-free** procedural program-synthesis substrate (random primitive
  compositions, held-out-execution graded, oracle-solvable 100%) — a reusable asset for elicitation claims
  with no memorization confound. On it, a neurosymbolic execution-feedback REPL loop did NOT beat
  matched-compute sampling (M2), but self-training on the 4B's own verified solutions banked capability into
  held-out single-shot (M3, +0.095). Extends C1 (executable intermediates) toward execution *feedback* and
  self-correction *training*.

## Current Read

Structured execution is one of the strongest imported signals. The next useful work is not another isolated
positive run; it is controlled comparison of representations and supervision sources — and, per C11, on a
CONTAMINATION-FREE substrate so that gains (and non-gains) are honestly measurable.
