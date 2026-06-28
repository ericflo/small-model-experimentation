# Initial Claims

These narrative claims are preserved for readability. The structured, validation-backed ledger is [claim_ledger.json](claim_ledger.json), with generated navigation in [index.md](index.md).

## C1: Structured intermediates can improve small-model reliability

Status: `Confirmed`

Evidence:

- [qwen_structural_latent_compiler_expansion](../../experiments/qwen_structural_latent_compiler_expansion/reports/structural_latent_compiler_expansion_report.md)
- [qwen35_4b_foofah_selective_program_fallback](../../experiments/qwen35_4b_foofah_selective_program_fallback/reports/report.md)
- [qwen35_4b_operator_inventory_search_pilot](../../experiments/qwen35_4b_operator_inventory_search_pilot/reports/qwen35_4b_operator_inventory_search_pilot_report.md)

Implication: create structured-output and executable-evidence baselines for future direct-answer tasks.

## C2: Candidate coverage does not imply deployable accuracy

Status: `Confirmed`

Evidence:

- [qwen35_4b_retrieval_adapt_verify_scale](../../experiments/qwen35_4b_retrieval_adapt_verify_scale/reports/final_report.md)
- [qwen35_4b_foofah_selective_program_fallback](../../experiments/qwen35_4b_foofah_selective_program_fallback/reports/report.md)

Implication: every candidate-pool experiment should report oracle coverage, deployable selection, false-pass rate, and abstention separately.

## C3: Naive retrieved skill cards are not enough

Status: `Negative`

Evidence:

- [qwen_verified_skill_memory_rag](../../experiments/qwen_verified_skill_memory_rag/reports/qwen_verified_skill_memory_rag_report.md)

Implication: future memory work should test memory as constraints, tests, candidates, or verifier inputs, not only prompt context.

## C4: The repository itself must be program-oriented

Status: `Confirmed`

Evidence:

- The imported archive required a program scaffold to avoid becoming a two-track snapshot.

Implication: new experiments should attach to programs, and new durable uncertainties should create programs.
