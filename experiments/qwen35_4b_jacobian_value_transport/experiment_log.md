# Qwen3.5-4B Jacobian Value Transport Experiment Log

## 2026-07-12 — intake and design freeze preparation

- Synchronized the clean `main` worktree with `origin/main` before creating the
  experiment.
- Ran related-work discovery for Jacobian transport, think-prefix value, causal
  coordinate patching, and counterfactual reflection.
- Selected `interpretability_and_diagnostics` as the primary program and named
  `qwen35_4b_activation_steering` as the closest near-duplicate.
- Separated final correctness from token-level credit: the design uses common-prefix
  sibling continuations and exact rollout value rather than broadcasting a trace
  label across all thought tokens.
- Added a mandatory Qwen-specific positive control before any value or capability
  claim, plus an immutable design boundary before real-model scientific work.
- No result-bearing model call has occurred in this experiment.

## 2026-07-12 — immutable design boundary

- The design commit was rebased onto concurrent `origin/main` work and finalized
  as `57fe5249e38f1e498e53a63ea4e9a72c0b48e2f0`.
- Frozen preregistration SHA-256:
  `a7e5711236f9c0dd0c39182c1ccad4c881cf18b604247a12f5362faafc627bae`.
- Frozen README SHA-256:
  `119b20dcbb41dbc578cc8aabbbcb7cf65739fc734180654dcfd0f5f69d12fdf0`.
- The run harness now fails closed unless this commit is an ancestor and both
  frozen-file digests match. Scientific GPU work remains unstarted.

## 2026-07-12 — real-model plumbing gate

- Recreated the pinned Transformers environment and verified torch 2.11.0,
  Transformers 5.13.0, flash-linear-attention 0.5.1, and causal-conv1d
  1.6.2.post1.
- Tokenizer audit confirmed the configured think tokens and all 24 positive-
  control concepts as single leading-space tokens.
- The first real targeted pullback fitted four token directions at layers 8,
  16, and 24 from two 64-token prompts with explicit equal causal-pair
  weighting. All directions were finite and nonzero.
- A first coordinate-write attempt exposed that `torch.linalg.pinv` rejects
  bf16 inputs. The implementation now reads coordinates in fp32 and casts only
  the final residual delta back to bf16; a regression test covers this path.
- The rerun passed: 26 CPU tests, 9.25 GiB peak allocated model memory, and a
  nonzero mean patch delta norm of 0.4045 in cache-free full-prefix generation.
  These are plumbing checks and carry no scientific evidence.

## 2026-07-12 — pre-result G0 implementation freeze

- Added the preregistered calibration-only coordinate scale sweep
  `{0.5, 1.0, 2.0, 4.0}` before any positive-control outcome was observed.
- Implemented batched fp32 pseudoinverse coordinate reads, next-token direct and
  downstream-consequence controls, deterministic random/logit/wrong controls,
  and selection-half versus confirmation-half isolation.
- G0 implementation changes will be committed and pushed before fitting the
  64-prompt scientific targeted lens.

## 2026-07-12 — G0 tokenizer preflight stop

- The 64-prompt targeted lens fit completed in 29.3 seconds at 12.3 GB peak;
  all 120 layer/concept directions were finite and nonzero.
- G0 then stopped before writing an outcome artifact because leading-space
  digit strings tokenize as two tokens (`space`, `digit`). No gate result was
  observed or reduced.
- Corrected the contract: direct concepts are scored as their fitted leading-
  space tokens after `Concept:`, while the `Value: ` prefix owns the space and
  the following digit is scored as one bare token.
