# Qwen Structural Compiler Attribution Ablation

**Status:** finished

This standalone experiment tests why a tuned executable latent compiler learns length-24 modular programs.

The arms isolate:

- Copy-based structural expansion.
- Same curriculum with a max-24 compiler from the start.
- Expansion with random initialization for newly introduced slots.
- No-curriculum max-24 training.
- Training only through length 16, then evaluating length 24.

Large checkpoints are stored outside this directory:

`/workspace/large_artifacts/qwen_structural_compiler_attribution_ablation/checkpoints`

