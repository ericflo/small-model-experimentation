# Qwen Structural Latent Compiler Expansion

This experiment tests whether an executable latent program compiler can be expanded structurally from short chains to longer chains without using beam search, candidate reranking, or text program generation.

The model receives a modular-arithmetic prompt plus fixed latent register positions. A Qwen backbone reads the prompt, a direct compiler head predicts an initial value and a sequence of typed operations/arguments, and a differentiable executor supervises the resulting latent program.

Large checkpoints are stored outside this directory under:

`/workspace/large_artifacts/qwen_structural_latent_compiler_expansion/checkpoints`

