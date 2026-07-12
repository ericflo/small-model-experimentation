# Source

This directory holds the pinned vLLM runner, copied procedural generators, long-horizon thought/scoring
operations, split builder, selector, and statistics. Training and checkpoint merging remain under `scripts/`
because they use a separate Transformers environment. Training code must not import held-family registries,
and no source file may import benchmark internals.
