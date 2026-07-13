# Source

Experiment-owned Transformers activation/caching code will live here after the
immutable scientific design is pushed. Bulk vLLM inference is intentionally not
used because the measurement requires layer activations and cache-local hooks.

The implementation must copy only the required mechanisms into this directory;
it may not import result-bearing parent modules at runtime.
