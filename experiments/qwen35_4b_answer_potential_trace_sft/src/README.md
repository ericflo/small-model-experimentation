# Source

This directory will contain the self-contained procedural atom generators, task split builder,
thought-only inference/scoring utilities, deterministic selector, statistics, and the pinned vLLM
runner. Training code remains under `scripts/` because it has a separate Transformers environment.

The calibration/train registry must not import held-family modules. Held-family code is loaded only
by the evaluation path. No file may import or reference `benchmarks/` internals.
