# Source

This directory contains the self-contained procedural atom generators, task split builder,
thought-only inference/scoring utilities, G0 statistics, and pinned vLLM runner. G0 failed, so the
full selector and training path was not implemented or executed; the full-stage guard refuses it.

The calibration/train registry must not import held-family modules. Held-family code is loaded only
by the evaluation path. No file may import or reference `benchmarks/` internals.
