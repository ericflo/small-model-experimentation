# Source

Intentionally empty of engine code. This experiment is an eval-only
measurement intake (the medium-tier think-budget probe): its only model
events run through the trusted aggregate gateway
(`scripts/run_benchmark_aggregate.py` at the repo root), which owns
engine setup, decoding, scoring, and budget policy behind a process
boundary and returns only the whitelisted aggregate fields. The
scaffold's `src/vllm_runner.py` and `tests/test_vllm_runner.py` were
removed on purpose: no code path in this cell may load model weights or
sample tokens outside the gateway.
