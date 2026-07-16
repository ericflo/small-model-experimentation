# Source — intentionally empty (gateway-only rule)

This experiment deliberately has NO experiment-local model engine. The
scaffold's `src/vllm_runner.py` (and its test) was removed, not left to
rot: every model execution in this cell goes through the trusted
aggregate gateway `scripts/run_benchmark_aggregate.py` at the repo root,
which is sha256-pinned by the design receipt, the seed-consuming runner,
and the harness. Tier choices are `quick|medium` only, and nothing under
`benchmarks/` is ever read from this experiment — only gateway receipts.

If you are tempted to add engine code here, stop: a local engine would
bypass the benchmark firewall (receipts-only, no benchmark data read)
and break the implementation-signature anchoring to the discovery event
(all six confirmation receipts must carry the discovery summary's exact
benchmark-implementation signature, fail closed).
